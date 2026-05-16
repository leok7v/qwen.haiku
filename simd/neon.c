// SPDX-License-Identifier: MIT
//
// neon.c -- ARM NEON kernel pack.
//
// Two tiers per kernel:
//   _dotprod  -- ARMv8.2-A with `sdot`/vdotq_s32      (Apple Silicon, A76+)
//   _baseline -- ARMv8.0-A ASIMD only                  (Cortex-A53, A72, A73)
//
// Both tiers are compiled into the same binary; simd.c dispatches at
// runtime via getauxval(AT_HWCAP)/sysctlbyname. The +dotprod tier is
// gated with __attribute__((target("+dotprod"))) so the compiler emits
// sdot only inside that function. The dispatcher (compiled at baseline)
// never calls a tier the CPU can't run.
//
// Block layouts come from quants.h (single source of truth).
//
// Algorithm parity: dotprod variants are line-for-line ports of the
// qwen.haiku/llm/neon.c kernels (themselves derived from llama.cpp
// ggml-cpu/arch/arm/quants.c). Baseline variants substitute
// vdotq_s32(acc, a, b) with vmull_s8 + vpadalq_s16:
//
//     int16x8_t lo = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
//     int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
//     acc = vpadalq_s16(acc, lo);
//     acc = vpadalq_s16(acc, hi);
//
// which is bit-identical for int8 inputs (max product 127*127 = 16129
// fits in int16 with margin; pair-add to int32 cannot overflow either).

#include "quants.h"
#include <arm_neon.h>

// ---------------------------------------------------------------------------
// Baseline (no-dotprod) helper: 16-byte int8 dot-and-accumulate into int32x4
// ---------------------------------------------------------------------------
static inline int32x4_t neon_vdot_baseline(int32x4_t acc,
                                           int8x16_t a, int8x16_t b) {
    int16x8_t lo = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    acc = vpadalq_s16(acc, lo);
    acc = vpadalq_s16(acc, hi);
    return acc;
}

// ===========================================================================
// Q8_0 x Q8_0 dot
//
// Block layout: { fp16 d; int8 qs[32]; }
// Algorithm (mirrors ggml ARM nrc=1 path, processes 2 blocks per iter
// for ILP):
//   for each pair of blocks (b, b+1):
//       sum0  += d_b   * sum_int8(x[b].qs   * y[b].qs)
//       sum1  += d_b+1 * sum_int8(x[b+1].qs * y[b+1].qs)
//   return sum0 + sum1
// ===========================================================================

__attribute__((target("+dotprod"), noinline))
static float q8_0_dot_q8_0_dotprod(const q8_0_block* x, const q8_0_block* y,
                                   int64_t k) {
    const int   nb = (int)(k / QK8_0);
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const q8_0_block* x0 = &x[ib + 0];
        const q8_0_block* x1 = &x[ib + 1];
        const q8_0_block* y0 = &y[ib + 0];
        const q8_0_block* y1 = &y[ib + 1];
        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), x0_0, y0_0);
        p0 = vdotq_s32(p0, x0_1, y0_1);
        int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), x1_0, y1_0);
        p1 = vdotq_s32(p1, x1_1, y1_1);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0),
                            q_fp16_to_f32(x0->d) * q_fp16_to_f32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1),
                            q_fp16_to_f32(x1->d) * q_fp16_to_f32(y1->d));
    }
    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    for (; ib < nb; ++ib) {
        int32_t sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int32_t)x[ib].qs[j] * (int32_t)y[ib].qs[j];
        }
        sumf += (float)sumi * (q_fp16_to_f32(x[ib].d) * q_fp16_to_f32(y[ib].d));
    }
    return sumf;
}

// ===========================================================================
// Q4_K x Q8_K dot   (256-elem super-blocks, 4-bit weights with scales+mins)
//
// Layout recap (see quants.h):
//   x[i]: { fp16 d, fp16 dmin, u8 scales[12], u8 qs[128] }
//   y[i]: { f32  d, i8 qs[256], i16 bsums[16] }
//
// Per super-block, the math is:
//   d    = y.d * fp16_to_f32(x.d)
//   dmin = y.d * fp16_to_f32(x.dmin)
//   isum_mins = sum over 8 sub-blocks j of (mins[j] * (bsums[2j] + bsums[2j+1]))
//   isum      = sum over 8 sub-blocks j of (scales[j] * sub_dot_int8_j)
//   row_sum  += d * isum - dmin * isum_mins
// ===========================================================================

__attribute__((target("+dotprod"), noinline))
static float q4k_row_dot_q8k_dotprod(int nb, const q4k_block* x,
                                     const q8k_block* y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),
                                            vld1q_s16(y[i].bsums + 8));
        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        uint32x2_t mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2)
                              | (((utmp[1] >> 6) & kmask3) << 4),
                              mins8, 1);
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;
        const int16x8_t mins = vreinterpretq_s16_u16(
            vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(
            vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * (float)vaddvq_s32(prod);
        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        int32_t sumi1 = 0, sumi2 = 0;
        for (int32_t j = 0; j < QK_K / 64; j++) {
            const uint8x16_t q4bits0 = vld1q_u8(q4);
            const uint8x16_t q4bits1 = vld1q_u8(q4 + 16);
            q4 += 32;
            int8x16_t q8b0 = vld1q_s8(q8);
            int8x16_t q8b1 = vld1q_s8(q8 + 16);
            q8 += 32;
            int8x16_t q4b0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            int8x16_t q4b1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4b0, q8b0),
                                           q4b1, q8b1);
            sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];
            q8b0 = vld1q_s8(q8);
            q8b1 = vld1q_s8(q8 + 16);
            q8 += 32;
            q4b0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            q4b1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4b0, q8b0),
                                           q4b1, q8b1);
            sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
        }
        sumf += d * (float)(sumi1 + sumi2);
    }
    return sumf;
}

__attribute__((noinline))
static float q4k_row_dot_q8k_baseline(int nb, const q4k_block* x,
                                      const q8k_block* y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),
                                            vld1q_s16(y[i].bsums + 8));
        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        uint32x2_t mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2)
                              | (((utmp[1] >> 6) & kmask3) << 4),
                              mins8, 1);
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;
        const int16x8_t mins = vreinterpretq_s16_u16(
            vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(
            vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * (float)vaddvq_s32(prod);
        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        int32_t sumi1 = 0, sumi2 = 0;
        for (int32_t j = 0; j < QK_K / 64; j++) {
            const uint8x16_t q4bits0 = vld1q_u8(q4);
            const uint8x16_t q4bits1 = vld1q_u8(q4 + 16);
            q4 += 32;
            int8x16_t q8b0 = vld1q_s8(q8);
            int8x16_t q8b1 = vld1q_s8(q8 + 16);
            q8 += 32;
            int8x16_t q4b0 = vreinterpretq_s8_u8(vandq_u8(q4bits0, m4b));
            int8x16_t q4b1 = vreinterpretq_s8_u8(vandq_u8(q4bits1, m4b));
            int32x4_t p1 = neon_vdot_baseline(mzero, q4b0, q8b0);
            p1 = neon_vdot_baseline(p1, q4b1, q8b1);
            sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];
            q8b0 = vld1q_s8(q8);
            q8b1 = vld1q_s8(q8 + 16);
            q8 += 32;
            q4b0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits0, 4));
            q4b1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits1, 4));
            int32x4_t p2 = neon_vdot_baseline(mzero, q4b0, q8b0);
            p2 = neon_vdot_baseline(p2, q4b1, q8b1);
            sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
        }
        sumf += d * (float)(sumi1 + sumi2);
    }
    return sumf;
}

// ===========================================================================
// Q5_K x Q8_K dot  (5-bit weights: low nibble in qs[], high bit in qh[])
// ===========================================================================

__attribute__((target("+dotprod"), noinline))
static float q5k_row_dot_q8k_dotprod(int nb, const q5k_block* x,
                                     const q8k_block* y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const uint8x16_t mone  = vdupq_n_u8(1);
    const uint8x16_t mtwo  = vdupq_n_u8(2);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),
                                            vld1q_s16(y[i].bsums + 8));
        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        const int16x8_t mins  = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod  = vaddq_s32(
            vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        const int32_t sumi_mins = vaddvq_s32(prod);
        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q5 = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        uint8x16_t qhbits0 = vld1q_u8(qh);
        uint8x16_t qhbits1 = vld1q_u8(qh + 16);
        int32_t sumi = 0;
        for (int32_t j = 0; j < QK_K / 64; j++) {
            const uint8x16_t q5bits0 = vld1q_u8(q5);
            const uint8x16_t q5bits1 = vld1q_u8(q5 + 16);
            q5 += 32;
            const int8x16_t q8b0 = vld1q_s8(q8);
            const int8x16_t q8b1 = vld1q_s8(q8 + 16);
            const int8x16_t q8b2 = vld1q_s8(q8 + 32);
            const int8x16_t q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            const uint8x16_t q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            const uint8x16_t q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            const uint8x16_t q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits0), 3);
            const uint8x16_t q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits1), 3);
            qhbits0 = vshrq_n_u8(qhbits0, 2);
            qhbits1 = vshrq_n_u8(qhbits1, 2);
            const int8x16_t q5b0 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8 (q5bits0, m4b), q5h0));
            const int8x16_t q5b1 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8 (q5bits1, m4b), q5h1));
            const int8x16_t q5b2 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q5bits0, 4),  q5h2));
            const int8x16_t q5b3 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q5bits1, 4),  q5h3));
            sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5b0, q8b0),
                                         q5b1, q8b1)) * scales[2 * j + 0];
            sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5b2, q8b2),
                                         q5b3, q8b3)) * scales[2 * j + 1];
        }
        sumf += d * (float)sumi - dmin * (float)sumi_mins;
    }
    return sumf;
}

__attribute__((noinline))
static float q5k_row_dot_q8k_baseline(int nb, const q5k_block* x,
                                      const q8k_block* y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const uint8x16_t mone  = vdupq_n_u8(1);
    const uint8x16_t mtwo  = vdupq_n_u8(2);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),
                                            vld1q_s16(y[i].bsums + 8));
        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        const int16x8_t mins  = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod  = vaddq_s32(
            vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        const int32_t sumi_mins = vaddvq_s32(prod);
        const uint8_t* scales = (const uint8_t*)utmp;
        const uint8_t* q5 = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        uint8x16_t qhbits0 = vld1q_u8(qh);
        uint8x16_t qhbits1 = vld1q_u8(qh + 16);
        int32_t sumi = 0;
        for (int32_t j = 0; j < QK_K / 64; j++) {
            const uint8x16_t q5bits0 = vld1q_u8(q5);
            const uint8x16_t q5bits1 = vld1q_u8(q5 + 16);
            q5 += 32;
            const int8x16_t q8b0 = vld1q_s8(q8);
            const int8x16_t q8b1 = vld1q_s8(q8 + 16);
            const int8x16_t q8b2 = vld1q_s8(q8 + 32);
            const int8x16_t q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            const uint8x16_t q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            const uint8x16_t q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            const uint8x16_t q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits0), 3);
            const uint8x16_t q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits1), 3);
            qhbits0 = vshrq_n_u8(qhbits0, 2);
            qhbits1 = vshrq_n_u8(qhbits1, 2);
            const int8x16_t q5b0 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8 (q5bits0, m4b), q5h0));
            const int8x16_t q5b1 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8 (q5bits1, m4b), q5h1));
            const int8x16_t q5b2 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q5bits0, 4),  q5h2));
            const int8x16_t q5b3 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q5bits1, 4),  q5h3));
            int32x4_t p1 = neon_vdot_baseline(mzero, q5b0, q8b0);
                       p1 = neon_vdot_baseline(p1,   q5b1, q8b1);
            sumi += vaddvq_s32(p1) * scales[2 * j + 0];
            int32x4_t p2 = neon_vdot_baseline(mzero, q5b2, q8b2);
                       p2 = neon_vdot_baseline(p2,   q5b3, q8b3);
            sumi += vaddvq_s32(p2) * scales[2 * j + 1];
        }
        sumf += d * (float)sumi - dmin * (float)sumi_mins;
    }
    return sumf;
}

// ===========================================================================
// Q6_K x Q8_K dot  (6-bit weights: 4 in ql, 2 in qh, offset -32)
// ===========================================================================

__attribute__((target("+dotprod"), noinline))
static float q6k_row_dot_q8k_dotprod(int nb, const q6k_block* x,
                                     const q8k_block* y) {
    const uint8x16_t m4b   = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    const uint8x16_t mone  = vdupq_n_u8(3);
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        const int16x8_t q8sums_lo = vld1q_s16(y[i].bsums);
        const int16x8_t q8sums_hi = vld1q_s16(y[i].bsums + 8);
        const int8x16_t scales_v  = vld1q_s8(x[i].scales);
        const int16x8_t q6scales_lo = vmovl_s8(vget_low_s8 (scales_v));
        const int16x8_t q6scales_hi = vmovl_s8(vget_high_s8(scales_v));
        const int32x4_t prod = vaddq_s32(
            vaddq_s32(
                vmull_s16(vget_low_s16 (q8sums_lo), vget_low_s16 (q6scales_lo)),
                vmull_s16(vget_high_s16(q8sums_lo), vget_high_s16(q6scales_lo))),
            vaddq_s32(
                vmull_s16(vget_low_s16 (q8sums_hi), vget_low_s16 (q6scales_hi)),
                vmull_s16(vget_high_s16(q8sums_hi), vget_high_s16(q6scales_hi))));
        const int32_t isum_mins = vaddvq_s32(prod);
        const uint8_t* q6    = x[i].ql;
        const uint8_t* qh    = x[i].qh;
        const int8_t*  q8    = y[i].qs;
        const int8_t*  scale = x[i].scales;
        int32_t isum = 0;
        for (int32_t j = 0; j < QK_K / 128; j++) {
            uint8x16_t qhbits0 = vld1q_u8(qh);
            uint8x16_t qhbits1 = vld1q_u8(qh + 16);
            qh += 32;
            const uint8x16_t q6bits0 = vld1q_u8(q6);
            const uint8x16_t q6bits1 = vld1q_u8(q6 + 16);
            const uint8x16_t q6bits2 = vld1q_u8(q6 + 32);
            const uint8x16_t q6bits3 = vld1q_u8(q6 + 48);
            q6 += 64;
            int8x16_t q8b0 = vld1q_s8(q8);
            int8x16_t q8b1 = vld1q_s8(q8 + 16);
            int8x16_t q8b2 = vld1q_s8(q8 + 32);
            int8x16_t q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            uint8x16_t q6h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            uint8x16_t q6h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits0, 2);
            uint8x16_t q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 2);
            uint8x16_t q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            int8x16_t q6b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits0, m4b), q6h0));
            int8x16_t q6b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits1, m4b), q6h1));
            int8x16_t q6b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits2, m4b), q6h2));
            int8x16_t q6b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits3, m4b), q6h3));
            isum += vaddvq_s32(vdotq_s32(vzero, q6b0, q8b0)) * scale[0]
                  + vaddvq_s32(vdotq_s32(vzero, q6b1, q8b1)) * scale[1]
                  + vaddvq_s32(vdotq_s32(vzero, q6b2, q8b2)) * scale[2]
                  + vaddvq_s32(vdotq_s32(vzero, q6b3, q8b3)) * scale[3];
            scale += 4;
            q8b0 = vld1q_s8(q8);
            q8b1 = vld1q_s8(q8 + 16);
            q8b2 = vld1q_s8(q8 + 32);
            q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            shifted = vshrq_n_u8(qhbits0, 4); q6h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 4); q6h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits0, 6); q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 6); q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            q6b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits0, 4), q6h0));
            q6b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits1, 4), q6h1));
            q6b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits2, 4), q6h2));
            q6b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits3, 4), q6h3));
            isum += vaddvq_s32(vdotq_s32(vzero, q6b0, q8b0)) * scale[0]
                  + vaddvq_s32(vdotq_s32(vzero, q6b1, q8b1)) * scale[1]
                  + vaddvq_s32(vdotq_s32(vzero, q6b2, q8b2)) * scale[2]
                  + vaddvq_s32(vdotq_s32(vzero, q6b3, q8b3)) * scale[3];
            scale += 4;
        }
        sum += (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}

__attribute__((noinline))
static float q6k_row_dot_q8k_baseline(int nb, const q6k_block* x,
                                      const q8k_block* y) {
    const uint8x16_t m4b   = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    const uint8x16_t mone  = vdupq_n_u8(3);
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        const int16x8_t q8sums_lo = vld1q_s16(y[i].bsums);
        const int16x8_t q8sums_hi = vld1q_s16(y[i].bsums + 8);
        const int8x16_t scales_v  = vld1q_s8(x[i].scales);
        const int16x8_t q6scales_lo = vmovl_s8(vget_low_s8 (scales_v));
        const int16x8_t q6scales_hi = vmovl_s8(vget_high_s8(scales_v));
        const int32x4_t prod = vaddq_s32(
            vaddq_s32(
                vmull_s16(vget_low_s16 (q8sums_lo), vget_low_s16 (q6scales_lo)),
                vmull_s16(vget_high_s16(q8sums_lo), vget_high_s16(q6scales_lo))),
            vaddq_s32(
                vmull_s16(vget_low_s16 (q8sums_hi), vget_low_s16 (q6scales_hi)),
                vmull_s16(vget_high_s16(q8sums_hi), vget_high_s16(q6scales_hi))));
        const int32_t isum_mins = vaddvq_s32(prod);
        const uint8_t* q6    = x[i].ql;
        const uint8_t* qh    = x[i].qh;
        const int8_t*  q8    = y[i].qs;
        const int8_t*  scale = x[i].scales;
        int32_t isum = 0;
        for (int32_t j = 0; j < QK_K / 128; j++) {
            uint8x16_t qhbits0 = vld1q_u8(qh);
            uint8x16_t qhbits1 = vld1q_u8(qh + 16);
            qh += 32;
            const uint8x16_t q6bits0 = vld1q_u8(q6);
            const uint8x16_t q6bits1 = vld1q_u8(q6 + 16);
            const uint8x16_t q6bits2 = vld1q_u8(q6 + 32);
            const uint8x16_t q6bits3 = vld1q_u8(q6 + 48);
            q6 += 64;
            int8x16_t q8b0 = vld1q_s8(q8);
            int8x16_t q8b1 = vld1q_s8(q8 + 16);
            int8x16_t q8b2 = vld1q_s8(q8 + 32);
            int8x16_t q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            uint8x16_t q6h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            uint8x16_t q6h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits0, 2);
            uint8x16_t q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 2);
            uint8x16_t q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            int8x16_t q6b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits0, m4b), q6h0));
            int8x16_t q6b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits1, m4b), q6h1));
            int8x16_t q6b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits2, m4b), q6h2));
            int8x16_t q6b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits3, m4b), q6h3));
            isum += vaddvq_s32(neon_vdot_baseline(vzero, q6b0, q8b0)) * scale[0]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b1, q8b1)) * scale[1]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b2, q8b2)) * scale[2]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b3, q8b3)) * scale[3];
            scale += 4;
            q8b0 = vld1q_s8(q8);
            q8b1 = vld1q_s8(q8 + 16);
            q8b2 = vld1q_s8(q8 + 32);
            q8b3 = vld1q_s8(q8 + 48);
            q8 += 64;
            shifted = vshrq_n_u8(qhbits0, 4); q6h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 4); q6h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits0, 6); q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 6); q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            q6b0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits0, 4), q6h0));
            q6b1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits1, 4), q6h1));
            q6b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits2, 4), q6h2));
            q6b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits3, 4), q6h3));
            isum += vaddvq_s32(neon_vdot_baseline(vzero, q6b0, q8b0)) * scale[0]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b1, q8b1)) * scale[1]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b2, q8b2)) * scale[2]
                  + vaddvq_s32(neon_vdot_baseline(vzero, q6b3, q8b3)) * scale[3];
            scale += 4;
        }
        sum += (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}

__attribute__((noinline))
static float q8_0_dot_q8_0_baseline(const q8_0_block* x, const q8_0_block* y,
                                    int64_t k) {
    const int   nb = (int)(k / QK8_0);
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const q8_0_block* x0 = &x[ib + 0];
        const q8_0_block* x1 = &x[ib + 1];
        const q8_0_block* y0 = &y[ib + 0];
        const q8_0_block* y1 = &y[ib + 1];
        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);
        int32x4_t p0 = neon_vdot_baseline(vdupq_n_s32(0), x0_0, y0_0);
        p0 = neon_vdot_baseline(p0, x0_1, y0_1);
        int32x4_t p1 = neon_vdot_baseline(vdupq_n_s32(0), x1_0, y1_0);
        p1 = neon_vdot_baseline(p1, x1_1, y1_1);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0),
                            q_fp16_to_f32(x0->d) * q_fp16_to_f32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1),
                            q_fp16_to_f32(x1->d) * q_fp16_to_f32(y1->d));
    }
    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    for (; ib < nb; ++ib) {
        int32_t sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int32_t)x[ib].qs[j] * (int32_t)y[ib].qs[j];
        }
        sumf += (float)sumi * (q_fp16_to_f32(x[ib].d) * q_fp16_to_f32(y[ib].d));
    }
    return sumf;
}

// ===========================================================================
// SiLU (x / (1 + exp(-x))) — 4-lane NEON port.
//
// Verbatim port of ggml-cpu/vec.h's NEON ggml_v_silu (which inlines
// ggml_v_expf). Bit-identical to the polynomial-expf path that
// llm/neon.c emits, which is the parity gate for `--chat-test`. Do
// NOT replace with libm calls — they're written to match ggml's bit
// output, not to be more accurate.
// ===========================================================================

#include <math.h>

static inline float32x4_t neon_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t  e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t  c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
        return vfmaq_f32(k, j, k);
    }
    const uint32x4_t  d  = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static inline float32x4_t neon_v_silu(float32x4_t x) {
    const float32x4_t one      = vdupq_n_f32(1.0f);
    const float32x4_t zero     = vdupq_n_f32(0.0f);
    const float32x4_t neg_x    = vsubq_f32(zero, x);
    const float32x4_t exp_neg  = neon_v_expf(neg_x);
    const float32x4_t denom    = vaddq_f32(one, exp_neg);
    return vdivq_f32(x, denom);
}

// 4-lane SiLU over `n` consecutive fp32 values. Tail (n mod 4) uses
// the scalar libm formula — bit-identical to ggml's ggml_silu_f32
// fallback. Single shared path for both dotprod and baseline NEON
// (the polynomial doesn't need sdot).
static void silu_f32_neon(int n, float * dst, const float * src) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(dst + i, neon_v_silu(vld1q_f32(src + i)));
    }
    for (; i < n; i++) {
        dst[i] = src[i] / (1.0f + expf(-src[i]));
    }
}
