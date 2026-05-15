// SPDX-License-Identifier: MIT
//
// neon.c - single-file NEON kernel pack, included by tensor.c.
//
// This file is the ONLY place that includes <arm_neon.h> and emits
// NEON intrinsics. tensor.c and llm.c stay scalar / portable. Layout
// is intentional: future avx2.c slots in as a sibling with the same
// function names and the dispatch macro selects via #if at the call
// site (see tensor.c "DOT_*" dispatch).
//
// What lives here today:
//   - q4k_dot_q8k_neon  : Q4_K x Q8_K int8 dot, 256 weights/block
//   - q5k_dot_q8k_neon  : Q5_K x Q8_K int8 dot
//   - q6k_dot_q8k_neon  : Q6_K x Q8_K int8 dot
//   - ggml_v_expf       : 4-lane fp32 expf, polynomial approximation,
//                         verbatim port from llama.cpp
//                         ggml/src/ggml-cpu/vec.h (NEON branch).
//   - ggml_v_silu       : 4-lane fp32 SiLU = x / (1 + expf(-x)),
//                         verbatim port from same source.
//   - neon_silu_vec_f32 : driver that maps ggml_v_silu over a row.
//
// The three Q*_K dots are line-for-line ports of llama.cpp's
// ggml-cpu/arch/arm/quants.c (__ARM_NEON branch); see the per-function
// comment for the exact source line range and ggml git rev.
//
// All functions are static inline / static so the file can be
// included into multiple translation units without ODR violations.
// In this codebase it is included once, from tensor.c.
//
// Block typedefs (q4k_block, q5k_block, q6k_block, q8k_block) are
// expected to be in scope at the point of inclusion. tensor.c defines
// them just above the `#include "neon.c"` directive.
//
// NOTE on bit-identical parity: ggml's NEON code is the parity
// target. Do NOT replace the algorithms here with "equivalent"
// formulations - even reordering an FMA chain or a vpaddq collapse
// changes the fp32 result by 1 ULP, which compounds across 24 layers
// to visible drift. The accompanying `tools/qwen-haiku` reference
// dumper is what gates "ok to refactor": if the post-refactor dump
// matches byte-for-byte, the change is safe.

#include <arm_neon.h>
#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Q4_K x Q8_K NEON dot. Source: llama.cpp ggml-cpu/arch/arm/quants.c
// ggml_vec_dot_q4_K_q8_K (__ARM_NEON branch, ~lines 2522-2583,
// commit b48e80f67 at time of port).
// ---------------------------------------------------------------------------

// Row-level Q4_K x Q8_K dot. Processes ALL nb super-blocks of one
// row with a single fp32 accumulator, exactly matching ggml's
// `ggml_vec_dot_q4_K_q8_K` (NEON nrc=1 branch) accumulation pattern:
//
//   float sumf = 0;
//   for (int i = 0; i < nb; ++i) {
//       sumf -= dmin_i * X_i;
//       sumf += d_i    * Y_i;
//   }
//
// The previous per-block kernel reset sumf=0 inside each block and
// returned `d_i*Y_i - dmin_i*X_i` as a single value; the outer loop
// then accumulated those into `acc`. Mathematically equivalent but
// the fp32 parenthesisation differs by 1 ULP for some weight values
// - it happened to round consistently on attn_qkv weights but
// produced visible drift on attn_gate weights. Row-level matches
// bit-for-bit.
static inline float q4k_row_dot_q8k_neon(int nb,
                                         const q4k_block * x,
                                         const q8k_block * y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d    = y[i].d * (float)x[i].d;
        const float dmin = y[i].d * (float)x[i].dmin;
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
        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * q4 = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        int32_t sumi1 = 0;
        int32_t sumi2 = 0;
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

static inline float q4k_dot_q8k_neon(const q4k_block * x,
                                     const q8k_block * y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const int32x4_t  mzero = vdupq_n_s32(0);
    const float d    = y->d * (float)x->d;
    const float dmin = y->d * (float)x->dmin;
    float sumf = 0.0f;
    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y->bsums),
                                        vld1q_s16(y->bsums + 8));
    uint32_t utmp[4];
    memcpy(utmp, x->scales, 12);
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
    const uint8_t * scales = (const uint8_t *)utmp;
    const uint8_t * q4 = x->qs;
    const int8_t  * q8 = y->qs;
    int32_t sumi1 = 0;
    int32_t sumi2 = 0;
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
    return sumf;
}

// ---------------------------------------------------------------------------
// Q5_K x Q8_K NEON dot. Source: llama.cpp ggml-cpu/arch/arm/quants.c
// ggml_vec_dot_q5_K_q8_K (__ARM_NEON branch, ~lines 2617-2683).
// ---------------------------------------------------------------------------

// Row-level Q5_K x Q8_K dot - same pattern as q4k_row_dot. The
// per-block expression in ggml is `sumf += d*sumi - dmin*sumi_mins`,
// which is a SINGLE add into a row-wide sumf (rather than two ops
// like Q4_K's split). The per-block kernel below already returns
// `d*sumi - dmin*sumi_mins` and the outer loop adds it to acc, so
// per-block and row-level are arithmetically the same expression.
// We still expose this row-level form to keep all three Q-quants
// uniform in the matmul caller.
static inline float q5k_row_dot_q8k_neon(int nb,
                                         const q5k_block * x,
                                         const q8k_block * y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const uint8x16_t mone  = vdupq_n_u8(1);
    const uint8x16_t mtwo  = vdupq_n_u8(2);
    const int32x4_t  mzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * (float)x[i].d;
        const float dmin = y[i].d * (float)x[i].dmin;
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums),
                                            vld1q_s16(y[i].bsums + 8));
        uint32_t utmp[4];
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2)
                | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8x8_t mins8 = vld1_u8((const uint8_t *)utmp + 8);
        const int16x8_t mins  = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod  = vaddq_s32(
            vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        const int32_t sumi_mins = vaddvq_s32(prod);
        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * q5 = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * q8 = y[i].qs;
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

static inline float q5k_dot_q8k_neon(const q5k_block * x,
                                     const q8k_block * y) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    const uint8x16_t m4b   = vdupq_n_u8(0xf);
    const uint8x16_t mone  = vdupq_n_u8(1);
    const uint8x16_t mtwo  = vdupq_n_u8(2);
    const int32x4_t  mzero = vdupq_n_s32(0);
    const float d    = y->d * (float)x->d;
    const float dmin = y->d * (float)x->dmin;
    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y->bsums),
                                        vld1q_s16(y->bsums + 8));
    uint32_t utmp[4];
    memcpy(utmp, x->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2)
            | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const uint8x8_t mins8 = vld1_u8((const uint8_t *)utmp + 8);
    const int16x8_t mins  = vreinterpretq_s16_u16(vmovl_u8(mins8));
    const int32x4_t prod  = vaddq_s32(
        vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
    const int32_t sumi_mins = vaddvq_s32(prod);
    const uint8_t * scales = (const uint8_t *)utmp;
    const uint8_t * q5 = x->qs;
    const uint8_t * qh = x->qh;
    const int8_t  * q8 = y->qs;
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
    return d * (float)sumi - dmin * (float)sumi_mins;
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_K NEON dot. Source: llama.cpp ggml-cpu/arch/arm/quants.c
// ggml_vec_dot_q6_K_q8_K (__ARM_NEON branch, ~lines 3225-3318). Uses
// the (q - 32) offset trick: dot in raw 6-bit space then subtract
// 32 * sum(scale * bsums).
// ---------------------------------------------------------------------------

// Row-level Q6_K x Q8_K dot - mirrors ggml's NEON nrc=1 path:
// `sum += d_all * y[i].d * (isum - 32 * isum_mins)` per super-block,
// with sum carried across the row.
static inline float q6k_row_dot_q8k_neon(int nb,
                                         const q6k_block * x,
                                         const q8k_block * y) {
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
        const uint8_t * q6    = x[i].ql;
        const uint8_t * qh    = x[i].qh;
        const int8_t  * q8    = y[i].qs;
        const int8_t  * scale = x[i].scales;
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
            int8x16_t q6b0 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q6bits0, m4b), q6h0));
            int8x16_t q6b1 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q6bits1, m4b), q6h1));
            int8x16_t q6b2 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q6bits2, m4b), q6h2));
            int8x16_t q6b3 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q6bits3, m4b), q6h3));
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
            shifted = vshrq_n_u8(qhbits0, 4);
            q6h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 4);
            q6h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits0, 6);
            q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits1, 6);
            q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            q6b0 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q6bits0, 4), q6h0));
            q6b1 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q6bits1, 4), q6h1));
            q6b2 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q6bits2, 4), q6h2));
            q6b3 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q6bits3, 4), q6h3));
            isum += vaddvq_s32(vdotq_s32(vzero, q6b0, q8b0)) * scale[0]
                  + vaddvq_s32(vdotq_s32(vzero, q6b1, q8b1)) * scale[1]
                  + vaddvq_s32(vdotq_s32(vzero, q6b2, q8b2)) * scale[2]
                  + vaddvq_s32(vdotq_s32(vzero, q6b3, q8b3)) * scale[3];
            scale += 4;
        }
        sum += (float)x[i].d * y[i].d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}

static inline float q6k_dot_q8k_neon(const q6k_block * x,
                                     const q8k_block * y) {
    const uint8x16_t m4b   = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    const uint8x16_t mone  = vdupq_n_u8(3);
    const int16x8_t q8sums_lo = vld1q_s16(y->bsums);
    const int16x8_t q8sums_hi = vld1q_s16(y->bsums + 8);
    const int8x16_t scales_v  = vld1q_s8(x->scales);
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
    const uint8_t * q6    = x->ql;
    const uint8_t * qh    = x->qh;
    const int8_t  * q8    = y->qs;
    const int8_t  * scale = x->scales;
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
        int8x16_t q6b0 = vreinterpretq_s8_u8(
            vorrq_u8(vandq_u8(q6bits0, m4b), q6h0));
        int8x16_t q6b1 = vreinterpretq_s8_u8(
            vorrq_u8(vandq_u8(q6bits1, m4b), q6h1));
        int8x16_t q6b2 = vreinterpretq_s8_u8(
            vorrq_u8(vandq_u8(q6bits2, m4b), q6h2));
        int8x16_t q6b3 = vreinterpretq_s8_u8(
            vorrq_u8(vandq_u8(q6bits3, m4b), q6h3));
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
        shifted = vshrq_n_u8(qhbits0, 4);
        q6h0 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits1, 4);
        q6h1 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits0, 6);
        q6h2 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits1, 6);
        q6h3 = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        q6b0 = vreinterpretq_s8_u8(
            vorrq_u8(vshrq_n_u8(q6bits0, 4), q6h0));
        q6b1 = vreinterpretq_s8_u8(
            vorrq_u8(vshrq_n_u8(q6bits1, 4), q6h1));
        q6b2 = vreinterpretq_s8_u8(
            vorrq_u8(vshrq_n_u8(q6bits2, 4), q6h2));
        q6b3 = vreinterpretq_s8_u8(
            vorrq_u8(vshrq_n_u8(q6bits3, 4), q6h3));
        isum += vaddvq_s32(vdotq_s32(vzero, q6b0, q8b0)) * scale[0]
              + vaddvq_s32(vdotq_s32(vzero, q6b1, q8b1)) * scale[1]
              + vaddvq_s32(vdotq_s32(vzero, q6b2, q8b2)) * scale[2]
              + vaddvq_s32(vdotq_s32(vzero, q6b3, q8b3)) * scale[3];
        scale += 4;
    }
    return (float)x->d * y->d * (float)(isum - 32 * isum_mins);
}

// ---------------------------------------------------------------------------
// Q8_0 x Q8_0 NEON dot. Source: llama.cpp ggml-cpu/arch/arm/quants.c
// ggml_vec_dot_q8_0_q8_0 (__ARM_NEON branch, ~lines 1086-1116). Both
// inputs are Q8_0-quantised (int8 + fp16 per-block scale). We use the
// nrc=1 single-row variant, not the MMLA 2-row variant - the latter
// requires producing two output rows together and would force a
// rewrite of the dispatch loop. The standard NEON path is in the same
// fp32 accumulation order as ggml dispatches when nrc=1.
//
// For Qwen3.5-0.8B the only Q8_0 weights are ssm_alpha and ssm_beta
// (per-layer, shape [hidden, n_heads]) - small matrices where the
// nrc=1 path is what llama.cpp's graph dispatcher actually selects.
// ---------------------------------------------------------------------------

#define QK8_0 32

// Scalar Q8_0 row quantisation. Reference impl per ggml-quants.c
// quantize_row_q8_0_ref. Identical bit output to the NEON variant
// when fed typical fp32 inputs (no denormals/inf/nan).
static inline void quantize_row_q8_0(const float * x, q8_0_block * y,
                                     int64_t k) {
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i * QK8_0 + j]);
            if (v > amax) { amax = v; }
        }
        float d  = amax / ((1 << 7) - 1);
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        y[i].d = (_Float16)d;
        for (int j = 0; j < QK8_0; j++) {
            float    x0 = x[i * QK8_0 + j] * id;
            int32_t  qv = (int32_t)roundf(x0);
            if (qv > 127)  { qv = 127;  }
            if (qv < -128) { qv = -128; }
            y[i].qs[j] = (int8_t)qv;
        }
    }
}

// Dot product of two Q8_0-quantized rows of length k (multiple of 32).
// Bit-identical to ggml's ggml_vec_dot_q8_0_q8_0 (nrc=1, __ARM_NEON
// branch). Assumes nb (number of blocks) is even - true for any
// k % 64 == 0 input, which all the qwen3-next ssm_alpha/ssm_beta
// rows satisfy.
static inline float q8_0_dot_q8_0_neon(const q8_0_block * x,
                                       const q8_0_block * y,
                                       int64_t k) {
    const int      nb    = (int)(k / QK8_0);
    float32x4_t    sumv0 = vdupq_n_f32(0.0f);
    float32x4_t    sumv1 = vdupq_n_f32(0.0f);
    int            ib    = 0;
    for (; ib + 1 < nb; ib += 2) {
        const q8_0_block * x0 = &x[ib + 0];
        const q8_0_block * x1 = &x[ib + 1];
        const q8_0_block * y0 = &y[ib + 0];
        const q8_0_block * y1 = &y[ib + 1];
        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);
        sumv0 = vmlaq_n_f32(
            sumv0,
            vcvtq_f32_s32(vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
            (float)x0->d * (float)y0->d);
        sumv1 = vmlaq_n_f32(
            sumv1,
            vcvtq_f32_s32(vaddq_s32(
                vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
            (float)x1->d * (float)y1->d);
    }
    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += (float)sumi * ((float)x[ib].d * (float)y[ib].d);
    }
    return sumf;
}

// ---------------------------------------------------------------------------
// 4-lane fp32 expf and SiLU. Verbatim port from llama.cpp
// ggml/src/ggml-cpu/vec.h (NEON branch, ~lines 1148-1179, commit
// b48e80f67). Polynomial approximation, NOT the libm expf - the
// approximation matches ggml's NEON path bit-for-bit, which is what
// we need for parity. The accompanying scalar libm path in ggml is
// only used as a tail loop when n is not multiple of 4, and
// ggml_silu_f32 (the inline) is the libm fallback shape.
//
// IMPORTANT: do NOT replace these with vector calls into libm
// (sincosf, expf etc.) - they are written to match ggml's bit
// output, not to be "more accurate". Use them as the parity target.
// ---------------------------------------------------------------------------

static inline float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
        return vfmaq_f32(k, j, k);
    }
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static inline float32x4_t ggml_v_silu(float32x4_t x) {
    const float32x4_t one      = vdupq_n_f32(1.0f);
    const float32x4_t zero     = vdupq_n_f32(0.0f);
    const float32x4_t neg_x    = vsubq_f32(zero, x);
    const float32x4_t exp_neg  = ggml_v_expf(neg_x);
    const float32x4_t denom    = vaddq_f32(one, exp_neg);
    return vdivq_f32(x, denom);
}

// Mirrors ggml_vec_silu_f32: 4-lane SiLU over `n` consecutive floats
// from `src` into `dst`. Tail (n mod 4) falls back to the same scalar
// formula libm/ggml use, so the output is bit-identical to ggml's
// SILU op on the same input - which is the whole point of including
// this here.
static inline void neon_silu_vec_f32(int n, float * dst, const float * src) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(dst + i, ggml_v_silu(vld1q_f32(src + i)));
    }
    for (; i < n; i++) {
        // Scalar fallback matches ggml's inline ggml_silu_f32.
        dst[i] = src[i] / (1.0f + expf(-src[i]));
    }
}
