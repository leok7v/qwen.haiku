// SPDX-License-Identifier: MIT
//
// avx.c -- x86_64 SSE/AVX kernel pack.
//
// Three tiers per kernel:
//   _avxvnni  -- AVX-VNNI 256-bit dpbusd            (Alder Lake+, Zen 4+)
//   _avx2     -- AVX2 + FMA + F16C                  (Haswell+, ~2013)
//   _avx1     -- AVX1 + F16C (no FMA, no int 256)   (Ivy Bridge, ~2012)
//
// AVX-512-VNNI is folded into the AVX-VNNI tier for Q8_0 specifically:
// the block is 32 bytes = one YMM, so a 512-bit kernel would waste half
// its lane. For Q4_K/Q5_K/Q6_K (256-elem blocks) we'll add a dedicated
// AVX-512 tier in the larger kernels.
//
// All tiers use __attribute__((target(...), noinline)) so the compiler
// emits the elevated ISA strictly inside the function body; the dispatcher
// (compiled at baseline) only calls a tier when CPUID confirms support.
//
// Algorithm: maddubs + madd for AVX2 (the standard ggml/llama.cpp x86
// path), dpbusd_avx_epi32 for AVX-VNNI, scalar tail + 128-bit SSE int
// for AVX1 (Ivy Bridge has no 256-bit integer ops).

#include "quants.h"
#include <immintrin.h>

// Macro to keep tier attributes terse.
#define SIMD_FN(attrs) __attribute__((target(attrs), noinline)) static

// ---------------------------------------------------------------------------
// Horizontal reduce __m256i (int32x8) -> int32. AVX2-only (uses
// _mm256_extracti128_si256). Marked with target attribute so the
// dispatcher TU (compiled at baseline) doesn't try to emit it.
// ---------------------------------------------------------------------------
__attribute__((target("avx2"), always_inline))
static inline int32_t hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    __m128i sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    s = _mm_add_epi32(s, sh);
    sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_epi32(s, sh);
    return _mm_cvtsi128_si32(s);
}

// Horizontal reduce __m128i (int32x4) -> int32. Used by AVX1 path.
static inline int32_t hsum_i32_4(__m128i v) {
    __m128i sh = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i s  = _mm_add_epi32(v, sh);
    sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_epi32(s, sh);
    return _mm_cvtsi128_si32(s);
}

// ===========================================================================
// Q8_0 x Q8_0 dot, AVX-VNNI tier
//
// Block: 32 int8 + fp16 scale. One block = one YMM register.
// VNNI: _mm256_dpbusd_avx_epi32(acc, u8, s8) -> int32x8
//   Requires unsigned-first-operand. Standard workaround:
//     u8 = |x|        (abs gives u8 in [0..127])
//     s8 = sign(x)*y  (y flipped where x is negative)
//   Result: dpbusd(acc, |x|, sign(x)*y) == acc + sum_4(x*y) per lane
// ===========================================================================
SIMD_FN("avxvnni,avx2,fma,f16c,bmi2")
float q8_0_dot_q8_0_avxvnni(const q8_0_block* x, const q8_0_block* y, int64_t k) {
    const int nb = (int)(k / QK8_0);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        __m256i xv = _mm256_loadu_si256((const __m256i*)x[i].qs);
        __m256i yv = _mm256_loadu_si256((const __m256i*)y[i].qs);
        __m256i ax = _mm256_abs_epi8(xv);
        __m256i sy = _mm256_sign_epi8(yv, xv);
        __m256i pi = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy);
        float scale = q_fp16_to_f32(x[i].d) * q_fp16_to_f32(y[i].d);
        __m256 fp = _mm256_cvtepi32_ps(pi);
        acc = _mm256_fmadd_ps(fp, _mm256_set1_ps(scale), acc);
    }
    // hsum __m256 -> float
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    return _mm_cvtss_f32(ss);
}

// ===========================================================================
// Q8_0 x Q8_0 dot, AVX2 tier
//
// Same |x|, sign(x)*y trick, but composed from maddubs + madd because
// AVX2 lacks dpbusd:
//   p16 = _mm256_maddubs_epi16(ax, sy)   // 16 x int16, each = ax_i*sy_i + ax_{i+1}*sy_{i+1}
//   p32 = _mm256_madd_epi16(p16, ONE16)  //  8 x int32, sum of pairs
// Throughput: 2 cycles per 32-byte block; AVX-VNNI's dpbusd is 1.
// ===========================================================================
SIMD_FN("avx2,fma,f16c")
float q8_0_dot_q8_0_avx2(const q8_0_block* x, const q8_0_block* y, int64_t k) {
    const int nb = (int)(k / QK8_0);
    const __m256i ONE16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        __m256i xv  = _mm256_loadu_si256((const __m256i*)x[i].qs);
        __m256i yv  = _mm256_loadu_si256((const __m256i*)y[i].qs);
        __m256i ax  = _mm256_abs_epi8(xv);
        __m256i sy  = _mm256_sign_epi8(yv, xv);
        __m256i p16 = _mm256_maddubs_epi16(ax, sy);
        __m256i pi  = _mm256_madd_epi16(p16, ONE16);
        float scale = q_fp16_to_f32(x[i].d) * q_fp16_to_f32(y[i].d);
        __m256 fp   = _mm256_cvtepi32_ps(pi);
        acc = _mm256_fmadd_ps(fp, _mm256_set1_ps(scale), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    return _mm_cvtss_f32(ss);
}

// ===========================================================================
// Q8_0 x Q8_0 dot, AVX1 tier (Ivy Bridge)
//
// No 256-bit integer ops on AVX1. We do the int8 dot in two 16-byte
// halves using SSE4.1 (sign_epi8, maddubs, madd_epi16), accumulate to
// scalar int32 per block, then convert+add to float in 256-bit float
// ops. F16C converts fp16->fp32 in hardware.
//
// Performance ceiling on this tier is ~1/2 of AVX2 throughput because
// the int side is 128-bit, but it's still ~5-8x scalar.
// ===========================================================================
SIMD_FN("avx,f16c,ssse3,sse4.1")
float q8_0_dot_q8_0_avx1(const q8_0_block* x, const q8_0_block* y, int64_t k) {
    const int nb = (int)(k / QK8_0);
    const __m128i ONE16 = _mm_set1_epi16(1);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        __m128i xv0 = _mm_loadu_si128((const __m128i*)(x[i].qs +  0));
        __m128i xv1 = _mm_loadu_si128((const __m128i*)(x[i].qs + 16));
        __m128i yv0 = _mm_loadu_si128((const __m128i*)(y[i].qs +  0));
        __m128i yv1 = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
        __m128i ax0 = _mm_abs_epi8(xv0);
        __m128i ax1 = _mm_abs_epi8(xv1);
        __m128i sy0 = _mm_sign_epi8(yv0, xv0);
        __m128i sy1 = _mm_sign_epi8(yv1, xv1);
        __m128i p160 = _mm_maddubs_epi16(ax0, sy0);
        __m128i p161 = _mm_maddubs_epi16(ax1, sy1);
        __m128i pi0  = _mm_madd_epi16(p160, ONE16);
        __m128i pi1  = _mm_madd_epi16(p161, ONE16);
        int32_t sumi = hsum_i32_4(_mm_add_epi32(pi0, pi1));
        sumf += (float)sumi * (q_fp16_to_f32(x[i].d) * q_fp16_to_f32(y[i].d));
    }
    return sumf;
}

// ===========================================================================
// Q4_K x Q8_K dot, AVX2 tier
//
// Per super-block (256 weights):
//   load 32 q4 bytes (one __m256i) -> 32 low nibbles + 32 high nibbles
//   load 64 q8 bytes (two __m256i)
//   for low nibbles vs q8[0..31]:   p16 = maddubs(q4_lo_u8, q8_lo_s8)
//                                   p32 = madd_epi16(p16, ONE16)         8x i32
//                                   acc_lo += mullo_epi32(p32, scale_lo)
//   for high nibbles vs q8[32..63]: same with scale_hi
//   isum = horizontal sum of acc
//   isum_mins side: bsums pair-summed via hadd_epi16, then madd with mins
//   sumf += d_eff * isum - dmin_eff * isum_mins
// ===========================================================================
SIMD_FN("avx2,fma,f16c")
float q4k_row_dot_q8k_avx2(int nb, const q4k_block* x, const q8k_block* y) {
    const __m256i M4   = _mm256_set1_epi8(0x0F);
    const __m256i ONE16 = _mm256_set1_epi16(1);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        // isum_mins: 16 bsums pair-summed to 8 sums, times 8 mins
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_u8 = _mm_loadl_epi64((const __m128i*)mins);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(mins_u8);
        const __m128i mins_prod = _mm_madd_epi16(bs_pair, mins_i16);  // 4 i32
        // hsum to scalar
        __m128i sh = _mm_shuffle_epi32(mins_prod, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mins_prod, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        // main int8 dot
        __m256i isum_v = _mm256_setzero_si256();
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m256i q4_bytes = _mm256_loadu_si256((const __m256i*)q4);
            __m256i q4_lo = _mm256_and_si256(q4_bytes, M4);
            __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), M4);
            __m256i q8_lo = _mm256_loadu_si256((const __m256i*)q8);
            __m256i q8_hi = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p16_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            __m256i p16_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);
            __m256i p32_lo = _mm256_madd_epi16(p16_lo, ONE16);
            __m256i p32_hi = _mm256_madd_epi16(p16_hi, ONE16);
            p32_lo = _mm256_mullo_epi32(p32_lo, _mm256_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm256_mullo_epi32(p32_hi, _mm256_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(p32_lo, p32_hi));
            q4 += 32;
            q8 += 64;
        }
        int32_t isum = hsum_i32_8(isum_v);
        __m256 cur = _mm256_set1_ps(d * (float)isum - dmin * (float)isum_mins);
        // Accumulate as 8 lanes (we only really need lane 0 -- but keep it in vector for ILP)
        acc_f = _mm256_add_ps(acc_f, cur);
    }
    // Only lane 0 was used per super-block, but cur was broadcast so all lanes have same value.
    // To avoid 8x overcount, divide by 8 at the end.
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

// ===========================================================================
// Q5_K x Q8_K dot, AVX2 tier
//
// Same maddubs+madd+mullo pattern as Q4_K, with an extra OR-in step for
// the 5th bit from x[i].qh. The 32-byte qh is held once and bits 2j,
// 2j+1 are extracted per outer iter (j = 0..3). Bit n in [0..7] can be
// extracted from a __m256i of bytes via srli_epi16(qh, n) & set1_epi8(1)
// -- cross-byte bleed within 16-bit lanes is masked off by the &.
// ===========================================================================
SIMD_FN("avx2,fma,f16c")
float q5k_row_dot_q8k_avx2(int nb, const q5k_block* x, const q8k_block* y) {
    const __m256i M4    = _mm256_set1_epi8(0x0F);
    const __m256i M1    = _mm256_set1_epi8(0x01);
    const __m256i ONE16 = _mm256_set1_epi16(1);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        // isum_mins (same as q4k)
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_u8 = _mm_loadl_epi64((const __m128i*)mins);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(mins_u8);
        const __m128i mins_prod = _mm_madd_epi16(bs_pair, mins_i16);
        __m128i sh = _mm_shuffle_epi32(mins_prod, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mins_prod, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        // main int8 dot with extra 5th-bit unpack from qh
        __m256i isum_v = _mm256_setzero_si256();
        __m256i qh_v = _mm256_loadu_si256((const __m256i*)x[i].qh);
        const uint8_t* q5 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m256i q5v = _mm256_loadu_si256((const __m256i*)q5);
            __m256i q5_low_nib  = _mm256_and_si256(q5v, M4);
            __m256i q5_high_nib = _mm256_and_si256(_mm256_srli_epi16(q5v, 4), M4);
            // bits 2j and 2j+1 of each qh byte, shifted left by 4 to add to nibble
            __m256i hb_lo = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2*j),     M1);
            __m256i hb_hi = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2*j + 1), M1);
            __m256i hb_lo_sh = _mm256_slli_epi16(hb_lo, 4);
            __m256i hb_hi_sh = _mm256_slli_epi16(hb_hi, 4);
            __m256i q5_lo = _mm256_or_si256(q5_low_nib,  hb_lo_sh);
            __m256i q5_hi = _mm256_or_si256(q5_high_nib, hb_hi_sh);
            __m256i q8_lo = _mm256_loadu_si256((const __m256i*)q8);
            __m256i q8_hi = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p16_lo = _mm256_maddubs_epi16(q5_lo, q8_lo);
            __m256i p16_hi = _mm256_maddubs_epi16(q5_hi, q8_hi);
            __m256i p32_lo = _mm256_madd_epi16(p16_lo, ONE16);
            __m256i p32_hi = _mm256_madd_epi16(p16_hi, ONE16);
            p32_lo = _mm256_mullo_epi32(p32_lo, _mm256_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm256_mullo_epi32(p32_hi, _mm256_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(p32_lo, p32_hi));
            q5 += 32;
            q8 += 64;
        }
        int32_t isum = hsum_i32_8(isum_v);
        acc_f = _mm256_add_ps(acc_f, _mm256_set1_ps(d * (float)isum - dmin * (float)isum_mins));
    }
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

// ===========================================================================
// Q6_K x Q8_K dot, AVX2 tier
//
// 6-bit weights: low 4 bits from ql, high 2 bits from qh. Dot in raw
// 6-bit (unsigned) space then subtract 32 * isum_mins at the end.
// (Each weight is (w - 32); sum(scale * (w-32) * q8) = sum(scale*w*q8)
// - 32*sum(scale*sum(q8)) = isum - 32 * isum_mins.)
// isum_mins side: 16 i16 bsums * 16 signed int8 scales (sign-extended).
// ===========================================================================
SIMD_FN("avx2,fma,f16c")
float q6k_row_dot_q8k_avx2(int nb, const q6k_block* x, const q8k_block* y) {
    const __m256i M4    = _mm256_set1_epi8(0x0F);
    const __m256i M3    = _mm256_set1_epi8(0x03);
    const __m256i ONE16 = _mm256_set1_epi16(1);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        // isum_mins = sum(bsums[j] * scales[j]) over 16 entries
        const __m256i bsums_v = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i sc8     = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i sc16    = _mm256_cvtepi8_epi16(sc8);  // sign-extend to i16x16
        const __m256i p_mins  = _mm256_madd_epi16(bsums_v, sc16);  // 8 i32
        int32_t isum_mins = hsum_i32_8(p_mins);

        __m256i isum_v = _mm256_setzero_si256();
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        const int8_t*  sc = x[i].scales;
        // 2 outer halves; each processes 128 weights as 4 sub-blocks of 32.
        for (int j = 0; j < QK_K / 128; j++) {
            __m256i qh_v   = _mm256_loadu_si256((const __m256i*)qh);  // 32 qh bytes
            // First half: low nibble of ql[0..63] + 2 high bits from qh (positions 0 and 2)
            // We process 32 weights at a time using 32 q8 bytes.
            //   weights[l]      = (ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)   for l in [0..32)  *scale[0..1]*
            //   weights[l+32]   = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)   for l in [0..32)  *scale[2..3]*
            //   weights[l+64]   = (ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4)   for l in [0..32)  *scale[4..5]*
            //   weights[l+96]   = (ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)   for l in [0..32)  *scale[6..7]*
            // But scales are applied per 16-elem sub-block, not 32. Each 32-elem chunk
            // above uses TWO consecutive scales [sc[0..1]], [sc[2..3]] etc.
            // To keep this readable, accumulate 32-wide partial dot per sub-block then
            // do partial mullo by *splitting* the 8 i32 lanes into halves where each
            // half gets a different scale. Simpler: process two 16-elem sub-blocks
            // separately so each gets its own scalar mullo. That doubles the inner
            // ops but keeps clarity.
            // For compactness here, we still use 32-elem chunks but post-process
            // the 8 i32 lanes by *blending* two scale broadcasts (low 4 lanes
            // get one scale, high 4 lanes get the next). Per maddubs+madd, lanes
            // 0..3 correspond to bytes 0..15 and lanes 4..7 to bytes 16..31 (the
            // shuffle is determined by the AVX2 lane semantics: each 128-bit
            // half is independent and madd preserves halves).
            __m256i qlv0   = _mm256_loadu_si256((const __m256i*)ql);        // ql[0..31]
            __m256i qlv1   = _mm256_loadu_si256((const __m256i*)(ql + 32)); // ql[32..63]
            __m256i ql0_lo = _mm256_and_si256(qlv0, M4);
            __m256i ql1_lo = _mm256_and_si256(qlv1, M4);
            __m256i ql0_hi = _mm256_and_si256(_mm256_srli_epi16(qlv0, 4), M4);
            __m256i ql1_hi = _mm256_and_si256(_mm256_srli_epi16(qlv1, 4), M4);
            // qh high 2-bit chunks
            __m256i qh_0 = _mm256_slli_epi16(_mm256_and_si256(qh_v, M3), 4);
            __m256i qh_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), M3), 4);
            __m256i qh_4 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), M3), 4);
            __m256i qh_6 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), M3), 4);
            __m256i w0 = _mm256_or_si256(ql0_lo, qh_0);
            __m256i w1 = _mm256_or_si256(ql1_lo, qh_2);
            __m256i w2 = _mm256_or_si256(ql0_hi, qh_4);
            __m256i w3 = _mm256_or_si256(ql1_hi, qh_6);
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +   0));
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 +  32));
            __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 +  64));
            __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 +  96));
            // Each 32-byte chunk's 8 i32 lanes: low 4 from low 16 bytes, high 4 from high 16 bytes.
            // sc[0] applies to bytes [0..15], sc[1] to [16..31]. Build a scale vector accordingly.
            __m256i sv0 = _mm256_set_epi32(sc[1],sc[1],sc[1],sc[1], sc[0],sc[0],sc[0],sc[0]);
            __m256i sv1 = _mm256_set_epi32(sc[3],sc[3],sc[3],sc[3], sc[2],sc[2],sc[2],sc[2]);
            __m256i sv2 = _mm256_set_epi32(sc[5],sc[5],sc[5],sc[5], sc[4],sc[4],sc[4],sc[4]);
            __m256i sv3 = _mm256_set_epi32(sc[7],sc[7],sc[7],sc[7], sc[6],sc[6],sc[6],sc[6]);
            __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, q8_0), ONE16);
            __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, q8_1), ONE16);
            __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(w2, q8_2), ONE16);
            __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(w3, q8_3), ONE16);
            p0 = _mm256_mullo_epi32(p0, sv0);
            p1 = _mm256_mullo_epi32(p1, sv1);
            p2 = _mm256_mullo_epi32(p2, sv2);
            p3 = _mm256_mullo_epi32(p3, sv3);
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }
        int32_t isum = hsum_i32_8(isum_v);
        float fp = (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
        acc_f = _mm256_add_ps(acc_f, _mm256_set1_ps(fp));
    }
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

// ===========================================================================
// AVX-VNNI variants of Q4_K / Q5_K / Q6_K.
//
// Identical structure to the AVX2 versions, but the inner per-32-byte
// dot becomes a single _mm256_dpbusd_avx_epi32 instead of
// (maddubs + madd). Weights are unsigned (4/5/6-bit fit in u8) and the
// q8 activation byte is signed int8, which is exactly dpbusd's u8 x s8
// signature -- no sign-trick needed (unlike Q8_0 which has signed-x-signed).
// ===========================================================================

SIMD_FN("avxvnni,avx2,fma,f16c")
float q4k_row_dot_q8k_avxvnni(int nb, const q4k_block* x, const q8k_block* y) {
    const __m256i M4 = _mm256_set1_epi8(0x0F);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        // isum_mins via hadd_epi16 + madd (same as AVX2)
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)mins));
        const __m128i mp = _mm_madd_epi16(bs_pair, mins_i16);
        __m128i sh = _mm_shuffle_epi32(mp, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mp, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        __m256i isum_v = _mm256_setzero_si256();
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m256i q4_bytes = _mm256_loadu_si256((const __m256i*)q4);
            __m256i q4_lo = _mm256_and_si256(q4_bytes, M4);
            __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), M4);
            __m256i q8_lo = _mm256_loadu_si256((const __m256i*)q8);
            __m256i q8_hi = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p32_lo = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q4_lo, q8_lo);
            __m256i p32_hi = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q4_hi, q8_hi);
            p32_lo = _mm256_mullo_epi32(p32_lo, _mm256_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm256_mullo_epi32(p32_hi, _mm256_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(p32_lo, p32_hi));
            q4 += 32;
            q8 += 64;
        }
        int32_t isum = hsum_i32_8(isum_v);
        acc_f = _mm256_add_ps(acc_f, _mm256_set1_ps(d * (float)isum - dmin * (float)isum_mins));
    }
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

SIMD_FN("avxvnni,avx2,fma,f16c")
float q5k_row_dot_q8k_avxvnni(int nb, const q5k_block* x, const q8k_block* y) {
    const __m256i M4 = _mm256_set1_epi8(0x0F);
    const __m256i M1 = _mm256_set1_epi8(0x01);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)mins));
        const __m128i mp = _mm_madd_epi16(bs_pair, mins_i16);
        __m128i sh = _mm_shuffle_epi32(mp, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mp, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        __m256i isum_v = _mm256_setzero_si256();
        __m256i qh_v = _mm256_loadu_si256((const __m256i*)x[i].qh);
        const uint8_t* q5 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m256i q5v = _mm256_loadu_si256((const __m256i*)q5);
            __m256i q5_low_nib  = _mm256_and_si256(q5v, M4);
            __m256i q5_high_nib = _mm256_and_si256(_mm256_srli_epi16(q5v, 4), M4);
            __m256i hb_lo = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2*j),     M1);
            __m256i hb_hi = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2*j + 1), M1);
            __m256i q5_lo = _mm256_or_si256(q5_low_nib,  _mm256_slli_epi16(hb_lo, 4));
            __m256i q5_hi = _mm256_or_si256(q5_high_nib, _mm256_slli_epi16(hb_hi, 4));
            __m256i q8_lo = _mm256_loadu_si256((const __m256i*)q8);
            __m256i q8_hi = _mm256_loadu_si256((const __m256i*)(q8 + 32));
            __m256i p32_lo = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q5_lo, q8_lo);
            __m256i p32_hi = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q5_hi, q8_hi);
            p32_lo = _mm256_mullo_epi32(p32_lo, _mm256_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm256_mullo_epi32(p32_hi, _mm256_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(p32_lo, p32_hi));
            q5 += 32;
            q8 += 64;
        }
        int32_t isum = hsum_i32_8(isum_v);
        acc_f = _mm256_add_ps(acc_f, _mm256_set1_ps(d * (float)isum - dmin * (float)isum_mins));
    }
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

SIMD_FN("avxvnni,avx2,fma,f16c")
float q6k_row_dot_q8k_avxvnni(int nb, const q6k_block* x, const q8k_block* y) {
    const __m256i M4 = _mm256_set1_epi8(0x0F);
    const __m256i M3 = _mm256_set1_epi8(0x03);
    __m256 acc_f = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const __m256i bsums_v = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i sc8     = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i sc16    = _mm256_cvtepi8_epi16(sc8);
        const __m256i p_mins  = _mm256_madd_epi16(bsums_v, sc16);
        int32_t isum_mins = hsum_i32_8(p_mins);
        __m256i isum_v = _mm256_setzero_si256();
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        const int8_t*  sc = x[i].scales;
        for (int j = 0; j < QK_K / 128; j++) {
            __m256i qh_v = _mm256_loadu_si256((const __m256i*)qh);
            __m256i qlv0 = _mm256_loadu_si256((const __m256i*)ql);
            __m256i qlv1 = _mm256_loadu_si256((const __m256i*)(ql + 32));
            __m256i ql0_lo = _mm256_and_si256(qlv0, M4);
            __m256i ql1_lo = _mm256_and_si256(qlv1, M4);
            __m256i ql0_hi = _mm256_and_si256(_mm256_srli_epi16(qlv0, 4), M4);
            __m256i ql1_hi = _mm256_and_si256(_mm256_srli_epi16(qlv1, 4), M4);
            __m256i qh_0 = _mm256_slli_epi16(_mm256_and_si256(qh_v, M3), 4);
            __m256i qh_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), M3), 4);
            __m256i qh_4 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), M3), 4);
            __m256i qh_6 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), M3), 4);
            __m256i w0 = _mm256_or_si256(ql0_lo, qh_0);
            __m256i w1 = _mm256_or_si256(ql1_lo, qh_2);
            __m256i w2 = _mm256_or_si256(ql0_hi, qh_4);
            __m256i w3 = _mm256_or_si256(ql1_hi, qh_6);
            __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +   0));
            __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 +  32));
            __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 +  64));
            __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 +  96));
            __m256i sv0 = _mm256_set_epi32(sc[1],sc[1],sc[1],sc[1], sc[0],sc[0],sc[0],sc[0]);
            __m256i sv1 = _mm256_set_epi32(sc[3],sc[3],sc[3],sc[3], sc[2],sc[2],sc[2],sc[2]);
            __m256i sv2 = _mm256_set_epi32(sc[5],sc[5],sc[5],sc[5], sc[4],sc[4],sc[4],sc[4]);
            __m256i sv3 = _mm256_set_epi32(sc[7],sc[7],sc[7],sc[7], sc[6],sc[6],sc[6],sc[6]);
            __m256i p0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w0, q8_0);
            __m256i p1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w1, q8_1);
            __m256i p2 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w2, q8_2);
            __m256i p3 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w3, q8_3);
            p0 = _mm256_mullo_epi32(p0, sv0);
            p1 = _mm256_mullo_epi32(p1, sv1);
            p2 = _mm256_mullo_epi32(p2, sv2);
            p3 = _mm256_mullo_epi32(p3, sv3);
            isum_v = _mm256_add_epi32(isum_v, _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }
        int32_t isum = hsum_i32_8(isum_v);
        float fp = (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
        acc_f = _mm256_add_ps(acc_f, _mm256_set1_ps(fp));
    }
    __m128 lo = _mm256_castps256_ps128(acc_f);
    __m128 hi = _mm256_extractf128_ps(acc_f, 1);
    __m128 sum_v = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum_v);
    __m128 sums = _mm_add_ps(sum_v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums) / 8.0f;
}

// ===========================================================================
// AVX1 (128-bit SSE4.1) variants of Q4_K / Q5_K / Q6_K.
//
// Ivy Bridge has no 256-bit integer ops -- only 256-bit FP. So we fall
// back to 128-bit SSE4.1 for the int8 dot work, then mul by scale and
// accumulate in __m128i. Each AVX2 32-byte iteration becomes two 16-byte
// SSE iterations fused in one outer step.
// ===========================================================================

SIMD_FN("avx,f16c,ssse3,sse4.1")
float q4k_row_dot_q8k_avx1(int nb, const q4k_block* x, const q8k_block* y) {
    const __m128i M4 = _mm_set1_epi8(0x0F);
    const __m128i ONE16 = _mm_set1_epi16(1);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)mins));
        const __m128i mp = _mm_madd_epi16(bs_pair, mins_i16);
        __m128i sh = _mm_shuffle_epi32(mp, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mp, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        __m128i isum_v = _mm_setzero_si128();
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m128i q4_a = _mm_loadu_si128((const __m128i*)q4);
            __m128i q4_b = _mm_loadu_si128((const __m128i*)(q4 + 16));
            __m128i q4_lo_a = _mm_and_si128(q4_a, M4);
            __m128i q4_lo_b = _mm_and_si128(q4_b, M4);
            __m128i q4_hi_a = _mm_and_si128(_mm_srli_epi16(q4_a, 4), M4);
            __m128i q4_hi_b = _mm_and_si128(_mm_srli_epi16(q4_b, 4), M4);
            __m128i q8_lo_a = _mm_loadu_si128((const __m128i*)q8);
            __m128i q8_lo_b = _mm_loadu_si128((const __m128i*)(q8 + 16));
            __m128i q8_hi_a = _mm_loadu_si128((const __m128i*)(q8 + 32));
            __m128i q8_hi_b = _mm_loadu_si128((const __m128i*)(q8 + 48));
            __m128i p32_lo = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(q4_lo_a, q8_lo_a), ONE16),
                _mm_madd_epi16(_mm_maddubs_epi16(q4_lo_b, q8_lo_b), ONE16));
            __m128i p32_hi = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(q4_hi_a, q8_hi_a), ONE16),
                _mm_madd_epi16(_mm_maddubs_epi16(q4_hi_b, q8_hi_b), ONE16));
            p32_lo = _mm_mullo_epi32(p32_lo, _mm_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm_mullo_epi32(p32_hi, _mm_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm_add_epi32(isum_v, _mm_add_epi32(p32_lo, p32_hi));
            q4 += 32;
            q8 += 64;
        }
        sh = _mm_shuffle_epi32(isum_v, _MM_SHUFFLE(1,0,3,2));
        __m128i isum_h = _mm_add_epi32(isum_v, sh);
        sh = _mm_shuffle_epi32(isum_h, _MM_SHUFFLE(2,3,0,1));
        isum_h = _mm_add_epi32(isum_h, sh);
        int32_t isum = _mm_cvtsi128_si32(isum_h);
        sumf += d * (float)isum - dmin * (float)isum_mins;
    }
    return sumf;
}

SIMD_FN("avx,f16c,ssse3,sse4.1")
float q5k_row_dot_q8k_avx1(int nb, const q5k_block* x, const q8k_block* y) {
    const __m128i M4 = _mm_set1_epi8(0x0F);
    const __m128i M1 = _mm_set1_epi8(0x01);
    const __m128i ONE16 = _mm_set1_epi16(1);
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d    = y[i].d * q_fp16_to_f32(x[i].d);
        const float dmin = y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t scales[8], mins[8];
        for (int j = 0; j < 8; j++) q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        const __m128i bs0 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs1 = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i bs_pair = _mm_hadd_epi16(bs0, bs1);
        const __m128i mins_i16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)mins));
        const __m128i mp = _mm_madd_epi16(bs_pair, mins_i16);
        __m128i sh = _mm_shuffle_epi32(mp, _MM_SHUFFLE(1,0,3,2));
        __m128i s  = _mm_add_epi32(mp, sh);
        sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
        s = _mm_add_epi32(s, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(s);
        __m128i isum_v = _mm_setzero_si128();
        __m128i qh_a = _mm_loadu_si128((const __m128i*)x[i].qh);
        __m128i qh_b = _mm_loadu_si128((const __m128i*)(x[i].qh + 16));
        const uint8_t* q5 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; j++) {
            __m128i q5v_a = _mm_loadu_si128((const __m128i*)q5);
            __m128i q5v_b = _mm_loadu_si128((const __m128i*)(q5 + 16));
            __m128i q5_lo_nib_a = _mm_and_si128(q5v_a, M4);
            __m128i q5_lo_nib_b = _mm_and_si128(q5v_b, M4);
            __m128i q5_hi_nib_a = _mm_and_si128(_mm_srli_epi16(q5v_a, 4), M4);
            __m128i q5_hi_nib_b = _mm_and_si128(_mm_srli_epi16(q5v_b, 4), M4);
            __m128i hb_lo_a = _mm_and_si128(_mm_srli_epi16(qh_a, 2*j),     M1);
            __m128i hb_lo_b = _mm_and_si128(_mm_srli_epi16(qh_b, 2*j),     M1);
            __m128i hb_hi_a = _mm_and_si128(_mm_srli_epi16(qh_a, 2*j + 1), M1);
            __m128i hb_hi_b = _mm_and_si128(_mm_srli_epi16(qh_b, 2*j + 1), M1);
            __m128i q5_lo_a = _mm_or_si128(q5_lo_nib_a, _mm_slli_epi16(hb_lo_a, 4));
            __m128i q5_lo_b = _mm_or_si128(q5_lo_nib_b, _mm_slli_epi16(hb_lo_b, 4));
            __m128i q5_hi_a = _mm_or_si128(q5_hi_nib_a, _mm_slli_epi16(hb_hi_a, 4));
            __m128i q5_hi_b = _mm_or_si128(q5_hi_nib_b, _mm_slli_epi16(hb_hi_b, 4));
            __m128i q8_lo_a = _mm_loadu_si128((const __m128i*)q8);
            __m128i q8_lo_b = _mm_loadu_si128((const __m128i*)(q8 + 16));
            __m128i q8_hi_a = _mm_loadu_si128((const __m128i*)(q8 + 32));
            __m128i q8_hi_b = _mm_loadu_si128((const __m128i*)(q8 + 48));
            __m128i p32_lo = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(q5_lo_a, q8_lo_a), ONE16),
                _mm_madd_epi16(_mm_maddubs_epi16(q5_lo_b, q8_lo_b), ONE16));
            __m128i p32_hi = _mm_add_epi32(
                _mm_madd_epi16(_mm_maddubs_epi16(q5_hi_a, q8_hi_a), ONE16),
                _mm_madd_epi16(_mm_maddubs_epi16(q5_hi_b, q8_hi_b), ONE16));
            p32_lo = _mm_mullo_epi32(p32_lo, _mm_set1_epi32((int32_t)scales[2*j+0]));
            p32_hi = _mm_mullo_epi32(p32_hi, _mm_set1_epi32((int32_t)scales[2*j+1]));
            isum_v = _mm_add_epi32(isum_v, _mm_add_epi32(p32_lo, p32_hi));
            q5 += 32;
            q8 += 64;
        }
        sh = _mm_shuffle_epi32(isum_v, _MM_SHUFFLE(1,0,3,2));
        __m128i isum_h = _mm_add_epi32(isum_v, sh);
        sh = _mm_shuffle_epi32(isum_h, _MM_SHUFFLE(2,3,0,1));
        isum_h = _mm_add_epi32(isum_h, sh);
        int32_t isum = _mm_cvtsi128_si32(isum_h);
        sumf += d * (float)isum - dmin * (float)isum_mins;
    }
    return sumf;
}

SIMD_FN("avx,f16c,ssse3,sse4.1")
float q6k_row_dot_q8k_avx1(int nb, const q6k_block* x, const q8k_block* y) {
    const __m128i M4 = _mm_set1_epi8(0x0F);
    const __m128i M3 = _mm_set1_epi8(0x03);
    const __m128i ONE16 = _mm_set1_epi16(1);
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        // isum_mins: 16 i16 bsums * 16 signed int8 scales, two 128-bit halves
        const __m128i bs_lo = _mm_loadu_si128((const __m128i*)(y[i].bsums + 0));
        const __m128i bs_hi = _mm_loadu_si128((const __m128i*)(y[i].bsums + 8));
        const __m128i sc8  = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i sc16_lo = _mm_cvtepi8_epi16(sc8);
        const __m128i sc16_hi = _mm_cvtepi8_epi16(_mm_srli_si128(sc8, 8));
        const __m128i p_mins = _mm_add_epi32(
            _mm_madd_epi16(bs_lo, sc16_lo),
            _mm_madd_epi16(bs_hi, sc16_hi));
        __m128i sh = _mm_shuffle_epi32(p_mins, _MM_SHUFFLE(1,0,3,2));
        __m128i ss  = _mm_add_epi32(p_mins, sh);
        sh = _mm_shuffle_epi32(ss, _MM_SHUFFLE(2,3,0,1));
        ss = _mm_add_epi32(ss, sh);
        int32_t isum_mins = _mm_cvtsi128_si32(ss);
        __m128i isum_v = _mm_setzero_si128();
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        const int8_t*  sc = x[i].scales;
        for (int j = 0; j < QK_K / 128; j++) {
            // Process 128 weights as 8 chunks of 16, each with its own scale.
            // Each chunk: weights = (ql_low4 | (qh_2bit << 4)) for first 4 chunks,
            //             weights = (ql_hi4  | (qh_2bit << 4)) for next 4 chunks.
            __m128i qh_a = _mm_loadu_si128((const __m128i*)qh);
            __m128i qh_b = _mm_loadu_si128((const __m128i*)(qh + 16));
            // 4 chunks of 16 from ql[0..63] low nibbles + qh bits {0..1, 0..1, 2..3, 2..3}
            // Inspecting the scalar:
            //   chunk 0:  ql[l]    & 0xF  | ((qh[l]    >> 0) & 3) << 4    *sc[0]*  l in [0..16)
            //   chunk 1:  ql[l+16] & 0xF  | ((qh[l+16] >> 0) & 3) << 4    *sc[1]*  l in [0..16)
            //   chunk 2:  ql[l+32] & 0xF  | ((qh[l]    >> 2) & 3) << 4    *sc[2]*  l in [0..16)
            //   chunk 3:  ql[l+48] & 0xF  | ((qh[l+16] >> 2) & 3) << 4    *sc[3]*  l in [0..16)
            //   chunk 4:  ql[l]    >>  4  | ((qh[l]    >> 4) & 3) << 4    *sc[4]*
            //   chunk 5:  ql[l+16] >>  4  | ((qh[l+16] >> 4) & 3) << 4    *sc[5]*
            //   chunk 6:  ql[l+32] >>  4  | ((qh[l]    >> 6) & 3) << 4    *sc[6]*
            //   chunk 7:  ql[l+48] >>  4  | ((qh[l+16] >> 6) & 3) << 4    *sc[7]*
            __m128i ql_lo_a = _mm_loadu_si128((const __m128i*)(ql +  0));
            __m128i ql_lo_b = _mm_loadu_si128((const __m128i*)(ql + 16));
            __m128i ql_lo_c = _mm_loadu_si128((const __m128i*)(ql + 32));
            __m128i ql_lo_d = _mm_loadu_si128((const __m128i*)(ql + 48));
            __m128i ql_n0 = _mm_and_si128(ql_lo_a, M4);
            __m128i ql_n1 = _mm_and_si128(ql_lo_b, M4);
            __m128i ql_n2 = _mm_and_si128(ql_lo_c, M4);
            __m128i ql_n3 = _mm_and_si128(ql_lo_d, M4);
            __m128i ql_n4 = _mm_and_si128(_mm_srli_epi16(ql_lo_a, 4), M4);
            __m128i ql_n5 = _mm_and_si128(_mm_srli_epi16(ql_lo_b, 4), M4);
            __m128i ql_n6 = _mm_and_si128(_mm_srli_epi16(ql_lo_c, 4), M4);
            __m128i ql_n7 = _mm_and_si128(_mm_srli_epi16(ql_lo_d, 4), M4);
            __m128i qh_a_0 = _mm_slli_epi16(_mm_and_si128(qh_a, M3), 4);
            __m128i qh_b_0 = _mm_slli_epi16(_mm_and_si128(qh_b, M3), 4);
            __m128i qh_a_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_a, 2), M3), 4);
            __m128i qh_b_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_b, 2), M3), 4);
            __m128i qh_a_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_a, 4), M3), 4);
            __m128i qh_b_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_b, 4), M3), 4);
            __m128i qh_a_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_a, 6), M3), 4);
            __m128i qh_b_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_b, 6), M3), 4);
            __m128i w0 = _mm_or_si128(ql_n0, qh_a_0);
            __m128i w1 = _mm_or_si128(ql_n1, qh_b_0);
            __m128i w2 = _mm_or_si128(ql_n2, qh_a_2);
            __m128i w3 = _mm_or_si128(ql_n3, qh_b_2);
            __m128i w4 = _mm_or_si128(ql_n4, qh_a_4);
            __m128i w5 = _mm_or_si128(ql_n5, qh_b_4);
            __m128i w6 = _mm_or_si128(ql_n6, qh_a_6);
            __m128i w7 = _mm_or_si128(ql_n7, qh_b_6);
            __m128i q8_0 = _mm_loadu_si128((const __m128i*)(q8 +   0));
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)(q8 +  16));
            __m128i q8_2 = _mm_loadu_si128((const __m128i*)(q8 +  32));
            __m128i q8_3 = _mm_loadu_si128((const __m128i*)(q8 +  48));
            __m128i q8_4 = _mm_loadu_si128((const __m128i*)(q8 +  64));
            __m128i q8_5 = _mm_loadu_si128((const __m128i*)(q8 +  80));
            __m128i q8_6 = _mm_loadu_si128((const __m128i*)(q8 +  96));
            __m128i q8_7 = _mm_loadu_si128((const __m128i*)(q8 + 112));
            __m128i p0 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w0, q8_0), ONE16), _mm_set1_epi32(sc[0]));
            __m128i p1 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w1, q8_1), ONE16), _mm_set1_epi32(sc[1]));
            __m128i p2 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w2, q8_2), ONE16), _mm_set1_epi32(sc[2]));
            __m128i p3 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w3, q8_3), ONE16), _mm_set1_epi32(sc[3]));
            __m128i p4 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w4, q8_4), ONE16), _mm_set1_epi32(sc[4]));
            __m128i p5 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w5, q8_5), ONE16), _mm_set1_epi32(sc[5]));
            __m128i p6 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w6, q8_6), ONE16), _mm_set1_epi32(sc[6]));
            __m128i p7 = _mm_mullo_epi32(_mm_madd_epi16(_mm_maddubs_epi16(w7, q8_7), ONE16), _mm_set1_epi32(sc[7]));
            __m128i sum01 = _mm_add_epi32(p0, p1);
            __m128i sum23 = _mm_add_epi32(p2, p3);
            __m128i sum45 = _mm_add_epi32(p4, p5);
            __m128i sum67 = _mm_add_epi32(p6, p7);
            isum_v = _mm_add_epi32(isum_v, _mm_add_epi32(_mm_add_epi32(sum01, sum23), _mm_add_epi32(sum45, sum67)));
            ql += 64;
            qh += 32;
            sc += 8;
            q8 += 128;
        }
        __m128i sh2 = _mm_shuffle_epi32(isum_v, _MM_SHUFFLE(1,0,3,2));
        __m128i isum_h = _mm_add_epi32(isum_v, sh2);
        sh2 = _mm_shuffle_epi32(isum_h, _MM_SHUFFLE(2,3,0,1));
        isum_h = _mm_add_epi32(isum_h, sh2);
        int32_t isum = _mm_cvtsi128_si32(isum_h);
        sum += (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}

// ===========================================================================
// fp16 4x8 HGEMM with explicit F16C conversion (vcvtph2ps).
//
// The portable simd_hgemm_tiled in simd.c uses q_fp16_to_f32 bit-twiddle
// because __FLT16_MANT_DIG__ isn't defined on x86 at baseline (clang
// only enables it under -mavx512fp16, which our CPUs don't have). The
// bit-twiddle is ~30x slower than vcvtph2ps. This target("f16c") variant
// loads 4 fp16 values into __m128i, converts to 4 fp32 via _mm_cvtph_ps,
// extracts scalars, and the rest of the inner loop is identical to the
// portable sgemm 4x8 kernel.
// ===========================================================================
SIMD_FN("avx,f16c")
void simd_hgemm_tiled_f16c(int M, int N, int K,
                           const uint16_t* A,  int lda,
                           const uint16_t* Bt, int ldb,
                           float* C, int ldc);

// Single-tile 4x8 kernel, F16C-using.
__attribute__((target("avx,f16c"), noinline))
static void hgemm_kernel_4x8_f16c(int K,
                                  const uint16_t* A,  int lda,
                                  const uint16_t* Bt, int ldb,
                                  float* C, int ldc) {
    float c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
    float c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
    float c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
    float c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;
    for (int k = 0; k < K; k++) {
        // 4 A values from 4 rows  ->  one __m128i (4 fp16 packed in low 64 bits)
        __m128i a16 = _mm_set_epi16(0, 0, 0, 0,
            (short)A[3*lda + k], (short)A[2*lda + k],
            (short)A[1*lda + k], (short)A[0*lda + k]);
        __m128 af = _mm_cvtph_ps(a16);
        float a0 = _mm_cvtss_f32(af);
        float a1 = _mm_cvtss_f32(_mm_shuffle_ps(af, af, _MM_SHUFFLE(1,1,1,1)));
        float a2 = _mm_cvtss_f32(_mm_shuffle_ps(af, af, _MM_SHUFFLE(2,2,2,2)));
        float a3 = _mm_cvtss_f32(_mm_shuffle_ps(af, af, _MM_SHUFFLE(3,3,3,3)));
        // 8 B values from 8 rows  ->  __m128i (8 fp16 = 128 bits) -> __m256 (8 fp32)
        __m128i b16 = _mm_set_epi16(
            (short)Bt[7*ldb + k], (short)Bt[6*ldb + k],
            (short)Bt[5*ldb + k], (short)Bt[4*ldb + k],
            (short)Bt[3*ldb + k], (short)Bt[2*ldb + k],
            (short)Bt[1*ldb + k], (short)Bt[0*ldb + k]);
        __m256 bf = _mm256_cvtph_ps(b16);
        float bvec[8] __attribute__((aligned(32)));
        _mm256_storeu_ps(bvec, bf);
        float b0=bvec[0],b1=bvec[1],b2=bvec[2],b3=bvec[3];
        float b4=bvec[4],b5=bvec[5],b6=bvec[6],b7=bvec[7];
        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
        c04 += a0*b4; c05 += a0*b5; c06 += a0*b6; c07 += a0*b7;
        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
        c14 += a1*b4; c15 += a1*b5; c16 += a1*b6; c17 += a1*b7;
        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
        c24 += a2*b4; c25 += a2*b5; c26 += a2*b6; c27 += a2*b7;
        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
        c34 += a3*b4; c35 += a3*b5; c36 += a3*b6; c37 += a3*b7;
    }
    C[0*ldc+0]=c00; C[0*ldc+1]=c01; C[0*ldc+2]=c02; C[0*ldc+3]=c03;
    C[0*ldc+4]=c04; C[0*ldc+5]=c05; C[0*ldc+6]=c06; C[0*ldc+7]=c07;
    C[1*ldc+0]=c10; C[1*ldc+1]=c11; C[1*ldc+2]=c12; C[1*ldc+3]=c13;
    C[1*ldc+4]=c14; C[1*ldc+5]=c15; C[1*ldc+6]=c16; C[1*ldc+7]=c17;
    C[2*ldc+0]=c20; C[2*ldc+1]=c21; C[2*ldc+2]=c22; C[2*ldc+3]=c23;
    C[2*ldc+4]=c24; C[2*ldc+5]=c25; C[2*ldc+6]=c26; C[2*ldc+7]=c27;
    C[3*ldc+0]=c30; C[3*ldc+1]=c31; C[3*ldc+2]=c32; C[3*ldc+3]=c33;
    C[3*ldc+4]=c34; C[3*ldc+5]=c35; C[3*ldc+6]=c36; C[3*ldc+7]=c37;
}

__attribute__((target("avx,f16c"), noinline))
static void hgemm_edge_f16c(int M, int N, int K,
                            const uint16_t* A, int lda,
                            const uint16_t* Bt, int ldb,
                            float* C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)A[m*lda+k])));
                float b = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)Bt[n*ldb+k])));
                acc += a * b;
            }
            C[m*ldc + n] = acc;
        }
    }
}

void simd_hgemm_tiled_f16c(int M, int N, int K,
                           const uint16_t* A,  int lda,
                           const uint16_t* Bt, int ldb,
                           float* C, int ldc) {
    const int MR = 4, NR = 8;
    int M_main = (M / MR) * MR;
    int N_main = (N / NR) * NR;
    for (int m = 0; m < M_main; m += MR) {
        for (int n = 0; n < N_main; n += NR) {
            hgemm_kernel_4x8_f16c(K, A + m*lda, lda, Bt + n*ldb, ldb,
                                  C + m*ldc + n, ldc);
        }
        if (N_main < N) {
            hgemm_edge_f16c(MR, N - N_main, K,
                            A + m*lda, lda, Bt + N_main*ldb, ldb,
                            C + m*ldc + N_main, ldc);
        }
    }
    if (M_main < M) {
        hgemm_edge_f16c(M - M_main, N, K,
                        A + M_main*lda, lda, Bt, ldb,
                        C + M_main*ldc, ldc);
    }
}

#undef SIMD_FN
