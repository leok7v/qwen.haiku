// SPDX-License-Identifier: MIT
//
// quants.h - portable block typedefs + scalar reference for the
// llama.cpp-style K-quants and Q8_0 used by neon.c / avx.c.
//
// This file is the ONLY place block layouts and bit-level conversions
// are defined. Everything that consumes them (neon.c, avx.c, the
// bench harness, the scalar references below) sees the same layout.
//
// Block layouts mirror ggml-quants.c (and llama.cpp/src/ggml-cpu).
// Field order and packed-ness are load-bearing: a half-byte different
// here breaks every SIMD kernel that loads via vld1q_u8 etc.
//
// Scalar reference dots match the INTEGER accumulation path that
// vdotq_s32 / _mm256_maddubs_epi16 produce. They do NOT dequantize
// first and then fp32-multiply -- that would round differently and
// drift from llama.cpp's emitted weights.

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// K-quant constants
// ---------------------------------------------------------------------------

#define QK_K   256              // super-block size for Q4_K / Q5_K / Q6_K
#define QK8_0   32              // block size for Q8_0

// ---------------------------------------------------------------------------
// Half-precision (IEEE 754 binary16) helpers.
//
// Used for the per-block scale `d` (and `dmin` where applicable) in
// every block type. We treat fp16 as raw uint16 bits and use the
// platform's _Float16 only when available.
//
// Conversion semantics: scale fields in K-quants are *exact* values
// at quantization time -- they were chosen to fit in fp16 and to be
// reproducible across implementations. So fp16<->fp32 round-trip
// must be deterministic. We use the standard IEEE binary16 rules
// (round-half-to-even) implemented by either:
//   - _Float16 cast on platforms that define __FLT16_MANT_DIG__
//     (clang on macOS arm64/x86_64, gcc 13+ on aarch64/x86_64)
//   - a portable bit-twiddling fallback for older toolchains
// ---------------------------------------------------------------------------

#if defined(__FLT16_MANT_DIG__)
    typedef _Float16 q_fp16_t;
    static inline float    q_fp16_to_f32(q_fp16_t h) { return (float)h; }
    static inline q_fp16_t q_f32_to_fp16(float    f) { return (q_fp16_t)f; }
#else
    // No native _Float16: store as uint16 with portable conversion.
    typedef uint16_t q_fp16_t;
    static inline float q_fp16_to_f32(q_fp16_t h) {
        uint32_t sign     = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp      = (h >> 10) & 0x1F;
        uint32_t mant     = h & 0x3FF;
        uint32_t out;
        if (exp == 0 && mant == 0) {
            out = sign;
        } else if (exp == 0x1F) {
            out = sign | (0xFFu << 23) | (mant << 13);
        } else if (exp != 0) {
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        } else {
            uint32_t e = 112u + 1u;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            out = sign | (e << 23) | (mant << 13);
        }
        float f;
        memcpy(&f, &out, sizeof(f));
        return f;
    }
    static inline q_fp16_t q_f32_to_fp16(float f) {
        uint32_t b;
        memcpy(&b, &f, sizeof(b));
        uint32_t sign = (b >> 16) & 0x8000u;
        int32_t  e    = (int32_t)((b >> 23) & 0xFF) - 112;
        uint32_t m    = (b & 0x7FFFFFu) >> 13;
        if (e >= 0x1F) { return (q_fp16_t)(sign | 0x7C00u); }
        if (e <= 0)    { return (q_fp16_t)sign; }
        return (q_fp16_t)(sign | ((uint32_t)e << 10) | m);
    }
#endif

// ---------------------------------------------------------------------------
// Block typedefs. Match ggml-quants.c byte-for-byte.
// ---------------------------------------------------------------------------

// Q8_K activation block: int8 + per-block fp32 scale + 16-wide bsums.
typedef struct {
    float   d;                  // per-block scale (1 / iscale)
    int8_t  qs[QK_K];           // 256 int8 weights
    int16_t bsums[QK_K / 16];   // 16 sums of 16-element chunks
} q8k_block;

// Q4_K: 4-bit weights with 6-bit scales+mins. 144 bytes/block.
typedef struct {
    q_fp16_t d;
    q_fp16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];      // 128 bytes of packed 4-bit weights
} q4k_block;

// Q5_K: 5-bit weights. 176 bytes/block.
typedef struct {
    q_fp16_t d;
    q_fp16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];      // high bits of 5-bit weights
    uint8_t  qs[QK_K / 2];      // low nibbles
} q5k_block;

// Q6_K: 6-bit weights (4 low + 2 high) + signed int8 scales. 210 bytes/block.
typedef struct {
    uint8_t  ql[QK_K / 2];      // 128 bytes low nibbles
    uint8_t  qh[QK_K / 4];      //  64 bytes high pairs
    int8_t   scales[QK_K / 16]; //  16 signed scales
    q_fp16_t d;                 // global per-block scale
} q6k_block;

// Q8_0: int8 + fp16 scale, 32-elem block.
typedef struct {
    q_fp16_t d;
    int8_t   qs[QK8_0];
} q8_0_block;

// ---------------------------------------------------------------------------
// Banker's-rounded nearest-int (matches llama.cpp nearest_int()). The
// rounding mode matters: differs from roundf for half-values, and the
// off-by-one drift compounds through matmul.
// ---------------------------------------------------------------------------
static inline int32_t q_nearest_int(float fval) {
    float val = fval + 12582912.f;          // 1.5 * 2^23
    int32_t i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

// ---------------------------------------------------------------------------
// Q8_K row quantizer. Port of llama.cpp's quantize_row_q8_K_ref.
//
// For each 256-elem super-block:
//   amax  = max(|src[i]|)              over i in [0..256)
//   xmax  = signed value at amax
//   d     = -xmax / 127                (the scale factor)
//   qs[i] = clip(nearest_int(src[i] / d), -128, 127)
// bsums[j] = sum(qs[j*16 + 0..15])     for j in [0..16)
//
// Convention from llama.cpp: iscale = -127 / xmax (note the minus).
// ---------------------------------------------------------------------------
static inline void q_quantize_row_q8_K(const float* src, q8k_block* dst, int64_t n) {
    int64_t nb = n / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const float* sb = src + b * QK_K;
        q8k_block* y = dst + b;
        float amax = 0.0f, xmax = 0.0f;
        for (int32_t i = 0; i < QK_K; i++) {
            float ax = fabsf(sb[i]);
            if (ax > amax) { amax = ax; xmax = sb[i]; }
        }
        if (amax == 0.0f) {
            y->d = 0.0f;
            memset(y->qs,    0, sizeof(y->qs));
            memset(y->bsums, 0, sizeof(y->bsums));
            continue;
        }
        float iscale = -127.0f / xmax;
        y->d = 1.0f / iscale;
        for (int32_t i = 0; i < QK_K; i++) {
            int32_t v = q_nearest_int(iscale * sb[i]);
            if (v > 127)  v = 127;
            if (v < -128) v = -128;
            y->qs[i] = (int8_t)v;
        }
        for (int32_t j = 0; j < QK_K / 16; j++) {
            int32_t s = 0;
            for (int32_t l = 0; l < 16; l++) s += y->qs[j * 16 + l];
            y->bsums[j] = (int16_t)s;
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 row quantizer. Per llama.cpp quantize_row_q8_0_ref.
// 32-elem block, d = amax/127, qs = clip(round(x/d)).
// ---------------------------------------------------------------------------
static inline void q_quantize_row_q8_0(const float* src, q8_0_block* dst, int64_t n) {
    int64_t nb = n / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int32_t j = 0; j < QK8_0; j++) {
            float v = fabsf(src[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d  = amax / 127.0f;
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        dst[i].d = q_f32_to_fp16(d);
        for (int32_t j = 0; j < QK8_0; j++) {
            int32_t v = q_nearest_int(src[i * QK8_0 + j] * id);
            if (v > 127)  v = 127;
            if (v < -128) v = -128;
            dst[i].qs[j] = (int8_t)v;
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K scale/min unpack. 12 bytes encode 8 pairs of (6-bit scale, 6-bit min).
// ---------------------------------------------------------------------------
static inline void q4k_get_scale_min(const uint8_t* sc, int32_t j,
                                     uint8_t* out_d, uint8_t* out_m) {
    if (j < 4) {
        *out_d = sc[j]     & 63;
        *out_m = sc[j + 4] & 63;
    } else {
        *out_d = (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4);
        *out_m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

// ===========================================================================
// SCALAR REFERENCE DOT PRODUCTS
//
// These are the bit-for-bit reference implementations every SIMD kernel
// is compared against. Each follows the same integer-accumulation
// pattern as ggml's NEON/AVX paths, NOT a dequantize-and-fp32-dot
// formulation. Specifically:
//
//   isum         = sum over j: (int8 weight) * (int8 activation)
//   isum_mins    = sum over j: scale_min * bsums
//   final_fp32   = d * isum - dmin * isum_mins
//
// The integer sums above are bit-exact across any SIMD implementation
// that uses the same accumulation tree (which sdot, maddubs, dpbusd
// all guarantee for int8 inputs).
// ===========================================================================

// Q8_0 x Q8_0 dot, k must be multiple of 32.
static inline float q_scalar_q8_0_dot_q8_0(const q8_0_block* x,
                                           const q8_0_block* y, int64_t k) {
    int64_t nb = k / QK8_0;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        int32_t sumi = 0;
        for (int32_t j = 0; j < QK8_0; j++) {
            sumi += (int32_t)x[i].qs[j] * (int32_t)y[i].qs[j];
        }
        sumf += (float)sumi * (q_fp16_to_f32(x[i].d) * q_fp16_to_f32(y[i].d));
    }
    return sumf;
}

// Q4_K x Q8_K dot. Processes `nb` super-blocks of one weight row vs
// one activation row.
//
// Per llama.cpp ggml_vec_dot_q4_K_q8_K (scalar GEN branch):
//   for each super-block i:
//       unpack 8 scales / 8 mins from x[i].scales
//       isum      = sum over 8 sub-blocks of (scale_d * sub_dot_int8)
//       isum_mins = sum over 8 sub-blocks of (scale_m * bsums[2j..2j+1])
//       sumf += d_eff * isum - dmin_eff * isum_mins
static inline float q_scalar_q4k_row_dot_q8k(int nb,
                                             const q4k_block* x,
                                             const q8k_block* y) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d    = (float)y[i].d * q_fp16_to_f32(x[i].d);
        float dmin = (float)y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t  scales[8], mins[8];
        for (int j = 0; j < 8; j++) {
            q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        }
        // sum of mins side: pair-up bsums (the q8 block has 16 bsums,
        // bsums[2j]+bsums[2j+1] corresponds to the 32 weights covered by
        // sub-block j's min)
        int32_t isum_mins = 0;
        for (int j = 0; j < 8; j++) {
            int32_t s = (int32_t)y[i].bsums[2*j] + (int32_t)y[i].bsums[2*j+1];
            isum_mins += s * (int32_t)mins[j];
        }
        // weight side: 8 sub-blocks, low nibbles use scales[2j+0] and
        // sit in qs[2j*16 .. 2j*16+15]; high nibbles use scales[2j+1]
        // and sit in qs[(2j+1)*16 .. (2j+1)*16+15]. q8 qs is contiguous.
        int32_t isum = 0;
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < QK_K/64; j++) {
            int32_t sumi1 = 0, sumi2 = 0;
            // low nibbles (32 weights, scales[2j+0])
            for (int l = 0; l < 32; l++) sumi1 += (int32_t)(q4[l] & 0x0F) * (int32_t)q8[l];
            // high nibbles (32 weights, scales[2j+1])
            for (int l = 0; l < 32; l++) sumi2 += (int32_t)(q4[l] >> 4)   * (int32_t)q8[32 + l];
            isum += sumi1 * (int32_t)scales[2*j+0]
                  + sumi2 * (int32_t)scales[2*j+1];
            q4 += 32;
            q8 += 64;
        }
        sumf += d * (float)isum - dmin * (float)isum_mins;
    }
    return sumf;
}

// Q5_K x Q8_K dot. Same shape as Q4_K but 5-bit weights (low nibble
// from qs[], high bit from qh[]).
static inline float q_scalar_q5k_row_dot_q8k(int nb,
                                             const q5k_block* x,
                                             const q8k_block* y) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d    = (float)y[i].d * q_fp16_to_f32(x[i].d);
        float dmin = (float)y[i].d * q_fp16_to_f32(x[i].dmin);
        uint8_t  scales[8], mins[8];
        for (int j = 0; j < 8; j++) {
            q4k_get_scale_min(x[i].scales, j, &scales[j], &mins[j]);
        }
        int32_t isum_mins = 0;
        for (int j = 0; j < 8; j++) {
            int32_t s = (int32_t)y[i].bsums[2*j] + (int32_t)y[i].bsums[2*j+1];
            isum_mins += s * (int32_t)mins[j];
        }
        int32_t isum = 0;
        const uint8_t* q5  = x[i].qs;
        const uint8_t* qh  = x[i].qh;
        const int8_t*  q8  = y[i].qs;
        uint8_t        u1  = 1, u2 = 2;
        for (int j = 0; j < QK_K/64; j++) {
            int32_t sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 32; l++) {
                int32_t w = (int32_t)(q5[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                sumi1 += w * (int32_t)q8[l];
            }
            for (int l = 0; l < 32; l++) {
                int32_t w = (int32_t)(q5[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
                sumi2 += w * (int32_t)q8[32 + l];
            }
            isum += sumi1 * (int32_t)scales[2*j+0]
                  + sumi2 * (int32_t)scales[2*j+1];
            q5 += 32;
            q8 += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += d * (float)isum - dmin * (float)isum_mins;
    }
    return sumf;
}

// Q6_K x Q8_K dot. 6-bit weights stored as (low 4 in ql, high 2 in qh)
// then offset by -32 to land in signed [-32..31]. Per llama.cpp scalar
// GEN branch: dot in raw 6-bit space then subtract 32 * sum(scale*bsums).
static inline float q_scalar_q6k_row_dot_q8k(int nb,
                                             const q6k_block* x,
                                             const q8k_block* y) {
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        // sum of mins side: pair-up bsums, multiply by signed int8 scales
        int32_t isum_mins = 0;
        for (int j = 0; j < QK_K/16; j++) {
            isum_mins += (int32_t)y[i].bsums[j] * (int32_t)x[i].scales[j];
        }
        int32_t isum = 0;
        const uint8_t* ql    = x[i].ql;
        const uint8_t* qh    = x[i].qh;
        const int8_t*  q8    = y[i].qs;
        const int8_t*  scale = x[i].scales;
        for (int j = 0; j < QK_K/128; j++) {
            // first 64 weights (4 sub-blocks of 16): low halves
            for (int l = 0; l < 16; l++) {
                int32_t w0 = (int32_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4));
                int32_t w1 = (int32_t)((ql[l + 16] & 0x0F) | (((qh[l + 16] >> 0) & 3) << 4));
                int32_t w2 = (int32_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4));
                int32_t w3 = (int32_t)((ql[l + 48] & 0x0F) | (((qh[l + 16] >> 2) & 3) << 4));
                isum += w0 * (int32_t)q8[l]      * (int32_t)scale[0]
                      + w1 * (int32_t)q8[l + 16] * (int32_t)scale[1]
                      + w2 * (int32_t)q8[l + 32] * (int32_t)scale[2]
                      + w3 * (int32_t)q8[l + 48] * (int32_t)scale[3];
            }
            // second 64 weights: high halves
            for (int l = 0; l < 16; l++) {
                int32_t w0 = (int32_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4));
                int32_t w1 = (int32_t)((ql[l + 16] >>   4) | (((qh[l + 16] >> 4) & 3) << 4));
                int32_t w2 = (int32_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4));
                int32_t w3 = (int32_t)((ql[l + 48] >>   4) | (((qh[l + 16] >> 6) & 3) << 4));
                isum += w0 * (int32_t)q8[l + 64]  * (int32_t)scale[4]
                      + w1 * (int32_t)q8[l + 80]  * (int32_t)scale[5]
                      + w2 * (int32_t)q8[l + 96]  * (int32_t)scale[6]
                      + w3 * (int32_t)q8[l + 112] * (int32_t)scale[7];
            }
            ql    += 64;
            qh    += 32;
            scale +=  8;
            q8    += 128;
        }
        sum += (float)q_fp16_to_f32(x[i].d) * y[i].d * (float)(isum - 32 * isum_mins);
    }
    return sum;
}
