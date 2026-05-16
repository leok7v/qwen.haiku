// SPDX-License-Identifier: MIT
//
// probe.c — parity probe: llm/neon.c (legacy NEON-dotprod-only) vs
// simd/simd.c (dispatched, dotprod + baseline + AVX tiers).
//
// Goal: prove that on Apple Silicon (dotprod tier) the simd dispatcher
// is BYTE-IDENTICAL to the kernels that produced the load-bearing
// parity hashes (d2ba984b228fff7a / f12add080343c386 / a99b9d0705cc0220
// for `--chat-test`). If this probe shows any divergence on any quant,
// the dispatcher swap is unsafe without re-baselining hashes.
//
// Build:
//     clang -O3 -std=c11 -Wall -mcpu=apple-m1 \
//         rnd/probe.c -o Build/cli/parity-probe
// Run:
//     ./Build/cli/parity-probe
//
// What it does:
//   1. Pull q4k_block/q5k_block/q6k_block/q8_0_block/q8k_block + scalar
//      refs from simd/quants.h (byte-compatible with llm/tensor.c's
//      block layouts — verified by static_assert below).
//   2. #include llm/neon.c to get q4k_row_dot_q8k_neon / _q5k / _q6k +
//      q8_0_dot_q8_0_neon.
//   3. #include simd/simd.c to get simd_q4k_row_dot_q8k / _q5k / _q6k +
//      simd_q8_0_dot_q8_0 (dispatcher selects dotprod tier on M-series).
//   4. Generate deterministic random block bytes (seeded RNG).
//   5. Run BOTH kernels on the same input; compare bitwise (memcmp of
//      the result fp32 word).
//   6. Print PASS / FAIL per quant type with the bit pattern.
//
// PASS means dispatcher = legacy on this CPU. If FAIL, the divergence
// can be only one of:
//   - struct layout drift (caught by static_assert)
//   - the C source actually differs (caught by visual diff)
//   - clang vectorization of the inner loop diverges between the two
//     translation units (rare, usually flagged by codegen diff).

#include "../simd/quants.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// llm/neon.c is wrapped in `#ifndef NEON_C` and expects q4k_block etc.
// to be in scope under those exact names. simd/quants.h provides them
// with byte-compatible layout but uses `q_fp16_t` for the d/dmin
// fields where llm/neon.c expects `_Float16`. On Apple Silicon both
// are the same primitive — verify and continue.
_Static_assert(sizeof(q4k_block) == 144, "q4k_block size drift");
_Static_assert(sizeof(q5k_block) == 176, "q5k_block size drift");
_Static_assert(sizeof(q6k_block) == 210, "q6k_block size drift");
_Static_assert(sizeof(q8_0_block) == 34, "q8_0_block size drift");

// llm/neon.c's `static inline` kernels live behind its own header
// guard; pull them in directly.
#include "../rnd/archive/neon.c"

// simd/simd.c brings the dispatcher + simd/neon.c (or avx.c).
#include "../simd/simd.c"

// Deterministic xorshift32 to fill block bytes reproducibly across
// hosts. Same RNG state on every run -> same input bytes.
static uint32_t s_state = 0xdeadbeefu;
static uint32_t xs32(void) {
    uint32_t x = s_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return s_state = x;
}
static void fill_bytes(void * p, size_t n) {
    uint8_t * b = (uint8_t *)p;
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(xs32() & 0xff);
}

// Compare two floats by bit pattern. Returns 1 if identical bytes.
static int bit_equal(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

// Sane d / dmin: random fp16 bytes occasionally land in NaN/Inf; force
// the per-block scales to a small finite value so the reduction stays
// in fp32 range. (Same convention as simd/simd.c's self-test.)
static void scrub_scales_q4k(q4k_block * x, int nb) {
    for (int i = 0; i < nb; i++) {
        x[i].d    = q_f32_to_fp16(0.01f);
        x[i].dmin = q_f32_to_fp16(0.5f);
    }
}
static void scrub_scales_q5k(q5k_block * x, int nb) {
    for (int i = 0; i < nb; i++) {
        x[i].d    = q_f32_to_fp16(0.01f);
        x[i].dmin = q_f32_to_fp16(0.5f);
    }
}
static void scrub_scales_q6k(q6k_block * x, int nb) {
    for (int i = 0; i < nb; i++) {
        x[i].d = q_f32_to_fp16(0.01f);
    }
}
static void scrub_scales_q8_0(q8_0_block * x, int nb) {
    for (int i = 0; i < nb; i++) {
        x[i].d = q_f32_to_fp16(0.01f);
    }
}

static int probe_q4k(int nb) {
    q4k_block * x = (q4k_block *)malloc((size_t)nb * sizeof(q4k_block));
    q8k_block * y = (q8k_block *)malloc((size_t)nb * sizeof(q8k_block));
    fill_bytes(x, (size_t)nb * sizeof(q4k_block));
    fill_bytes(y, (size_t)nb * sizeof(q8k_block));
    scrub_scales_q4k(x, nb);
    // q8k_block has fp32 d (not fp16) — set to a sane finite value.
    for (int i = 0; i < nb; i++) y[i].d = 0.02f;

    float got_legacy = q4k_row_dot_q8k_neon(nb, x, y);
    float got_simd   = simd_q4k_row_dot_q8k(nb, x, y);
    int   ok        = bit_equal(got_legacy, got_simd);
    uint32_t bl, bs;
    memcpy(&bl, &got_legacy, 4);
    memcpy(&bs, &got_simd,   4);
    printf("[q4k_dot ] legacy=%.7g (0x%08x)  simd(%s)=%.7g (0x%08x)  %s\n",
           got_legacy, bl, simd_dispatch_label(), got_simd, bs,
           ok ? "PASS" : "FAIL");
    free(x); free(y);
    return ok;
}

static int probe_q5k(int nb) {
    q5k_block * x = (q5k_block *)malloc((size_t)nb * sizeof(q5k_block));
    q8k_block * y = (q8k_block *)malloc((size_t)nb * sizeof(q8k_block));
    fill_bytes(x, (size_t)nb * sizeof(q5k_block));
    fill_bytes(y, (size_t)nb * sizeof(q8k_block));
    scrub_scales_q5k(x, nb);
    for (int i = 0; i < nb; i++) y[i].d = 0.02f;

    float got_legacy = q5k_row_dot_q8k_neon(nb, x, y);
    float got_simd   = simd_q5k_row_dot_q8k(nb, x, y);
    int   ok        = bit_equal(got_legacy, got_simd);
    uint32_t bl, bs;
    memcpy(&bl, &got_legacy, 4);
    memcpy(&bs, &got_simd,   4);
    printf("[q5k_dot ] legacy=%.7g (0x%08x)  simd(%s)=%.7g (0x%08x)  %s\n",
           got_legacy, bl, simd_dispatch_label(), got_simd, bs,
           ok ? "PASS" : "FAIL");
    free(x); free(y);
    return ok;
}

static int probe_q6k(int nb) {
    q6k_block * x = (q6k_block *)malloc((size_t)nb * sizeof(q6k_block));
    q8k_block * y = (q8k_block *)malloc((size_t)nb * sizeof(q8k_block));
    fill_bytes(x, (size_t)nb * sizeof(q6k_block));
    fill_bytes(y, (size_t)nb * sizeof(q8k_block));
    scrub_scales_q6k(x, nb);
    for (int i = 0; i < nb; i++) y[i].d = 0.02f;

    float got_legacy = q6k_row_dot_q8k_neon(nb, x, y);
    float got_simd   = simd_q6k_row_dot_q8k(nb, x, y);
    int   ok        = bit_equal(got_legacy, got_simd);
    uint32_t bl, bs;
    memcpy(&bl, &got_legacy, 4);
    memcpy(&bs, &got_simd,   4);
    printf("[q6k_dot ] legacy=%.7g (0x%08x)  simd(%s)=%.7g (0x%08x)  %s\n",
           got_legacy, bl, simd_dispatch_label(), got_simd, bs,
           ok ? "PASS" : "FAIL");
    free(x); free(y);
    return ok;
}

static int probe_q8_0(int nb) {
    q8_0_block * x = (q8_0_block *)malloc((size_t)nb * sizeof(q8_0_block));
    q8_0_block * y = (q8_0_block *)malloc((size_t)nb * sizeof(q8_0_block));
    fill_bytes(x, (size_t)nb * sizeof(q8_0_block));
    fill_bytes(y, (size_t)nb * sizeof(q8_0_block));
    scrub_scales_q8_0(x, nb);
    scrub_scales_q8_0(y, nb);

    float got_legacy = q8_0_dot_q8_0_neon(x, y, (int64_t)nb * QK8_0);
    float got_simd   = simd_q8_0_dot_q8_0(x, y, (int64_t)nb * QK8_0);
    int   ok        = bit_equal(got_legacy, got_simd);
    uint32_t bl, bs;
    memcpy(&bl, &got_legacy, 4);
    memcpy(&bs, &got_simd,   4);
    printf("[q8_0_dot] legacy=%.7g (0x%08x)  simd(%s)=%.7g (0x%08x)  %s\n",
           got_legacy, bl, simd_dispatch_label(), got_simd, bs,
           ok ? "PASS" : "FAIL");
    free(x); free(y);
    return ok;
}

// SiLU probe: compare neon_silu_vec_f32 (from llm/neon.c) to
// simd_silu_f32 (dispatcher). Both polynomials on Apple Silicon —
// MUST be byte-identical. n=6144 + 2048 (actual conv_dim / V_dim
// values used by the Qwen3.5 SSM block).
static int probe_silu(int n) {
    float * in  = (float *)malloc((size_t)n * sizeof(float));
    float * a   = (float *)malloc((size_t)n * sizeof(float));
    float * b   = (float *)malloc((size_t)n * sizeof(float));
    fill_bytes(in, (size_t)n * sizeof(float));
    // Scrub NaN/Inf from random fp32 bytes — sample a realistic range.
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &in[i], 4);
        // Keep finite, range roughly [-4, 4] which matches the Qwen
        // post-conv1d activations the SSM block actually feeds in.
        in[i] = ((float)(u % 8000) - 4000.0f) / 1000.0f;
    }
    neon_silu_vec_f32(n, a, in);
    simd_silu_f32   (n, b, in);
    int   bad = 0;
    int   first_bad = -1;
    for (int i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], 4) != 0) {
            bad++;
            if (first_bad < 0) first_bad = i;
        }
    }
    int ok = (bad == 0);
    if (ok) {
        printf("[silu_n=%-5d]  legacy=simd over all %d elements  PASS\n",
               n, n);
    } else {
        uint32_t ua, ub;
        memcpy(&ua, &a[first_bad], 4); memcpy(&ub, &b[first_bad], 4);
        printf("[silu_n=%-5d]  %d/%d differ; first @i=%d: legacy=%.9g (0x%08x)  simd=%.9g (0x%08x)  FAIL\n",
               n, bad, n, first_bad, a[first_bad], ua, b[first_bad], ub);
    }
    free(in); free(a); free(b);
    return ok;
}

int main(void) {
    simd_init();
    printf("dispatch: %s\n", simd_dispatch_label());
    const int NB = 16;  // 16 super-blocks = realistic single-row dot size

    // Run several seeds to catch input-dependent divergence.
    int all = 1;
    for (uint32_t seed = 1; seed <= 4; seed++) {
        s_state = 0xdeadbeefu ^ (seed * 0x9e3779b9u);
        printf("--- seed %u\n", seed);
        all &= probe_q4k(NB);
        all &= probe_q5k(NB);
        all &= probe_q6k(NB);
        all &= probe_q8_0(NB);
    }
    // SiLU probe: conv_dim=6144 and V_dim=2048 are the actual sizes the
    // SSM block feeds in; SwiGLU FFN runs at hidden=1024 * 3 = 3072.
    printf("--- silu\n");
    s_state = 0xcafebabeu;
    all &= probe_silu(6144);
    all &= probe_silu(2048);
    all &= probe_silu(3072);
    printf("parity-probe: %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
