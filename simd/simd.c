// SPDX-License-Identifier: MIT
//
// simd.c -- single-file SIMD dispatcher + tiled SGEMM reference.
//
// Build modes:
//   library:        cc -O3 -c simd.c -o simd.o
//   self-test:      cc -O3 -DSIMD_TEST simd.c -o simd_test && ./simd_test
//
// Architecture:
//   - Per-target SIMD kernels live in neon.c (aarch64) and avx.c (x86_64).
//     They are #include'd from this file based on __aarch64__/__x86_64__,
//     producing a single translation unit. Function-multi-versioning via
//     __attribute__((target("..."))) means one binary contains every ISA
//     tier; the dispatcher routes calls to the best one available at run
//     time on the current CPU.
//   - Dispatcher itself is compiled at baseline ISA (SSE2 on x86, ASIMD
//     on aarch64). It must never call a tier the CPU can't run, so we
//     gate every assignment to a function pointer behind a CPUID/HWCAP
//     check.
//   - Public API exposes only the dispatched entry points; per-tier
//     functions are file-static. There is no way for a caller to force
//     a particular tier -- if you need that for benchmarking, build the
//     self-test which prints which tier got selected.
//
// Currently implemented (Q8_0 only as proof-of-concept; q4k/q5k/q6k +
// activations will land in a subsequent pass once this pipeline is
// validated on Mac/khadas/x.local/halo2/agi/win):
//   simd_init()                run once at startup
//   simd_q8_0_dot_q8_0()       int8 quantized dot
//   simd_sgemm_tiled()         fp32 4x8 register-tiled GEMM (reference)
//   simd_dispatch_label()      "AVX2-FMA"/"NEON+dotprod"/... for logging
//
// The tiled SGEMM is a verbatim port of
// /Users/leo/github.com/leok7v/kittens.cpu/cpu/tensor.c:1316-1406. Pure
// C99; the compiler vectorizes it on every target. Useful as the fp32
// baseline for comparing quantized GFlops against.

// Linux glibc gates clock_gettime/CLOCK_MONOTONIC behind POSIX 2001+;
// macOS gates BSD extension types (u_char/u_short used by sys/proc.h)
// behind _DARWIN_C_SOURCE. Define both before any system header.
#if defined(__APPLE__)
  #define _DARWIN_C_SOURCE
#else
  #ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 200809L
  #endif
#endif

#include "quants.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Architecture branch: pull in the per-ISA kernel pack
// ---------------------------------------------------------------------------

#if defined(__aarch64__) || defined(__arm__)
  #define SIMD_ARM 1
  #include "neon.c"
#elif defined(__x86_64__) || defined(__i386__)
  #define SIMD_X86 1
  #include "avx.c"
#else
  #define SIMD_NONE 1
#endif

// ---------------------------------------------------------------------------
// CPU feature detection. ISA-specific, BASELINE-compiled.
// ---------------------------------------------------------------------------

typedef struct {
    bool armv8_dotprod;     // ARMv8.2-A sdot/udot
    bool x86_avx;
    bool x86_avx2;
    bool x86_fma;
    bool x86_f16c;          // half-precision conversion (Ivy Bridge+)
    bool x86_avx_vnni;      // YMM VNNI (Alder Lake+, Zen 4+)
    bool x86_avx512f;
    bool x86_avx512vnni;
} simd_cpu_t;

static simd_cpu_t g_cpu;

#ifdef SIMD_ARM
  #if defined(__APPLE__)
    #include <sys/sysctl.h>
    static bool sysctl_bool(const char* name) {
        int  v = 0;
        size_t sz = sizeof(v);
        if (sysctlbyname(name, &v, &sz, NULL, 0) != 0) return false;
        return v != 0;
    }
    static void detect_cpu(void) {
        // Apple Silicon: dotprod is mandatory on every M-series chip.
        // The sysctl key documents the FEAT_DotProd capability bit.
        g_cpu.armv8_dotprod = sysctl_bool("hw.optional.arm.FEAT_DotProd");
        if (!g_cpu.armv8_dotprod) {
            // Older sysctl on macOS 11 named it differently; the bare
            // existence of arm64 implies ARMv8.5+ on real Macs, so we
            // can fall through to "true" for safety.
            g_cpu.armv8_dotprod = sysctl_bool("hw.optional.AdvSIMD");
        }
    }
  #elif defined(__linux__)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
    static void detect_cpu(void) {
        unsigned long hw = getauxval(AT_HWCAP);
        #ifdef HWCAP_ASIMDDP
            g_cpu.armv8_dotprod = (hw & HWCAP_ASIMDDP) != 0;
        #else
            g_cpu.armv8_dotprod = false;
        #endif
    }
  #else
    static void detect_cpu(void) {
        // Unknown OS on arm. Conservative.
        g_cpu.armv8_dotprod = false;
    }
  #endif
#endif

#ifdef SIMD_X86
  // CPUID portable wrapper. Avoid name collision with gcc <cpuid.h>'s
  // __cpuid macro: use inline asm directly.
  static inline void simd_cpuidex(uint32_t leaf, uint32_t subleaf,
                                  uint32_t out[4]) {
    #if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, (int)leaf, (int)subleaf);
        out[0] = regs[0]; out[1] = regs[1];
        out[2] = regs[2]; out[3] = regs[3];
    #else
        __asm__ volatile("cpuid"
                         : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                         : "a"(leaf), "c"(subleaf));
    #endif
  }
  static void detect_cpu(void) {
        uint32_t r[4];
        simd_cpuidex(1, 0, r);
        g_cpu.x86_avx  = (r[2] & (1u << 28)) != 0;
        g_cpu.x86_fma  = (r[2] & (1u << 12)) != 0;
        g_cpu.x86_f16c = (r[2] & (1u << 29)) != 0;
        simd_cpuidex(7, 0, r);
        g_cpu.x86_avx2       = (r[1] & (1u <<  5)) != 0;
        g_cpu.x86_avx512f    = (r[1] & (1u << 16)) != 0;
        g_cpu.x86_avx512vnni = (r[2] & (1u << 11)) != 0;
        simd_cpuidex(7, 1, r);
        g_cpu.x86_avx_vnni   = (r[0] & (1u <<  4)) != 0;
  }
#endif

#ifdef SIMD_NONE
  static void detect_cpu(void) { /* nothing */ }
#endif

// ---------------------------------------------------------------------------
// Dispatcher function-pointer table + initialization
// ---------------------------------------------------------------------------

typedef float (*q8_0_dot_fn)(const q8_0_block*, const q8_0_block*, int64_t);
typedef float (*q4k_dot_fn) (int nb, const q4k_block*, const q8k_block*);
typedef float (*q5k_dot_fn) (int nb, const q5k_block*, const q8k_block*);
typedef float (*q6k_dot_fn) (int nb, const q6k_block*, const q8k_block*);

static float scalar_q8_0_wrapper(const q8_0_block* x, const q8_0_block* y, int64_t k) {
    return q_scalar_q8_0_dot_q8_0(x, y, k);
}
static float scalar_q4k_wrapper(int nb, const q4k_block* x, const q8k_block* y) {
    return q_scalar_q4k_row_dot_q8k(nb, x, y);
}
static float scalar_q5k_wrapper(int nb, const q5k_block* x, const q8k_block* y) {
    return q_scalar_q5k_row_dot_q8k(nb, x, y);
}
static float scalar_q6k_wrapper(int nb, const q6k_block* x, const q8k_block* y) {
    return q_scalar_q6k_row_dot_q8k(nb, x, y);
}

static q8_0_dot_fn  g_q8_0_dot = scalar_q8_0_wrapper;
static q4k_dot_fn   g_q4k_dot  = scalar_q4k_wrapper;
static q5k_dot_fn   g_q5k_dot  = scalar_q5k_wrapper;
static q6k_dot_fn   g_q6k_dot  = scalar_q6k_wrapper;
static const char*  g_dispatch_label = "scalar";

static bool g_simd_initialized = false;

void simd_init(void) {
    if (g_simd_initialized) return;
    detect_cpu();
    #if defined(SIMD_ARM)
        if (g_cpu.armv8_dotprod) {
            g_q8_0_dot       = q8_0_dot_q8_0_dotprod;
            g_q4k_dot        = q4k_row_dot_q8k_dotprod;
            g_q5k_dot        = q5k_row_dot_q8k_dotprod;
            g_q6k_dot        = q6k_row_dot_q8k_dotprod;
            g_dispatch_label = "NEON+dotprod";
        } else {
            g_q8_0_dot       = q8_0_dot_q8_0_baseline;
            g_q4k_dot        = q4k_row_dot_q8k_baseline;
            g_q5k_dot        = q5k_row_dot_q8k_baseline;
            g_q6k_dot        = q6k_row_dot_q8k_baseline;
            g_dispatch_label = "NEON-baseline";
        }
    #elif defined(SIMD_X86)
        // q4k AVX paths land in a follow-up. Today the x86 dispatch picks
        // the q8_0 tier for the label and leaves q4k on the scalar reference.
        if (g_cpu.x86_avx_vnni && g_cpu.x86_avx2 && g_cpu.x86_fma) {
            g_q8_0_dot       = q8_0_dot_q8_0_avxvnni;
            g_q4k_dot        = q4k_row_dot_q8k_avxvnni;
            g_q5k_dot        = q5k_row_dot_q8k_avxvnni;
            g_q6k_dot        = q6k_row_dot_q8k_avxvnni;
            g_dispatch_label = "AVX-VNNI";
        } else if (g_cpu.x86_avx2 && g_cpu.x86_fma) {
            g_q8_0_dot       = q8_0_dot_q8_0_avx2;
            g_q4k_dot        = q4k_row_dot_q8k_avx2;
            g_q5k_dot        = q5k_row_dot_q8k_avx2;
            g_q6k_dot        = q6k_row_dot_q8k_avx2;
            g_dispatch_label = "AVX2-FMA";
        } else if (g_cpu.x86_avx) {
            g_q8_0_dot       = q8_0_dot_q8_0_avx1;
            g_q4k_dot        = q4k_row_dot_q8k_avx1;
            g_q5k_dot        = q5k_row_dot_q8k_avx1;
            g_q6k_dot        = q6k_row_dot_q8k_avx1;
            g_dispatch_label = "AVX1";
        }
    #endif
    g_simd_initialized = true;
}

const char* simd_dispatch_label(void) { return g_dispatch_label; }

float simd_q8_0_dot_q8_0(const q8_0_block* x, const q8_0_block* y, int64_t k) {
    return g_q8_0_dot(x, y, k);
}

float simd_q4k_row_dot_q8k(int nb, const q4k_block* x, const q8k_block* y) {
    return g_q4k_dot(nb, x, y);
}

float simd_q5k_row_dot_q8k(int nb, const q5k_block* x, const q8k_block* y) {
    return g_q5k_dot(nb, x, y);
}

float simd_q6k_row_dot_q8k(int nb, const q6k_block* x, const q8k_block* y) {
    return g_q6k_dot(nb, x, y);
}

// ===========================================================================
// SiLU dispatcher: 4-lane NEON polynomial on ARM, scalar libm on x86
// (until a faithful AVX expf polynomial port lands here too).
//
// On Apple Silicon / Snapdragon / Khadas the NEON polynomial is
// bit-identical to llm/neon.c's neon_silu_vec_f32 — that's the
// parity gate for `--chat-test` hashes. On x86 the scalar libm path
// produces a per-element ULP drift that compounds through layers; the
// cross-host sweep records a per-ISA hash baseline rather than
// expecting byte-identical hashes across architectures.
// ===========================================================================

__attribute__((unused))
static void silu_f32_scalar(int n, float* dst, const float* src) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] / (1.0f + expf(-src[i]));
    }
}

void simd_silu_f32(int n, float* dst, const float* src) {
    #if defined(SIMD_ARM)
        silu_f32_neon(n, dst, src);
    #else
        silu_f32_scalar(n, dst, src);
    #endif
}

// ===========================================================================
// fp32 4x8 register-tiled SGEMM (reference baseline).
// Verbatim port of kittens.cpu/cpu/tensor.c:1316-1406.
//
// Computes C(M,N) = A(M,K) @ Bt(N,K)^T  in row-major.
// All matrices contiguous, leading dims lda/ldb/ldc in elements.
//
// Algorithm: tile output into 4x8 blocks. Each block uses 32 fp32
// accumulators (clang allocates to NEON/AVX registers at -O3) and
// streams K elements of A's 4 rows + Bt's 8 rows through them.
//
// Edge cases (non-multiple of 4 / 8) fall to a naive scalar handler.
// ===========================================================================

static void sgemm_kernel_4x8(int K,
                             const float* A,  int lda,
                             const float* Bt, int ldb,
                             float* C, int ldc) {
    float c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
    float c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
    float c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
    float c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;
    for (int k = 0; k < K; k++) {
        float a0 = A[0*lda + k], a1 = A[1*lda + k];
        float a2 = A[2*lda + k], a3 = A[3*lda + k];
        float b0 = Bt[0*ldb + k], b1 = Bt[1*ldb + k];
        float b2 = Bt[2*ldb + k], b3 = Bt[3*ldb + k];
        float b4 = Bt[4*ldb + k], b5 = Bt[5*ldb + k];
        float b6 = Bt[6*ldb + k], b7 = Bt[7*ldb + k];
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

static void sgemm_edge(int M, int N, int K,
                       const float* A,  int lda,
                       const float* Bt, int ldb,
                       float* C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[m*lda + k] * Bt[n*ldb + k];
            C[m*ldc + n] = acc;
        }
    }
}

void simd_sgemm_tiled(int M, int N, int K,
                      const float* A,  int lda,
                      const float* Bt, int ldb,
                      float* C, int ldc) {
    const int MR = 4, NR = 8;
    int M_main = (M / MR) * MR;
    int N_main = (N / NR) * NR;
    for (int m = 0; m < M_main; m += MR) {
        for (int n = 0; n < N_main; n += NR) {
            sgemm_kernel_4x8(K, A + m*lda, lda, Bt + n*ldb, ldb,
                             C + m*ldc + n, ldc);
        }
        if (N_main < N) {
            sgemm_edge(MR, N - N_main, K,
                       A + m*lda, lda, Bt + N_main*ldb, ldb,
                       C + m*ldc + N_main, ldc);
        }
    }
    if (M_main < M) {
        sgemm_edge(M - M_main, N, K,
                   A + M_main*lda, lda, Bt, ldb,
                   C + M_main*ldc, ldc);
    }
}

// ===========================================================================
// fp16 4x8 tiled HGEMM   (measuring stick for quantization tradeoffs)
//
// Inputs: A and Bt are fp16 (uint16_t storage). Output C is fp32.
// Compute: per-element fp16->fp32 conversion via q_fp16_to_f32 (header
// inline), then identical 4x8 register tile to simd_sgemm_tiled with
// 32 fp32 accumulators.
//
// Platform speedup story (handled by the COMPILER, not us):
//   - x86 with F16C   : q_fp16_to_f32 lowers to vcvtph2ps (1 uop port 5)
//   - aarch64 + fphp  : ditto vcvt_f32_f16
//   - everywhere else : bit-twiddle fallback; still vectorizable by clang
//
// q_fp16_to_f32 is static-inline in quants.h, so clang can specialise per
// caller. Result: same source on all 7 hosts; compiler picks best path.
// ===========================================================================

static void hgemm_kernel_4x8(int K,
                             const uint16_t* A,  int lda,
                             const uint16_t* Bt, int ldb,
                             float* C, int ldc) {
    float c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
    float c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
    float c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
    float c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;
    for (int k = 0; k < K; k++) {
        float a0 = q_fp16_to_f32((q_fp16_t)A[0*lda + k]);
        float a1 = q_fp16_to_f32((q_fp16_t)A[1*lda + k]);
        float a2 = q_fp16_to_f32((q_fp16_t)A[2*lda + k]);
        float a3 = q_fp16_to_f32((q_fp16_t)A[3*lda + k]);
        float b0 = q_fp16_to_f32((q_fp16_t)Bt[0*ldb + k]);
        float b1 = q_fp16_to_f32((q_fp16_t)Bt[1*ldb + k]);
        float b2 = q_fp16_to_f32((q_fp16_t)Bt[2*ldb + k]);
        float b3 = q_fp16_to_f32((q_fp16_t)Bt[3*ldb + k]);
        float b4 = q_fp16_to_f32((q_fp16_t)Bt[4*ldb + k]);
        float b5 = q_fp16_to_f32((q_fp16_t)Bt[5*ldb + k]);
        float b6 = q_fp16_to_f32((q_fp16_t)Bt[6*ldb + k]);
        float b7 = q_fp16_to_f32((q_fp16_t)Bt[7*ldb + k]);
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

static void hgemm_edge(int M, int N, int K,
                       const uint16_t* A,  int lda,
                       const uint16_t* Bt, int ldb,
                       float* C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += q_fp16_to_f32((q_fp16_t)A[m*lda + k]) *
                       q_fp16_to_f32((q_fp16_t)Bt[n*ldb + k]);
            }
            C[m*ldc + n] = acc;
        }
    }
}

// Forward declaration of the F16C-using variant from avx.c (only exists
// on x86 builds; SIMD_X86 gate).
#ifdef SIMD_X86
void simd_hgemm_tiled_f16c(int M, int N, int K,
                           const uint16_t* A, int lda,
                           const uint16_t* Bt, int ldb,
                           float* C, int ldc);
#endif

// Dispatched hgemm entry: x86+F16C uses the explicit vcvtph2ps kernel
// (~30x faster than bit-twiddle); everything else uses the portable
// 4x8 kernel where the compiler vectorizes via _Float16 when available.
void simd_hgemm_tiled(int M, int N, int K,
                      const uint16_t* A,  int lda,
                      const uint16_t* Bt, int ldb,
                      float* C, int ldc) {
#ifdef SIMD_X86
    if (g_cpu.x86_f16c) {
        simd_hgemm_tiled_f16c(M, N, K, A, lda, Bt, ldb, C, ldc);
        return;
    }
#endif
    const int MR = 4, NR = 8;
    int M_main = (M / MR) * MR;
    int N_main = (N / NR) * NR;
    for (int m = 0; m < M_main; m += MR) {
        for (int n = 0; n < N_main; n += NR) {
            hgemm_kernel_4x8(K, A + m*lda, lda, Bt + n*ldb, ldb,
                             C + m*ldc + n, ldc);
        }
        if (N_main < N) {
            hgemm_edge(MR, N - N_main, K,
                       A + m*lda, lda, Bt + N_main*ldb, ldb,
                       C + m*ldc + N_main, ldc);
        }
    }
    if (M_main < M) {
        hgemm_edge(M - M_main, N, K,
                   A + M_main*lda, lda, Bt, ldb,
                   C + M_main*ldc, ldc);
    }
}

// ===========================================================================
// Self-tests. Activate with -DSIMD_TEST.
//
// Tests two things:
//   1. Q8_0 dispatched kernel produces the same fp32 result as the
//      scalar reference (within float epsilon * len).
//   2. Tiled SGEMM produces the same C as a naive triple-loop SGEMM.
// ===========================================================================

#ifdef SIMD_TEST

#include <time.h>

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// Compiler memory barrier. Inside a timing loop, prevents clang -O3 from
// hoisting a "pure-from-its-POV" function call out and replicating the
// scalar add. Necessary because our quant inputs don't change per iter,
// so without this clang sees SIMD_FN(nb,x,y) as loop-invariant.
#define BARRIER() __asm__ volatile("" : : : "memory")

static uint32_t rng = 1;
static uint32_t rand32(void) {
    uint32_t z = (rng += 0x6D2B79F5UL);
    z = (z ^ (z >> 15)) * (z | 1UL);
    z ^= z + (z ^ (z >> 7)) * (z | 61UL);
    return z ^ (z >> 14);
}
static float randf(void) {
    return (float)rand32() / (float)UINT32_MAX - 0.5f;
}

// Generic q*k tester: fabricates a quant-block array with random bytes and
// compares scalar reference vs dispatched on the same byte stream.
#define DEFINE_QK_TEST(NAME, BLOCKTYPE, SIMD_FN, SCALAR_FN)             \
static int test_##NAME(void) {                                          \
    const int nb = 16;                                                  \
    BLOCKTYPE* x = (BLOCKTYPE*)malloc((size_t)nb * sizeof(BLOCKTYPE));  \
    q8k_block* y = (q8k_block*)malloc((size_t)nb * sizeof(q8k_block));  \
    float*     a = (float*)    malloc((size_t)nb * QK_K * sizeof(float)); \
    uint8_t*   xb = (uint8_t*)x;                                        \
    for (size_t i = 0; i < (size_t)nb * sizeof(BLOCKTYPE); i++)         \
        xb[i] = (uint8_t)(rand32() & 0xFF);                             \
    /* Reasonable d/dmin so isum*d stays in fp32 range */               \
    for (int i = 0; i < nb; i++) {                                      \
        ((BLOCKTYPE*)x)[i].d    = q_f32_to_fp16(0.01f);                 \
    }                                                                   \
    for (int i = 0; i < nb * QK_K; i++) a[i] = randf();                 \
    q_quantize_row_q8_K(a, y, (int64_t)nb * QK_K);                      \
    float ref = SCALAR_FN(nb, x, y);                                    \
    float got = SIMD_FN  (nb, x, y);                                    \
    float tol = 1e-2f * fabsf(ref);                                     \
    if (tol < 1e-3f) tol = 1e-3f;                                       \
    int ok = (fabsf(ref - got) <= tol);                                 \
    printf("[%-9s] ref=%.4f  simd(%-15s)=%.4f  delta=%.2e  %s\n",       \
           #NAME, ref, simd_dispatch_label(), got, ref - got,           \
           ok ? "OK" : "FAIL");                                         \
    const int reps = 100;                                               \
    double t0 = now_seconds();                                          \
    volatile float sink = 0;                                            \
    for (int r = 0; r < reps; r++) { sink += SIMD_FN(nb, x, y); BARRIER(); } \
    double dt = now_seconds() - t0;                                     \
    double gflops = (2.0 * (double)QK_K * (double)nb * (double)reps) / (dt * 1e9); \
    printf("[%-9s] %d reps x %d blocks x 256: %.2f GFlops  (sink=%.3f)\n", \
           #NAME, reps, nb, gflops, (double)sink);                      \
    free(x); free(y); free(a);                                          \
    return ok;                                                          \
}

DEFINE_QK_TEST(q6k_dot, q6k_block, simd_q6k_row_dot_q8k, q_scalar_q6k_row_dot_q8k)

// q5k has d + dmin like q4k; the generic macro only sets d so we'd leave
// dmin as random fp16 bytes (often NaN). Bespoke test.
static int test_q5k(void) {
    const int nb = 16;
    q5k_block* x = (q5k_block*)malloc((size_t)nb * sizeof(q5k_block));
    q8k_block* y = (q8k_block*)malloc((size_t)nb * sizeof(q8k_block));
    float*     a = (float*)    malloc((size_t)nb * QK_K * sizeof(float));
    uint8_t* xb = (uint8_t*)x;
    for (size_t i = 0; i < (size_t)nb * sizeof(q5k_block); i++) xb[i] = (uint8_t)(rand32() & 0xFF);
    for (int i = 0; i < nb; i++) {
        x[i].d    = q_f32_to_fp16(0.01f);
        x[i].dmin = q_f32_to_fp16(0.5f);
    }
    for (int i = 0; i < nb * QK_K; i++) a[i] = randf();
    q_quantize_row_q8_K(a, y, (int64_t)nb * QK_K);
    float ref = q_scalar_q5k_row_dot_q8k(nb, x, y);
    float got = simd_q5k_row_dot_q8k   (nb, x, y);
    float tol = 1e-2f * fabsf(ref);
    if (tol < 1e-3f) tol = 1e-3f;
    int ok = (fabsf(ref - got) <= tol);
    printf("[q5k_dot]   ref=%.4f  simd(%-15s)=%.4f  delta=%.2e  %s\n",
           ref, simd_dispatch_label(), got, ref - got, ok ? "OK" : "FAIL");
    const int reps = 100;
    double t0 = now_seconds();
    volatile float sink = 0;
    for (int r = 0; r < reps; r++) { sink += simd_q5k_row_dot_q8k(nb, x, y); BARRIER(); }
    double dt = now_seconds() - t0;
    double gflops = (2.0 * (double)QK_K * (double)nb * (double)reps) / (dt * 1e9);
    printf("[q5k_dot]   %d reps x %d blocks x 256: %.2f GFlops  (sink=%.3f)\n",
           reps, nb, gflops, (double)sink);
    free(x); free(y); free(a);
    return ok;
}

// q4k test stays bespoke because q4k_block has BOTH d and dmin fields,
// while the macro only sets d. Keep it explicit.
static int test_q4k(void) {
    // We don't have a Q4_K quantizer; fabricate plausible block bytes
    // (random scales / qs work because the scalar reference and the SIMD
    // path both interpret the same bytes via q4k_get_scale_min and
    // identical bit shifts). The activation side IS quantized via
    // q_quantize_row_q8_K from a random fp32 row.
    const int nb = 16;
    q4k_block* x = (q4k_block*)malloc((size_t)nb * sizeof(q4k_block));
    q8k_block* y = (q8k_block*)malloc((size_t)nb * sizeof(q8k_block));
    float*     a = (float*)malloc((size_t)nb * QK_K * sizeof(float));
    uint8_t* xb = (uint8_t*)x;
    for (size_t i = 0; i < (size_t)nb * sizeof(q4k_block); i++) {
        xb[i] = (uint8_t)(rand32() & 0xFF);
    }
    // Force d / dmin to reasonable magnitudes so isum*d stays in fp32 range
    for (int i = 0; i < nb; i++) {
        x[i].d    = q_f32_to_fp16(0.01f);
        x[i].dmin = q_f32_to_fp16(0.5f);
    }
    for (int i = 0; i < nb * QK_K; i++) a[i] = randf();
    q_quantize_row_q8_K(a, y, (int64_t)nb * QK_K);
    float ref = q_scalar_q4k_row_dot_q8k(nb, x, y);
    float got = simd_q4k_row_dot_q8k   (nb, x, y);
    float tol = 1e-2f * fabsf(ref);     // relative; fp accum order can differ
    if (tol < 1e-3f) tol = 1e-3f;
    int ok = (fabsf(ref - got) <= tol);
    printf("[q4k_dot]   ref=%.4f  simd(%-15s)=%.4f  delta=%.2e  %s\n",
           ref, simd_dispatch_label(), got, ref - got, ok ? "OK" : "FAIL");
    const int reps = 100;
    double t0 = now_seconds();
    volatile float sink = 0;
    for (int r = 0; r < reps; r++) { sink += simd_q4k_row_dot_q8k(nb, x, y); BARRIER(); }
    double dt = now_seconds() - t0;
    // 2 flops per int8 mac (mul + add) over QK_K elements * nb blocks * reps
    double gflops = (2.0 * (double)QK_K * (double)nb * (double)reps) / (dt * 1e9);
    printf("[q4k_dot]   %d reps x %d blocks x 256: %.2f GFlops  (sink=%.3f)\n",
           reps, nb, gflops, (double)sink);
    free(x); free(y); free(a);
    return ok;
}

static int test_q8_0(void) {
    const int64_t k = 32 * 128;   // 128 blocks
    const int     nb = (int)(k / QK8_0);
    float* a = malloc((size_t)k * sizeof(float));
    float* b = malloc((size_t)k * sizeof(float));
    q8_0_block* qa = malloc((size_t)nb * sizeof(q8_0_block));
    q8_0_block* qb = malloc((size_t)nb * sizeof(q8_0_block));
    for (int64_t i = 0; i < k; i++) { a[i] = randf(); b[i] = randf(); }
    q_quantize_row_q8_0(a, qa, k);
    q_quantize_row_q8_0(b, qb, k);
    float ref = q_scalar_q8_0_dot_q8_0(qa, qb, k);
    float got = simd_q8_0_dot_q8_0   (qa, qb, k);
    float tol = 1e-4f * (float)k;
    int ok = (fabsf(ref - got) <= tol);
    printf("[q8_0_dot]  ref=%.6f  simd(%-15s)=%.6f  delta=%.2e  %s\n",
           ref, simd_dispatch_label(), got, ref - got, ok ? "OK" : "FAIL");
    // crude perf number
    const int reps = 200;
    double t0 = now_seconds();
    volatile float sink = 0;
    for (int r = 0; r < reps; r++) { sink += simd_q8_0_dot_q8_0(qa, qb, k); BARRIER(); }
    double dt = now_seconds() - t0;
    double gflops = (2.0 * (double)k * (double)reps) / (dt * 1e9);
    printf("[q8_0_dot]  %d reps x %lldx32 ints: %.2f GFlops  (sink=%.3f)\n",
           reps, (long long)nb, gflops, (double)sink);
    free(a); free(b); free(qa); free(qb);
    return ok;
}

static void naive_sgemm(int M, int N, int K,
                        const float* A, int lda,
                        const float* Bt, int ldb,
                        float* C, int ldc) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[m*lda + k] * Bt[n*ldb + k];
            C[m*ldc + n] = s;
        }
}

static int test_sgemm(int M, int N, int K) {
    float* A   = malloc((size_t)M * K * sizeof(float));
    float* Bt  = malloc((size_t)N * K * sizeof(float));
    float* C1  = malloc((size_t)M * N * sizeof(float));
    float* C2  = malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M*K; i++) A[i]  = randf();
    for (int i = 0; i < N*K; i++) Bt[i] = randf();
    naive_sgemm     (M, N, K, A, K, Bt, K, C1, N);
    simd_sgemm_tiled(M, N, K, A, K, Bt, K, C2, N);
    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float d = fabsf(C1[i] - C2[i]);
        if (d > max_err) max_err = d;
    }
    int ok = (max_err <= 1e-3f * (float)K);
    // timing
    const int reps = (M*N >= 65536) ? 5 : 50;
    double t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        simd_sgemm_tiled(M, N, K, A, K, Bt, K, C2, N);
        BARRIER();
    }
    double dt = now_seconds() - t0;
    double gflops = (2.0 * (double)M * (double)N * (double)K * (double)reps) / (dt * 1e9);
    printf("[sgemm_4x8] M=%-4d N=%-4d K=%-4d  max_err=%.2e  tiled=%.2f GFlops  %s\n",
           M, N, K, max_err, gflops, ok ? "OK" : "FAIL");
    free(A); free(Bt); free(C1); free(C2);
    return ok;
}

static int test_hgemm(int M, int N, int K) {
    uint16_t* A   = malloc((size_t)M * K * sizeof(uint16_t));
    uint16_t* Bt  = malloc((size_t)N * K * sizeof(uint16_t));
    float*    C1  = malloc((size_t)M * N * sizeof(float));
    float*    C2  = malloc((size_t)M * N * sizeof(float));
    for (int i = 0; i < M*K; i++) A[i]  = (uint16_t)q_f32_to_fp16(randf());
    for (int i = 0; i < N*K; i++) Bt[i] = (uint16_t)q_f32_to_fp16(randf());
    // naive fp16 ref
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += q_fp16_to_f32((q_fp16_t)A[m*K + k]) *
                     q_fp16_to_f32((q_fp16_t)Bt[n*K + k]);
            C1[m*N + n] = s;
        }
    simd_hgemm_tiled(M, N, K, A, K, Bt, K, C2, N);
    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float d = fabsf(C1[i] - C2[i]);
        if (d > max_err) max_err = d;
    }
    int ok = (max_err <= 1e-3f * (float)K);
    const int reps = (M*N >= 65536) ? 5 : 50;
    double t0 = now_seconds();
    for (int r = 0; r < reps; r++) {
        simd_hgemm_tiled(M, N, K, A, K, Bt, K, C2, N);
        BARRIER();
    }
    double dt = now_seconds() - t0;
    double gflops = (2.0 * (double)M * (double)N * (double)K * (double)reps) / (dt * 1e9);
    printf("[hgemm_4x8] M=%-4d N=%-4d K=%-4d  max_err=%.2e  tiled=%.2f GFlops  %s\n",
           M, N, K, max_err, gflops, ok ? "OK" : "FAIL");
    free(A); free(Bt); free(C1); free(C2);
    return ok;
}

int main(void) {
    simd_init();
    printf("dispatch: %s\n", simd_dispatch_label());
    int ok = 1;
    ok &= test_q8_0();
    ok &= test_q4k();
    ok &= test_q5k();
    ok &= test_q6k_dot();
    ok &= test_sgemm(128, 128, 128);
    ok &= test_sgemm(256, 256, 256);
    ok &= test_sgemm(512, 512, 512);
    ok &= test_hgemm(128, 128, 128);
    ok &= test_hgemm(256, 256, 256);
    ok &= test_hgemm(512, 512, 512);
    return ok ? 0 : 1;
}

#endif  // SIMD_TEST
