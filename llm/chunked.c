// SPDX-License-Identifier: MIT
//
// chunked.c - chunked Gated DeltaNet SSM kernel for qwen3-next.
//
// Port of llama.cpp src/models/qwen3next.cpp build_delta_net_chunking
// (~lines 97-356), scalar fp32. Selected over the recurrent
// (autoregressive) path when llm_forward_batch sees n_tokens > 1;
// produces the same algebra but with ~10x throughput because every
// chunk's intra-token attention becomes a small matmul instead of
// k_hd sequential mul-adds.
//
// Algorithm (per head, per chunk of size CHUNK_SIZE = 64):
//
//   1. g_cumsum[t]    = Σ_{s<=t} g_log[s]                     // running decay
//   2. decay_mask[i,j] = exp(g_cumsum[i] - g_cumsum[j])
//                       if i>j else (0 if i<j, 0 on diagonal)
//   3. attn_lower[i,j] = -(K[i,:] · K_beta[j,:]) * decay_mask[i,j]
//                       restricted to strict lower-tri
//      lhs[i,j]        = I[i,j] - attn_lower[i,j]
//      attn            = SOLVE_TRI(lhs, attn_lower, lower=1, unit=1)
//                        with the identity added on the diagonal
//   4. v_eff           = attn^T @ V_beta       (within-chunk corrected V)
//   5. g_exp[t]        = exp(g_cumsum[t])
//      k_cumdecay      = attn @ (K_beta * g_exp)^T            // decay-weighted keys
//      attn_kq[i,j]    = (K[i,:] · Q[j,:]) * decay_mask[i,j]  // intra-chunk attention
//   6. Per-chunk loop:
//        q_g_exp        = Q * g_exp
//        attn_inter     = state @ q_g_exp^T                    // (carry from prior chunks)
//        v_prime        = state @ k_cumdecay^T                 // predicted V from state
//        v_new          = V - v_prime                          // correction
//        v_attn         = attn^T @ v_new                       // within-chunk
//        out_chunk      = attn_inter + v_attn
//        g_last         = exp(g_cumsum[-1])
//        key_gdiff      = K * exp(g_cumsum[-1] - g_cumsum)
//        state         := state * g_last + key_gdiff^T @ v_new // carry update
//
// Note: g is the pre-exp scalar gate (`ssm_a * softplus(α+dt_bias)`).
// It's negative for typical Qwen3.5 weights, so exp(g_cumsum) decays
// monotonically across a chunk - this is what makes the SSM
// "forget" stale state.
//
// Reference dims for Qwen3.5-0.8B (CHUNK_SIZE = 64):
//   k_hd = v_hd = 128, n_heads = 16, K_dim = V_dim = 2048
//
// Degeneracy check: for n_tokens=1 (chunk padded with 63 zeros on
// the right of q/k/v/beta and end of g_log), this collapses cleanly
// to the autoregressive update. With state_init = 0:
//
//   attn_inter   = state @ (q * gexp)   = 0
//   v_eff[0]     = v_beta[0] = V[0] * beta[0]
//   v_prime[0]   = state @ k_cumdecay   = 0
//   v_new[0]     = v_eff[0] - 0          = V[0] * beta[0]
//   attn[0,0]    = 1 (SOLVE_TRI on trivial system + identity)
//   attn_kq[0,0] = K[0] · Q[0]
//   v_attn[0]    = attn_kq[0,0] * v_new[0] = (K · Q) · V[0] · beta[0]
//   out[0]       = attn_inter + v_attn   = (K · Q) · V[0] · beta[0]
//
// which is exactly what the autoregressive step produces for the
// same inputs. So bit-parity-testing this kernel against the
// existing autoregressive path on a single token is meaningful (the
// math is identical, only the accumulation order differs).
//
// Numerical reference: ggml's compute kernel - the SOLVE_TRI step is
// a pure forward-substitution on a lower-triangular system, fp32. We
// match `ggml_compute_forward_solve_tri_f32` (ops.cpp ~line 10293) bit
// for bit. Everything else is straightforward fp32 matmul / hadamard /
// exp; the only choice that matters is the exp routine, which uses
// `expf` from libm (matches ggml's non-Accelerate, non-vectorised
// scalar path).

#include <math.h>
#include <string.h>

#define CHUNK_SIZE 64

// Forward substitution on a strict-lower-triangular system (I - L)X = B
// where L has zero diagonal. Solution overwrites X (B may be the same
// buffer). All matrices are stored row-major; dimensions are square
// (n × n) for the matrix and (n × n) for B/X here since the right-hand
// side in our use is itself a chunk-by-chunk attn matrix.
//
// Mirrors `ggml_compute_forward_solve_tri_f32`'s inner loop:
//
//   for (i = 0; i < n; ++i) {
//       sum = sum_{t<i} A[i,t] * X[t,j]
//       X[i,j] = (B[i,j] - sum) / A[i,i]
//   }
//
// We hand-roll this rather than calling tensor_matmul_f32 because the
// system is tiny (n = CHUNK_SIZE = 64) and the per-row dependency
// prevents matmul-style parallelism anyway.
static void solve_tri_lower_unit_f32(int n,
                                     const float * A,
                                     const float * B,
                                     float * X) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int t = 0; t < i; t++) {
                sum += A[i * n + t] * X[t * n + j];
            }
            // A has unit diagonal so the division is by 1.0 — keep
            // the divide explicit to match ggml's bit output exactly
            // (the float `(B - sum) / 1.0f` is identical to `B - sum`,
            // but writing it the same way avoids surprises if someone
            // later changes the diagonal convention).
            float diag = A[i * n + i];
            X[i * n + j] = (B[i * n + j] - sum) / diag;
        }
    }
}

// Scalar fp32 matrix multiply: C[m, n] = A[m, k] @ B[k, n].
// Row-major. Used inside chunked_ssm_step for the small (chunk_size,
// chunk_size, k_hd, v_hd) matmuls. Accumulator is fp32 — matches
// ggml's GGML_OP_MUL_MAT on F32 inputs (which is also scalar fp32 acc
// per output element in the generic-C path).
static void matmul_f32_ref(int m, int k, int n,
                           const float * A, const float * B, float * C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += A[i * k + t] * B[t * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

// Chunked Gated DeltaNet SSM step for a single chunk of size
// CHUNK_SIZE (caller pads/segments multi-chunk prompts).
//
// Inputs (single head, single chunk; caller iterates per head):
//   q[CHUNK_SIZE * k_hd]      - post-L2-norm + temperature-scaled Q
//   k[CHUNK_SIZE * k_hd]      - post-L2-norm K
//   v[CHUNK_SIZE * v_hd]      - raw V (post conv1d + silu)
//   g_log[CHUNK_SIZE]         - pre-exp gate (g_log = ssm_a * softplus(α))
//   beta[CHUNK_SIZE]          - post-sigmoid β
//
// In/out:
//   state[k_hd * v_hd]        - recurrent state (row-major k×v).
//                               Caller initialises to zeros for a fresh
//                               conversation, then this function carries
//                               it across chunks within a batched prefill.
//
// Output:
//   out[CHUNK_SIZE * v_hd]    - per-token v outputs (pre-gated-rmsnorm).
//
// Scratch (caller-allocated):
//   gcs        [CHUNK_SIZE]                      g_cumsum
//   gexp       [CHUNK_SIZE]                      exp(g_cumsum)
//   decay_mask [CHUNK_SIZE * CHUNK_SIZE]         decay weights, masked
//   k_beta     [CHUNK_SIZE * k_hd]               K * β (per-token bcast)
//   v_beta     [CHUNK_SIZE * v_hd]               V * β
//   kk_dot     [CHUNK_SIZE * CHUNK_SIZE]         K · K_β^T scaled by decay
//   lhs        [CHUNK_SIZE * CHUNK_SIZE]         (I - attn_lower)
//   attn       [CHUNK_SIZE * CHUNK_SIZE]         SOLVE_TRI result + I
//   v_eff      [CHUNK_SIZE * v_hd]               attn^T @ v_beta
//   kbeta_gexp [CHUNK_SIZE * k_hd]               K_β * gexp
//   k_cumdecay [CHUNK_SIZE * k_hd]               (attn @ kbeta_gexp^T)^T
//   attn_kq    [CHUNK_SIZE * CHUNK_SIZE]         intra-chunk attention
//   q_g_exp    [CHUNK_SIZE * k_hd]               Q * gexp
//   attn_inter [CHUNK_SIZE * v_hd]               state @ q_g_exp
//   v_prime    [CHUNK_SIZE * v_hd]               state @ k_cumdecay^T
//   v_new      [CHUNK_SIZE * v_hd]               V_eff - v_prime
//   v_attn     [CHUNK_SIZE * v_hd]               attn^T @ v_new
//   key_gdiff  [CHUNK_SIZE * k_hd]               K * exp(gcs[-1] - gcs)
//   kgd_vnew   [k_hd * v_hd]                     key_gdiff^T @ v_new
//
// All scratch buffers are stack-allocated when chunk_size and head
// dims are small enough; the caller passes them in to share across
// heads / chunks and keep this function arena-free.
static void chunked_ssm_step_f32(int k_hd, int v_hd,
                                 const float * q,
                                 const float * k_in,
                                 const float * v_in,
                                 const float * g_log,
                                 const float * beta,
                                 float * state,
                                 float * out,
                                 // scratch — caller supplies
                                 float * gcs,
                                 float * gexp,
                                 float * decay_mask,
                                 float * k_beta,
                                 float * v_beta,
                                 float * kk_dot,
                                 float * lhs,
                                 float * attn,
                                 float * v_eff,
                                 float * kbeta_gexp,
                                 float * k_cumdecay,
                                 float * attn_kq,
                                 float * q_g_exp,
                                 float * attn_inter,
                                 float * v_prime,
                                 float * v_new,
                                 float * v_attn,
                                 float * key_gdiff,
                                 float * kgd_vnew) {
    const int N = CHUNK_SIZE;
    // 1. g_cumsum
    float acc = 0.0f;
    for (int t = 0; t < N; t++) {
        acc += g_log[t];
        gcs[t] = acc;
    }
    // 5a. gexp = exp(gcs)
    for (int t = 0; t < N; t++) {
        gexp[t] = expf(gcs[t]);
    }
    // 2. decay_mask[i, j] = exp(gcs[i] - gcs[j]) for i > j, else 0
    //    Diagonal is 0 (multiplied by diag_mask which is 0 on diag in ggml).
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            decay_mask[i * N + j] = (i > j) ? expf(gcs[i] - gcs[j]) : 0.0f;
        }
    }
    // 3a. k_beta[t, d] = k_in[t, d] * beta[t]
    //     v_beta[t, e] = v_in[t, e] * beta[t]
    for (int t = 0; t < N; t++) {
        float b = beta[t];
        for (int d = 0; d < k_hd; d++) {
            k_beta[t * k_hd + d] = k_in[t * k_hd + d] * b;
        }
        for (int e = 0; e < v_hd; e++) {
            v_beta[t * v_hd + e] = v_in[t * v_hd + e] * b;
        }
    }
    // 3b. kk_dot[i, j] = (K[i, :] · K_beta[j, :]) * decay_mask[i, j]
    //     attn_lower (strict lower tri) = -kk_dot * causal_lower (i.e. negate, keep i>j)
    //     lhs = I - attn_lower = I + kk_dot * causal_lower
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i > j) {
                float s = 0.0f;
                for (int d = 0; d < k_hd; d++) {
                    s += k_in[i * k_hd + d] * k_beta[j * k_hd + d];
                }
                kk_dot[i * N + j] = s * decay_mask[i * N + j];
                lhs[i * N + j]    = kk_dot[i * N + j];     // I + (-attn_lower) = I + kk_dot
                // also the matrix passed as RHS to solve_tri is -attn_lower's
                // strict-lower part, which equals kk_dot here. ggml's attn
                // before solve is -(kk_dot * decay * causal); since we
                // already include decay and causal_lower above, the RHS
                // matrix's strict-lower part IS kk_dot. Upper part is 0.
            } else if (i == j) {
                kk_dot[i * N + j] = 0.0f;
                lhs[i * N + j]    = 1.0f;
            } else {
                kk_dot[i * N + j] = 0.0f;
                lhs[i * N + j]    = 0.0f;
            }
        }
    }
    // 3c. SOLVE_TRI(lhs, kk_dot, lower=1, unit=1) -> in-place into attn,
    //     then add identity on the diagonal so attn = lin_solve + I.
    //     The strict-lower part of attn after solve is the "within-chunk
    //     attention coefficient" between tokens i and j with j < i.
    solve_tri_lower_unit_f32(N, lhs, kk_dot, attn);
    for (int i = 0; i < N; i++) {
        attn[i * N + i] = 1.0f;                            // add identity on diag
        for (int j = i + 1; j < N; j++) {
            attn[i * N + j] = 0.0f;                        // mask upper (causal)
        }
    }
    // 4. v_eff = attn^T @ v_beta    (shape [N, v_hd])
    //    v_eff[t, e] = sum_s attn[s, t] * v_beta[s, e]
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int u = 0; u < N; u++) {
                s += attn[u * N + t] * v_beta[u * v_hd + e];
            }
            v_eff[t * v_hd + e] = s;
        }
    }
    // 5b. kbeta_gexp[t, d] = k_beta[t, d] * gexp[t]
    for (int t = 0; t < N; t++) {
        float ge = gexp[t];
        for (int d = 0; d < k_hd; d++) {
            kbeta_gexp[t * k_hd + d] = k_beta[t * k_hd + d] * ge;
        }
    }
    // 5c. k_cumdecay = (attn @ kbeta_gexp^T)^T -> shape [N, k_hd]
    //     Equivalent: k_cumdecay[t, d] = sum_s attn[t, s] * kbeta_gexp[s, d]
    for (int t = 0; t < N; t++) {
        for (int d = 0; d < k_hd; d++) {
            float s = 0.0f;
            for (int u = 0; u < N; u++) {
                s += attn[t * N + u] * kbeta_gexp[u * k_hd + d];
            }
            k_cumdecay[t * k_hd + d] = s;
        }
    }
    // 5d. attn_kq[i, j] = (K[i, :] · Q[j, :]) * decay_mask[i, j]  (causal i>j only)
    //     For i == j: causal includes the diagonal? ggml uses
    //     attn_kq = (K@Q) * decay_mask * diag_mask. diag_mask is 1
    //     on i > j AND on i == j (it's the lower-or-equal mask).
    //     decay_mask on diag = 1 (exp(0)). Actually ggml's diag_mask
    //     is "below diagonal inclusive" — verify when wiring.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i >= j) {
                float s = 0.0f;
                for (int d = 0; d < k_hd; d++) {
                    s += k_in[i * k_hd + d] * q[j * k_hd + d];
                }
                // decay_mask above is 0 on diagonal — but for attn_kq
                // we WANT the diagonal contribution (a token attending
                // to itself). Use the un-masked decay value, exp(0) = 1.
                float dm = (i == j) ? 1.0f : decay_mask[i * N + j];
                attn_kq[i * N + j] = s * dm;
            } else {
                attn_kq[i * N + j] = 0.0f;
            }
        }
    }
    // 6. Per-chunk update (single chunk per call, no outer loop here).
    // 6a. q_g_exp[t, d] = q[t, d] * gexp[t]
    for (int t = 0; t < N; t++) {
        float ge = gexp[t];
        for (int d = 0; d < k_hd; d++) {
            q_g_exp[t * k_hd + d] = q[t * k_hd + d] * ge;
        }
    }
    // 6b. attn_inter[t, e] = sum_d state[d, e] * q_g_exp[t, d]
    //     state is [k_hd, v_hd] row-major.
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * q_g_exp[t * k_hd + d];
            }
            attn_inter[t * v_hd + e] = s;
        }
    }
    // 6c. v_prime[t, e] = sum_d state[d, e] * k_cumdecay[t, d]
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * k_cumdecay[t * k_hd + d];
            }
            v_prime[t * v_hd + e] = s;
        }
    }
    // 6d. v_new = v_eff - v_prime
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            v_new[t * v_hd + e] = v_eff[t * v_hd + e] - v_prime[t * v_hd + e];
        }
    }
    // 6e. v_attn[t, e] = sum_s attn_kq[t, s] * v_new[s, e]   (attn_kq @ v_new)
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int u = 0; u < N; u++) {
                s += attn_kq[t * N + u] * v_new[u * v_hd + e];
            }
            v_attn[t * v_hd + e] = s;
        }
    }
    // 6f. out = attn_inter + v_attn
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            out[t * v_hd + e] = attn_inter[t * v_hd + e] + v_attn[t * v_hd + e];
        }
    }
    // 6g. key_gdiff[t, d] = k_in[t, d] * exp(gcs[N-1] - gcs[t])
    float g_last_scalar = gcs[N - 1];
    float g_last_exp    = expf(g_last_scalar);
    for (int t = 0; t < N; t++) {
        float diff = expf(g_last_scalar - gcs[t]);
        for (int d = 0; d < k_hd; d++) {
            key_gdiff[t * k_hd + d] = k_in[t * k_hd + d] * diff;
        }
    }
    // 6h. kgd_vnew[d, e] = sum_t key_gdiff[t, d] * v_new[t, e]
    //     (key_gdiff^T @ v_new -> [k_hd, v_hd])
    for (int d = 0; d < k_hd; d++) {
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int t = 0; t < N; t++) {
                s += key_gdiff[t * k_hd + d] * v_new[t * v_hd + e];
            }
            kgd_vnew[d * v_hd + e] = s;
        }
    }
    // 6i. state := state * g_last_exp + kgd_vnew
    for (int d = 0; d < k_hd; d++) {
        for (int e = 0; e < v_hd; e++) {
            state[d * v_hd + e] = state[d * v_hd + e] * g_last_exp
                                + kgd_vnew[d * v_hd + e];
        }
    }
    // Note: unused — but referenced in the per-chunk math above for
    // clarity. The matmul_f32_ref / solve_tri_lower_unit_f32 helpers
    // and the cumsum scratch are organised so a future SIMD pass can
    // replace each step in isolation with bit-parity testing per op.
    (void)matmul_f32_ref;
}
