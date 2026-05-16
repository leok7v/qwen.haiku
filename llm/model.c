// SPDX-License-Identifier: Apache-2.0
//
// model.c -- model-agnostic transformer plumbing.
//
// `#include`-d from qwen.c (or any other sibling arch file) as
// part of the single-TU build. Owns:
//   - the shared struct definitions (slm_config, slm_layer_w,
//     slm_weights, slm_kv, slm_model, slm_ctx, slm_tensor_ref)
//   - generic weight resolution (slm_resolve_tensor) and matmul
//     dispatch (matmul_dispatch + weights_as_f32_view)
//   - KV cache management (kv_init/kv_free/kv_row_*)
//   - SSM cache (slm_ssm_cache + ssm_cache_*) — kept here pragmatically
//     since slm_ctx embeds it by value; should eventually become a
//     void* arch_state when smollm.c / lfm2.c land. TODO.
//   - the GQA + RoPE attention block (slm_forward_attn / _attn_batch)
//   - sampling primitives (sample_argmax, struct rng + rng_*)
//   - tracing helpers (slm_trace_*, slm_dump_row, slm_fnv1a64,
//     slm_emit_floats — generic JSONL parity dump)
//   - the two-level lifecycle (slm_model_load, slm_ctx_create, ...)
//
// What stays in qwen.c (or any arch sibling):
//   - struct slm_ssm_cache *math* (only the storage is here)
//   - slm_forward_ssm / _ssm_batch (Gated DeltaNet)
//   - slm_forward_step / _forward_batch (per-layer attn/ssm dispatcher)
//   - autoregressive_ref, chunked_self_test, qwen_self_test
//   - slm_load_config (qwen35.* GGUF KV reader)
//   - slm_load_weights (blk.L.ssm_* / blk.L.attn_* resolver)
//   - the ARENA_F32 macro (file-scope to qwen.c)
//
// Lifecycle calls slm_load_config / slm_load_weights as forward-
// declared static entry points; the arch file (qwen.c) defines them.
// When smollm.c is added, the dispatch becomes a tiny switch on
// the GGUF `general.architecture` value.

#ifndef MODEL_C
#define MODEL_C

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "utils/maps.c"  // oom, struct chars, struct arr, struct map
#include "tensor.c"      // struct tensor + ops (incl. neon.c + chunked.c)
#include "slm.h"         // struct slm_ctrl + public API types
#include "gguf.c"        // GGUF v3 reader

__attribute__((unused))
static double slm_monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Arch-specific entry points provided by the sibling architecture
// file (qwen.c / smollm.c / lfm2.c). model.c calls these from
// slm_model_load; the sibling defines them after model.c is
// included. Static at the TU level — they're not API surface.
struct slm_config;
struct slm_weights;
static int32_t slm_load_config(const struct gguf * g, struct slm_config * c);
static int32_t slm_load_weights(const struct gguf * g,
                                const struct slm_config * cfg,
                                struct slm_weights * w);

struct slm_config {
    int32_t  n_layers;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  hidden_dim;
    int32_t  ffn_dim;
    int32_t  vocab_size;
    int32_t  max_position;
    float    rope_theta;
    int32_t  rope_dim;              // see footnote (1)
    float    norm_eps;
    int32_t  bos_id;
    int32_t  eos_id;
    int32_t  eot_id;                // see footnote (4)
    int32_t  stop_ids[8];           // see footnote (5); -1 in unused slots
    int32_t  n_stop_ids;
    int32_t  attn_output_gate;      // see footnote (2)
    // qwen35 hybrid-only (Gated DeltaNet linear-attention block):
    int32_t  is_hybrid;
    int32_t  full_attn_interval;
    int32_t  linear_n_heads;        // num_k_heads = num_v_heads, =16
    int32_t  linear_k_head_dim;     // =128
    int32_t  linear_v_head_dim;     // =128
    int32_t  linear_conv_kernel;    // =4
    int32_t  rope_sections[4];      // see footnote (3)
};

// slm_config footnotes:
//
// (1) rope_dim: partial rotary - rotate only the first rope_dim
//     entries of each head's vector (64 of 256 for qwen35). 0 means
//     rotate all.
//
// (2) attn_output_gate: 1 if the Q projection includes the qwen35
//     output gate (Q proj emits n_heads*head_dim*2, split into Q
//     and a sigmoid gate); 0 for plain Qwen3.
//
// (3) rope_sections: mrope T / H / W / E sections. qwen35 ships
//     [11, 11, 10, 0]. All zeros falls back to standard NEOX RoPE.
//
// (4) eot_id: secondary stop token for chat turns. eos_id is
//     `<|endoftext|>` (end-of-document); chat turns end with
//     `<|im_end|>` and the model otherwise keeps generating into a
//     hallucinated next role. Looked up by string at load and
//     defaulted to -1 when absent.
//
// (5) stop_ids[]: full set of end-of-generation tokens that
//     llama.cpp recognizes for qwen35. Populated at tokenizer load
//     by string lookup. Mirror of llama.cpp's load-time print:
//     `<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`,
//     `<|file_sep|>`. Without these we run until max_new on any
//     completion that wants to emit `<|endoftext|>` (the model card
//     uses it as the document terminator and emits it readily under
//     non-chat prompts).

// ---------------------------------------------------------------------------
// Tokenizer (byte-level BPE, simplified). Lives in tokenizer.c so the
// `tokenizer_*` namespace stays out of model.c. Pulled in here, after
// struct slm_config is defined, because tokenizer_load takes a config
// pointer.
// ---------------------------------------------------------------------------
#include "tokenizer.c"

// ---------------------------------------------------------------------------
// Model weights - pointers into the mmap'd GGUF, plus type tags.
// ---------------------------------------------------------------------------

struct slm_tensor_ref {
    const void * data;
    int32_t      type;        // GGUF_TT_*
    int32_t      n_dims;
    int64_t      shape[4];
};

struct slm_layer_w {
    int32_t               is_ssm;       // 1 for qwen35 SSM layer, 0 for attention
    // Common (both attention and SSM):
    struct slm_tensor_ref attn_norm;    // RMSNorm before mixer
    struct slm_tensor_ref ffn_norm;     // RMSNorm before FFN (called
                                        // post_attention_norm in qwen35)
    struct slm_tensor_ref ffn_gate;
    struct slm_tensor_ref ffn_up;
    struct slm_tensor_ref ffn_down;
    // Attention-only:
    struct slm_tensor_ref attn_q;
    struct slm_tensor_ref attn_k;
    struct slm_tensor_ref attn_v;
    struct slm_tensor_ref attn_q_norm;
    struct slm_tensor_ref attn_k_norm;
    struct slm_tensor_ref attn_out;
    // SSM-only (qwen35 hybrid):
    struct slm_tensor_ref attn_qkv;     // 1024 -> 6144 input projection
    struct slm_tensor_ref attn_gate;    // 1024 -> 2048 output gate
    struct slm_tensor_ref ssm_a;        // (16,) per-group A_log
    struct slm_tensor_ref ssm_alpha;    // 1024 -> 16 (DeltaNet alpha)
    struct slm_tensor_ref ssm_beta;     // 1024 -> 16 (DeltaNet beta)
    struct slm_tensor_ref ssm_conv1d;   // (4, 6144) depthwise causal conv
    struct slm_tensor_ref ssm_dt_bias;  // (16,) per-group dt bias
    struct slm_tensor_ref ssm_norm;     // (128,) group-wise RMSNorm
    struct slm_tensor_ref ssm_out;      // 2048 -> 1024 output projection
};

struct slm_weights {
    struct slm_tensor_ref tok_embd;
    struct slm_tensor_ref output_norm;
    struct slm_tensor_ref output;       // may equal tok_embd (tied)
    struct slm_layer_w * layers;        // [n_layers]
};
static int32_t slm_resolve_tensor(const struct gguf * g, const char * name,
                                  struct slm_tensor_ref * out,
                                  int32_t required) {
    const struct gguf_tensor * t = gguf_find_tensor(g, name);
    int32_t r = 0;
    if (t == NULL) {
        if (required) {
            fprintf(stderr, "llm: missing required tensor: %s\n", name);
            r = -1;
        }
    } else {
        out->data   = t->data;
        out->type   = (int32_t)t->type;
        out->n_dims = (int32_t)t->n_dims;
        for (int32_t i = 0; i < 4; i++) {
            out->shape[i] = (i < (int32_t)t->n_dims) ? (int64_t)t->shape[i] : 1;
        }
    }
    return r;
}
static void slm_free_weights(struct slm_weights * w) {
    if (w->layers) {
        free(w->layers);
        w->layers = NULL;
    }
}
// Dispatch matmul based on weight tensor's quantization type. Weight
// shape in GGUF is (in_features, out_features) - first dim is the
// inner dim that contracts with x.
static struct tensor * matmul_dispatch(const struct slm_tensor_ref * w,
                                       struct tensor * x) {
    int64_t k     = w->shape[0];
    int64_t out_f = w->shape[1];
    assert(x->ne[0] == k);
    struct tensor * r = NULL;
    switch (w->type) {
        case GGUF_TT_F32: {
            // Wrap the mmap'd weights in a tensor-like view for the
            // fp32 matmul. We use a temporary stack tensor to avoid a
            // copy; matmul_f32 only reads from it.
            struct tensor wt;
            memset(&wt, 0, sizeof(wt));
            wt.ndim  = 2;
            wt.ne[0] = k; wt.ne[1] = out_f;
            for (int i = 2; i < TENSOR_MAX_DIMS; i++) wt.ne[i] = 1;
            wt.nb[0] = sizeof(float);
            wt.nb[1] = k * sizeof(float);
            wt.data  = (float *)w->data;
            wt.arena = x->arena;
            r = tensor_matmul_f32(&wt, x);
            break;
        }
        case GGUF_TT_Q4_K:
            r = tensor_matmul_q4k_f32((const q4k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q5_K:
            r = tensor_matmul_q5k_f32((const q5k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q6_K:
            r = tensor_matmul_q6k_f32((const q6k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q8_0:
            r = tensor_matmul_q8_0_f32((const q8_0_block *)w->data,
                                       out_f, k, x);
            break;
        default:
            fprintf(stderr, "llm: matmul: unsupported weight type %d\n",
                    w->type);
            abort();
    }
    return r;
}

// Cast an mmap'd weights ref to a struct tensor (read-only view) for
// RMSNorm weight tensors etc., which GGUF stores as F32.
static struct tensor weights_as_f32_view(const struct slm_tensor_ref * w,
                                         struct arena * a) {
    struct tensor t;
    memset(&t, 0, sizeof(t));
    t.ndim = w->n_dims;
    for (int32_t i = 0; i < TENSOR_MAX_DIMS; i++) {
        t.ne[i] = (i < w->n_dims) ? w->shape[i] : 1;
    }
    t.nb[0] = sizeof(float);
    for (int32_t i = 1; i < TENSOR_MAX_DIMS; i++) {
        t.nb[i] = t.nb[i - 1] * t.ne[i - 1];
    }
    t.data  = (float *)w->data;
    t.arena = a;
    return t;
}
// ---------------------------------------------------------------------------
// KV cache (fp16 storage; fp32 working buffers for now)
// ---------------------------------------------------------------------------

struct slm_kv {
    int32_t    n_layers;
    int32_t    n_kv_heads;
    int32_t    head_dim;
    int32_t    max_position;
    int32_t    used;
    _Float16 * k;       // [n_layers][max_position][n_kv_heads][head_dim]
    _Float16 * v;
};

static int32_t kv_init(struct slm_kv * c, int32_t n_layers, int32_t n_kv_heads,
                       int32_t head_dim, int32_t max_position) {
    memset(c, 0, sizeof(*c));
    c->n_layers     = n_layers;
    c->n_kv_heads   = n_kv_heads;
    c->head_dim     = head_dim;
    c->max_position = max_position;
    c->used         = 0;
    size_t per_layer = (size_t)max_position * n_kv_heads * head_dim;
    c->k = (_Float16 *)oom(calloc((size_t)n_layers * per_layer,
                                      sizeof(_Float16)));
    c->v = (_Float16 *)oom(calloc((size_t)n_layers * per_layer,
                                      sizeof(_Float16)));
    return 0;
}

static void kv_free(struct slm_kv * c) {
    free(c->k); c->k = NULL;
    free(c->v); c->v = NULL;
}

static _Float16 * kv_row_k(struct slm_kv * c, int32_t L, int32_t pos) {
    size_t per_row = (size_t)c->n_kv_heads * c->head_dim;
    size_t row     = (size_t)L * c->max_position + pos;
    return c->k + row * per_row;
}

static _Float16 * kv_row_v(struct slm_kv * c, int32_t L, int32_t pos) {
    size_t per_row = (size_t)c->n_kv_heads * c->head_dim;
    size_t row     = (size_t)L * c->max_position + pos;
    return c->v + row * per_row;
}
// ---------------------------------------------------------------------------
// SSM state cache (qwen35 hybrid only)
//
// Per SSM layer we carry:
//   - conv_state: a (conv_kernel - 1) * n_channels rolling buffer for
//                 the depthwise causal conv1d. We use a circular index
//                 within a fixed [conv_kernel] buffer so the conv
//                 takes O(conv_kernel) per step.
//   - ssm_state:  the recurrent state per group, shape (group_count,
//                 state_size). Updated in place every step. Stored as
//                 fp32 (state values can be large in magnitude during
//                 long sequences; fp16 would saturate).
//
// Allocated once for every layer in the model; non-SSM layers leave
// their slots zero and untouched. Cheap enough to skip the bookkeeping.
// ---------------------------------------------------------------------------

struct slm_ssm_cache {
    int32_t   n_layers;
    int32_t   n_channels;     // 6144
    int32_t   conv_kernel;    // 4
    int32_t   group_count;    // 16
    int32_t   state_size;     // 128
    float *   conv_state;     // [n_layers][conv_kernel][n_channels] (ring)
    int32_t * conv_head;      // [n_layers] - circular index into conv_state
    float *   ssm_state;      // [n_layers][group_count][state_size]
};

static int32_t ssm_cache_init(struct slm_ssm_cache * s,
                              const struct slm_config * cfg) {
    memset(s, 0, sizeof(*s));
    if (cfg->is_hybrid) {
        s->n_layers    = cfg->n_layers;
        // conv_dim = 2*K_dim + V_dim = 2*(n_heads*head_k) + n_heads*head_v.
        int32_t K_dim  = cfg->linear_n_heads * cfg->linear_k_head_dim;
        int32_t V_dim  = cfg->linear_n_heads * cfg->linear_v_head_dim;
        s->n_channels  = 2 * K_dim + V_dim;
        s->conv_kernel = cfg->linear_conv_kernel;
        s->group_count = cfg->linear_n_heads;
        s->state_size  = cfg->linear_k_head_dim;
        // Recurrent state per head is (head_k_dim, head_v_dim). Total
        // state per layer = n_heads * head_k_dim * head_v_dim.
        size_t cb = (size_t)s->n_layers * s->conv_kernel *
                    s->n_channels * sizeof(float);
        size_t sb = (size_t)s->n_layers * cfg->linear_n_heads *
                    cfg->linear_k_head_dim * cfg->linear_v_head_dim *
                    sizeof(float);
        s->conv_state = (float *)oom(calloc(1, cb));
        s->ssm_state  = (float *)oom(calloc(1, sb));
        s->conv_head  = (int32_t *)oom(
            calloc((size_t)s->n_layers, sizeof(int32_t)));
    }
    return 0;
}

static void ssm_cache_free(struct slm_ssm_cache * s) {
    free(s->conv_state); s->conv_state = NULL;
    free(s->ssm_state);  s->ssm_state  = NULL;
    free(s->conv_head);  s->conv_head  = NULL;
}

// Reset recurrent state to "fresh conversation". The KV cache is
// overwritten by the next forward pass so it does not need
// clearing here; only the SSM/conv recurrent buffers do.
static void ssm_cache_reset(struct slm_ssm_cache * s,
                            const struct slm_config * cfg) {
    if (s->conv_state != NULL) {
        size_t cb = (size_t)s->n_layers * s->conv_kernel *
                    s->n_channels * sizeof(float);
        memset(s->conv_state, 0, cb);
    }
    if (s->ssm_state != NULL) {
        size_t sb = (size_t)s->n_layers * cfg->linear_n_heads *
                    cfg->linear_k_head_dim * cfg->linear_v_head_dim *
                    sizeof(float);
        memset(s->ssm_state, 0, sb);
    }
    if (s->conv_head != NULL) {
        memset(s->conv_head, 0, (size_t)s->n_layers * sizeof(int32_t));
    }
}
// ---------------------------------------------------------------------------
// Forward pass - single token, with KV cache update.
//
// Returns logits as a (vocab_size,) tensor. Uses the per-call arena
// and dispatch matmul; KV cache pre-fill of past tokens is handled
// by feeding tokens through this same function in sequence.
// ---------------------------------------------------------------------------

// Two-level lifecycle (2026-05-15 split):
//
//   struct slm_model — the GGUF mmap + parsed config + weights +
//     tokenizer. Immutable post-load. Sharable across many ctxs
//     in principle (one model, multiple concurrent conversations).
//     Created/destroyed by slm_model_load / slm_model_unload.
//
//   struct slm_ctx — per-conversation mutable state: KV cache, SSM
//     ring, write position, sampler arena, ctrl knobs, stats.
//     References the model by borrowed pointer (lifetime is the
//     caller's responsibility: don't unload a model while any ctx
//     still points at it). Created/destroyed by slm_ctx_create /
//     slm_ctx_destroy.
//
// `slm_reset` is gone: drop the old ctx and create a fresh one for
// a new conversation. The model survives unchanged so the GGUF
// doesn't need to be re-mmapped.
struct slm_model {
    struct gguf            gguf;
    struct slm_config      cfg;
    struct slm_weights     W;
    struct tokenizer       tok;
    int32_t                loaded;
    char                   err[256];
    char *                 chat_template;   // see footnote (1)
};

struct slm_ctx {
    struct slm_model *     model;           // borrowed; NOT owned
    struct slm_kv          kv;
    struct slm_ssm_cache   ssm;
    struct arena *         arena;
    int32_t                dump_layer;      // see footnote (2)
    double                 t_prefill_s;     // see footnote (3)
    double                 t_gen_s;
    int32_t                n_prefill;
    int32_t                n_generated;
    int32_t                pos;             // see footnote (4)
    struct slm_ctrl        ctrl;            // see footnote (5)
    char *                 system_prompt;   // see footnote (6)
};

// slm_model / slm_ctx footnotes:
//
// (1) chat_template: Jinja string from GGUF KV
//     `tokenizer.chat_template`, copied as a NUL-terminated heap
//     string (GGUF stores it length-prefixed). NULL if the GGUF lacks
//     the KV (base completion models). Freed in slm_model_unload.
//
// (2) dump_layer: --dump-layer L diagnostic. When >= 0, the forward
//     path prints the first 8 values of selected intermediate
//     tensors for layer L (and only that layer) before/after each
//     major op. Used for layer-by-layer parity diffing against
//     llama-eval-callback.
//
// (3) t_prefill_s / t_gen_s / n_prefill / n_generated: filled by
//     slm_generate's prefill and decode loops, read back via the
//     slm_pp_per_sec / slm_tg_per_sec / slm_n_* accessors. Zero
//     before any call has completed.
//
// (4) pos: next free position in the KV cache. Carries across
//     consecutive slm_generate calls so multi-turn chat does not
//     need to reformat and re-prefill prior turns — tokenize only
//     the new delta (e.g. `<|im_start|>user\nQ<|im_end|>\n` +
//     gen header) and call slm_generate again. A fresh ctx_create
//     starts pos = 0.
//
// (5) ctrl: per-conversation behavior knobs (tools / think / effort
//     / debug). Initialized by slm_ctx_create from its `ctrl`
//     argument (or defaults if NULL); callers can override via
//     slm_set_ctrl(). slm_generate reads tools and debug from here
//     on every call. think + effort feed into slm_chat_format_delta
//     on the first turn; once committed to KV they are locked in
//     for the rest of the conversation (changing them mid-stream
//     produces a mixed history — that's why ctx is the granularity
//     of "switch these settings").
//
// (6) system_prompt: a heap-owned copy of the system prompt passed
//     to slm_ctx_create. Stored for record / future use (eventual
//     auto-prefill at ctx_create time). Today the caller still
//     passes it explicitly via slm_chat_format_delta on the first
//     turn — the ctx field is informational. NULL if no system
//     prompt was specified. Freed in slm_ctx_destroy.
static int g_dump_layer = -1;
int g_no_q8k_rt = 0;
// Token-level trace: when set, slm_generate prints each sampled
// token ID to stderr as "[tok] %d\n". Used by tools/bench.sh's
// token-level parity mode, which survives ULP drift across
// perf rewrites that change the underlying logits but keep the
// argmax stable.
static int g_trace_tokens = 1;
__attribute__((unused)) static int g_min_new = 0;

// DUMP(label, data, n) - one-line dump-when-this-layer-is-selected.
// Captures `c` (the slm_ctx) and `L` (the current layer index) from
// the calling scope; both are in scope at every dump site in
// slm_forward_ssm / slm_forward_attn / slm_forward_step's layer loop.
// Off (no fprintf, no read of `data`) when c->dump_layer != L.
#define DUMP(label, data, n) \
    do { \
        if (c->dump_layer == L) { \
            slm_dump_row((label), (data), (n)); \
        } \
    } while (0)
// ---------------------------------------------------------------------------
// qwen_trace: machine-comparable per-tensor JSONL dump, enabled when the
// QH_TRACE_OUT env var is set to a writable path. The schema matches
// what llama.cpp/tools/qwen-haiku/llama-qwen-haiku emits, so a side-by-side
// proofdiff of the two files reveals the first divergence on a fixed
// prompt + greedy run. Disabled at runtime when the file pointer is
// NULL - the slm_trace_f32 call sites then degenerate to a single
// pointer comparison.
//
// Emitted fields per line:
//   name, op, type=f32, ne[4], nb[4] (synthetic dense-row strides),
//   nbytes, fnv64 (raw byte hash for binary equality), head/head_hex,
//   tail/tail_hex (first/last up-to-8 floats as both %.9g and as raw
//   fp32 hex bits), sum (drift-sensitive single scalar), n_nan, n_inf.
// ---------------------------------------------------------------------------

static FILE * g_qh_trace_fp = NULL;

__attribute__((unused))
static void slm_trace_open(void) {
    if (g_qh_trace_fp == NULL) {
        const char * p = getenv("QH_TRACE_OUT");
        if (p != NULL && p[0] != '\0') {
            g_qh_trace_fp = fopen(p, "w");
            if (g_qh_trace_fp != NULL) {
                fprintf(stderr, "qwen_trace: writing JSONL to %s\n", p);
            }
        }
    }
}

__attribute__((unused))
static void slm_trace_close(void) {
    if (g_qh_trace_fp != NULL) {
        fclose(g_qh_trace_fp);
        g_qh_trace_fp = NULL;
    }
}

// FNV-1a 64-bit over raw bytes. Stable across runs and across
// implementations; identical bytes -> identical hash. We do NOT need
// cryptographic strength, just "did the bytes match?" - collision
// probability for typical tensor sizes is negligible.
static uint64_t slm_fnv1a64(const uint8_t * data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void slm_emit_floats(FILE * f, const char * key,
                           const float * v, int n) {
    fprintf(f, ",\"%s\":[", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) { fputc(',', f); }
        if (isnan(v[i])) {
            fputs("\"nan\"", f);
        } else if (isinf(v[i])) {
            fputs(v[i] < 0.0f ? "\"-inf\"" : "\"inf\"", f);
        } else {
            fprintf(f, "%.9g", v[i]);
        }
    }
    fputs("]", f);
    fprintf(f, ",\"%s_hex\":[", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) { fputc(',', f); }
        uint32_t u;
        memcpy(&u, &v[i], 4);
        fprintf(f, "\"%08x\"", u);
    }
    fputs("]", f);
}

static void slm_trace_f32(const char * name, const char * op,
                         const float * data,
                         int64_t ne0, int64_t ne1,
                         int64_t ne2, int64_t ne3) {
    FILE *  f = g_qh_trace_fp;
    int64_t n = ne0 * ne1 * ne2 * ne3;
    if (f != NULL && n > 0) {
        size_t   nbytes = (size_t)n * sizeof(float);
        uint64_t fnv    = slm_fnv1a64((const uint8_t *)data, nbytes);
        int64_t  k      = n < 8 ? n : 8;
        int64_t  tstart = n - k > 0 ? n - k : 0;
        float    head[8] = {0,0,0,0,0,0,0,0};
        float    tail[8] = {0,0,0,0,0,0,0,0};
        for (int64_t i = 0; i < k; i++) { head[i] = data[i];          }
        for (int64_t i = 0; i < k; i++) { tail[i] = data[tstart + i]; }
        float sum   = 0.0f;
        int   n_nan = 0;
        int   n_inf = 0;
        for (int64_t i = 0; i < n; i++) {
            float v = data[i];
            if (isnan(v))      { n_nan++; }
            else if (isinf(v)) { n_inf++; }
            else               { sum += v; }
        }
        // Synthetic dense-row strides: nb[0]=4, then ne0*4, ne0*ne1*4,
        // ne0*ne1*ne2*4. This matches what llama.cpp emits for
        // contiguous fp32 tensors.
        int64_t nb1 = ne0 * 4;
        int64_t nb2 = ne0 * ne1 * 4;
        int64_t nb3 = ne0 * ne1 * ne2 * 4;
        fprintf(f,
                "{\"name\":\"%s\",\"op\":\"%s\",\"type\":\"f32\","
                "\"ne\":[%lld,%lld,%lld,%lld],"
                "\"nb\":[4,%lld,%lld,%lld],"
                "\"nbytes\":%zu,\"fnv64\":\"%016llx\"",
                name, op,
                (long long)ne0, (long long)ne1,
                (long long)ne2, (long long)ne3,
                (long long)nb1, (long long)nb2, (long long)nb3,
                nbytes, (unsigned long long)fnv);
        slm_emit_floats(f, "head", head, (int)k);
        slm_emit_floats(f, "tail", tail, (int)k);
        fprintf(f, ",\"sum\":%.9g,\"n_nan\":%d,\"n_inf\":%d}\n",
                sum, n_nan, n_inf);
    }
}

// Convenience: trace a 1D row of n fp32 values.
static inline void slm_trace_row(const char * name, const char * op,
                                const float * data, int64_t n) {
    slm_trace_f32(name, op, data, n, 1, 1, 1);
}

// Convenience: trace a 2D batch tensor of shape (dim, n_tokens).
// Emits one JSONL line covering all n_tokens; the receiving side
// (llama.cpp/qwen-haiku reference dumper) uses the same shape when
// it runs chunked prefill, so byte-comparing the two files works.
static inline void slm_trace_batch(const char * name, const char * op,
                                  const float * data,
                                  int64_t dim, int64_t n_tokens) {
    slm_trace_f32(name, op, data, dim, n_tokens, 1, 1);
}
static void slm_dump_row(const char * label, const float * data, int32_t n) {
    // Mirror llama-eval-callback's format: head + tail + sum. The
    // sum across the whole tensor is the cheapest single number that
    // exposes drift in middle elements (head/tail can be identical
    // while the bulk diverges).
    double dsum = 0.0;
    for (int32_t i = 0; i < n; i++) { dsum += data[i]; }
    fprintf(stderr, "[dump] %s: ", label);
    int32_t k_head = n > 4 ? 4 : n;
    for (int32_t i = 0; i < k_head; i++) {
        fprintf(stderr, "%9.4f ", data[i]);
    }
    if (n >= 1024) {
        fprintf(stderr, " | mid: ");
        int32_t mids[] = { n / 8, n / 4, n / 2, (3 * n) / 4 };
        for (int32_t j = 0; j < 4; j++) {
            fprintf(stderr, "[%d]=%9.4f ", (int)mids[j], data[mids[j]]);
        }
    }
    if (n > 8) {
        fprintf(stderr, "...");
        for (int32_t i = n - 3; i < n; i++) {
            fprintf(stderr, "%9.4f ", data[i]);
        }
    }
    fprintf(stderr, "  sum=%.6f\n", dsum);
}

// ---------------------------------------------------------------------------
// Attention block (GQA + RoPE) - model-agnostic, used by Qwen3/Qwen3.5
// attention layers, classic Qwen3, and any other transformer that
// reads attn_q/k/v/out + optional attn_q_norm/attn_k_norm with a head
// gate. The Qwen3.5 hybrid SSM layers use the SSM forward in qwen.c
// instead; the per-layer dispatcher (slm_forward_step in qwen.c)
// picks which one based on layer_w.is_ssm.
// ---------------------------------------------------------------------------

static struct tensor * slm_forward_attn(struct slm_ctx * c,
                                        int32_t L, int32_t pos,
                                        struct tensor * h) {
    struct arena * a = c->arena;
    struct slm_layer_w * Lw = &c->model->W.layers[L];
    int32_t hd         = c->model->cfg.head_dim;
    int32_t n_h        = c->model->cfg.n_heads;
    int32_t n_kvh      = c->model->cfg.n_kv_heads;
    // attn_inner is the attention output width before attn_output.
    // For pure Qwen3 it equals hidden_dim, for qwen35 it does not
    // (head_dim=256 * n_heads=8 = 2048, hidden=1024).
    int32_t attn_inner = n_h * hd;
    int32_t kv_hidden  = n_kvh * hd;
    // RMSNorm.
    struct tensor attn_norm_w =
        weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->model->cfg.norm_eps);
    DUMP("[A]attn_nm", h_norm->data, c->model->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        slm_trace_row(nm, "RMS_NORM", h_norm->data, c->model->cfg.hidden_dim);
    }
    // Q / K / V projections. Qwen3.5 has attn_output_gate=true: the
    // Q projection emits n_heads*head_dim*2, split into (Q, gate).
    // The gate is sigmoid'd and multiplied with the attention output
    // before attn_output. attn_q.weight shape is therefore
    // (hidden, 2*n_heads*head_dim).
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Qfull-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", q_raw->data, attn_inner * 2);
        snprintf(nm, sizeof(nm), "Kcur-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", k->data, kv_hidden);
        snprintf(nm, sizeof(nm), "Vcur-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT", v->data, kv_hidden);
    }
    if (c->dump_layer == L) {
        slm_dump_row("[A]Qfull  ", q_raw->data, attn_inner * 2);
        slm_dump_row("[A]Kcur   ", k->data, n_kvh * hd);
        slm_dump_row("[A]Vcur   ", v->data, n_kvh * hd);
        // One-shot diagnostics for attn_v block 0.
        float dqv[QK_K];
        const q6k_block * vrow0 = (const q6k_block *)Lw->attn_v.data;
        q6k_dequant_block(vrow0, dqv);
        slm_dump_row("[A]Vw[0]  ", dqv, 16);
        const uint8_t * raw = (const uint8_t *)vrow0;
        fprintf(stderr, "[dump] Vw[0] block bytes: d=0x%02x%02x"
                " scales[0..3]=%d %d %d %d  qh[0]=0x%02x  ql[0]=0x%02x\n",
                raw[209], raw[208],
                (int)((const int8_t *)raw)[192],
                (int)((const int8_t *)raw)[193],
                (int)((const int8_t *)raw)[194],
                (int)((const int8_t *)raw)[195],
                raw[128], raw[0]);
        fprintf(stderr, "[dump] sizeof q6k_block = %zu (should be 210)\n",
                sizeof(q6k_block));
        double s = 0.0;
        for (int32_t b = 0; b < 4; b++) {
            q6k_dequant_block(vrow0 + b, dqv);
            for (int32_t i = 0; i < QK_K; i++) {
                s += (double)dqv[i] * (double)h_norm->data[b * QK_K + i];
            }
        }
        fprintf(stderr, "[dump] manual Vw0 . hnorm = %.6f (v[0]=%.6f)\n",
                s, v->data[0]);
    }
    // Split q_raw (2*attn_inner) into Q and gate when qwen35's
    // attn_output_gate is on. Layout is HEAD-INTERLEAVED per
    // llama.cpp src/models/qwen35.cpp:491-518: per head h, the
    // 2*hd block holds Q[hd] then gate[hd].
    struct tensor * q = tensor_new_3d(a, hd, n_h, 1);
    struct tensor * attn_gate_v = NULL;
    if (c->model->cfg.attn_output_gate) {
        attn_gate_v = tensor_new_2d(a, attn_inner, 1);
        for (int32_t h2 = 0; h2 < n_h; h2++) {
            const float * src = q_raw->data + h2 * (2 * hd);
            for (int32_t i = 0; i < hd; i++) {
                q->data[h2 * hd + i]           = src[i];
                attn_gate_v->data[h2 * hd + i] = src[hd + i];
            }
        }
    } else {
        for (int32_t i = 0; i < attn_inner; i++) {
            q->data[i] = q_raw->data[i];
        }
    }
    // Reshape K / V to (head_dim, n_kv_heads, 1). q is already
    // shaped (head_dim, n_heads, 1).
    k->ndim = 3; k->ne[0] = hd; k->ne[1] = n_kvh; k->ne[2] = 1;
    k->ne[3] = 1; tensor_set_packed_strides(k);
    v->ndim = 3; v->ne[0] = hd; v->ne[1] = n_kvh; v->ne[2] = 1;
    v->ne[3] = 1; tensor_set_packed_strides(v);
    // Optional Q/K per-head RMSNorm (Qwen3 specific).
    if (Lw->attn_q_norm.data != NULL) {
        struct tensor qn_w = weights_as_f32_view(&Lw->attn_q_norm, a);
        struct tensor q2 = *q;
        q2.ndim = 2;
        q2.ne[0] = hd; q2.ne[1] = n_h;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) q2.ne[i] = 1;
        tensor_set_packed_strides(&q2);
        struct tensor * qnorm =
            tensor_rms_norm(&q2, &qn_w, c->model->cfg.norm_eps);
        qnorm->ndim = 3;
        qnorm->ne[0] = hd; qnorm->ne[1] = n_h; qnorm->ne[2] = 1;
        tensor_set_packed_strides(qnorm);
        q = qnorm;
    }
    if (Lw->attn_k_norm.data != NULL) {
        struct tensor kn_w = weights_as_f32_view(&Lw->attn_k_norm, a);
        struct tensor k2 = *k;
        k2.ndim = 2;
        k2.ne[0] = hd; k2.ne[1] = n_kvh;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) k2.ne[i] = 1;
        tensor_set_packed_strides(&k2);
        struct tensor * knorm =
            tensor_rms_norm(&k2, &kn_w, c->model->cfg.norm_eps);
        knorm->ndim = 3;
        knorm->ne[0] = hd; knorm->ne[1] = n_kvh; knorm->ne[2] = 1;
        tensor_set_packed_strides(knorm);
        k = knorm;
    }
    // Partial RoPE on q and k. For qwen35 this is the interleaved-
    // mrope variant ported from ggml-cpu/ops.cpp. For plain Qwen3
    // it falls back to standard NEOX-style RoPE when sections are
    // all zero.
    int32_t rotary_dim = c->model->cfg.rope_dim > 0 ? c->model->cfg.rope_dim : hd;
    int32_t use_imrope = c->model->cfg.rope_sections[0] ||
                         c->model->cfg.rope_sections[1] ||
                         c->model->cfg.rope_sections[2] ||
                         c->model->cfg.rope_sections[3];
    struct tensor * q_rope;
    struct tensor * k_rope;
    if (use_imrope) {
        q_rope = tensor_rope_mrope_i(q, pos, c->model->cfg.rope_theta,
                                     rotary_dim, c->model->cfg.rope_sections);
        k_rope = tensor_rope_mrope_i(k, pos, c->model->cfg.rope_theta,
                                     rotary_dim, c->model->cfg.rope_sections);
    } else {
        q_rope = tensor_rope(q, pos, c->model->cfg.rope_theta, rotary_dim);
        k_rope = tensor_rope(k, pos, c->model->cfg.rope_theta, rotary_dim);
    }
    // Write new K/V row into KV cache (cast to fp16).
    _Float16 * kdst = kv_row_k(&c->kv, L, pos);
    _Float16 * vdst = kv_row_v(&c->kv, L, pos);
    for (int32_t i = 0; i < kv_hidden; i++) {
        kdst[i] = (_Float16)k_rope->data[i];
        vdst[i] = (_Float16)v->data[i];
    }
    // Build K/V cache views for attention over positions 0..pos.
    int32_t kv_len = pos + 1;
    struct tensor * k_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    struct tensor * v_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    for (int32_t p = 0; p < kv_len; p++) {
        const _Float16 * ks = kv_row_k(&c->kv, L, p);
        const _Float16 * vs = kv_row_v(&c->kv, L, p);
        float * kd = k_all->data + p * n_kvh * hd;
        float * vd = v_all->data + p * n_kvh * hd;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kd[i] = (float)ks[i];
            vd[i] = (float)vs[i];
        }
    }
    // Attention with causal k_offset = pos so kv_max = pos+1.
    // Passing 0 here was a historical catastrophic bug.
    struct tensor * ctx_t =
        tensor_attention(q_rope, k_all, v_all, /*k_offset=*/pos);
    ctx_t->ndim = 2;
    ctx_t->ne[0] = attn_inner; ctx_t->ne[1] = 1;
    for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) ctx_t->ne[i] = 1;
    tensor_set_packed_strides(ctx_t);
    DUMP("[A]Qrope  ", q_rope->data, attn_inner);
    DUMP("[A]Krope  ", k_rope->data, n_kvh * hd);
    DUMP("[A]attn   ", ctx_t->data, attn_inner);
    // Output gate: attn_output *= sigmoid(gate). qwen35 only.
    if (c->model->cfg.attn_output_gate && attn_gate_v != NULL) {
        for (int32_t i = 0; i < attn_inner; i++) {
            float g_v = attn_gate_v->data[i];
            float sig = 1.0f / (1.0f + expf(-g_v));
            ctx_t->data[i] *= sig;
        }
        DUMP("[A]gated  ", ctx_t->data, attn_inner);
    }
    // Output projection.
    struct tensor * attn_out_t = matmul_dispatch(&Lw->attn_out, ctx_t);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_out-%d", (int)L);
        slm_trace_row(nm, "MUL_MAT",
                     attn_out_t->data, c->model->cfg.hidden_dim);
    }
    return attn_out_t;
}
static struct tensor * slm_forward_attn_batch(struct slm_ctx * c,
                                              int32_t L,
                                              int32_t pos_start,
                                              int32_t n,
                                              struct tensor * h) {
    struct arena * a = c->arena;
    struct slm_layer_w * Lw = &c->model->W.layers[L];
    int32_t hd         = c->model->cfg.head_dim;
    int32_t n_h        = c->model->cfg.n_heads;
    int32_t n_kvh      = c->model->cfg.n_kv_heads;
    int32_t attn_inner = n_h * hd;
    int32_t kv_hidden  = n_kvh * hd;
    // RMSNorm over the n tokens. tensor_rms_norm iterates ne[1] = n.
    struct tensor attn_norm_w =
        weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->model->cfg.norm_eps);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        slm_trace_batch(nm, "RMS_NORM",
                       h_norm->data, c->model->cfg.hidden_dim, n);
    }
    // Q/K/V projections — matmul_dispatch handles ne[1]=n natively.
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Qfull-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", q_raw->data, attn_inner * 2, n);
        snprintf(nm, sizeof(nm), "Kcur-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", k->data, kv_hidden, n);
        snprintf(nm, sizeof(nm), "Vcur-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT", v->data, kv_hidden, n);
    }
    // Split q_raw into (q, gate) per token when attn_output_gate=true.
    // q_raw has shape [2*attn_inner, n] in head-interleaved layout.
    struct tensor * q = tensor_new_3d(a, hd, n_h, n);
    struct tensor * attn_gate_v = NULL;
    if (c->model->cfg.attn_output_gate) {
        attn_gate_v = tensor_new_2d(a, attn_inner, n);
        for (int32_t t = 0; t < n; t++) {
            const float * src = q_raw->data + (size_t)t * (2 * attn_inner);
            float * qdst   = q->data           + (size_t)t * attn_inner;
            float * gdst   = attn_gate_v->data + (size_t)t * attn_inner;
            for (int32_t h2 = 0; h2 < n_h; h2++) {
                const float * sh = src + h2 * (2 * hd);
                for (int32_t i = 0; i < hd; i++) {
                    qdst[h2 * hd + i] = sh[i];
                    gdst[h2 * hd + i] = sh[hd + i];
                }
            }
        }
    } else {
        for (int32_t t = 0; t < n; t++) {
            memcpy(q->data + (size_t)t * attn_inner,
                   q_raw->data + (size_t)t * attn_inner,
                   (size_t)attn_inner * sizeof(float));
        }
    }
    // Reshape k/v to (hd, n_kvh, n).
    k->ndim = 3; k->ne[0] = hd; k->ne[1] = n_kvh; k->ne[2] = n;
    k->ne[3] = 1; tensor_set_packed_strides(k);
    v->ndim = 3; v->ne[0] = hd; v->ne[1] = n_kvh; v->ne[2] = n;
    v->ne[3] = 1; tensor_set_packed_strides(v);
    // Per-head Q / K RMS norm (Qwen3 specific). View as 2D
    // (hd, n_h*n) / (hd, n_kvh*n) so tensor_rms_norm normalises
    // each head-vector independently.
    if (Lw->attn_q_norm.data != NULL) {
        struct tensor qn_w = weights_as_f32_view(&Lw->attn_q_norm, a);
        struct tensor q2 = *q;
        q2.ndim = 2;
        q2.ne[0] = hd; q2.ne[1] = (int64_t)n_h * n;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) q2.ne[i] = 1;
        tensor_set_packed_strides(&q2);
        struct tensor * qnorm =
            tensor_rms_norm(&q2, &qn_w, c->model->cfg.norm_eps);
        qnorm->ndim = 3;
        qnorm->ne[0] = hd; qnorm->ne[1] = n_h; qnorm->ne[2] = n;
        tensor_set_packed_strides(qnorm);
        q = qnorm;
    }
    if (Lw->attn_k_norm.data != NULL) {
        struct tensor kn_w = weights_as_f32_view(&Lw->attn_k_norm, a);
        struct tensor k2 = *k;
        k2.ndim = 2;
        k2.ne[0] = hd; k2.ne[1] = (int64_t)n_kvh * n;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) k2.ne[i] = 1;
        tensor_set_packed_strides(&k2);
        struct tensor * knorm =
            tensor_rms_norm(&k2, &kn_w, c->model->cfg.norm_eps);
        knorm->ndim = 3;
        knorm->ne[0] = hd; knorm->ne[1] = n_kvh; knorm->ne[2] = n;
        tensor_set_packed_strides(knorm);
        k = knorm;
    }
    // RoPE on q [hd, n_h, n] and k [hd, n_kvh, n] starting at pos_start.
    // tensor_rope* already iterate seq axis (ne[2]) and apply
    // pos_t = pos_offset + s.
    int32_t rotary_dim = c->model->cfg.rope_dim > 0 ? c->model->cfg.rope_dim : hd;
    int32_t use_imrope = c->model->cfg.rope_sections[0] ||
                         c->model->cfg.rope_sections[1] ||
                         c->model->cfg.rope_sections[2] ||
                         c->model->cfg.rope_sections[3];
    struct tensor * q_rope;
    struct tensor * k_rope;
    if (use_imrope) {
        q_rope = tensor_rope_mrope_i(q, pos_start, c->model->cfg.rope_theta,
                                     rotary_dim, c->model->cfg.rope_sections);
        k_rope = tensor_rope_mrope_i(k, pos_start, c->model->cfg.rope_theta,
                                     rotary_dim, c->model->cfg.rope_sections);
    } else {
        q_rope = tensor_rope(q, pos_start, c->model->cfg.rope_theta, rotary_dim);
        k_rope = tensor_rope(k, pos_start, c->model->cfg.rope_theta, rotary_dim);
    }
    // Write the n new (K, V) rows into the KV cache.
    for (int32_t t = 0; t < n; t++) {
        _Float16 * kdst = kv_row_k(&c->kv, L, pos_start + t);
        _Float16 * vdst = kv_row_v(&c->kv, L, pos_start + t);
        const float * ksrc = k_rope->data + (size_t)t * kv_hidden;
        const float * vsrc = v->data      + (size_t)t * kv_hidden;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kdst[i] = (_Float16)ksrc[i];
            vdst[i] = (_Float16)vsrc[i];
        }
    }
    // Build full K/V views [hd, n_kvh, kv_len] over the whole cache
    // up to pos_start + n - 1.
    int32_t kv_len = pos_start + n;
    struct tensor * k_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    struct tensor * v_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    for (int32_t p = 0; p < kv_len; p++) {
        const _Float16 * ks = kv_row_k(&c->kv, L, p);
        const _Float16 * vs = kv_row_v(&c->kv, L, p);
        float * kd = k_all->data + (size_t)p * kv_hidden;
        float * vd = v_all->data + (size_t)p * kv_hidden;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kd[i] = (float)ks[i];
            vd[i] = (float)vs[i];
        }
    }
    // Batched attention: causal mask handled by tensor_attention via
    // k_offset = pos_start (query t attends keys 0..pos_start+t).
    struct tensor * ctx_t =
        tensor_attention(q_rope, k_all, v_all, /*k_offset=*/pos_start);
    // View as (attn_inner, n) for output projection.
    ctx_t->ndim = 2;
    ctx_t->ne[0] = attn_inner;
    ctx_t->ne[1] = n;
    for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) ctx_t->ne[i] = 1;
    tensor_set_packed_strides(ctx_t);
    // Output gate (per token): ctx *= sigmoid(gate).
    if (c->model->cfg.attn_output_gate && attn_gate_v != NULL) {
        for (int32_t t = 0; t < n; t++) {
            float * out_row = ctx_t->data       + (size_t)t * attn_inner;
            float * gat_row = attn_gate_v->data + (size_t)t * attn_inner;
            for (int32_t i = 0; i < attn_inner; i++) {
                float sig = 1.0f / (1.0f + expf(-gat_row[i]));
                out_row[i] *= sig;
            }
        }
    }
    // Output projection [hidden_dim, n].
    struct tensor * attn_out_t = matmul_dispatch(&Lw->attn_out, ctx_t);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_out-%d", (int)L);
        slm_trace_batch(nm, "MUL_MAT",
                       attn_out_t->data, c->model->cfg.hidden_dim, n);
    }
    return attn_out_t;
}

// ---------------------------------------------------------------------------
// Sampling primitives. sample_argmax for greedy; struct rng + rng_*
// for the temperature/top-k/top-p chain in sampler.c. Pure scalar,
// model-agnostic.
// ---------------------------------------------------------------------------

static int32_t sample_argmax(const struct tensor * logits) {
    int64_t n = tensor_nelements(logits);
    int32_t best = 0;
    float v = logits->data[0];
    for (int64_t i = 1; i < n; i++) {
        if (logits->data[i] > v) {
            v    = logits->data[i];
            best = (int32_t)i;
        }
    }
    return best;
}
struct rng { uint64_t s0, s1; };

static inline uint64_t rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(struct rng * r) {
    uint64_t s0    = r->s0;
    uint64_t s1    = r->s1;
    uint64_t res   = rng_rotl(s0 * 5, 7) * 9;
    s1            ^= s0;
    r->s0          = rng_rotl(s0, 24) ^ s1 ^ (s1 << 16);
    r->s1          = rng_rotl(s1, 37);
    return res;
}

// SplitMix64 to expand a single 64-bit seed into the two-word state.
__attribute__((unused))
static void rng_seed(struct rng * r, uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s0 = z ^ (z >> 31);
    z     = (r->s0 + 0x9e3779b97f4a7c15ULL);
    z     = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z     = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s1 = z ^ (z >> 31);
    // Avoid the all-zero state, which would lock xoroshiro at zero.
    if (r->s0 == 0 && r->s1 == 0) { r->s0 = 1; }
}

static inline float rng_uniform(struct rng * r) {
    // 24-bit mantissa, divided to [0, 1).
    return (float)(rng_next(r) >> 40) / (float)(1u << 24);
}

// ---------------------------------------------------------------------------
// Two-level lifecycle: slm_model owns the immutable mmap+weights+
// tokenizer; slm_ctx owns the per-conversation mutable state (KV
// cache, SSM ring, write position, sampler arena, ctrl knobs).
// slm_load_config / slm_load_weights are arch-specific entry points
// provided by qwen.c (forward-declared above).
// ---------------------------------------------------------------------------

struct slm_model * slm_model_load(const char * path) {
    // Pick the best int8 dot / SiLU implementation for this CPU. Safe
    // to call multiple times (early-return if already initialized);
    // a no-op when the same model is reloaded.
    simd_init();
    struct slm_model * m =
        (struct slm_model *)oom(calloc(1, sizeof(struct slm_model)));
    if (gguf_open(&m->gguf, path) != 0) {
        snprintf(m->err, sizeof(m->err), "gguf open failed");
        return m;
    }
    if (slm_load_config(&m->gguf, &m->cfg) != 0) {
        snprintf(m->err, sizeof(m->err), "config load failed");
        return m;
    }
    if (slm_load_weights(&m->gguf, &m->cfg, &m->W) != 0) {
        snprintf(m->err, sizeof(m->err), "weights resolve failed");
        return m;
    }
    if (tokenizer_load(&m->tok, &m->gguf, &m->cfg) != 0) {
        snprintf(m->err, sizeof(m->err), "tokenizer load failed");
        return m;
    }
    // Chat-turn stop token: look up `<|im_end|>` in the vocab now
    // that the tokenizer's string->id map is built. Stays -1 when the
    // model isn't a chat-tuned vocab.
    m->cfg.eot_id = s2i_get(&m->tok.vocab_to_id,
                            "<|im_end|>", 10, -1);
    // Full end-of-generation set: mirrors what llama.cpp prints on
    // load as "EOG tokens" for qwen35. Generation stops on any of
    // these. We can fit 8; the model declares 5.
    static const char * stop_strs[] = {
        "<|endoftext|>",
        "<|im_end|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        NULL,
    };
    m->cfg.n_stop_ids = 0;
    for (int32_t i = 0; stop_strs[i] != NULL; i++) {
        int32_t id = s2i_get(&m->tok.vocab_to_id,
                             stop_strs[i],
                             (int32_t)strlen(stop_strs[i]), -1);
        if (id >= 0 && m->cfg.n_stop_ids <
            (int32_t)(sizeof(m->cfg.stop_ids) / sizeof(m->cfg.stop_ids[0]))) {
            m->cfg.stop_ids[m->cfg.n_stop_ids++] = id;
        }
    }
    // Stash the Jinja chat template from `tokenizer.chat_template`
    // as a null-terminated copy. Optional KV - base completion
    // models won't have one; instruct/chat models do.
    const struct gguf_kv * ct = gguf_find_kv(&m->gguf,
                                             "tokenizer.chat_template");
    if (ct != NULL && ct->v.type == GGUF_VT_STR) {
        size_t cc = 0;
        uint64_t n = gr_u64(ct->v.raw, &cc);
        m->chat_template = (char *)malloc(n + 1);
        if (m->chat_template != NULL) {
            memcpy(m->chat_template, ct->v.raw + cc, n);
            m->chat_template[n] = '\0';
        }
    }
    m->loaded = 1;
    return m;
}

void slm_model_unload(struct slm_model * m) {
    if (m != NULL) {
        tokenizer_free(&m->tok);
        slm_free_weights(&m->W);
        if (m->chat_template) { free(m->chat_template); }
        gguf_close(&m->gguf);
        free(m);
    }
}

bool slm_model_loaded(const struct slm_model * m) {
    return m != NULL && m->loaded != 0;
}

const char * slm_model_error(const struct slm_model * m) {
    return m == NULL ? "no model" : m->err;
}

struct slm_ctx * slm_ctx_create(struct slm_model * m,
                                 const char * system_prompt,
                                 const struct slm_ctrl * ctrl) {
    struct slm_ctx * c = NULL;
    if (m != NULL && m->loaded) {
        c = (struct slm_ctx *)oom(calloc(1, sizeof(struct slm_ctx)));
        c->model = m;
        c->ctrl  = (ctrl != NULL) ? *ctrl : slm_ctrl_defaults();
        kv_init(&c->kv, m->cfg.n_layers, m->cfg.n_kv_heads,
                m->cfg.head_dim, m->cfg.max_position);
        ssm_cache_init(&c->ssm, &m->cfg);
        c->arena      = arena_new(64 * 1024 * 1024);
        c->dump_layer = g_dump_layer;
        c->pos        = 0;
        if (system_prompt != NULL && system_prompt[0] != '\0') {
            // Record the system prompt as the conversation's
            // identity. We don't auto-prefill it (yet) — the
            // caller continues to thread it through
            // slm_chat_format_delta on turn 1 — but holding a
            // copy here lets future ctx_create wiring shift the
            // first-turn framing into the C side without changing
            // the API again.
            size_t n = strlen(system_prompt);
            c->system_prompt = (char *)oom(malloc(n + 1));
            memcpy(c->system_prompt, system_prompt, n + 1);
        }
    }
    return c;
}

void slm_ctx_destroy(struct slm_ctx * c) {
    if (c != NULL) {
        ssm_cache_free(&c->ssm);
        kv_free(&c->kv);
        if (c->arena)         { arena_free(c->arena); c->arena = NULL; }
        if (c->system_prompt) { free(c->system_prompt); }
        free(c);
    }
}

void slm_set_ctrl(struct slm_ctx * c, const struct slm_ctrl * ctrl) {
    if (c != NULL && ctrl != NULL) { c->ctrl = *ctrl; }
}

struct slm_ctrl slm_get_ctrl(const struct slm_ctx * c) {
    return c != NULL ? c->ctrl : slm_ctrl_defaults();
}

// slm_reset: file-internal, NOT exported. Drops the per-ctx pos +
// SSM ring back to a fresh-conversation state WITHOUT reallocating
// the KV cache or losing the model pointer. Used by agent.c's
// iteration loop and the CLI chat-test (which verifies reset
// reproducibility). The public way to start a new conversation is
// slm_ctx_destroy + slm_ctx_create.
__attribute__((unused))
static void slm_reset(struct slm_ctx * c) {
    if (c != NULL) {
        ssm_cache_reset(&c->ssm, &c->model->cfg);
        c->pos = 0;
    }
}

#endif  // MODEL_C
