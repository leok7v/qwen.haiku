// SPDX-License-Identifier: Apache-2.0
//
// model.c -- model-agnostic transformer plumbing.
//
// `#include`-d from qwen.c (or any other sibling arch file) as
// part of the single-TU build. Owns:
//   - the shared struct definitions (slm_config, slm_layer,
//     slm_weights, slm_kv, slm_model, slm_ctx, slm_tensor)
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
#include "utils/text.c"  // struct text (vector of owned UTF-8 strings)

// model.c owns the trace ring storage: this is the ONE TU that
// defines TRACE_IMPLEMENTATION. Every other consumer (Swift via
// bridge.h, or any future C consumer) just #includes utils/trace.c
// without the define and gets the declarations only — the linker
// resolves the extern functions back to this TU's copy.
#define TRACE_IMPLEMENTATION
#include "utils/trace.c"
#include "tensor.c"      // struct tensor + ops (incl. neon.c + chunked.c)
#include "slm.h"         // struct slm_ctrl + public API types
#include "gguf.c"        // GGUF v3 reader

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

struct slm_tensor {
    const void * data;
    int32_t      type;        // GGUF_TT_*
    int32_t      n_dims;
    int64_t      shape[4];
};

struct slm_layer {
    int32_t               is_ssm;       // 1 for qwen35 SSM layer, 0 for attention
    // Common (both attention and SSM):
    struct slm_tensor attn_norm;    // RMSNorm before mixer
    struct slm_tensor ffn_norm;     // RMSNorm before FFN (called
                                        // post_attention_norm in qwen35)
    struct slm_tensor ffn_gate;
    struct slm_tensor ffn_up;
    struct slm_tensor ffn_down;
    // Attention-only:
    struct slm_tensor attn_q;
    struct slm_tensor attn_k;
    struct slm_tensor attn_v;
    struct slm_tensor attn_q_norm;
    struct slm_tensor attn_k_norm;
    struct slm_tensor attn_out;
    // SSM-only (qwen35 hybrid):
    struct slm_tensor attn_qkv;     // 1024 -> 6144 input projection
    struct slm_tensor attn_gate;    // 1024 -> 2048 output gate
    struct slm_tensor ssm_a;        // (16,) per-group A_log
    struct slm_tensor ssm_alpha;    // 1024 -> 16 (DeltaNet alpha)
    struct slm_tensor ssm_beta;     // 1024 -> 16 (DeltaNet beta)
    struct slm_tensor ssm_conv1d;   // (4, 6144) depthwise causal conv
    struct slm_tensor ssm_dt_bias;  // (16,) per-group dt bias
    struct slm_tensor ssm_norm;     // (128,) group-wise RMSNorm
    struct slm_tensor ssm_out;      // 2048 -> 1024 output projection
};

struct slm_weights {
    struct slm_tensor tok_embd;
    struct slm_tensor output_norm;
    struct slm_tensor output;       // may equal tok_embd (tied)
    struct slm_layer * layers;        // [n_layers]
};

static int32_t slm_resolve_tensor(const struct gguf * g, const char * name,
                                  struct slm_tensor * out,
                                  int32_t required) {
    const struct gguf_tensor * t = gguf_find_tensor(g, name);
    int32_t r = 0;
    if (t == NULL) {
        if (required) {
            trace("missing required tensor: %s\n", name);
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
static struct tensor * matmul_dispatch(const struct slm_tensor * w,
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
            trace("matmul: unsupported weight type %d\n", w->type);
            abort();
    }
    return r;
}

// Cast an mmap'd weights ref to a struct tensor (read-only view) for
// RMSNorm weight tensors etc., which GGUF stores as F32.
static struct tensor weights_as_f32_view(const struct slm_tensor * w,
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

// ssm_cache_reset is not in the current architecture.
// Reason it could come back is branched conversations
// "save SSM snapshot, run a hypothetical, restore".
// That needs a slm_ssm_cache_clone + slm_ssm_cache_assign, not a zero-reset.

__attribute__((unused))
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
// Sampling primitives. sample_argmax for greedy; struct rng + rng_*
// for the temperature/top-k/top-p chain in sampler.c. Defined above
// struct slm_ctx because the ctx embeds a `struct rng` by value (one
// rng per conversation, seeded in slm_ctx_create, advanced across
// calls).
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
    struct chars           err;
    char *                 chat_template;   // see footnote (1)
};

define_array(int32_t, slm_tokens);

struct slm_ctx {
    struct slm_model *     model;           // borrowed; NOT owned
    struct slm_kv          kv;
    struct slm_ssm_cache   ssm;
    struct arena *         arena;
    struct rng             rng;             // see footnote (2)
    double                 t_prefill_s;     // see footnote (3)
    double                 t_gen_s;
    int32_t                n_prefill;
    int32_t                n_generated;
    int32_t                pos;             // see footnote (4)
    struct slm_ctrl        ctrl;            // see footnote (5)
    bool                   system_committed; // see footnote (6)
    struct slm_tokens      ids;             // see footnote (7)
    struct text            messages;        // see footnote (8)
};

// slm_model / slm_ctx footnotes:
//
// (1) chat_template: Jinja string from GGUF KV
//     `tokenizer.chat_template`, copied as a NUL-terminated heap
//     string (GGUF stores it length-prefixed). NULL if the GGUF lacks
//     the KV. Freed in slm_model_unload.
//
// (2) rng: per-conversation PRNG state. Seeded in slm_ctx_create
//     from time(NULL) so unseeded chat is non-deterministic across
//     runs. slm_generate's `seed` parameter re-seeds this rng before
//     decode when caller passes a non-zero value (parity-gate /
//     reproducibility path). Advances across slm_generate calls
//     within one ctx when caller passes seed=0.
//
// (3) t_prefill_s / t_gen_s / n_prefill / n_generated: filled by
//     slm_generate's prefill and decode loops, read back via the
//     slm_pp_per_sec / slm_tg_per_sec / slm_n_* accessors. Zero
//     before any call has completed.
//
// (4) pos: next free position in the KV cache. Carries across
//     consecutive slm_generate calls — multi-turn chat tokenizes
//     only the new delta and continues from there.
//
// (5) ctrl: per-conversation behavior knobs. Mutable via
//     slm_ctx_ctrl() until slm_ctx_system_prompt commits the
//     framing; tools/think become read-only after that. debug
//     stays mutable on the fly.
//
// (6) system_committed: flipped true by slm_ctx_system_prompt on
//     success. slm_generate checks this and refuses to run if false
//     (the framing must be prefilled first, even when system text
//     is empty).
//
// (7) ids: append-only history of every token the ctx has seen.
//     Includes system framing, every prefilled user turn, every
//     generated assistant token, and every tool dialogue token.
//     Exposed via slm_ctx_tokens(); pointer stable until the next
//     ctx call.
//
// (8) messages: append-only vector of UTF-8 fragments emitted via
//     the stream callback. Each user prompt is appended as one
//     string; each emitted reasoning / content / tool fragment is
//     appended as its own string. Exposed via
//     slm_ctx_messages_count / slm_ctx_message. The callback's
//     content/reasoning/call/response pointers point INTO this
//     array's owned strings.
// g_no_q8k_rt is the matmul Q8_K runtime gate — build/perf knob, not
// per-conversation state, so kept file-static here (see tensor.c
// for the consumer). Set via NO_Q8K_RT env from slm.c's main().
int g_no_q8k_rt = 0;

// slm_dump(c, L, label, data, n) - layer-gated one-line tensor dump.
// Used inside slm_forward_ssm / slm_forward_attn / slm_forward_step's
// per-layer loop for parity diffing against llama-eval-callback. The
// raw printer is slm_dump_row (defined below) — call it directly when
// you've already gated a block with `if (c->ctrl.dump_layer == L)`.
static void slm_dump_row(const char * label, const float * data, int32_t n);
static inline void slm_dump(const struct slm_ctx * c, int32_t L,
                            const char * label,
                            const float * data, int32_t n) {
    if (c->ctrl.dump_layer == L) {
        slm_dump_row(label, data, n);
    }
}
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

// Tag the JSONL trace with a printf-formatted name. The whole emit
// path is gated on g_qh_trace_fp at the call site so snprintf isn't
// executed when tracing is off (the common case). clang's -Wformat
// type-checks the snprintf inside the macro just like a real call.
//
// Call sites read as one line:
//   slm_trace_row("RMS_NORM", h_norm->data, hidden_dim,
//                 "attn_norm-%d", L);
//
// Literal names: pass a no-conversion format string:
//   slm_trace_row("GET_ROWS", h->data, hidden_dim, "inp_embd");

#define slm_trace_row(op, data, n, fmt, ...) \
    do { \
        if (g_qh_trace_fp != NULL) { \
            char _trace_nm_[64]; \
            snprintf(_trace_nm_, sizeof(_trace_nm_), (fmt), \
                     ##__VA_ARGS__); \
            slm_trace_f32(_trace_nm_, (op), (data), (n), 1, 1, 1); \
        } \
    } while (0)

// Same shape for 2D batch tensors (dim, n_tokens) — used by the
// chunked prefill path so byte-comparing JSONL against llama.cpp's
// qwen-haiku reference dumper works.
#define slm_trace_batch(op, data, dim, n_tokens, fmt, ...) \
    do { \
        if (g_qh_trace_fp != NULL) { \
            char _trace_nm_[64]; \
            snprintf(_trace_nm_, sizeof(_trace_nm_), (fmt), \
                     ##__VA_ARGS__); \
            slm_trace_f32(_trace_nm_, (op), (data), \
                          (dim), (n_tokens), 1, 1); \
        } \
    } while (0)

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
    struct slm_layer * Lw = &c->model->W.layers[L];
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
    slm_dump(c, L, "[A]attn_nm", h_norm->data, c->model->cfg.hidden_dim);
    slm_trace_row("RMS_NORM", h_norm->data, c->model->cfg.hidden_dim,
                  "attn_norm-%d", (int)L);
    // Q / K / V projections. Qwen3.5 has attn_output_gate=true: the
    // Q projection emits n_heads*head_dim*2, split into (Q, gate).
    // The gate is sigmoid'd and multiplied with the attention output
    // before attn_output. attn_q.weight shape is therefore
    // (hidden, 2*n_heads*head_dim).
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    slm_trace_row("MUL_MAT", q_raw->data, attn_inner * 2,
                  "Qfull-%d", (int)L);
    slm_trace_row("MUL_MAT", k->data,     kv_hidden,
                  "Kcur-%d",  (int)L);
    slm_trace_row("MUL_MAT", v->data,     kv_hidden,
                  "Vcur-%d",  (int)L);
    if (c->ctrl.dump_layer == L) {
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
    slm_dump(c, L, "[A]Qrope  ", q_rope->data, attn_inner);
    slm_dump(c, L, "[A]Krope  ", k_rope->data, n_kvh * hd);
    slm_dump(c, L, "[A]attn   ", ctx_t->data, attn_inner);
    // Output gate: attn_output *= sigmoid(gate). qwen35 only.
    if (c->model->cfg.attn_output_gate && attn_gate_v != NULL) {
        for (int32_t i = 0; i < attn_inner; i++) {
            float g_v = attn_gate_v->data[i];
            float sig = 1.0f / (1.0f + expf(-g_v));
            ctx_t->data[i] *= sig;
        }
        slm_dump(c, L, "[A]gated  ", ctx_t->data, attn_inner);
    }
    // Output projection.
    struct tensor * attn_out_t = matmul_dispatch(&Lw->attn_out, ctx_t);
    slm_trace_row("MUL_MAT", attn_out_t->data, c->model->cfg.hidden_dim,
                  "attn_out-%d", (int)L);
    return attn_out_t;
}

static struct tensor * slm_forward_attn_batch(struct slm_ctx * c,
                                              int32_t L,
                                              int32_t pos_start,
                                              int32_t n,
                                              struct tensor * h) {
    struct arena * a = c->arena;
    struct slm_layer * Lw = &c->model->W.layers[L];
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
    slm_trace_batch("RMS_NORM", h_norm->data, c->model->cfg.hidden_dim, n,
                    "attn_norm-%d", (int)L);
    // Q/K/V projections — matmul_dispatch handles ne[1]=n natively.
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    slm_trace_batch("MUL_MAT", q_raw->data, attn_inner * 2, n,
                    "Qfull-%d", (int)L);
    slm_trace_batch("MUL_MAT", k->data,     kv_hidden,      n,
                    "Kcur-%d",  (int)L);
    slm_trace_batch("MUL_MAT", v->data,     kv_hidden,      n,
                    "Vcur-%d",  (int)L);
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
    slm_trace_batch("MUL_MAT", attn_out_t->data,
                    c->model->cfg.hidden_dim, n,
                    "attn_out-%d", (int)L);
    return attn_out_t;
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
        chars_printf(&m->err, "gguf open failed");
        return m;
    }
    if (slm_load_config(&m->gguf, &m->cfg) != 0) {
        chars_printf(&m->err, "config load failed");
        return m;
    }
    if (slm_load_weights(&m->gguf, &m->cfg, &m->W) != 0) {
        chars_printf(&m->err, "weights resolve failed");
        return m;
    }
    if (tokenizer_load(&m->tok, &m->gguf, &m->cfg) != 0) {
        chars_printf(&m->err, "tokenizer load failed");
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
        chars_free(&m->err);
        free(m);
    }
}

bool slm_model_loaded(const struct slm_model * m) {
    return m != NULL && m->loaded != 0;
}

const char * slm_model_error(const struct slm_model * m) {
    const char * r = "no model";
    if (m != NULL) {
        r = m->err.data != NULL ? m->err.data : "";
    }
    return r;
}

struct slm_ctx * slm_ctx_create(struct slm_model * m) {
    struct slm_ctx * c = NULL;
    if (m != NULL && m->loaded) {
        c = (struct slm_ctx *)oom(calloc(1, sizeof(struct slm_ctx)));
        c->model = m;
        c->ctrl  = slm_ctrl_defaults();
        kv_init(&c->kv, m->cfg.n_layers, m->cfg.n_kv_heads,
                m->cfg.head_dim, m->cfg.max_position);
        ssm_cache_init(&c->ssm, &m->cfg);
        c->arena      = arena_new(64 * 1024 * 1024);
        c->pos        = 0;
        c->system_committed = false;
        // Per-conversation rng. Seeded from wall-clock so unseeded
        // chat is non-deterministic across runs. slm_generate's
        // `seed` parameter re-seeds this rng before decode when the
        // caller passes a non-zero value (the parity-gate path).
        rng_seed(&c->rng, (uint64_t)time(NULL));
        // c->ids and c->messages are zero-initialized via calloc;
        // first append in slm_ctx_system_prompt / slm_generate
        // grows the vectors on demand.
    }
    return c;
}

void slm_ctx_destroy(struct slm_ctx * c) {
    if (c != NULL) {
        ssm_cache_free(&c->ssm);
        kv_free(&c->kv);
        if (c->arena) { arena_free(c->arena); c->arena = NULL; }
        free(c->ids.data);
        text_free(&c->messages);
        free(c);
    }
}

// ---------------------------------------------------------------------------
// Snapshot / restore. Heap-owned deep copy of every mutable cache the
// forward pass writes to. Sized to the snapshot's `pos` for KV (so
// short-conversation snapshots stay small), full for the fixed SSM
// state.
// ---------------------------------------------------------------------------

struct slm_snapshot {
    int32_t    pos;
    size_t     ids_count;
    struct rng rng;
    _Float16 * k;            // [n_layers][pos][n_kv_heads][head_dim]
    _Float16 * v;            // same layout as k
    float    * conv_state;   // mirrors slm_ssm_cache.conv_state (or NULL)
    float    * ssm_state;    // mirrors slm_ssm_cache.ssm_state  (or NULL)
    int32_t  * conv_head;    // [n_layers] (or NULL)
};

struct slm_snapshot * slm_ctx_snapshot(const struct slm_ctx * c) {
    struct slm_snapshot * s = NULL;
    if (c != NULL) {
        s = (struct slm_snapshot *)oom(calloc(1, sizeof(*s)));
        s->pos       = c->pos;
        s->ids_count = c->ids.count;
        s->rng       = c->rng;
        int32_t L   = c->kv.n_layers;
        int32_t pos = c->pos;
        size_t  per = (size_t)c->kv.n_kv_heads * c->kv.head_dim;
        if (L > 0 && pos > 0 && per > 0) {
            size_t n_per_layer = (size_t)pos * per;
            size_t bytes       = (size_t)L * n_per_layer * sizeof(_Float16);
            s->k = (_Float16 *)oom(malloc(bytes));
            s->v = (_Float16 *)oom(malloc(bytes));
            for (int32_t l = 0; l < L; l++) {
                size_t src = (size_t)l * (size_t)c->kv.max_position * per;
                size_t dst = (size_t)l * n_per_layer;
                memcpy(s->k + dst, c->kv.k + src,
                       n_per_layer * sizeof(_Float16));
                memcpy(s->v + dst, c->kv.v + src,
                       n_per_layer * sizeof(_Float16));
            }
        }
        if (c->ssm.conv_state != NULL) {
            size_t cb = (size_t)c->ssm.n_layers * c->ssm.conv_kernel *
                        c->ssm.n_channels * sizeof(float);
            s->conv_state = (float *)oom(malloc(cb));
            memcpy(s->conv_state, c->ssm.conv_state, cb);
        }
        if (c->ssm.ssm_state != NULL) {
            size_t sb = (size_t)c->ssm.n_layers *
                        c->model->cfg.linear_n_heads *
                        c->model->cfg.linear_k_head_dim *
                        c->model->cfg.linear_v_head_dim *
                        sizeof(float);
            s->ssm_state = (float *)oom(malloc(sb));
            memcpy(s->ssm_state, c->ssm.ssm_state, sb);
        }
        if (c->ssm.conv_head != NULL) {
            size_t hb = (size_t)c->ssm.n_layers * sizeof(int32_t);
            s->conv_head = (int32_t *)oom(malloc(hb));
            memcpy(s->conv_head, c->ssm.conv_head, hb);
        }
    }
    return s;
}

void slm_ctx_restore(struct slm_ctx * c, const struct slm_snapshot * s) {
    if (c != NULL && s != NULL) {
        c->pos = s->pos;
        c->rng = s->rng;
        if (s->ids_count <= c->ids.count) {
            c->ids.count = s->ids_count;
        }
        int32_t L   = c->kv.n_layers;
        int32_t pos = s->pos;
        size_t  per = (size_t)c->kv.n_kv_heads * c->kv.head_dim;
        if (L > 0 && pos > 0 && per > 0 && s->k != NULL && s->v != NULL) {
            size_t n_per_layer = (size_t)pos * per;
            for (int32_t l = 0; l < L; l++) {
                size_t dst = (size_t)l * (size_t)c->kv.max_position * per;
                size_t src = (size_t)l * n_per_layer;
                memcpy(c->kv.k + dst, s->k + src,
                       n_per_layer * sizeof(_Float16));
                memcpy(c->kv.v + dst, s->v + src,
                       n_per_layer * sizeof(_Float16));
            }
        }
        if (s->conv_state != NULL && c->ssm.conv_state != NULL) {
            size_t cb = (size_t)c->ssm.n_layers * c->ssm.conv_kernel *
                        c->ssm.n_channels * sizeof(float);
            memcpy(c->ssm.conv_state, s->conv_state, cb);
        }
        if (s->ssm_state != NULL && c->ssm.ssm_state != NULL) {
            size_t sb = (size_t)c->ssm.n_layers *
                        c->model->cfg.linear_n_heads *
                        c->model->cfg.linear_k_head_dim *
                        c->model->cfg.linear_v_head_dim *
                        sizeof(float);
            memcpy(c->ssm.ssm_state, s->ssm_state, sb);
        }
        if (s->conv_head != NULL && c->ssm.conv_head != NULL) {
            memcpy(c->ssm.conv_head, s->conv_head,
                   (size_t)c->ssm.n_layers * sizeof(int32_t));
        }
    }
}

void slm_snapshot_free(struct slm_snapshot * s) {
    if (s != NULL) {
        free(s->k);
        free(s->v);
        free(s->conv_state);
        free(s->ssm_state);
        free(s->conv_head);
        free(s);
    }
}

struct slm_ctrl * slm_ctx_ctrl(struct slm_ctx * c) {
    return &c->ctrl;
}

const int32_t * slm_ctx_tokens(const struct slm_ctx * c, int32_t * out_n) {
    const int32_t * p = NULL;
    int32_t n = 0;
    if (c != NULL && c->ids.count > 0) {
        p = c->ids.data;
        n = (int32_t)c->ids.count;
    }
    if (out_n != NULL) { *out_n = n; }
    return p;
}

int32_t slm_ctx_messages_count(const struct slm_ctx * c) {
    return c == NULL ? 0 : (int32_t)c->messages.count;
}

const char * slm_ctx_message(const struct slm_ctx * c, int32_t i) {
    const char * out = NULL;
    if (c != NULL && i >= 0 && (size_t)i < c->messages.count) {
        out = c->messages.data[i];
    }
    return out;
}

__attribute__((unused))
static void slm_ctx_ids_append(struct slm_ctx * c,
                               const int32_t * src, size_t n) {
    if (n > 0) {
        arr_grow((struct arr *)&c->ids, sizeof(int32_t), c->ids.count + n);
        memcpy(c->ids.data + c->ids.count, src, n * sizeof(int32_t));
        c->ids.count += n;
    }
}

#endif  // MODEL_C
