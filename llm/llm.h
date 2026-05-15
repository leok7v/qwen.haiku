// SPDX-License-Identifier: Apache-2.0
//
// llm.h - public C API for the Qwen3.5-0.8B Q4_K_M GGUF runner.
//
// This header is the boundary between the C implementation (llm.c +
// tensor.c, ~3200 LOC) and any caller: Swift via the Xcode bridging
// header (app/bridge.h), Objective-C, Python ctypes, another C
// program, etc. Everything in llm.c that isn't declared here is
// internal.
//
// Lifecycle:
//   1. llm_create(path)         - mmap the GGUF, allocate all KV/SSM
//                                 caches, parse weights. Returns NULL
//                                 only on out-of-memory; check
//                                 llm_loaded(ctx) for parse success.
//   2. llm_tokenize(...)        - byte-level BPE encode UTF-8 text
//                                 into token IDs.
//   3. llm_generate(...)        - streaming generation. The callback
//                                 fires once per generated token with
//                                 the decoded UTF-8 piece.
//   4. llm_destroy(ctx)         - free everything; safe on NULL.
//
// Thread-safety: a single llm_ctx is NOT thread-safe. It holds a
// streaming KV cache and per-layer SSM state that mutates on every
// forward pass. Use one ctx per concurrent inference stream.

#ifndef LLM_H
#define LLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Callers should never dereference; treat it as
// `void *` with type safety.
struct llm_ctx;

// Token streaming callback. `utf8` is a NUL-terminated UTF-8 piece
// (single token's decoded text, may be a partial Unicode codepoint
// for BBPE; concatenating successive pieces always yields valid
// UTF-8). Return non-zero to abort generation.
typedef int (*llm_token_cb)(const char * utf8, void * user);

// Open a GGUF and prepare the context. `path` is a filesystem path
// to a qwen35 architecture Q4_K_M GGUF (e.g. unsloth's release).
// On failure to parse, returns a non-NULL ctx with llm_loaded() == 0
// and an error string available via llm_get_error().
struct llm_ctx * llm_create(const char * path);

// Free ctx and all owned resources. Safe to call with NULL.
void llm_destroy(struct llm_ctx * ctx);

// Clear the per-context recurrent state (SSM + conv1d rings) so the
// next generate() starts as if from a fresh conversation. The KV
// cache is naturally overwritten by the next forward pass and does
// not need explicit clearing. Use before re-feeding a cumulative
// conversation prompt in CLI multi-turn mode.
void llm_reset(struct llm_ctx * ctx);

// 1 if the GGUF was successfully parsed and weights are usable.
// 0 if anything during load failed; call llm_get_error() to find
// out what.
int llm_loaded(const struct llm_ctx * ctx);

// Returns a NUL-terminated error message describing why load or a
// previous call failed. Empty string if no error. Lifetime is tied
// to the ctx; do not free.
const char * llm_get_error(const struct llm_ctx * ctx);

// Encode `text` into token IDs. Writes up to `max_ids` ids into
// `out_ids`. Returns the number of tokens written, or a negative
// value on overflow / error.
int llm_tokenize(struct llm_ctx * ctx, const char * text,
                 int32_t * out_ids, int max_ids);

// Sampler parameters. Mirrors llama.cpp's `common_params_sampling`
// for the subset we care about. Zero-initialized struct = greedy
// (argmax), no filters, no penalty. See `llm_sampler_default` for a
// chat-friendly preset.
struct llm_sampler {
    float    temperature;        // 0 = greedy; >0 = softmax temperature
    int      top_k;              // 0 or 1 = greedy; >1 = keep top k
    float    top_p;              // 0 or 1 = off; (0,1) = nucleus sampling
    float    min_p;              // 0 = off; (0,1) = keep tokens with
                                 // prob >= min_p * top_prob (post-softmax)
    float    repetition_penalty; // 1.0 = off; 1.05-1.3 typical. Logit
                                 // of any token in the recent-history
                                 // window is divided by this when
                                 // positive, multiplied when negative.
    int      repetition_window;  // 0 = penalize against all tokens
                                 // emitted in this generate() call so
                                 // far; >0 = only the last N.
};

// Chat preset: T=1.0, top_k=20, top_p=0.95, min_p=0, rep=1.05, win=64.
// Roughly matches Qwen3.5 non-thinking recommended sampling.
struct llm_sampler llm_sampler_default(void);

// im.ai conversational preset: T=0.7, top_k=40, top_p=0.9, min_p=0.05,
// rep=1.25, win=64. Values lifted from im.ai's Sampler.swift defaults
// (well-tuned for back-and-forth chat — tighter filters and stronger
// repetition penalty than the Qwen3.5 card's recommendation).
struct llm_sampler llm_sampler_im_ai(void);

// ---------------------------------------------------------------------------
// <think>...</think> stream filter (C side, server-side state machine)
//
// Qwen3 models emit a reasoning scratchpad between `<think>` and
// `</think>` markers BEFORE producing visible content. Most chat
// surfaces want to render the two streams differently — reasoning
// muted / collapsible, content prominent. This filter is a small
// byte-stream state machine that splits one llm_generate stream
// into two callbacks: `content_cb` (outside think blocks) and
// `reasoning_cb` (inside). Marker bytes themselves are NOT
// emitted to either callback.
//
// Lifecycle: init -> push... -> finish. Stack-allocated; no
// malloc, no destroy. Reset via re-init.
//
// Inspired by im.ai's ReasoningState.swift table-driven parser; we
// implement just the Qwen3 `<think>` / `</think>` pair here.
// Easy to extend with more marker rows when porting other models.
//
// One emission from the split-stream callback. Exactly one of
// these fields is non-NULL per call; the others are NULL.
// NUL-terminated UTF-8 pieces, same semantics as llm_token_cb's
// `utf8`.
//
// content        - visible reply text. Display in the chat bubble.
// reasoning      - text inside a `<think>...</think>` block. Hide
//                  or render muted; do not feed back into history.
// tool_call      - text inside a `<tool_call>...</tool_call>` block
//                  the model emitted. Display as a "tool: …"
//                  indicator while the C side dispatches.
// tool_response  - text the C-side tool returned for that call.
//                  Display as a "result: …" indicator. After this
//                  chunk fires, the C side automatically continues
//                  generation with the tool's response prefilled
//                  into the model's context — the caller stays a
//                  pure stream consumer; no agent-loop wiring.
struct llm_stream_chunk {
    const char * content;
    const char * reasoning;
    const char * tool_call;
    const char * tool_response;
};

// Return non-zero to abort generation (same convention as
// llm_token_cb).
typedef int (*llm_stream_cb)(const struct llm_stream_chunk * chunk,
                             void * user);

// Like llm_generate, but the raw model output is run through an
// internal <think>...</think> state machine and emitted to `cb` in
// two streams. `cb` is invoked once per chunk with exactly one of
// `chunk->content` / `chunk->reasoning` set (the marker bytes
// themselves are not emitted). Pass NULL for `cb` to discard the
// stream while still advancing the model.
//
// The filter internals (push/finish/state struct) intentionally
// stay file-static in llm.c — callers don't need that surface, and
// hiding it keeps the public ABI small.
int llm_generate_split(struct llm_ctx * ctx,
                       const int32_t * prompt_ids, int prompt_n,
                       int max_new, int min_new,
                       const struct llm_sampler * sampler,
                       uint64_t seed,
                       llm_stream_cb cb,
                       void * user);

// Run inference: prefill the prompt token ids, then sample up to
// `max_new` tokens, calling `cb(chunk, user)` once per emitted
// piece. The raw token stream is run through an internal
// `<think>...</think>` filter, so each chunk has EXACTLY ONE of
// `chunk->content` / `chunk->reasoning` non-NULL — the marker
// bytes themselves are never emitted. Callers that just want
// visible bytes read `chunk->content` and ignore the other field.
// Stops early on EOS or when `cb` returns non-zero. Pass NULL for
// `cb` to discard the stream while still advancing the model.
//
// `sampler` controls how each token is chosen. Pass NULL for the
// zero-initialized (greedy) default. See `struct llm_sampler`.
// `seed` initializes a per-call PRNG (xoroshiro128**) used by the
// stochastic sampler paths; pass 0 to derive from the wall clock.
// Greedy sampling (temperature == 0) ignores the seed entirely.
// `min_new` clamps generation to at least that many tokens by
// suppressing eos / eot at the sampling step (logit -> -infinity)
// until the count is reached. 0 disables. Useful for chat where
// the model occasionally emits `<|im_end|>` as its very first
// reply token under high-temperature sampling, producing an empty
// bubble.
//
// Returns the number of tokens actually generated.
int llm_generate(struct llm_ctx * ctx,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct llm_sampler * sampler,
                 uint64_t seed,
                 llm_stream_cb cb, void * user);

// Model metadata accessors. Cheap (O(1)).
int llm_vocab_size(const struct llm_ctx * ctx);
int llm_eos_id    (const struct llm_ctx * ctx);
int llm_bos_id    (const struct llm_ctx * ctx);

// Timing stats from the most recent llm_generate() call.
// pp_per_sec: prompt-prefill throughput, tokens / second (wall time
//   of the prefill loop divided by prompt_n).
// tg_per_sec: token-generation throughput, tokens / second (wall
//   time of the decode loop divided by tokens generated).
// n_prefill, n_generated: token counts from the most recent call.
// All return 0 before any generate call has completed.
double  llm_pp_per_sec (const struct llm_ctx * ctx);
double  llm_tg_per_sec (const struct llm_ctx * ctx);
int32_t llm_n_prefill  (const struct llm_ctx * ctx);
int32_t llm_n_generated(const struct llm_ctx * ctx);

// The chat template Jinja string as stored in the GGUF under the
// `tokenizer.chat_template` KV. NULL if the file has no such KV
// (e.g. base completion models). Lifetime is tied to ctx; do not
// free, the string is mmap-backed.
//
// Mirrors llama.cpp's `llama_model_chat_template(model, NULL)`: the
// caller is expected to apply it, either by piping into a Jinja
// engine, or (recommended for this repo) by using the state machine
// in docs/DESIGN.md whose structure mirrors this template.
const char * llm_chat_template(const struct llm_ctx * ctx);

// ---------------------------------------------------------------------------
// Chat-template formatting (hand-translated Qwen3.5 Jinja, see
// llm/jinja-template.c). These let Swift / Obj-C / Python callers
// produce the model-facing prompt bytes without re-implementing the
// state machine on their side. Returned strings are heap-allocated;
// callers free with `free()` from <stdlib.h>.

// Roles for llm_chat_message.role. Numbers match jinja-template.c
// internals but are part of the public API contract.
enum llm_chat_role {
    LLM_CHAT_ROLE_SYSTEM    = 0,
    LLM_CHAT_ROLE_USER      = 1,
    LLM_CHAT_ROLE_ASSISTANT = 2,
    LLM_CHAT_ROLE_TOOL      = 3,
};

struct llm_chat_message {
    int          role;     // one of llm_chat_role
    const char * content;  // NUL-terminated UTF-8, may be NULL
};

// Render a full conversation. `messages` is `n_messages` entries.
// `add_generation_prompt` (1/0) controls whether the trailing
// `<|im_start|>assistant\n...` block is emitted so the model
// continues from there. `enable_thinking` (1/0) controls whether the
// generation prompt opens a `<think>\n` block (model will fill it
// before producing content) or pre-fills the empty
// `<think>\n\n</think>\n\n` block (skip-reasoning default).
// Returns NULL if `messages` is NULL or `n_messages <= 0`.
char * llm_chat_format(const struct llm_chat_message * messages,
                       int n_messages,
                       int add_generation_prompt,
                       int enable_thinking);

// Render ONE new user turn for persistent-KV chat. Produces the
// bytes you tokenize and feed into the next llm_generate() call,
// given an already-warm context. `system_prefix` is rendered INLINE
// with the user message (no separate <|im_start|>system block) and
// should be passed only on the FIRST turn of a conversation, NULL
// otherwise. Returns NULL if `user_message` is NULL.
char * llm_chat_format_delta(const char * user_message,
                             const char * system_prefix,
                             int enable_thinking);

#ifdef __cplusplus
}
#endif

#endif // LLM_H
