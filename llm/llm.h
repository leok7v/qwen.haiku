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
//   1. slm_create(path)         - mmap the GGUF, allocate all KV/SSM
//                                 caches, parse weights. Returns NULL
//                                 only on out-of-memory; check
//                                 slm_loaded(ctx) for parse success.
//   2. slm_tokenize(...)        - byte-level BPE encode UTF-8 text
//                                 into token IDs.
//   3. slm_generate(...)        - streaming generation. The callback
//                                 fires once per generated token with
//                                 the decoded UTF-8 piece.
//   4. slm_destroy(ctx)         - free everything; safe on NULL.
//
// Thread-safety: a single slm_ctx is NOT thread-safe. It holds a
// streaming KV cache and per-layer SSM state that mutates on every
// forward pass. Use one ctx per concurrent inference stream.

// Roadmap (still TODO):
//   1. Split `struct slm_ctx` into `slm_model` (mmap + weights +
//      tokenizer, immutable post-load, shareable across ctxs) and
//      `slm_ctx` (KV / SSM / pos / ctrl, one per conversation).
//   2. Replace slm_create / slm_destroy / slm_reset with
//      `slm_model_load / _unload` + `slm_ctx_create / _destroy`.
//      slm_ctx_create takes (model, system_prompt, slm_ctrl *).
//
// Done in the current pass: `struct slm_ctrl` exists (tools / think
// / effort / debug); the latter three pieces are stored on the ctx
// and consulted by slm_generate, so the API change above is purely
// a lifecycle refactor — the data is already in place.

#ifndef SLM_H
#define SLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Callers should never dereference; treat it as
// `void *` with type safety.
struct slm_ctx;

// (`slm_token_cb`, the raw per-token callback, used to live here
// but was internal-only. It now lives as a private typedef in llm.c
// — every public caller goes through `slm_stream_cb` below.)

// Open a GGUF and prepare the context. `path` is a filesystem path
// to a qwen35 architecture Q4_K_M GGUF (e.g. unsloth's release).
// On failure to parse, returns a non-NULL ctx with slm_loaded() == 0
// and an error string available via slm_get_error().
struct slm_ctx * slm_create(const char * path);

// Free ctx and all owned resources. Safe to call with NULL.
void slm_destroy(struct slm_ctx * ctx);

// Clear the per-context recurrent state (SSM + conv1d rings) so the
// next generate() starts as if from a fresh conversation. The KV
// cache is naturally overwritten by the next forward pass and does
// not need explicit clearing. Use before re-feeding a cumulative
// conversation prompt in CLI multi-turn mode.
void slm_reset(struct slm_ctx * ctx);

// 1 if the GGUF was successfully parsed and weights are usable.
// 0 if anything during load failed; call slm_get_error() to find
// out what.
int slm_loaded(const struct slm_ctx * ctx);

// Returns a NUL-terminated error message describing why load or a
// previous call failed. Empty string if no error. Lifetime is tied
// to the ctx; do not free.
const char * slm_get_error(const struct slm_ctx * ctx);

// Encode `text` into token IDs. Allocates an int32_t array sized to
// the worst-case bound (strlen(text), since byte-level BPE never
// produces more tokens than input bytes); caller takes ownership and
// frees with `free()`. Returns the number of tokens actually written.
// Aborts via `oom()` on allocation failure — there is no error path
// to recover from here.
//
// Callers commonly want the array AND its length, so the count comes
// back via the return value and the pointer via the out-param:
//
//     int32_t * ids = NULL;
//     int n = slm_tokenize(ctx, text, &ids);
//     // use ids[0..n) ...
//     free(ids);
int slm_tokenize(struct slm_ctx * ctx, const char * text,
                 int32_t ** out_ids);

// Sampler parameters. Mirrors llama.cpp's `common_params_sampling`
// for the subset we care about. Zero-initialized struct = greedy
// (argmax). Prefer slm_sampler_defaults() below for the chat-tuned
// defaults.
//
// Feature toggles (tool dispatch / reasoning / debug verbosity)
// used to live here. They moved to `struct slm_ctrl` below — the
// sampler stays purely about how-to-pick-the-next-token; tools /
// think / debug describe conversation-level behaviour, set once
// per ctx and consulted by slm_generate.
struct slm_sampler {
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

// Per-context behavior knobs. Stored on slm_ctx; slm_generate reads
// them on every call. Conceptually:
//   - `tools` / `think` / `effort` are "what kind of conversation
//      is this" — set once at conversation start (the eventual
//      slm_ctx_create call) and left alone. Changing them mid-
//      conversation works mechanically but produces a mixed history
//      (the KV cache already holds the first-turn framing).
//   - `debug` is a verbosity level (0 = quiet, 9 = chatty,
//     intermediate values reserved). Free to flip on the fly.
struct slm_ctrl {
    bool         tools;          // enable embedded agent dispatch in
                                 // slm_generate. With false, the
                                 // <tool_call> / </tool_call> markers
                                 // are NOT recognised — model output
                                 // streams as plain content.
    bool         think;          // enable reasoning. Hooked into
                                 // slm_chat_format_delta: with true,
                                 // the generation prompt opens
                                 // <think>\n; with false, it pre-fills
                                 // <think>\n\n</think>\n\n and the
                                 // model jumps to content.
    const char * effort;         // "low" / "medium" / "high" / NULL.
                                 // slm_chat_format_delta prepends a
                                 // hint to the first-turn system
                                 // prompt accordingly. NULL == "medium"
                                 // == no hint.
    int32_t      debug;          // verbosity. > 0 enables chunk->call
                                 // / chunk->response visibility chunks
                                 // from slm_generate. Future levels
                                 // can carry richer trace.
};

// Default ctrl: tools on, think off, effort = medium (no hint),
// debug = 1 (visibility chunks on).
struct slm_ctrl slm_ctrl_defaults(void);

// Set / get the ctrl currently stored on `ctx`. `set` copies the
// struct; the caller's storage can go out of scope immediately.
void            slm_set_ctrl(struct slm_ctx * ctx,
                             const struct slm_ctrl * ctrl);
struct slm_ctrl slm_get_ctrl(const struct slm_ctx * ctx);

// Default chat sampler: T=0.7, top_k=40, top_p=0.9, min_p=0.05,
// rep=1.25, win=64, tools=true, think=false, debug=true. Lifted
// from im.ai's Sampler.swift defaults — empirically the best fit
// for Qwen3.5-0.8B in chat + tools mode. The CLI hardcodes this;
// callers can override individual fields.
//
// Function name intentionally differs from the struct tag: when
// the C struct and a function share a name (C allows it — tag and
// ordinary namespaces are separate), Clang imports BOTH into Swift
// under the same identifier and `slm_sampler(field: value, ...)`
// at a Swift call site silently resolved to the no-arg getter
// (`tools = true`, ignoring what the caller passed). Different
// names side-steps the importer collision; this stays even after
// the eventual slm_ rename (see top-of-file TODO).
struct slm_sampler slm_sampler_defaults(void);

// ---------------------------------------------------------------------------
// <think>...</think> stream filter (C side, server-side state machine)
//
// Qwen3 models emit a reasoning scratchpad between `<think>` and
// `</think>` markers BEFORE producing visible content. Most chat
// surfaces want to render the two streams differently — reasoning
// muted / collapsible, content prominent. This filter is a small
// byte-stream state machine that splits one slm_generate stream
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
// NUL-terminated UTF-8 pieces — concatenating successive pieces
// always yields valid UTF-8 (the C side holds back partial codepoints
// across calls).
//
// content        - visible reply text. Display in the chat bubble.
// reasoning      - text inside a `<think>...</think>` block. Hide
//                  or render muted; do not feed back into history.
// call           - text inside a `<tool_call>...</tool_call>` block
//                  the model emitted. Display as a "tool: …"
//                  indicator while the C side dispatches.
// response       - text the C-side tool returned for that call.
//                  Display as a "result: …" indicator. After this
//                  chunk fires, the C side automatically continues
//                  generation with the tool's response prefilled
//                  into the model's context — the caller stays a
//                  pure stream consumer; no agent-loop wiring.
// prefilled      - signal-only flag. Set to true on the one chunk
//                  fired when prompt prefill finishes (the four
//                  string fields above are NULL on that emission).
//                  Useful for hiding a "thinking…" spinner / loading
//                  bar once decode starts. There is no granular
//                  prefill progress — for Qwen3.5-0.8B prefill takes
//                  ~1s and a per-token bar is more noise than value.
struct slm_stream_chunk {
    const char * content;
    const char * reasoning;
    const char * call;
    const char * response;
    bool         prefilled;
};

// Return non-zero to abort generation.
typedef int (*slm_stream_cb)(const struct slm_stream_chunk * chunk,
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
// zero-initialized (greedy) default. See `struct slm_sampler`.
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
int slm_generate(struct slm_ctx * ctx,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct slm_sampler * sampler,
                 uint64_t seed,
                 slm_stream_cb cb, void * user);

// Model metadata accessors. Cheap (O(1)).
int slm_vocab_size(const struct slm_ctx * ctx);
int slm_eos_id    (const struct slm_ctx * ctx);
int slm_bos_id    (const struct slm_ctx * ctx);

// Timing stats from the most recent slm_generate() call.
// pp_per_sec: prompt-prefill throughput, tokens / second (wall time
//   of the prefill loop divided by prompt_n).
// tg_per_sec: token-generation throughput, tokens / second (wall
//   time of the decode loop divided by tokens generated).
// n_prefill, n_generated: token counts from the most recent call.
// All return 0 before any generate call has completed.
double  slm_pp_per_sec (const struct slm_ctx * ctx);
double  slm_tg_per_sec (const struct slm_ctx * ctx);
int32_t slm_n_prefill  (const struct slm_ctx * ctx);
int32_t slm_n_generated(const struct slm_ctx * ctx);

// The chat template Jinja string as stored in the GGUF under the
// `tokenizer.chat_template` KV. NULL if the file has no such KV
// (e.g. base completion models). Lifetime is tied to ctx; do not
// free, the string is mmap-backed.
//
// Mirrors llama.cpp's `llama_model_chat_template(model, NULL)`: the
// caller is expected to apply it, either by piping into a Jinja
// engine, or (recommended for this repo) by using the state machine
// in docs/DESIGN.md whose structure mirrors this template.
const char * slm_chat_template(const struct slm_ctx * ctx);

// ---------------------------------------------------------------------------
// Chat-template formatting (hand-translated Qwen3.5 Jinja, see
// llm/jinja-template.c). These let Swift / Obj-C / Python callers
// produce the model-facing prompt bytes without re-implementing the
// state machine on their side. Returned strings are heap-allocated;
// callers free with `free()` from <stdlib.h>.

// Roles for slm_chat_message.role. Numbers match jinja-template.c
// internals but are part of the public API contract.
enum slm_chat_role {
    SLM_CHAT_ROLE_SYSTEM    = 0,
    SLM_CHAT_ROLE_USER      = 1,
    SLM_CHAT_ROLE_ASSISTANT = 2,
    SLM_CHAT_ROLE_TOOL      = 3,
};

struct slm_chat_message {
    int          role;     // one of slm_chat_role
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
char * slm_chat_format(const struct slm_chat_message * messages,
                       int n_messages,
                       int add_generation_prompt,
                       int enable_thinking);

// Render ONE new user turn for persistent-KV chat. Produces the
// bytes you tokenize and feed into the next slm_generate() call,
// given an already-warm context. `system_prefix` is rendered INLINE
// with the user message (no separate <|im_start|>system block) and
// should be passed only on the FIRST turn of a conversation, NULL
// otherwise. `enable_tools` (1/0) controls whether the FIRST-turn
// frame advertises the websearch / fetch / distill tools to the
// model (the Jinja `# Tools` system block). With 0, no tools are
// listed and the model produces plain content; pairs with
// `slm_sampler.tools = false` so the runtime's embedded agent loop
// is also dormant. Subsequent turns (system_prefix NULL) carry no
// tool advertisement either way — KV holds whatever was advertised
// on turn 1. Returns NULL if `user_message` is NULL.
char * slm_chat_format_delta(const char * user_message,
                             const char * system_prefix,
                             int enable_thinking, // bool think
                             int enable_tools); // bool tools

#ifdef __cplusplus
}
#endif

#endif // SLM_H
