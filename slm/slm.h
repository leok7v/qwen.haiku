// SPDX-License-Identifier: Apache-2.0
//
// slm.h - public C API for the Qwen3.5-0.8B Q4_K_M GGUF runner.
//
// The boundary between C (slm.c + tensor.c + simd/, ~4 KLoC) and any
// caller — Swift via app/bridge.h, ObjC, Python ctypes, another C
// program. Everything in slm.c that isn't declared here is internal.
//
// Two-level lifecycle:
//
//   1. slm_model_load(path)
//        mmap the GGUF, parse weights + tokenizer. Check
//        slm_model_loaded(). One model can be shared by many ctxs.
//   2. slm_ctx_create(model)
//        per-conversation state (KV cache + SSM ring + tokens
//        history + messages). Returns NULL on failure.
//   3. slm_ctx_ctrl(ctx)->tools = ... ; slm_ctx_ctrl(ctx)->think = ...
//        Tune the conversation BEFORE the first system_prompt /
//        generate call. Once the system block has been prefilled,
//        tools/think are frozen for the rest of the ctx.
//   4. slm_ctx_system_prompt(ctx, "...", cb)
//        Sync prefill of the system framing — must be called once
//        (even with "") before the first slm_generate. Cancellable
//        via cb (return non-zero from cb->callback).
//   5. slm_generate(ctx, "user prompt", cb)
//        One chat turn. The runtime frames the delta in canonical
//        Qwen3 form, tokenizes, prefills the new tokens, decodes
//        until eos/eot or cb says stop. Streams via cb. The user
//        prompt + the model's reasoning + the model's response are
//        all appended to ctx->messages and ctx->ids in order.
//   6. slm_ctx_destroy(ctx) / slm_model_unload(model)
//
// Thread-safety: a single slm_ctx is NOT thread-safe (it holds a
// streaming KV cache and per-layer SSM state mutated on every
// forward pass). One ctx per concurrent inference. The model is
// read-only after load and may be shared by many ctxs.

#ifndef SLM_H
#define SLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles. Treat as `void *`; never dereference.
struct slm_model;
struct slm_ctx;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

// Open a GGUF, mmap weights, parse the tokenizer. Returns non-NULL
// even on failure; check slm_model_loaded() / slm_model_error().
struct slm_model * slm_model_load(const char * path);
void               slm_model_unload(struct slm_model * model);
bool               slm_model_loaded(const struct slm_model * model);
const char *       slm_model_error(const struct slm_model * model);

// ---------------------------------------------------------------------------
// Per-conversation context
// ---------------------------------------------------------------------------

// Allocate a fresh ctx. Returns NULL if the model isn't loaded.
struct slm_ctx * slm_ctx_create(struct slm_model * model);
void             slm_ctx_destroy(struct slm_ctx * ctx);

// ---------------------------------------------------------------------------
// Snapshot / restore — capture & roll back per-conversation mutable state.
//
// Used by the atomic-tool flow: take a snapshot before the model starts
// emitting a turn, dispatch the tool internally, restore the ctx so the
// model never sees the round-trip, then inject the curated tool result
// as preamble before decoding the final answer.
//
// Cheap because the Mamba-2/DeltaNet half is fixed-size memcpy-able
// state; the KV half copies only [0..pos) rows (not the full
// max_position-sized arena).
// ---------------------------------------------------------------------------

struct slm_snapshot;  // opaque

// Capture the ctx's mutable state at the current moment:
//   - KV cache rows [0..pos)
//   - SSM conv_state ring + ssm_state + conv_head (when hybrid)
//   - pos, rng state, ids count
//
// Caller owns the result; free with slm_snapshot_free. Returns NULL
// when ctx is NULL.
//
// NOT captured: ctx->ids contents past the saved count, ctx->messages
// (caller-managed history); on restore, c->ids.count is truncated back
// but the underlying token buffer is not reallocated.
struct slm_snapshot * slm_ctx_snapshot(const struct slm_ctx * ctx);

// Restore the ctx from `snap`. Caller must guarantee the snapshot
// came from the same model (same n_layers / n_kv_heads / head_dim /
// linear_* layout) — otherwise behavior is undefined. The KV rows
// beyond [0..snap->pos) are left as-is and will be overwritten by
// the next forward step.
void slm_ctx_restore(struct slm_ctx *            ctx,
                     const struct slm_snapshot * snap);

// Release a snapshot's heap allocations.
void slm_snapshot_free(struct slm_snapshot * snap);

// ---------------------------------------------------------------------------
// Sampler + conversation control
// ---------------------------------------------------------------------------

// Sampler parameters. Pure how-to-pick-the-next-token; conversation-
// level knobs live in slm_ctrl. Zero-initialized = greedy (argmax).
struct slm_sampler {
    float    temperature;        // 0 = greedy; >0 = softmax temperature
    int      top_k;              // 0 or 1 = greedy; >1 = top-k
    float    top_p;              // 0 = off; (0,1) = nucleus
    float    min_p;              // 0 = off; (0,1) = min-prob filter
    float    repetition_penalty; // 1.0 = off; 1.05..1.3 typical
    int      repetition_window;  // 0 = all of generation so far; >0 = last N
};

// Chat-tuned default sampler: T=0.7, top_k=40, top_p=0.9, min_p=0.05,
// rep=1.25, win=64. Best fit for Qwen3.5-0.8B in chat + tools mode.
struct slm_sampler slm_sampler_defaults(void);

// Per-conversation behavior knobs. Mutable until the system block
// has been committed to KV (via slm_ctx_system_prompt). After that,
// changes to tools/think are ignored by the runtime — the
// in-context framing is already locked in. debug stays live; the
// diagnostic knobs (trace_tokens, dump_layer) are mutable on the fly.
struct slm_ctrl {
    bool         tools;          // see footnote (1)
    bool         think;          // see footnote (2)
    const char * effort;         // see footnote (3)
    int32_t      debug;          // see footnote (4)
    bool         trace_tokens;   // see footnote (5)
    int32_t      dump_layer;     // see footnote (6)
};

// slm_ctrl footnotes:
//
// (1) tools: advertise tools in the system framing, recognise
//     `<tool_call>` blocks in the stream, run the embedded agent
//     dispatch loop. With false, the model never sees a tools
//     advertisement and the `<tool_call>` filter is disabled.
//     Locked in by slm_ctx_system_prompt.
//
// (2) think: ask for reasoning. With true, the assistant turn opens
//     `<think>\n` and the model is free to fill in reasoning before
//     `</think>`. With false, the runtime pre-fills the empty
//     `<think>\n\n</think>\n\n` block so the model jumps straight
//     to content. Locked in by slm_ctx_system_prompt.
//
// (3) effort: prepended as a hint to the system block. Recognised:
//     "low"    -> "(brief: 1-2 sentences)\n\n"
//     "medium" or NULL -> no prefix
//     "high"   -> "(think carefully step-by-step ...)\n\n"
//     Locked in by slm_ctx_system_prompt.
//
// (4) debug: verbosity level (0 = quiet, 9 = chatty). Free to flip
//     on the fly between generate calls. Gates whether tool-call /
//     tool-response visibility chunks fire in the stream callback,
//     and how chatty the internal trace() ring is during a turn.
//     Graduated thresholds:
//       0  silent except errors
//       1+ turn-start, snapshot/restore, tool dispatch ok/err
//       3+ tool args, hit count, chosen URL
//       5+ top-3 hit titles
//       7+ user prompt prefix, preamble prefix, DDG body size,
//          fetch HTTP status
//       9+ raw DDG body excerpt, distill output excerpt
//
// (5) trace_tokens: when set, slm_generate prints each sampled token
//     id to stderr as "[tok] %d\n". Used by tools/bench.sh's token-
//     level parity mode, which survives ULP drift across perf
//     rewrites that change the underlying logits but keep the
//     argmax stable. Default OFF so non-CLI consumers (test-nihs
//     etc.) get clean output; slm.c's main() sets it from the
//     LLM_TRACE_TOKENS env var when LLM_CLI is defined. Mutable on
//     the fly.
//
// (6) dump_layer: -1 = off; >= 0 = dump per-op intermediate tensors
//     for that layer to stderr (one line per op, head + tail + sum
//     mirroring llama-eval-callback's format). Parity-diff knob —
//     drive two implementations with the same dump_layer and the
//     first divergence is layer-local. Mutable on the fly.

// Defaults: tools=true, think=false, effort=NULL, debug=1,
// trace_tokens=false, dump_layer=-1.
struct slm_ctrl slm_ctrl_defaults(void);

// Read-write access to the ctx's ctrl. Mutate fields BEFORE the
// first slm_ctx_system_prompt call to lock them into the framing.
// Aborts via NULL deref on a NULL ctx — caller error.
struct slm_ctrl * slm_ctx_ctrl(struct slm_ctx * ctx);

// ---------------------------------------------------------------------------
// Streaming callback (functor)
// ---------------------------------------------------------------------------

// Callback for slm_ctx_system_prompt + slm_generate streaming. The
// runtime writes the chunk fields below, then invokes `callback`.
// Embed in your own struct for extra state (first-field cast):
//
//   struct my_cb {
//       struct slm_stream_callback base;
//       int     my_tokens_seen;
//       FILE *  my_log;
//   };
//   static int my_handler(const struct slm_stream_callback * cb) {
//       struct my_cb * me = (struct my_cb *)cb;
//       if (cb->content)   { ... write to me->my_log ... }
//       if (cb->prefilled) { ... }
//       return 0;  // non-zero aborts
//   }
//
// Field semantics (at most one of content/reasoning/call/response
// is non-NULL per invocation; pp_pos < pp_total during prefill;
// prefilled=true on the single chunk fired when prefill finishes):
//
//   content    visible reply text. UTF-8 piece; concatenating
//              successive pieces yields valid UTF-8.
//   reasoning  text inside a <think>…</think> block. Hide or render
//              muted; do not feed back into history.
//   call       text inside a <tool_call>…</tool_call> block the
//              model emitted. With ctrl.tools, the runtime
//              dispatches; the result arrives as a `response`
//              chunk and generation continues.
//   response   text the dispatched tool returned for that call.
//   prefilled  signal-only. Fires once between prefill and decode.
//   pp_pos     during prefill: number of prompt tokens processed.
//   pp_total   during prefill: total prompt tokens to process.
//              Both 0 outside prefill.
//
// String pointers remain valid for the lifetime of the ctx (they
// point into ctx->messages), but treat them as borrowed for this
// callback only — the next emission may add new strings; the
// pointers you got stay valid but you'd want to re-read for any
// newly-appended content. Do not free.
struct slm_stream_callback {
    void *       that;       // opaque user pointer (or embed-cast)
    // chunk view — runtime writes before each invoke:
    const char * content;
    const char * reasoning;
    const char * call;
    const char * response;
    bool         prefilled;
    int32_t      pp_pos;
    int32_t      pp_total;
    // function pointer (return non-zero to abort):
    int        (*callback)(const struct slm_stream_callback *);
};

// ---------------------------------------------------------------------------
// Tokenizer (exposed for clients that want to inspect / count)
// ---------------------------------------------------------------------------

// Encode `text` into token IDs. Allocates an int32_t array (worst
// case = strlen(text)); caller frees with `free()`. Returns the
// number of tokens.
int slm_tokenize(struct slm_ctx * ctx, const char * text,
                 int32_t ** out_ids);

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// Prefill the system framing into the model. Called once per ctx
// before any slm_generate(). Pass "" or NULL for a system-prompt-
// less conversation (the canonical empty system block is still
// emitted so the framing is well-formed). Returns true on success,
// false if the callback aborted (cb->callback returned non-zero)
// or the ctx is unusable. After a successful return, the ctx is
// committed and ctrl.tools/think are frozen.
//
// `cb` may be NULL when no progress / cancellation is needed. When
// non-NULL, the runtime fires pp_pos/pp_total chunks during prefill
// and a single `prefilled = true` chunk when done.
bool slm_ctx_system_prompt(struct slm_ctx * ctx,
                           const char * text,
                           struct slm_stream_callback * cb);

// Run one user-turn chat round. Frames the delta as canonical
// Qwen3 (<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…),
// tokenizes, prefills the new tokens, decodes up to `max_new`
// tokens calling `cb` per emitted UTF-8 piece. The user prompt
// and the model's reply (reasoning + content + tool dialogue) all
// land in ctx->messages.
//
// Returns the number of decode tokens emitted. Aborts cleanly when
// cb returns non-zero or eos/eot is sampled. With `sampler` NULL,
// uses slm_sampler_defaults(). `seed = 0` derives from wall-clock;
// greedy (temperature == 0) ignores seed. `min_new` suppresses
// eos/eot until that many decode tokens have been emitted (helps
// avoid empty bubbles from high-temperature sampling).
int slm_generate(struct slm_ctx *                   ctx,
                 const char *                       prompt,
                 int                                max_new,
                 int                                min_new,
                 const struct slm_sampler *         sampler,
                 uint64_t                           seed,
                 struct slm_stream_callback *       cb);

// ---------------------------------------------------------------------------
// History accessors (append-only state on the ctx)
// ---------------------------------------------------------------------------

// All token ids prefilled or generated since ctx_create. Pointer
// stable until the next call into the ctx; *out_n receives the
// count. Returns NULL when ctx is NULL or empty.
const int32_t * slm_ctx_tokens(const struct slm_ctx * ctx, int32_t * out_n);

// Number of UTF-8 message strings on the ctx (each emitted in a
// callback chunk is appended).
int32_t      slm_ctx_messages_count(const struct slm_ctx * ctx);

// i-th message string. NUL-terminated, owned by ctx. NULL if i is
// out of range. The role is implied by the surrounding emission
// pattern: user prompts arrive as one string each; assistant
// reasoning / content / tool fragments arrive in order.
const char * slm_ctx_message(const struct slm_ctx * ctx, int32_t i);

// ---------------------------------------------------------------------------
// Metadata + stats
// ---------------------------------------------------------------------------

int  slm_vocab_size(const struct slm_ctx * ctx);
int  slm_eos_id    (const struct slm_ctx * ctx);
int  slm_bos_id    (const struct slm_ctx * ctx);

// Most-recent timing stats. 0 before any slm_generate call.
double  slm_pp_per_sec (const struct slm_ctx * ctx);
double  slm_tg_per_sec (const struct slm_ctx * ctx);
int32_t slm_n_prefill  (const struct slm_ctx * ctx);
int32_t slm_n_generated(const struct slm_ctx * ctx);

// Raw Jinja chat template from the GGUF (`tokenizer.chat_template`).
// NULL if absent. Lifetime tied to ctx; do not free.
const char * slm_chat_template(const struct slm_ctx * ctx);

// ---------------------------------------------------------------------------
// Chat-template formatting (exposed for clients that want the
// pre-tokenizer string view — slm_generate already runs this
// internally on every call).
// ---------------------------------------------------------------------------

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

// Render a full conversation. Caller frees the result with free().
char * slm_chat_format(const struct slm_chat_message * messages,
                       int                              n_messages,
                       int                              add_generation_prompt,
                       int                              enable_thinking);

// ---------------------------------------------------------------------------
// Self-test battery
// ---------------------------------------------------------------------------

// Run the C self-tests in sequence:
//   qwen_self_test       — synthetic forward pass (data-flow sanity)
//   chunked_self_test    — chunked vs autoregressive SSM parity
//   jinja_self_test      — jinja template fixtures
//   agent_parser_test    — <tool_call> parser fixtures
//   tools_self_test      — websearch / fetch / distill (LIVE network)
//   slm_think_test       — <think> stream-filter fixtures
//
// Progress and pass/fail lines are emitted via the trace ring buffer
// (utils/trace.h) and to stderr. Returns the total count of failing
// suites (0 = all PASS).
//
// Note: tools_self_test makes real DuckDuckGo HTTP requests; offline
// or rate-limited runs will FAIL that suite. The macOS app's Debug
// tab calls this and surfaces the failure count.
int slm_run_all_tests(void);

#ifdef __cplusplus
}
#endif

#endif // SLM_H
