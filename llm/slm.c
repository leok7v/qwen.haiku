// slm.c -- single-file CPU runner for Qwen3.5-0.8B-Q4_K_M.gguf.
//
// Reads the upstream Q4_K_M GGUF directly (no offline converter
// step) and runs forward passes against tensor.c kernels.
// See PLAN.md for the design rationale and what's deferred.
//
// Style: single-file C17, single-entry/single-exit, no goto, no
// mid-loop break/continue except inside switch. Math lives in
// tensor.c (included whole). Auxiliary primitives - chars/arr/map
// - are inlined here so the file stays standalone.
//
// Build: `make` in this directory produces ./llm. The binary
// supports three entry points (see main() at the bottom):
//
//   ./llm --self-test
//        Allocate synthetic weights, run one forward pass on a
//        dummy token sequence, assert shapes propagate and no
//        NaN/Inf escapes. No model file required - sanity check
//        that the math kernels and data flow are wired correctly.
//
//   ./llm --single "your prompt"
//        Load the hard-coded Q4_K_M GGUF, run greedy decode for
//        --max-new tokens (default 64), print the result.
//
//   ./llm --repl
//        Interactive chat loop using the Qwen3 chat template,
//        streamed token-by-token.
//
// File layout: this file holds the model-agnostic surface — the
// public API (slm_create / slm_generate / ...), the sampler
// chain, the <think>/<tool_call> state-machine filter that splits
// the raw token stream into content/reasoning/tool_call/
// tool_response chunks, the embedded agent loop, and the CLI
// driver. Model-specific code (GGUF parser, byte-level BPE
// tokenizer, Q4_K_M weight loader, forward pass, RNG primitives)
// lives in qwen.c, `#include`d below.

// Feature-test macros must be defined BEFORE any system header is
// processed (transitively via utils/maps.c → arrays.c → stdlib.h).
// Linux glibc gates clock_gettime/CLOCK_MONOTONIC behind POSIX
// 2001+ and memmem() behind _GNU_SOURCE. macOS gates BSD types
// (u_char/u_short used by sys/proc.h) behind _DARWIN_C_SOURCE.
#if defined(__APPLE__)
  #ifndef _DARWIN_C_SOURCE
  #define _DARWIN_C_SOURCE
  #endif
#else
  #ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 200809L
  #endif
  #ifndef _GNU_SOURCE
  #define _GNU_SOURCE
  #endif
#endif

// Include utils/maps.c FIRST so the `oom()` allocator wrapper and
// struct chars / struct arr layouts are in scope before tensor.c
// (chunked.c's self-test, included via tensor.c, calls `oom()` on
// its CHUNK_SIZE × CHUNK_SIZE scratch allocations).
#include "utils/maps.c"

#include "tensor.c"
#include "slm.h"

#ifdef LLM_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// utils/maps.c is already brought in above (before tensor.c) so
// chunked.c's self-test can reach `oom()`.

#include "gguf.c"             // GGUF v3 reader; used by qwen.c below
#include "qwen.c"
#include "jinja.c"           // renamed from llm/jinja-template.c
#include "tools.c"

// Sampler chain (slm_sampler_defaults + sample_with + helpers).
// Pulled in AFTER qwen.c so it can use struct rng / rng_uniform /
// sample_argmax defined there.
#include "sampler.c"

// ---------------------------------------------------------------------------
// Public-ish API used by Swift bridge AND by main()
//
// slm_model_load / slm_model_unload / slm_ctx_create / slm_ctx_destroy
// / slm_set_ctrl / slm_get_ctrl + the file-internal slm_reset() moved
// to model.c (with the structs they own). The chat-template accessor
// + chat formatting + the agent loop + the stream filter live below.
// ---------------------------------------------------------------------------

const char * slm_chat_template(const struct slm_ctx * c) {
    return (c == NULL) ? NULL : c->model->chat_template;
}

// Thin wrappers exposing jinja-template.c's chat-formatting through
// the public slm_chat_format / slm_chat_format_delta API. We copy
// the public slm_chat_message[] into the internal jinja_message
// layout so the public ABI stays minimal (no tool_calls /
// reasoning_content surface — text-only chat covers the iOS/macOS
// app's needs). Tool support, when wired up later, can either grow
// the public struct or take a richer "advanced" entry point.
char * slm_chat_format(const struct slm_chat_message * messages,
                       int n_messages,
                       int add_generation_prompt,
                       int enable_thinking) {
    char * result = NULL;
    if (messages != NULL && n_messages > 0) {
        struct jinja_message * tmp =
            (struct jinja_message *)calloc((size_t)n_messages,
                                           sizeof(struct jinja_message));
        if (tmp != NULL) {
            for (int i = 0; i < n_messages; i++) {
                tmp[i].role    = messages[i].role;
                tmp[i].content = messages[i].content;
                // reasoning_content / tool_calls left NULL — the
                // assistant branch will derive reasoning from a
                // <think>...</think> block in content if present.
            }
            result = jinja_apply(tmp, n_messages, NULL, 0,
                                 add_generation_prompt, enable_thinking);
            free(tmp);
        }
    }
    return result;
}

// slm_chat_format_delta is defined further down — its full body
// needs the AGENT_TOOL_* tool-spec JSON strings (declared in
// agent.c), so the definition lives below the `#include "agent.c"`
// site. See the matching block right after the agent.c include.

// File-internal token-stream callback. slm_generate_raw emits one
// utf8 piece per generated token to this; the public slm_generate
// trampolines into the <think>-filter on top of this (see further
// down in this file).
// File-internal token-stream callback. `utf8` is a NUL-terminated
// UTF-8 piece — possibly a partial Unicode codepoint for byte-level
// BPE; concatenating successive pieces always yields valid UTF-8.
// Return non-zero to abort generation. Used by slm_generate_raw
// internally; public callers go through `slm_stream_cb` in slm.h.
typedef int (*slm_token_cb)(const char * utf8, void * user);

static int slm_generate_raw(struct slm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct slm_sampler * sampler_in,
                 uint64_t seed,
                 slm_token_cb cb, void * user,
                 slm_stream_cb progress_cb, void * progress_user) {
    struct slm_sampler sp = {0};
    if (sampler_in != NULL) { sp = *sampler_in; }
    struct rng rng;
    rng_seed(&rng, seed != 0 ? seed : (uint64_t)time(NULL));
    slm_trace_open();
    int32_t   generated = 0;
    int32_t   pos       = c->pos;        // resume after previous call
    bool      stop      = false;
    double    t0        = slm_monotonic_seconds();
    // Repetition-penalty history: include the full prompt so the
    // model is discouraged from immediately echoing the input back,
    // plus everything it generates in this call. Capacity grows on
    // demand (single realloc up front sized for the worst case).
    int32_t * history = (int32_t *)oom(
        calloc((size_t)(prompt_n + max_new + 1), sizeof(int32_t)));
    int32_t   hist_n  = 0;
    for (int32_t i = 0; i < prompt_n; i++) { history[hist_n++] = prompt_ids[i]; }
    // Pre-fill prompt. If LLM_USE_FORWARD_BATCH is set AND the prompt
    // is multi-token AND fits in one chunk, run the chunked-SSM
    // batched forward in a single call. Otherwise loop the
    // autoregressive single-token forward as before. Decode (after
    // prefill) always uses single-token forward.
    static int8_t use_forward_batch = -1;
    if (use_forward_batch < 0) {
        use_forward_batch = (getenv("LLM_USE_FORWARD_BATCH") != NULL) ? 1 : 0;
    }
    if (use_forward_batch && prompt_n > 1) {
        // slm_forward_ssm_batch has a multi-chunk loop, so any
        // prompt_n is allowed (CHUNK_SIZE no longer caps the call).
        struct tensor * logits = slm_forward_batch(c, prompt_ids,
                                                   prompt_n, pos);
        (void)logits;
        pos += prompt_n;
        slm_trace_close();
    } else {
        for (int32_t i = 0; i < prompt_n && !stop; i++) {
            struct tensor * logits = slm_forward_step(c, prompt_ids[i], pos);
            // Trace facility is for parity diagnostics on the FIRST
            // forward pass only - one prompt token -> ~1850 JSONL
            // lines matches the llama-qwen-haiku reference dump
            // shape. Close after step 0 to keep file size bounded.
            if (i == 0) { slm_trace_close(); }
            (void)logits;
            pos++;
        }
    }
    if (progress_cb != NULL && prompt_n > 0) {
        // One terminal "prefill done" chunk after the prefill loop
        // (either path — autoregressive or batched). Useful for
        // hiding a spinner / progress indicator once decode starts.
        // No granular per-token progress: at this model size prefill
        // is ~1s and a per-token bar is more noise than value.
        struct slm_stream_chunk ch = {0};
        ch.prefilled = true;
        (void)progress_cb(&ch, progress_user);
    }
    double t1 = slm_monotonic_seconds();
    c->t_prefill_s = t1 - t0;
    c->n_prefill   = prompt_n;
    int32_t last = prompt_n > 0 ? prompt_ids[prompt_n - 1] : c->model->cfg.bos_id;
    // Decode loop.
    while (!stop && generated < max_new && pos < c->model->cfg.max_position) {
        struct tensor * logits = slm_forward_step(c, last, pos);
        // While below min_new, force the model to keep producing
        // content by zeroing out the eos / eot logits (effectively
        // -infinity once normalized). Restored automatically once
        // generated >= min_new.
        float saved_eos = 0.0f;
        float saved_eot = 0.0f;
        bool  mask      = (generated < min_new);
        if (mask) {
            if (c->model->cfg.eos_id >= 0 && c->model->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                saved_eos = logits->data[c->model->cfg.eos_id];
                logits->data[c->model->cfg.eos_id] = -INFINITY;
            }
            if (c->model->cfg.eot_id >= 0 && c->model->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                saved_eot = logits->data[c->model->cfg.eot_id];
                logits->data[c->model->cfg.eot_id] = -INFINITY;
            }
        }
        int32_t next = sample_with(logits, &sp, &rng, history, hist_n);
        history[hist_n++] = next;
        if (mask) {
            if (c->model->cfg.eos_id >= 0 && c->model->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->model->cfg.eos_id] = saved_eos;
            }
            if (c->model->cfg.eot_id >= 0 && c->model->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->model->cfg.eot_id] = saved_eot;
            }
        }
        pos++;
        generated++;
        if (g_trace_tokens) { fprintf(stderr, "[tok] %d\n", (int)next); }
        bool is_stop = (next == c->model->cfg.eos_id || next == c->model->cfg.eot_id);
        for (int32_t si = 0; !is_stop && si < c->model->cfg.n_stop_ids; si++) {
            if (next == c->model->cfg.stop_ids[si]) { is_stop = true; }
        }
        if (is_stop) {
            stop = true;
        } else {
            struct chars piece = {0};
            tokenizer_decode_one(&c->model->tok, next, &piece);
            chars_put(&piece, "", 0);  // ensure null-term
            if (cb != NULL && piece.data != NULL) {
                if (cb(piece.data, user) != 0) { stop = true; }
            }
            chars_free(&piece);
            last = next;
        }
    }
    double t2 = slm_monotonic_seconds();
    c->t_gen_s     = t2 - t1;
    c->n_generated = generated;
    c->pos         = pos;                // persist across calls
    free(history);
    slm_trace_close();
    return generated;
}

// ---------------------------------------------------------------------------
// <think> stream filter. State machine that splits slm_generate's
// single output stream into two: content (outside `<think>`...
// `</think>` blocks) and reasoning (inside). Marker bytes are NOT
// emitted; both streams are routed through ONE slm_stream_cb
// callback with a struct slm_stream_chunk that carries exactly one
// non-NULL pointer per call.
//
// The model emits both streams interleaved; we hold back the last
// few bytes (up to a marker's length) so a marker split across two
// callback invocations is still detected before any of its bytes
// leak as visible output.
//
// Markers we currently recognise (Qwen3 family):
//   - "<think>"   (7 bytes) -> enter reasoning
//   - "</think>"  (8 bytes) -> leave reasoning back to content
//
// Adding a new model is a marker-table edit (see THINK_MARKERS
// below). The state machine is otherwise marker-agnostic.
//
// Public surface (slm.h) is just `slm_generate_split`. The filter
// struct + push/finish helpers are file-static here: clients should
// not need the streaming pieces directly because slm_generate_split
// already drives one filter end-to-end per generate call.
// ---------------------------------------------------------------------------

// Filter state-machine modes. The filter walks through the model's
// raw byte stream and assigns each byte to one of these modes based
// on the marker tags it has seen. THINK_MODE_NONE is a sentinel
// used in the marker table for "this marker triggers an action but
// doesn't change mode" (no live state ever holds it).
enum slm_think_mode {
    THINK_MODE_CONTENT   = 0,  // visible reply text
    THINK_MODE_REASONING = 1,  // inside <think>...</think>
    THINK_MODE_TOOL_CALL = 2,  // inside <tool_call>...</tool_call>
    THINK_MODE_NONE      = -1, // sentinel for marker.mode (no-op)
};

// Marker-table row indices. Used to identify a specific marker
// from its position in THINK_MARKERS without comparing tag strings.
enum slm_think_marker_row {
    SLM_MARKER_THINK_OPEN  = 0,
    SLM_MARKER_THINK_CLOSE = 1,
    SLM_MARKER_TOOL_OPEN   = 2,
    SLM_MARKER_TOOL_CLOSE  = 3,
};

struct slm_think_filter {
    enum slm_think_mode mode;
    // Partial-marker hold-back: a fixed 64-byte sliding window. The
    // size is NOT incidental — it's the algorithmic flush trigger in
    // slm_think_filter_push: once hold_n == sizeof(hold_data), the
    // loop emits the front and keeps the trailing
    // SLM_THINK_MAX_MARKER-1 (=11) bytes as a potential prefix.
    // Backing this with `struct chars` would let it grow unbounded
    // and the flush would never happen, so we keep the fixed array.
    char                hold_data[64];
    int                 hold_n;
    // <tool_call>...</tool_call> body accumulator. Grows as needed
    // via chars_put. Caller (slm_generate) calls `chars_free` after
    // reading tool_call.data; the filter itself does no allocation
    // bookkeeping outside chars_put.
    struct chars        tool_call;
    // Flipped true by the filter the moment a </tool_call> marker
    // is consumed. The public slm_generate loop reads this after
    // the generate_raw call returns to dispatch the tool + re-feed
    // the model. Reset by init().
    bool                tool_call_ready;
    // Per-call configuration (set by slm_generate from sampler
    // flags before kicking the trampoline). recognize_tool_calls
    // = false makes find_marker ignore <tool_call> / </tool_call>
    // entries so the body streams as plain content. emit_visibility
    // = false suppresses chunk->tool_call emission (tool dispatch
    // still happens; the UI just doesn't see the trace).
    bool                recognize_tool_calls;
    bool                emit_visibility;
};

struct slm_think_marker {
    const char *        tag;   // null-terminated literal
    int                 len;   // strlen(tag), cached
    enum slm_think_mode mode;  // mode to switch INTO on match
                               // (THINK_MODE_NONE = no-op)
};

static const struct slm_think_marker THINK_MARKERS[] = {
    [SLM_MARKER_THINK_OPEN]  = { "<think>",      7, THINK_MODE_REASONING },
    [SLM_MARKER_THINK_CLOSE] = { "</think>",     8, THINK_MODE_CONTENT   },
    [SLM_MARKER_TOOL_OPEN]   = { "<tool_call>", 11, THINK_MODE_TOOL_CALL },
    [SLM_MARKER_TOOL_CLOSE]  = { "</tool_call>",12, THINK_MODE_CONTENT   },
};
#define SLM_THINK_N_MARKERS \
    ((int)(sizeof(THINK_MARKERS) / sizeof(THINK_MARKERS[0])))
#define SLM_THINK_MAX_MARKER 12  // strlen("</tool_call>")

// Caller is expected to zero-init the filter (or its containing
// struct, e.g. `struct slm_split_box box = {0};`) BEFORE calling
// init the first time — init does not chars_free the tool_call
// buffer, so it would leak any previous heap allocation.
static void slm_think_filter_init(struct slm_think_filter * f) {
    f->mode = THINK_MODE_CONTENT;
    f->hold_n = 0;
    f->hold_data[0] = '\0';
    f->tool_call = (struct chars){0};
    f->tool_call_ready = false;
    f->recognize_tool_calls = true;  // default on; slm_generate overrides
    f->emit_visibility      = true;
}

// Emit `[start, end)` bytes through `cb`, tagging the chunk's text
// per the filter's current mode. Mode 2 (tool_call) routes bytes
// into the filter's internal tool_call_data buffer instead of
// firing `cb` — the buffered call is dispatched as one atomic
// chunk when </tool_call> arrives. Returns the callback's return
// value, or 0 when there's no work / no callback / mode is 2.
static int slm_think_emit(struct slm_think_filter * f,
                          const char * start, int n,
                          slm_stream_cb cb, void * user) {
    int rc = 0;
    if (n > 0) {
        if (f->mode == THINK_MODE_TOOL_CALL) {
            // Buffering inside a tool_call block — don't emit yet;
            // grow as needed. No upper bound: the runtime tool-call
            // bodies are well under 1 KB in practice, but if a model
            // ever emits a giant one, we won't truncate it now.
            chars_put(&f->tool_call, start, (size_t)n);
        } else if (cb != NULL) {
            char tmp[80];
            int  copy = n < (int)sizeof(tmp) - 1
                      ? n : (int)sizeof(tmp) - 1;
            memcpy(tmp, start, (size_t)copy);
            tmp[copy] = '\0';
            struct slm_stream_chunk chunk = {0};
            if (f->mode == THINK_MODE_REASONING) { chunk.reasoning = tmp; }
            else                                 { chunk.content   = tmp; }
            rc = cb(&chunk, user);
            if (n > copy && rc == 0) {
                int rc2 = slm_think_emit(f, start + copy, n - copy,
                                         cb, user);
                if (rc2 != 0) { rc = rc2; }
            }
        }
    }
    return rc;
}

// Try to find the earliest complete marker fully present in hold[].
// Returns marker index (0..N-1) and writes the start offset to *pos,
// or -1 if no complete marker is in hold[].
static int slm_think_find_marker(const struct slm_think_filter * f,
                                 int * pos) {
    int best  = -1;
    int best_pos = f->hold_n;
    for (int m = 0; m < SLM_THINK_N_MARKERS; m++) {
        // Skip the tool_call markers when the sampler turned tools
        // off — their bytes will stream as plain content instead.
        bool is_tool_call = (m == SLM_MARKER_TOOL_OPEN) ||
                            (m == SLM_MARKER_TOOL_CLOSE);
        bool skip = (!f->recognize_tool_calls && is_tool_call);
        if (!skip) {
            const char * tag = THINK_MARKERS[m].tag;
            int tlen = THINK_MARKERS[m].len;
            if (tlen <= f->hold_n) {
                int last_start = f->hold_n - tlen;
                for (int i = 0; i <= last_start; i++) {
                    bool match = true;
                    for (int k = 0; k < tlen && match; k++) {
                        if (f->hold_data[i + k] != tag[k]) {
                            match = false;
                        }
                    }
                    if (match && i < best_pos) {
                        best = m;
                        best_pos = i;
                    }
                }
            }
        }
    }
    if (best >= 0) { *pos = best_pos; }
    return best;
}

// Walk `safe` back to the nearest UTF-8 codepoint boundary. A
// multibyte codepoint starts at a byte with high bits != 10xxxxxx
// (i.e. 0xxxxxxx or 11xxxxxx) and continues with 10xxxxxx
// continuation bytes; we must not split that sequence across a
// callback, or the receiver sees a half-decoded grapheme and the UI
// renders garbage / a replacement glyph.
//
// We DO emit at codepoint boundaries but do NOT delay across
// zero-width joiners (U+200D, three bytes 0xE2 0x80 0x8D). A ZWJ
// sequence (e.g. man + ZWJ + woman + ZWJ + girl as a family glyph)
// can be split across callback emissions; the renderer sees the
// component glyphs briefly until the next chunk lands, then they
// regroup. Full grapheme-cluster awareness would need ICU and is
// scope creep for a small chat surface.
static int slm_think_safe_utf8(const char * buf, int hold_n, int safe) {
    while (safe > 0 && safe < hold_n &&
           ((unsigned char)buf[safe] & 0xC0) == 0x80) {
        safe--;
    }
    return safe;
}

// Returns true if hold[]'s last `tail_n` bytes could be a strict
// prefix of any marker (i.e. we should hold them back rather than emit).
static bool slm_think_could_be_prefix(const struct slm_think_filter * f,
                                      int tail_start) {
    bool possible = false;
    int tail_n = f->hold_n - tail_start;
    if (tail_n > 0) {
        for (int m = 0; m < SLM_THINK_N_MARKERS && !possible; m++) {
            const char * tag = THINK_MARKERS[m].tag;
            int tlen = THINK_MARKERS[m].len;
            if (tail_n < tlen) {
                bool match = true;
                for (int k = 0; k < tail_n && match; k++) {
                    if (f->hold_data[tail_start + k] != tag[k]) {
                        match = false;
                    }
                }
                if (match) { possible = true; }
            }
        }
    }
    return possible;
}

int slm_think_filter_push(struct slm_think_filter * f,
                          const char * utf8,
                          slm_stream_cb cb,
                          void * user) {
    int rc = 0;
    if (utf8 != NULL) {
        // Append the new chunk to the holdback. The hold buffer is
        // sized generously (64 bytes) so multi-token streaming
        // never overflows; if a stray giant piece does arrive we
        // flush the surplus first.
        int avail = (int)sizeof(f->hold_data) - 1 - f->hold_n;
        const char * src = utf8;
        while (*src != '\0' && rc == 0) {
            if (avail <= 0) {
                // Hold is full; emit the front (keeping last
                // SLM_THINK_MAX_MARKER-1 bytes as potential prefix).
                // Same UTF-8 safety as the per-iter safe-emit below.
                int keep = SLM_THINK_MAX_MARKER - 1;
                int emit_n = f->hold_n - keep;
                emit_n = slm_think_safe_utf8(f->hold_data, f->hold_n,
                                             emit_n);
                if (emit_n > 0) {
                    int r2 = slm_think_emit(f, f->hold_data,
                                            emit_n, cb, user);
                    if (r2 != 0) { rc = r2; }
                    int remain = f->hold_n - emit_n;
                    memmove(f->hold_data, f->hold_data + emit_n,
                            (size_t)remain);
                    f->hold_n = remain;
                    f->hold_data[f->hold_n] = '\0';
                }
                avail = (int)sizeof(f->hold_data) - 1 - f->hold_n;
            }
            // Copy as many bytes as fit, then look for markers.
            int chunk_len = 0;
            while (src[chunk_len] != '\0' && chunk_len < avail) {
                chunk_len++;
            }
            memcpy(f->hold_data + f->hold_n, src, (size_t)chunk_len);
            f->hold_n += chunk_len;
            f->hold_data[f->hold_n] = '\0';
            src   += chunk_len;
            avail -= chunk_len;
            // Drain as many complete markers as we can.
            bool more = true;
            while (more && rc == 0) {
                int pos = 0;
                int m   = slm_think_find_marker(f, &pos);
                if (m < 0) {
                    more = false;
                } else {
                    // Emit prefix in current mode.
                    int r2 = slm_think_emit(f, f->hold_data, pos,
                                            cb, user);
                    if (r2 != 0) { rc = r2; }
                    enum slm_think_mode prev_mode = f->mode;
                    // Flip mode (or no-op for THINK_MODE_NONE markers).
                    if (THINK_MARKERS[m].mode != THINK_MODE_NONE) {
                        f->mode = THINK_MARKERS[m].mode;
                    }
                    // tool_call -> other: block just finished.
                    // Emit chunk->tool_call with the buffered body
                    // (debug only) and signal the outer loop to
                    // dispatch + inject.
                    if (prev_mode == THINK_MODE_TOOL_CALL &&
                        f->mode  != THINK_MODE_TOOL_CALL) {
                        if (cb != NULL && f->tool_call.count > 0 &&
                            f->emit_visibility) {
                            struct slm_stream_chunk chunk = {0};
                            chunk.call = f->tool_call.data;
                            int r3 = cb(&chunk, user);
                            if (r3 != 0) { rc = r3; }
                        }
                        f->tool_call_ready = true;
                        // Force slm_generate_raw's decode loop to
                        // exit so the public slm_generate can run
                        // the dispatch + inject step before any
                        // further sampling.
                        rc = 1;
                        more = false;
                    }
                    // other -> tool_call: entering a tool_call block;
                    // reset the accumulator (keep heap buffer for
                    // reuse — chars_put will overwrite from offset 0).
                    if (prev_mode != THINK_MODE_TOOL_CALL &&
                        f->mode  == THINK_MODE_TOOL_CALL) {
                        f->tool_call.count = 0;
                        if (f->tool_call.data != NULL) {
                            f->tool_call.data[0] = '\0';
                        }
                    }
                    int after = pos + THINK_MARKERS[m].len;
                    int remain = f->hold_n - after;
                    if (remain > 0) {
                        memmove(f->hold_data, f->hold_data + after,
                                (size_t)remain);
                    }
                    f->hold_n = remain;
                    f->hold_data[f->hold_n] = '\0';
                }
            }
            // Emit the part of hold that's safely past any partial
            // marker. The trailing (SLM_THINK_MAX_MARKER-1) bytes
            // stay as potential prefix.
            int safe = f->hold_n - (SLM_THINK_MAX_MARKER - 1);
            if (safe > 0 && rc == 0) {
                // Trim the safe region further: don't emit bytes
                // that could still extend into a marker tail (i.e.
                // if the LAST byte we'd emit is a '<', keep it).
                while (safe > 0 &&
                       slm_think_could_be_prefix(f, safe)) {
                    safe--;
                }
                // Also don't split a multibyte UTF-8 codepoint —
                // hold continuation bytes back until the codepoint
                // is complete (next push() will bring the tail).
                safe = slm_think_safe_utf8(f->hold_data, f->hold_n,
                                           safe);
                if (safe > 0) {
                    int r2 = slm_think_emit(f, f->hold_data, safe,
                                            cb, user);
                    if (r2 != 0) { rc = r2; }
                    int remain = f->hold_n - safe;
                    if (remain > 0) {
                        memmove(f->hold_data,
                                f->hold_data + safe,
                                (size_t)remain);
                    }
                    f->hold_n = remain;
                    f->hold_data[f->hold_n] = '\0';
                }
            }
            avail = (int)sizeof(f->hold_data) - 1 - f->hold_n;
        }
    }
    return rc;
}

int slm_think_filter_finish(struct slm_think_filter * f,
                            slm_stream_cb cb,
                            void * user) {
    int rc = 0;
    // No-more-data flush: any pending markers won't get longer, so
    // emit one last pass through the marker scan, then dump the
    // residue in current mode.
    bool more = true;
    while (more && rc == 0) {
        int pos = 0;
        int m   = slm_think_find_marker(f, &pos);
        if (m < 0) {
            more = false;
        } else {
            int r2 = slm_think_emit(f, f->hold_data, pos, cb, user);
            if (r2 != 0) { rc = r2; }
            if (THINK_MARKERS[m].mode != THINK_MODE_NONE) {
                f->mode = THINK_MARKERS[m].mode;
            }
            int after = pos + THINK_MARKERS[m].len;
            int remain = f->hold_n - after;
            if (remain > 0) {
                memmove(f->hold_data, f->hold_data + after,
                        (size_t)remain);
            }
            f->hold_n = remain;
            f->hold_data[f->hold_n] = '\0';
        }
    }
    if (f->hold_n > 0 && rc == 0) {
        rc = slm_think_emit(f, f->hold_data, f->hold_n, cb, user);
        f->hold_n = 0;
        f->hold_data[0] = '\0';
    }
    return rc;
}

// Trampoline box for slm_generate_split.
struct slm_split_box {
    struct slm_think_filter filter;
    slm_stream_cb           cb;
    void *                  user;
};

static int slm_split_trampoline(const char * utf8, void * user) {
    struct slm_split_box * b = (struct slm_split_box *)user;
    return slm_think_filter_push(&b->filter, utf8, b->cb, b->user);
}

// The public `slm_generate` definition lives AFTER `#include
// "agent.c"` so the tools-enabled build can reach
// agent_parse_tool_calls + agent_dispatch (defined in agent.c).
// See the matching definition further down in this file.

// Self-test for the think filter. Feeds known streams through it
// piece-by-piece (mimicking slm_generate's per-token callback) and
// asserts the routed content / reasoning outputs match hand-traced
// goldens. Returns 0 on PASS.
struct slm_think_test_capture {
    char content  [512];
    int  content_n;
    char reasoning[512];
    int  reasoning_n;
};

static int slm_think_test_cb(const struct slm_stream_chunk * chunk,
                             void * user) {
    struct slm_think_test_capture * cap =
        (struct slm_think_test_capture *)user;
    if (chunk->content != NULL) {
        int n = (int)strlen(chunk->content);
        int avail = (int)sizeof(cap->content) - cap->content_n - 1;
        int copy = n < avail ? n : avail;
        if (copy > 0) {
            memcpy(cap->content + cap->content_n,
                   chunk->content, (size_t)copy);
            cap->content_n += copy;
            cap->content[cap->content_n] = '\0';
        }
    }
    if (chunk->reasoning != NULL) {
        int n = (int)strlen(chunk->reasoning);
        int avail = (int)sizeof(cap->reasoning) - cap->reasoning_n - 1;
        int copy = n < avail ? n : avail;
        if (copy > 0) {
            memcpy(cap->reasoning + cap->reasoning_n,
                   chunk->reasoning, (size_t)copy);
            cap->reasoning_n += copy;
            cap->reasoning[cap->reasoning_n] = '\0';
        }
    }
    return 0;
}

// Feed `chunks[0..n)` into the filter sequentially, then finish.
// Verifies content / reasoning outputs against the goldens.
// Returns 1 on fail (caller increments failure counter).
static int slm_think_test_case(const char * name,
                               const char ** chunks, int n_chunks,
                               const char * want_content,
                               const char * want_reasoning) {
    int failed = 0;
    // Zero-init the filter (the tool_call accumulator's heap buffer
    // is the only allocating field; init assumes a cleared start).
    struct slm_think_filter f = {0};
    slm_think_filter_init(&f);
    struct slm_think_test_capture cap;
    cap.content_n = 0;
    cap.content[0] = '\0';
    cap.reasoning_n = 0;
    cap.reasoning[0] = '\0';
    for (int i = 0; i < n_chunks; i++) {
        slm_think_filter_push(&f, chunks[i],
                              slm_think_test_cb, &cap);
    }
    slm_think_filter_finish(&f, slm_think_test_cb, &cap);
    if (strcmp(cap.content, want_content) != 0) {
        fprintf(stderr,
            "think-test %s: content MISMATCH\n  got:  %s\n  want: %s\n",
            name, cap.content, want_content);
        failed = 1;
    }
    if (strcmp(cap.reasoning, want_reasoning) != 0) {
        fprintf(stderr,
            "think-test %s: reasoning MISMATCH\n  got:  %s\n  want: %s\n",
            name, cap.reasoning, want_reasoning);
        failed = 1;
    }
    chars_free(&f.tool_call);
    return failed;
}

__attribute__((unused))
static int32_t slm_think_test(void) {
    int failures = 0;
    // A: plain content, no markers.
    {
        const char * chunks[] = { "Hello, world." };
        failures += slm_think_test_case("A", chunks, 1,
                                        "Hello, world.", "");
    }
    // B: leading <think> block emptied by gen prompt pre-fill,
    //    then content. Simulates Qwen3.5 non-thinking mode where
    //    the prompt ends "<think>\n\n</think>\n\n" and the model
    //    occasionally emits the closer first.
    {
        const char * chunks[] = {
            "</think>\n\nHello."
        };
        failures += slm_think_test_case("B", chunks, 1,
                                        "\n\nHello.", "");
    }
    // C: real reasoning block then content.
    {
        const char * chunks[] = {
            "<think>let me think</think>\n\nAnswer."
        };
        failures += slm_think_test_case("C", chunks, 1,
                                        "\n\nAnswer.",
                                        "let me think");
    }
    // D: marker split across two chunks (the holdback case).
    {
        const char * chunks[] = {
            "abc<thi", "nk>secret</thi", "nk>end"
        };
        failures += slm_think_test_case("D", chunks, 3,
                                        "abcend", "secret");
    }
    // E: stray `<` that doesn't lead anywhere (must emit, not eat).
    {
        const char * chunks[] = {
            "x<y<not_a_marker>z"
        };
        failures += slm_think_test_case("E", chunks, 1,
                                        "x<y<not_a_marker>z", "");
    }
    // F: byte-at-a-time delivery (worst-case for the holdback).
    {
        const char * src = "a<think>b</think>c";
        // 18 1-byte chunks.
        const char * chunks[32];
        char single[32];
        int n = (int)strlen(src);
        for (int i = 0; i < n; i++) {
            single[i] = src[i];
            single[i + 1] = '\0';
            // Have to dup so the char* survives — but chunks[i] in
            // this scope can just point into a stack array of
            // 2-byte strings; do it inline.
        }
        char twos[32][2];
        for (int i = 0; i < n; i++) {
            twos[i][0] = src[i];
            twos[i][1] = '\0';
            chunks[i] = twos[i];
        }
        failures += slm_think_test_case("F", chunks, n, "ac", "b");
    }
    if (failures == 0) {
        printf("think-test: PASS (6 fixtures)\n");
    } else {
        printf("think-test: FAIL (%d fixture(s) failed)\n",
               (int)failures);
    }
    return failures;
}

double slm_pp_per_sec(const struct slm_ctx * c) {
    double r = 0.0;
    if (c != NULL && c->t_prefill_s > 0.0) {
        r = (double)c->n_prefill / c->t_prefill_s;
    }
    return r;
}

double slm_tg_per_sec(const struct slm_ctx * c) {
    double r = 0.0;
    if (c != NULL && c->t_gen_s > 0.0) {
        r = (double)c->n_generated / c->t_gen_s;
    }
    return r;
}

int32_t slm_n_prefill(const struct slm_ctx * c) {
    return (c == NULL) ? 0 : c->n_prefill;
}

int32_t slm_n_generated(const struct slm_ctx * c) {
    return (c == NULL) ? 0 : c->n_generated;
}

// Convenience wrappers exported for the Swift bridge.
//
// slm_tokenize allocates the output array sized to a strict
// worst-case bound (one token per input byte; byte-level BPE never
// exceeds this). Caller takes ownership; `free()` releases the
// buffer.
int  slm_tokenize(struct slm_ctx * c, const char * text,
                  int32_t ** out_ids) {
    size_t bytes = (text != NULL) ? strlen(text) : 0;
    size_t cap   = bytes + 1;        // +1 so an empty input still
                                      // gets a valid (zero-length)
                                      // allocation that the caller
                                      // can free uniformly.
    int32_t * ids = (int32_t *)oom(calloc(cap, sizeof(int32_t)));
    int n = tokenizer_encode(&c->model->tok, text, ids, (int)cap);
    *out_ids = ids;
    return n;
}

int  slm_vocab_size(const struct slm_ctx * c) { return c->model->cfg.vocab_size; }
int  slm_eos_id    (const struct slm_ctx * c) { return c->model->cfg.eos_id; }
int  slm_bos_id    (const struct slm_ctx * c) { return c->model->cfg.bos_id; }

// ---------------------------------------------------------------------------

// Agent helpers (parser + tool dispatcher) — `#include`-d here so
// slm_generate's embedded agent loop can reach them. agent.c uses
// slm_ctx + tokenizer_encode + chars / chars_put + tools_*, all of which
// are in scope by this point.
#include "agent.c"

// Public delta-format helper: produces the bytes the C runner
// tokenizes for one new user turn given an already-warm KV cache.
//
// Behavior depends on `system_prefix`:
//
//   - First turn (system_prefix non-NULL): emit a FULL system+user
//     frame including the websearch / fetch / distill tools
//     advertisement (the Jinja's `# Tools` system block). The model
//     sees the tools list once and remembers it via the persistent
//     KV cache.
//   - Subsequent turns (system_prefix NULL): emit a bare delta —
//     `<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
//     <think>\n\n</think>\n\n`. The model still "knows" about the
//     tools from turn 1's system frame.
//
// Without this, the Swift chat path never advertised tools to the
// model — the embedded agent loop in slm_generate fires only when
// the model emits `<tool_call>` markers, which only happens when
// it's been told tools exist.
char * slm_chat_format_delta(const char * user_message,
                             const char * system_prefix,
                             int enable_thinking,
                             int enable_tools,
                             const char * effort) {
    char * result = NULL;
    if (user_message != NULL) {
        if (system_prefix != NULL && system_prefix[0] != '\0') {
            // First turn: full frame, with tool advertisements only
            // when the caller asked for them. Effort prefix prepended
            // to the system content per the docstring:
            //   "low"    → brevity hint
            //   "high"   → think-step-by-step hint
            //   "medium" / NULL → no prefix
            const char * prefix = "";
            if (effort != NULL) {
                if (strcmp(effort, "low") == 0) {
                    prefix = "(brief: 1-2 sentences)\n\n";
                } else if (strcmp(effort, "high") == 0) {
                    prefix = "(think carefully step-by-step before "
                             "answering)\n\n";
                }
            }
            struct chars system_block = {0};
            chars_put(&system_block, prefix, strlen(prefix));
            chars_put(&system_block, system_prefix, strlen(system_prefix));
            chars_put(&system_block, "", 0);
            struct jinja_message msgs[2];
            memset(msgs, 0, sizeof(msgs));
            msgs[0].role    = JINJA_ROLE_SYSTEM;
            msgs[0].content = system_block.data != NULL
                            ? system_block.data
                            : system_prefix;
            msgs[1].role    = JINJA_ROLE_USER;
            msgs[1].content = user_message;
            struct jinja_tool tools[3] = {
                { AGENT_TOOL_WEBSEARCH },
                { AGENT_TOOL_FETCH },
                { AGENT_TOOL_DISTILL },
            };
            int n_tools = enable_tools ? 3 : 0;
            result = jinja_apply(msgs, 2, tools, n_tools,
                                 /*add_gen=*/1,
                                 enable_thinking);
            chars_free(&system_block);
        } else {
            // Subsequent turn: bare delta (KV holds whatever was
            // advertised on turn 1, including no tools at all).
            // `effort` is ignored here — it lives in the system
            // block which is committed to KV on turn 1.
            (void)effort;
            result = jinja_apply_delta(user_message, NULL,
                                       enable_thinking);
        }
    }
    return result;
}

// Public slm_generate. The think + tool-call stream filter (set up
// via slm_split_box / slm_split_trampoline above) routes the
// model's bytes through slm_stream_cb. A completed
// <tool_call>...</tool_call> block stops slm_generate_raw, the
// body is parsed, the named tool dispatches synchronously, and
// the result is tokenized + prefilled back into the KV cache for
// the next iteration. The loop continues until the model produces
// content without another tool_call or an iteration cap kicks in.
//
// Behaviour is controlled by sampler->tools / sampler->debug:
//   - tools=false : the agent loop is bypassed entirely. The
//                   <tool_call> markers are still scanned by the
//                   filter (cheap), but the body emits as content
//                   so the caller sees raw model output rather
//                   than dispatched results.
//   - debug=true  : emit chunk->tool_call and chunk->tool_response
//                   visibility chunks alongside content / reasoning
//                   so the UI can render the agent trace. With
//                   debug=false those chunks are suppressed; the
//                   tool still dispatches, the caller just doesn't
//                   see it.
#define SLM_GENERATE_TOOL_ITER_CAP 6
int slm_generate(struct slm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct slm_sampler * sampler_in,
                 uint64_t seed,
                 slm_stream_cb cb,
                 void * user) {
    // Defaults when sampler isn't provided (rare; CLI / Swift
    // always pass one).
    // tools / debug live on the ctx now (via struct slm_ctrl). The
    // sampler is purely about how-to-pick-the-next-token.
    bool with_tools = c->ctrl.tools;
    bool with_debug = (c->ctrl.debug > 0);
    int  total_gen  = 0;
    bool done       = false;
    int32_t * cur_ids =
        (int32_t *)oom(calloc((size_t)(prompt_n > 0 ? prompt_n : 1),
                                  sizeof(int32_t)));
    if (prompt_n > 0) {
        memcpy(cur_ids, prompt_ids,
               (size_t)prompt_n * sizeof(int32_t));
    }
    int cur_n = prompt_n;
    int iter  = 0;
    while (!done && iter < SLM_GENERATE_TOOL_ITER_CAP &&
           total_gen < max_new) {
        // Zero-init the box so slm_think_filter_init can assume the
        // tool_call accumulator starts cleared (init does NOT
        // chars_free the previous heap buffer — see its header).
        struct slm_split_box box = {0};
        slm_think_filter_init(&box.filter);
        // Toggle marker recognition + visibility chunk emission per
        // sampler flags. The filter consults these when scanning
        // for markers and when emitting chunk->tool_call.
        box.filter.recognize_tool_calls = with_tools;
        box.filter.emit_visibility      = with_debug;
        box.cb   = cb;
        box.user = user;
        int budget = max_new - total_gen;
        // LLM_AGENT_TRACE: print the prefill size + KV pos at each
        // agent-loop iteration so we can verify that the loop
        // APPENDS to the KV (cur_n == size of the new delta only,
        // c->pos grows monotonically) rather than re-sending the full
        // conversation each iteration. Off by default.
        if (getenv("LLM_AGENT_TRACE") != NULL) {
            fprintf(stderr,
                    "[agent] iter=%d pos=%d prefill_n=%d budget=%d\n",
                    iter, c->pos, cur_n, budget);
        }
        int n = slm_generate_raw(c, cur_ids, cur_n,
                                 budget, min_new,
                                 sampler_in, seed,
                                 slm_split_trampoline, &box,
                                 cb, user);
        total_gen += n;
        slm_think_filter_finish(&box.filter, cb, user);
        free(cur_ids);
        cur_ids = NULL;
        cur_n   = 0;
        if (with_tools && box.filter.tool_call_ready &&
            total_gen < max_new) {
            // Rewrap the buffered body in <tool_call>...</tool_call>
            // since the parser scans for those markers.
            struct chars wrapped = {0};
            const char * pre  = "<tool_call>";
            const char * post = "</tool_call>";
            chars_put(&wrapped, pre, strlen(pre));
            if (box.filter.tool_call.count > 0) {
                chars_put(&wrapped, box.filter.tool_call.data,
                          box.filter.tool_call.count);
            }
            chars_put(&wrapped, post, strlen(post));
            chars_put(&wrapped, "", 0);
            struct agent_call calls[4];
            int nc = agent_parse_tool_calls(wrapped.data, calls, 4);
            chars_free(&wrapped);
            struct chars result = {0};
            for (int i = 0; i < nc; i++) {
                struct tool_result r = {0};
                agent_dispatch(&calls[i], &r);
                const char * payload =
                    r.ok && r.body != NULL ? r.body
                  : r.error != NULL        ? r.error
                                           : "(no result)";
                if (i > 0) { chars_put(&result, "\n\n", 2); }
                chars_put(&result, payload, strlen(payload));
                tools_result_free(&r);
            }
            chars_put(&result, "", 0);
            agent_free_calls(calls, nc);
            // Visibility chunk: tell the caller what the tool
            // returned (debug only). The model never sees this
            // chunk — only the framed tool_response below is fed
            // back into KV.
            if (cb != NULL && result.data != NULL && with_debug) {
                struct slm_stream_chunk ch = {0};
                ch.response = result.data;
                cb(&ch, user);
            }
            struct chars inject = {0};
            const char * head =
                "<|im_end|>\n<|im_start|>user\n<tool_response>\n";
            chars_put(&inject, head, strlen(head));
            if (result.data != NULL) {
                chars_put(&inject, result.data, result.count);
            }
            const char * tail =
                "\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n";
            chars_put(&inject, tail, strlen(tail));
            chars_put(&inject, "", 0);
            int32_t * ids = (int32_t *)oom(
                calloc(16384, sizeof(int32_t)));
            int n_ids = tokenizer_encode(&c->model->tok, inject.data,
                                   ids, 16384);
            cur_ids = ids;
            cur_n   = n_ids;
            chars_free(&inject);
            chars_free(&result);
        } else {
            done = true;
        }
        // The filter's tool_call accumulator was the only heap-owning
        // field; free it now that the iteration's read of it is past.
        chars_free(&box.filter.tool_call);
        iter++;
    }
    free(cur_ids);
    return total_gen;
}

#ifdef LLM_CLI

// ---------------------------------------------------------------------------
// CLI - main() and friends. Compiled only when LLM_CLI is defined
// (the standalone `llm` binary built by Makefile). Xcode/iOS builds
// compile this file as a library without LLM_CLI, so this whole
// block is elided to keep the symbol surface clean.
// ---------------------------------------------------------------------------

static const char * slm_cli_gguf_path(void) {
    const char * env = getenv("QWEN_GGUF");
    if (env != NULL && env[0] != '\0') { return env; }
    return SLM_GGUF_PATH_DEFAULT;
}

static int32_t print_cb(const struct slm_stream_chunk * chunk,
                        void * user) {
    (void)user;
    if (chunk->content != NULL) {
        fputs(chunk->content, stdout);
        fflush(stdout);
    }
    // Reasoning is discarded in the bare CLI surface. --ask (agent
    // mode) and the SwiftUI app are where reasoning gets routed
    // somewhere user-visible.
    return 0;
}

// Token callback for the chat CLI: capture content into `chars`.
// Reasoning is the C-side filter's responsibility now — by the time
// we get here, `<think>...</think>` blocks have already been
// stripped and routed to chunk->reasoning, which we drop.
static int32_t capture_cb(const struct slm_stream_chunk * chunk,
                          void * user) {
    struct chars * out = (struct chars *)user;
    if (chunk->content != NULL) {
        chars_put(out, chunk->content, strlen(chunk->content));
    }
    return 0;
}

// Strip reasoning content from a captured assistant reply before
// feeding it back into the framed history. Mirrors the Qwen3 Jinja:
//     content = content.split('</think>')[-1].lstrip('\n')
// i.e. take everything AFTER THE LAST `</think>`, then strip leading
// whitespace. When no `</think>` is present, just strip leading
// whitespace. Past-turn assistant entries in chat history must
// contain only `content` (no `<think>` tags, no reasoning text);
// the only place `<think>\n\n</think>\n\n` legitimately appears is
// in the generation prompt for the NEXT turn.
static void strip_reasoning_for_history(struct chars * s) {
    if (s->data == NULL || s->count == 0) { return; }
    const char close[] = "</think>";
    size_t close_len = sizeof(close) - 1;
    size_t last_close_end = 0;
    int    found = 0;
    for (size_t i = 0; i + close_len <= s->count; i++) {
        if (memcmp(s->data + i, close, close_len) == 0) {
            last_close_end = i + close_len;
            found = 1;
        }
    }
    size_t cursor = found ? last_close_end : 0;
    while (cursor < s->count &&
           (s->data[cursor] == '\n' || s->data[cursor] == '\r' ||
            s->data[cursor] == ' '  || s->data[cursor] == '\t')) {
        cursor++;
    }
    size_t kept = s->count - cursor;
    if (kept > 0) { memmove(s->data, s->data + cursor, kept); }
    s->count = kept;
    s->data[kept] = '\0';
}

// Multi-turn chat mode. Each --prompt is one user turn. The runner
// re-prefills the full conversation each turn (KV cache overwrites,
// SSM cache cleared via slm_reset() between turns). The optional
// --system string is prepended to the FIRST user turn's body (no
// `<|im_start|>system` block), matching im.ai's observed framing.
static int32_t run_chat(const char ** prompts, int32_t n_prompts,
                        const char * system_prompt,
                        const struct slm_sampler * sp, uint64_t seed,
                        int32_t max_new) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t r = 0;
    if (!c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        // Persistent KV: only the DELTA gets tokenized each turn.
        // Turn 0 carries the system prefix inline with the first
        // user message (matches im.ai's framing). Subsequent turns
        // tokenize a bare `<|im_start|>user\n{Q}<|im_end|>\n` +
        // assistant gen header. slm_generate advances c->pos so the
        // next call's prefill writes into KV at the correct offset.
        int32_t * ids = (int32_t *)oom(calloc(16384, sizeof(int32_t)));
        for (int32_t t = 0; t < n_prompts && r == 0; t++) {
            // Build this turn's delta via jinja-template.c. On turn 0
            // we pass system_prompt as the inline prefix; subsequent
            // turns get NULL so the prefix block isn't repeated.
            const char * sys = (t == 0) ? system_prompt : NULL;
            char * delta_str = jinja_apply_delta(prompts[t], sys, 0);
            printf("\n--- turn %d/%d ---\n", (int)(t + 1), (int)n_prompts);
            if (t == 0 && system_prompt != NULL && system_prompt[0] != '\0') {
                printf("[system inline] %s\n", system_prompt);
            }
            printf("[user] %s\n[assistant] ", prompts[t]);
            fflush(stdout);
            int32_t nids = tokenizer_encode(&c->model->tok, delta_str, ids, 16384);
            struct chars reply = {0};
            slm_generate(c, ids, nids, max_new, g_min_new, sp, seed,
                         capture_cb, &reply);
            chars_put(&reply, "", 0);
            strip_reasoning_for_history(&reply);
            const char eot[] = "<|im_end|>";
            size_t eot_len = sizeof(eot) - 1;
            if (reply.count >= eot_len &&
                memcmp(reply.data + reply.count - eot_len,
                       eot, eot_len) == 0) {
                reply.count -= eot_len;
                reply.data[reply.count] = '\0';
            }
            fwrite(reply.data, 1, reply.count, stdout);
            fflush(stdout);
            printf("\n");
            fprintf(stderr,
                    "pp: %.2f tok/s (%d tok)  tg: %.2f tok/s (%d tok)  "
                    "kv pos=%d\n",
                    slm_pp_per_sec(c), (int)slm_n_prefill(c),
                    slm_tg_per_sec(c), (int)slm_n_generated(c),
                    (int)c->pos);
            chars_free(&reply);
            free(delta_str);
        }
        free(ids);
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    return r;
}

// Capture-and-hash callback: accumulates the decoded UTF-8 stream
// into a chars buffer (for the visible reply) AND folds each piece
// into a running FNV-1a 64-bit hash (for compact pass/fail comparison
// across runs). The hash is what --chat-test compares between
// run-1-pre-reset and run-2-post-reset; the captured text is shown
// to the user only when something diverges.
struct chat_test_capture {
    struct chars text;
    uint64_t     hash;
};

// Hash BOTH streams together so the pre-filter byte-stream
// equivalence is preserved across the API rename. content +
// reasoning combined exactly reconstructs what the model emitted
// (modulo `<think>`/`</think>` marker bytes, which the C-side
// filter discards). The hash thus captures the same "did the
// generator behave identically" signal as before; if reasoning
// shape changes between runs, that's surfaced.
static int chat_test_cb(const struct slm_stream_chunk * chunk,
                        void * user) {
    struct chat_test_capture * cap = (struct chat_test_capture *)user;
    const char * piece = (chunk->content != NULL) ? chunk->content
                                                  : chunk->reasoning;
    if (piece != NULL) {
        size_t n = strlen(piece);
        chars_put(&cap->text, piece, n);
        for (size_t i = 0; i < n; i++) {
            cap->hash ^= (uint64_t)(uint8_t)piece[i];
            cap->hash *= 0x100000001b3ULL;
        }
    }
    return 0;
}

// Multi-turn scripted chat test. Runs three deterministic turns
// twice — once on a fresh context, then again after slm_reset() — and
// verifies the reply hashes match across the two passes. This exercises
// the chat-template state machine, the persistent KV cache, the
// chunked-SSM prefill, AND the slm_reset() contract that state is
// fully cleared between conversations.
//
// Fixed inputs (not user-configurable) so a CI run produces a stable
// PASS/FAIL signal: im.ai sampler preset, seed=42, three short turns.
// Failure prints both passes' replies side-by-side for diagnosis.
static int32_t run_chat_test(int32_t max_new) {
    static const char * const turns[] = {
        "Hi! Just say hello back.",
        "What did I just ask?",
        "Thanks!"
    };
    enum { N_TURNS = 3 };
    struct slm_sampler sp = slm_sampler_defaults();
    uint64_t seed = 42;
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t r = 0;
    if (!c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        struct chat_test_capture caps_a[N_TURNS] = {0};
        struct chat_test_capture caps_b[N_TURNS] = {0};
        int32_t * ids = (int32_t *)oom(calloc(16384,
                                                  sizeof(int32_t)));
        for (int32_t pass = 0; pass < 2 && r == 0; pass++) {
            struct chat_test_capture * caps = (pass == 0) ? caps_a : caps_b;
            // Reset before pass 1; pass 0 starts from a fresh ctx so
            // the SSM state and pos are already zeroed by slm_create.
            if (pass == 1) { slm_reset(c); }
            for (int32_t t = 0; t < N_TURNS && r == 0; t++) {
                struct chat_test_capture * cap = &caps[t];
                cap->hash = 0xcbf29ce484222325ULL;
                char * delta_str = jinja_apply_delta(turns[t], NULL, 0);
                int32_t nids = tokenizer_encode(&c->model->tok, delta_str,
                                          ids, 16384);
                int32_t gen = slm_generate(c, ids, nids, max_new,
                                           g_min_new, &sp, seed,
                                           chat_test_cb, cap);
                chars_put(&cap->text, "", 0);
                fprintf(stderr,
                        "chat-test pass %d turn %d: gen=%d "
                        "hash=%016llx pos=%d\n",
                        (int)pass, (int)(t + 1), (int)gen,
                        (unsigned long long)cap->hash,
                        (int)c->pos);
                free(delta_str);
            }
        }
        free(ids);
        int32_t mismatch = -1;
        for (int32_t t = 0; t < N_TURNS && mismatch < 0; t++) {
            if (caps_a[t].hash != caps_b[t].hash) { mismatch = t; }
        }
        if (mismatch < 0) {
            printf("chat-test: PASS (3 turns x 2 passes,"
                   " hashes match across slm_reset)\n");
        } else {
            printf("chat-test: FAIL at turn %d\n", (int)(mismatch + 1));
            printf("  user:       %s\n", turns[mismatch]);
            printf("  pass A:     %s\n", caps_a[mismatch].text.data);
            printf("  pass B:     %s\n", caps_b[mismatch].text.data);
            r = 1;
        }
        for (int32_t t = 0; t < N_TURNS; t++) {
            chars_free(&caps_a[t].text);
            chars_free(&caps_b[t].text);
        }
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    return r;
}

// One-shot agent run: feed `question` to the model with websearch /
// fetch / distill tools advertised, let it iterate up to max_iters
// tool calls, print the final answer. --trace dumps per-iteration
// trace to stderr so the user can watch what the agent does.
static int32_t run_ask(const char * question,
                       const struct slm_sampler * sp, uint64_t seed,
                       int32_t max_iters, int32_t max_new,
                       int32_t trace) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t rc = 0;
    if (!c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        rc = 1;
    } else {
        printf("[user] %s\n", question);
        fflush(stdout);
        char * answer = agent_run(c, question, sp, seed,
                                  max_iters, max_new, trace);
        printf("\n[assistant] %s\n", answer != NULL ? answer : "");
        free(answer);
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    tools_global_cleanup();
    return rc;
}

static int32_t run_single(const char * prompt, int32_t max_new,
                          const struct slm_sampler * sp, uint64_t seed) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t r = 0;
    if (!c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        printf("model: %d layers, %d heads (%d kv), head_dim=%d, "
               "hidden=%d, ffn=%d, vocab=%d\n",
               (int)c->model->cfg.n_layers, (int)c->model->cfg.n_heads,
               (int)c->model->cfg.n_kv_heads, (int)c->model->cfg.head_dim,
               (int)c->model->cfg.hidden_dim, (int)c->model->cfg.ffn_dim,
               (int)c->model->cfg.vocab_size);
        printf("rope: dim=%d base=%.1f sections=[%d,%d,%d,%d]\n",
               (int)c->model->cfg.rope_dim, c->model->cfg.rope_theta,
               (int)c->model->cfg.rope_sections[0], (int)c->model->cfg.rope_sections[1],
               (int)c->model->cfg.rope_sections[2], (int)c->model->cfg.rope_sections[3]);
        int32_t ids[2048];
        int32_t n = tokenizer_encode(&c->model->tok, prompt, ids, 2048);
        printf("prompt tokens (%d): ", (int)n);
        for (int32_t i = 0; i < n; i++) printf("%d ", (int)ids[i]);
        printf("\n---\n%s", prompt);
        fflush(stdout);
        slm_generate(c, ids, n, max_new, g_min_new, sp, seed,
                     print_cb, NULL);
        printf("\n");
        fprintf(stderr,
                "pp: %.2f tok/s (%d tok)  tg: %.2f tok/s (%d tok)\n",
                slm_pp_per_sec(c), (int)slm_n_prefill(c),
                slm_tg_per_sec(c), (int)slm_n_generated(c));
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    return r;
}

// --bench: cross-host throughput probe. Runs a fixed prompt
// through prefill + greedy decode three times and prints one
// machine-grep-friendly summary line:
//
//   bench: pp=<float> tg=<float> n_pp=<int> n_tg=<int> dispatch=<label>
//
// pp/tg are the median of the three runs (in tok/s). Greedy decode
// (temperature=0) makes the output deterministic, so the timing
// numbers aren't perturbed by sampler RNG. Used by rnd/sweep.sh
// across the multi-host fleet — each host's summary line goes into
// rnd/sweep-results/<host>.out and a final table aggregates the
// fleet's pp/tg by ISA tier.
static int32_t run_bench(void) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx *   c =
        slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t r = 0;
    if (c == NULL || !c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        // Fixed prompt: ~30 tokens once tokenized, enough to give a
        // stable pp signal across hosts (very short prompts mostly
        // measure model-load tail noise, not prefill throughput).
        const char * prompt =
            "Explain in five short sentences why benchmarking large "
            "language models across diverse hardware matters. Avoid "
            "vague generalities; be concrete.";
        const int32_t bench_max_new = 64;
        struct slm_sampler sp = slm_sampler_defaults();
        sp.temperature = 0.0f;   // greedy: deterministic for re-runs
        int32_t ids[2048];
        int32_t n = tokenizer_encode(&c->model->tok, prompt, ids, 2048);
        // 3 runs; reset between for clean state. Take median.
        const int N = 3;
        float pps[N], tgs[N];
        int32_t n_pp = 0, n_tg = 0;
        for (int run = 0; run < N; run++) {
            slm_reset(c);
            slm_generate(c, ids, n, bench_max_new, g_min_new, &sp,
                         /*seed=*/1u, /*cb=*/NULL, /*user=*/NULL);
            pps[run] = slm_pp_per_sec(c);
            tgs[run] = slm_tg_per_sec(c);
            n_pp     = (int32_t)slm_n_prefill(c);
            n_tg     = (int32_t)slm_n_generated(c);
            fprintf(stderr, "  run %d: pp=%.2f tg=%.2f\n",
                    run + 1, pps[run], tgs[run]);
        }
        // Median of 3: pick the middle element after a 3-element sort.
        // Tiny enough to inline a sort net.
        #define SORT3(a) do {                                         \
            if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }\
            if (a[1] > a[2]) { float t = a[1]; a[1] = a[2]; a[2] = t; }\
            if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }\
        } while (0)
        SORT3(pps);
        SORT3(tgs);
        #undef SORT3
        printf("bench: pp=%.2f tg=%.2f n_pp=%d n_tg=%d dispatch=%s\n",
               pps[1], tgs[1], n_pp, n_tg, simd_dispatch_label());
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    return r;
}

static int32_t run_repl(const struct slm_sampler * sp,
                        uint64_t seed, int32_t max_new) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
    int32_t r = 0;
    if (!c->model->loaded) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        char line[4096];
        printf("repl: type a message, Enter to send, Ctrl-D to exit.\n");
        while (fgets(line, sizeof(line), stdin) != NULL) {
            size_t n = strlen(line);
            while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) {
                line[--n] = '\0';
            }
            if (n > 0) {
                char framed[8192];
                snprintf(framed, sizeof(framed),
                         "<|im_start|>user\n%s<|im_end|>\n"
                         "<|im_start|>assistant\n", line);
                int32_t ids[2048];
                int32_t nids = tokenizer_encode(&c->model->tok, framed, ids, 2048);
                printf("\nassistant: ");
                fflush(stdout);
                slm_generate(c, ids, nids, max_new, g_min_new, sp, seed,
                             print_cb, NULL);
                printf("\n\n> ");
                fflush(stdout);
            }
        }
    }
    slm_ctx_destroy(c);
    slm_model_unload(m);
    return r;
}

#define LLM_CLI_MAX_TURNS 32

int main(int argc, char ** argv) {
    enum cli_mode {
        CLI_HELP            =  0,
        CLI_SELF_TEST       =  1,
        CLI_SINGLE          =  2,
        CLI_REPL            =  3,
        CLI_CHAT            =  4,
        CLI_CHUNKED_TEST    =  5,
        CLI_CHAT_TEST       =  6,
        CLI_PRINT_TEMPLATE  =  7,
        CLI_JINJA_TEST      =  8,
        CLI_TOOLS_TEST      =  9,
        CLI_AGENT_TEST      = 10,
        CLI_ASK             = 11,
        CLI_THINK_TEST      = 12,
        CLI_BENCH           = 13,
    };
    enum cli_mode mode = CLI_HELP;
    const char * prompt = "Hello, my name is";
    const char * system_prompt   = NULL;
    const char * chat_prompts[LLM_CLI_MAX_TURNS];
    int32_t      chat_n          = 0;
    // Hardcoded sampler default: im.ai's chat-tuned profile
    // (temp 0.7, top_k 40, top_p 0.9, min_p 0.05, rep 1.25, win 64).
    // Empirically the best fit for Qwen3.5-0.8B in chat + tools mode.
    // Individual flags below can override any field.
    struct slm_sampler sp = slm_sampler_defaults();
    uint64_t seed         = 0;          // 0 = derive from wall clock
    int32_t  max_new      = 64;
    int32_t  max_iters    = 4;          // agent-loop cap (--ask)
    int32_t  trace_agent  = 0;          // --trace
    int32_t  dump_layer   = -1;
    for (int32_t i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            mode = CLI_SELF_TEST;
        } else if (strcmp(argv[i], "--chunked-test") == 0) {
            mode = CLI_CHUNKED_TEST;
        } else if (strcmp(argv[i], "--single") == 0) {
            mode = CLI_SINGLE;
            if (i + 1 < argc) { prompt = argv[++i]; }
        } else if (strcmp(argv[i], "--repl") == 0) {
            mode = CLI_REPL;
        } else if (strcmp(argv[i], "--chat") == 0) {
            mode = CLI_CHAT;
        } else if (strcmp(argv[i], "--chat-test") == 0) {
            mode = CLI_CHAT_TEST;
        } else if (strcmp(argv[i], "--print-chat-template") == 0) {
            mode = CLI_PRINT_TEMPLATE;
        } else if (strcmp(argv[i], "--jinja-test") == 0) {
            mode = CLI_JINJA_TEST;
        } else if (strcmp(argv[i], "--tools-test") == 0) {
            mode = CLI_TOOLS_TEST;
        } else if (strcmp(argv[i], "--agent-test") == 0) {
            mode = CLI_AGENT_TEST;
        } else if (strcmp(argv[i], "--think-test") == 0) {
            mode = CLI_THINK_TEST;
        } else if (strcmp(argv[i], "--bench") == 0) {
            mode = CLI_BENCH;
        } else if (strcmp(argv[i], "--ask") == 0 && i + 1 < argc) {
            mode = CLI_ASK;
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 ||
                    strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            if (chat_n < LLM_CLI_MAX_TURNS) {
                chat_prompts[chat_n++] = argv[++i];
            } else {
                fprintf(stderr, "llm: too many --prompt turns (max %d)\n",
                        LLM_CLI_MAX_TURNS);
                i++;
            }
        } else if ((strcmp(argv[i], "-sys") == 0 ||
                    strcmp(argv[i], "--system") == 0) && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            sp.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            sp.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            sp.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            sp.min_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            sp.repetition_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-window") == 0 && i + 1 < argc) {
            sp.repetition_window = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace_agent = 1;
        } else if (strcmp(argv[i], "--min-new") == 0 && i + 1 < argc) {
            g_min_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump-layer") == 0 && i + 1 < argc) {
            dump_layer = atoi(argv[++i]);
        }
    }
    g_dump_layer   = dump_layer;
    g_no_q8k_rt    = getenv("NO_Q8K_RT")       != NULL;
    g_trace_tokens = getenv("LLM_TRACE_TOKENS") != NULL;
    int rc = 0;
    if (mode == CLI_SELF_TEST) {
        rc = qwen_self_test();
    } else if (mode == CLI_CHUNKED_TEST) {
        rc = chunked_self_test();
    } else if (mode == CLI_SINGLE) {
        rc = run_single(prompt, max_new, &sp, seed);
    } else if (mode == CLI_REPL) {
        rc = run_repl(&sp, seed, max_new);
    } else if (mode == CLI_CHAT) {
        if (chat_n == 0) {
            fprintf(stderr, "llm: --chat needs at least one -p/--prompt\n");
            rc = 1;
        } else {
            rc = run_chat(chat_prompts, chat_n, system_prompt,
                          &sp, seed, max_new);
        }
    } else if (mode == CLI_CHAT_TEST) {
        rc = run_chat_test(max_new);
    } else if (mode == CLI_PRINT_TEMPLATE) {
        struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m, NULL, NULL) : NULL;
        if (!c->model->loaded) {
            fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
            rc = 1;
        } else {
            const char * tpl = slm_chat_template(c);
            if (tpl != NULL) { fputs(tpl, stdout); }
        }
        slm_ctx_destroy(c);
    slm_model_unload(m);
    } else if (mode == CLI_JINJA_TEST) {
        rc = jinja_self_test();
    } else if (mode == CLI_TOOLS_TEST) {
        rc = tools_self_test();
        tools_global_cleanup();
    } else if (mode == CLI_AGENT_TEST) {
        rc = agent_parser_test();
    } else if (mode == CLI_ASK) {
        rc = run_ask(prompt, &sp, seed, max_iters,
                     max_new > 0 ? max_new : 256, trace_agent);
    } else if (mode == CLI_THINK_TEST) {
        rc = slm_think_test();
    } else if (mode == CLI_BENCH) {
        rc = run_bench();
    } else {
        printf("usage (set QWEN_GGUF=/path/to/model.gguf to override default):\n"
               "  llm --self-test\n"
               "  llm --bench\n"
               "  llm --single \"prompt\" [--max-new N] [sampler flags]\n"
               "  llm --repl [--max-new N] [sampler flags]\n"
               "  llm --chat -p \"turn1\" [-p \"turn2\" ...] "
                       "[-sys \"system prompt\"] [sampler flags]\n"
               "  llm --chat-test [--max-new N]\n"
               "  llm --jinja-test\n"
               "  llm --tools-test          (live: needs network for DDG)\n"
               "  llm --agent-test          (parser fixtures, no network)\n"
               "  llm --ask \"question\" [--max-iters N] [--max-new N]"
                       " [--trace] [sampler flags]\n"
               "  llm --print-chat-template\n"
               "\n"
               "sampler flags:\n"
               "  --temperature T                   0 = greedy, >0 = softmax T\n"
               "  --top-k K                         keep K best logits\n"
               "  --top-p P                         nucleus cutoff in (0,1)\n"
               "  --min-p P                         keep prob >= P * top_prob\n"
               "  --rep-penalty F                   1.0 = off, 1.05-1.3 typical\n"
               "  --rep-window N                    history window for rep penalty\n"
               "  --seed S                          0 = wall-clock derived\n");
        rc = 0;
    }
    return rc;
}

#endif // LLM_CLI
