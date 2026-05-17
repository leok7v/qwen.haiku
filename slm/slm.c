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

// trace() declarations — model.c (further down) supplies the
// implementation via #define TRACE_IMPLEMENTATION. We need the
// declarations now because gguf.c (included below) uses trace().
#include "utils/trace.c"

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
// / slm_ctx_ctrl + the file-internal slm_reset() moved
// to model.c (with the structs they own). The chat-template accessor
// + chat formatting + the agent loop + the stream filter live below.
// ---------------------------------------------------------------------------

const char * slm_chat_template(const struct slm_ctx * c) {
    return (c == NULL) ? NULL : c->model->chat_template;
}

// Thin wrapper exposing jinja-template.c's chat-formatting through
// the public slm_chat_format API. We copy the public slm_chat_message[]
// into the internal jinja_message layout so the public ABI stays
// minimal.
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
            }
            result = jinja_apply(tmp, n_messages, NULL, 0,
                                 add_generation_prompt, enable_thinking);
            free(tmp);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Streaming callback helper. Writes chunk fields into `cb`, invokes
// the callback, then zeros the fields so a stale pointer from one
// call isn't visible at the next. Returns the callback's return
// value (non-zero = abort), or 0 when cb is NULL.
// ---------------------------------------------------------------------------
static int slm_cb_emit(struct slm_stream_callback * cb,
                       const char * content,
                       const char * reasoning,
                       const char * call,
                       const char * response,
                       bool         prefilled,
                       int32_t      pp_pos,
                       int32_t      pp_total) {
    int rc = 0;
    if (cb != NULL && cb->callback != NULL) {
        cb->content   = content;
        cb->reasoning = reasoning;
        cb->call      = call;
        cb->response  = response;
        cb->prefilled = prefilled;
        cb->pp_pos    = pp_pos;
        cb->pp_total  = pp_total;
        rc = cb->callback(cb);
        cb->content   = NULL;
        cb->reasoning = NULL;
        cb->call      = NULL;
        cb->response  = NULL;
        cb->prefilled = false;
        cb->pp_pos    = 0;
        cb->pp_total  = 0;
    }
    return rc;
}

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Emit a long text payload through the trace ring as a sequence of
// ~180-byte chunks (TRACE_MESSAGE_MAX is 232; leave room for the
// "<label> [start..end of total]: " prefix). Used by the debug>=9
// "full text I/O" dumps so the user sees system framing / user
// prompts / model output verbatim instead of truncated at 232 chars.
static void slm_trace_long(const char * label,
                           const char * text, size_t n) {
    if (text != NULL && n > 0) {
        const size_t CHUNK = 180;
        size_t       pos   = 0;
        while (pos < n) {
            size_t take = (n - pos < CHUNK) ? (n - pos) : CHUNK;
            trace("%s [%zu..%zu of %zu]: %.*s\n",
                  label, pos, pos + take, n,
                  (int)take, text + pos);
            pos += take;
        }
    }
}

// File-internal token-stream callback. slm_generate_raw emits one
// utf8 piece per generated token to this; slm_generate trampolines
// into the <think>-filter on top of this (see further down in this
// file). `utf8` is a NUL-terminated UTF-8 piece. Return non-zero to
// abort generation.

typedef int (*slm_token_cb)(const char * utf8, void * user);

// Optional functor fired after the prefill phase finishes and BEFORE
// the decode loop starts. Same first-field-cast idiom as
// slm_stream_callback: embed in your own struct, store extra state
// alongside, the on_prefilled function receives `self` and reads its
// state via the cast — no separate `void * user` plumbing.
//
// slm_generate uses this to snapshot the ctx at the model's "decision
// point" (KV holds user delta + assistant prefix; no decode tokens
// yet). A tool_call dispatched mid-decode restores to this exact
// point and just appends a preamble, skipping any re-prefill.
struct slm_after_prefill {
    void (*on_prefilled)(const struct slm_after_prefill * self,
                         struct slm_ctx *                 c);
};

// slm_generate's snap-capture functor — first-field-cast over
// slm_after_prefill. `slot` is where the captured snapshot pointer
// lands; consumed and NULLed in the tool-dispatch branch.
struct slm_snap_capture {
    struct slm_after_prefill base;
    struct slm_snapshot **   slot;
};

static void slm_snap_capture_fn(const struct slm_after_prefill * self,
                                struct slm_ctx *                 c) {
    const struct slm_snap_capture * me =
        (const struct slm_snap_capture *)self;
    *me->slot = slm_ctx_snapshot(c);
}

static int slm_generate_raw(struct slm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct slm_sampler * sampler_in,
                 uint64_t seed,
                 slm_token_cb cb, void * user,
                 struct slm_stream_callback *     progress_cb,
                 const struct slm_after_prefill * after_prefill) {
    struct slm_sampler sp = {0};
    if (sampler_in != NULL) { sp = *sampler_in; }
    // rng lives on the ctx (seeded in slm_ctx_create from time(NULL)).
    // Caller-supplied non-zero `seed` re-seeds it before this call —
    // that's the parity-gate / reproducibility path. seed=0 advances
    // c->rng across calls in the same ctx (real chat sampling).
    if (seed != 0) { rng_seed(&c->rng, seed); }
    slm_trace_open();
    int32_t   generated = 0;
    int32_t   pos       = c->pos;        // resume after previous call
    bool      stop      = false;
    double    t0        = seconds();
    // Repetition-penalty history: include the full prompt so the
    // model is discouraged from immediately echoing the input back,
    // plus everything it generates in this call.
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
        struct tensor * logits = slm_forward_batch(c, prompt_ids,
                                                   prompt_n, pos);
        (void)logits;
        pos += prompt_n;
        slm_trace_close();
    } else {
        for (int32_t i = 0; i < prompt_n && !stop; i++) {
            struct tensor * logits = slm_forward_step(c, prompt_ids[i], pos);
            if (i == 0) { slm_trace_close(); }
            (void)logits;
            pos++;
        }
    }
    // Fire the after-prefill hook with c->pos already advanced to
    // post-prefill so a snapshot taken here covers the user delta.
    c->pos = pos;
    if (after_prefill != NULL && after_prefill->on_prefilled != NULL) {
        after_prefill->on_prefilled(after_prefill, c);
    }
    if (progress_cb != NULL && prompt_n > 0) {
        slm_cb_emit(progress_cb, NULL, NULL, NULL, NULL,
                    /*prefilled=*/true, 0, 0);
    }
    double t1 = seconds();
    c->t_prefill_s = t1 - t0;
    c->n_prefill   = prompt_n;
    int32_t last = prompt_n > 0 ? prompt_ids[prompt_n - 1] : c->model->cfg.bos_id;
    // Decode loop.
    while (!stop && generated < max_new && pos < c->model->cfg.max_position) {
        struct tensor * logits = slm_forward_step(c, last, pos);
        float saved_eos = 0.0f;
        float saved_eot = 0.0f;
        bool  mask      = (generated < min_new);
        if (mask) {
            if (c->model->cfg.eos_id >= 0 &&
                c->model->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                saved_eos = logits->data[c->model->cfg.eos_id];
                logits->data[c->model->cfg.eos_id] = -INFINITY;
            }
            if (c->model->cfg.eot_id >= 0 &&
                c->model->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                saved_eot = logits->data[c->model->cfg.eot_id];
                logits->data[c->model->cfg.eot_id] = -INFINITY;
            }
        }
        int32_t next = sample_with(logits, &sp, &c->rng, history, hist_n);
        history[hist_n++] = next;
        if (mask) {
            if (c->model->cfg.eos_id >= 0 &&
                c->model->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->model->cfg.eos_id] = saved_eos;
            }
            if (c->model->cfg.eot_id >= 0 &&
                c->model->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->model->cfg.eot_id] = saved_eot;
            }
        }
        pos++;
        generated++;
        if (c->ctrl.trace_tokens) {
            fprintf(stderr, "[tok] %d\n", (int)next);
        }
        bool is_stop = (next == c->model->cfg.eos_id ||
                        next == c->model->cfg.eot_id);
        for (int32_t si = 0; !is_stop && si < c->model->cfg.n_stop_ids; si++) {
            if (next == c->model->cfg.stop_ids[si]) { is_stop = true; }
        }
        if (is_stop) {
            stop = true;
        } else {
            struct chars piece = {0};
            tokenizer_decode_one(&c->model->tok, next, &piece);
            chars_put(&piece, "", 0);
            if (cb != NULL && piece.data != NULL) {
                if (cb(piece.data, user) != 0) { stop = true; }
            }
            chars_free(&piece);
            last = next;
        }
    }
    double t2 = seconds();
    c->t_gen_s     = t2 - t1;
    c->n_generated = generated;
    c->pos         = pos;
    free(history);
    slm_trace_close();
    return generated;
}

// ---------------------------------------------------------------------------
// <think> stream filter. State machine that splits slm_generate's
// single output stream into two: content (outside `<think>`...
// `</think>` blocks) and reasoning (inside). Marker bytes are NOT
// emitted; both streams are routed through ONE slm_stream_callback
// callback with chunk fields that carry exactly one non-NULL pointer
// per call.
// ---------------------------------------------------------------------------

enum slm_think_mode {
    THINK_MODE_CONTENT   = 0,
    THINK_MODE_REASONING = 1,
    THINK_MODE_TOOL_CALL = 2,
    THINK_MODE_NONE      = -1,
};

enum slm_think_marker_row {
    SLM_MARKER_THINK_OPEN  = 0,
    SLM_MARKER_THINK_CLOSE = 1,
    SLM_MARKER_TOOL_OPEN   = 2,
    SLM_MARKER_TOOL_CLOSE  = 3,
};

struct slm_think_filter {
    enum slm_think_mode mode;
    char                hold_data[64];
    int                 hold_n;
    struct chars        tool_call;
    bool                tool_call_ready;
    bool                recognize_tool_calls;
    bool                emit_visibility;
};

struct slm_think_marker {
    const char *        tag;
    int                 len;
    enum slm_think_mode mode;
};

static const struct slm_think_marker THINK_MARKERS[] = {
    [SLM_MARKER_THINK_OPEN]  = { "<think>",      7, THINK_MODE_REASONING },
    [SLM_MARKER_THINK_CLOSE] = { "</think>",     8, THINK_MODE_CONTENT   },
    [SLM_MARKER_TOOL_OPEN]   = { "<tool_call>", 11, THINK_MODE_TOOL_CALL },
    [SLM_MARKER_TOOL_CLOSE]  = { "</tool_call>",12, THINK_MODE_CONTENT   },
};
#define SLM_THINK_N_MARKERS \
    ((int)(sizeof(THINK_MARKERS) / sizeof(THINK_MARKERS[0])))
#define SLM_THINK_MAX_MARKER 12

static void slm_think_filter_init(struct slm_think_filter * f) {
    f->mode = THINK_MODE_CONTENT;
    f->hold_n = 0;
    f->hold_data[0] = '\0';
    f->tool_call = (struct chars){0};
    f->tool_call_ready = false;
    f->recognize_tool_calls = true;
    f->emit_visibility      = true;
}

// Emit `[start, start+n)` bytes through `cb`, tagging the chunk
// per the filter's current mode. Mode TOOL_CALL routes bytes into
// the filter's internal tool_call buffer instead of firing cb.
// Returns the callback's return value, or 0 when no work/no cb.

static int slm_think_emit(struct slm_think_filter * f,
                          const char * start, int n,
                          struct slm_stream_callback * cb) {
    int rc = 0;
    if (n > 0) {
        if (f->mode == THINK_MODE_TOOL_CALL) {
            chars_put(&f->tool_call, start, (size_t)n);
        } else if (cb != NULL) {
            char tmp[80];
            int  copy = n < (int)sizeof(tmp) - 1
                      ? n : (int)sizeof(tmp) - 1;
            memcpy(tmp, start, (size_t)copy);
            tmp[copy] = '\0';
            if (f->mode == THINK_MODE_REASONING) {
                rc = slm_cb_emit(cb, NULL, tmp, NULL, NULL,
                                 false, 0, 0);
            } else {
                rc = slm_cb_emit(cb, tmp, NULL, NULL, NULL,
                                 false, 0, 0);
            }
            if (n > copy && rc == 0) {
                int rc2 = slm_think_emit(f, start + copy, n - copy, cb);
                if (rc2 != 0) { rc = rc2; }
            }
        }
    }
    return rc;
}

static int slm_think_find_marker(const struct slm_think_filter * f,
                                 int * pos) {
    int best     = -1;
    int best_pos = f->hold_n;
    for (int m = 0; m < SLM_THINK_N_MARKERS; m++) {
        bool is_tool_call = (m == SLM_MARKER_TOOL_OPEN) ||
                            (m == SLM_MARKER_TOOL_CLOSE);
        bool skip = (!f->recognize_tool_calls && is_tool_call);
        if (!skip) {
            const char * tag  = THINK_MARKERS[m].tag;
            int          tlen = THINK_MARKERS[m].len;
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
                        best     = m;
                        best_pos = i;
                    }
                }
            }
        }
    }
    if (best >= 0) { *pos = best_pos; }
    return best;
}

static int slm_think_safe_utf8(const char * buf, int hold_n, int safe) {
    while (safe > 0 && safe < hold_n &&
           ((unsigned char)buf[safe] & 0xC0) == 0x80) {
        safe--;
    }
    return safe;
}

static bool slm_think_could_be_prefix(const struct slm_think_filter * f,
                                      int tail_start) {
    bool possible = false;
    int  tail_n   = f->hold_n - tail_start;
    if (tail_n > 0) {
        for (int m = 0; m < SLM_THINK_N_MARKERS && !possible; m++) {
            const char * tag  = THINK_MARKERS[m].tag;
            int          tlen = THINK_MARKERS[m].len;
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
                          struct slm_stream_callback * cb) {
    int rc = 0;
    if (utf8 != NULL) {
        int avail = (int)sizeof(f->hold_data) - 1 - f->hold_n;
        const char * src = utf8;
        while (*src != '\0' && rc == 0) {
            if (avail <= 0) {
                int keep   = SLM_THINK_MAX_MARKER - 1;
                int emit_n = f->hold_n - keep;
                emit_n = slm_think_safe_utf8(f->hold_data, f->hold_n,
                                             emit_n);
                if (emit_n > 0) {
                    int r2 = slm_think_emit(f, f->hold_data,
                                            emit_n, cb);
                    if (r2 != 0) { rc = r2; }
                    int remain = f->hold_n - emit_n;
                    memmove(f->hold_data, f->hold_data + emit_n,
                            (size_t)remain);
                    f->hold_n = remain;
                    f->hold_data[f->hold_n] = '\0';
                }
                avail = (int)sizeof(f->hold_data) - 1 - f->hold_n;
            }
            int chunk_len = 0;
            while (src[chunk_len] != '\0' && chunk_len < avail) {
                chunk_len++;
            }
            memcpy(f->hold_data + f->hold_n, src, (size_t)chunk_len);
            f->hold_n += chunk_len;
            f->hold_data[f->hold_n] = '\0';
            src   += chunk_len;
            avail -= chunk_len;
            bool more = true;
            while (more && rc == 0) {
                int pos = 0;
                int m   = slm_think_find_marker(f, &pos);
                if (m < 0) {
                    more = false;
                } else {
                    int r2 = slm_think_emit(f, f->hold_data, pos, cb);
                    if (r2 != 0) { rc = r2; }
                    enum slm_think_mode prev_mode = f->mode;
                    if (THINK_MARKERS[m].mode != THINK_MODE_NONE) {
                        f->mode = THINK_MARKERS[m].mode;
                    }
                    if (prev_mode == THINK_MODE_TOOL_CALL &&
                        f->mode  != THINK_MODE_TOOL_CALL) {
                        if (cb != NULL && f->tool_call.count > 0 &&
                            f->emit_visibility) {
                            char * call_text = f->tool_call.data;
                            slm_cb_emit(cb, NULL, NULL, call_text,
                                        NULL, false, 0, 0);
                        }
                        f->tool_call_ready = true;
                        rc = 1;
                        more = false;
                    }
                    if (prev_mode != THINK_MODE_TOOL_CALL &&
                        f->mode  == THINK_MODE_TOOL_CALL) {
                        f->tool_call.count = 0;
                        if (f->tool_call.data != NULL) {
                            f->tool_call.data[0] = '\0';
                        }
                    }
                    int after  = pos + THINK_MARKERS[m].len;
                    int remain = f->hold_n - after;
                    if (remain > 0) {
                        memmove(f->hold_data, f->hold_data + after,
                                (size_t)remain);
                    }
                    f->hold_n = remain;
                    f->hold_data[f->hold_n] = '\0';
                }
            }
            int safe = f->hold_n - (SLM_THINK_MAX_MARKER - 1);
            if (safe > 0 && rc == 0) {
                while (safe > 0 &&
                       slm_think_could_be_prefix(f, safe)) {
                    safe--;
                }
                safe = slm_think_safe_utf8(f->hold_data, f->hold_n,
                                           safe);
                if (safe > 0) {
                    int r2 = slm_think_emit(f, f->hold_data, safe, cb);
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
                            struct slm_stream_callback * cb) {
    int rc = 0;
    bool more = true;
    while (more && rc == 0) {
        int pos = 0;
        int m   = slm_think_find_marker(f, &pos);
        if (m < 0) {
            more = false;
        } else {
            int r2 = slm_think_emit(f, f->hold_data, pos, cb);
            if (r2 != 0) { rc = r2; }
            if (THINK_MARKERS[m].mode != THINK_MODE_NONE) {
                f->mode = THINK_MARKERS[m].mode;
            }
            int after  = pos + THINK_MARKERS[m].len;
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
        rc = slm_think_emit(f, f->hold_data, f->hold_n, cb);
        f->hold_n = 0;
        f->hold_data[0] = '\0';
    }
    return rc;
}

// Trampoline box for the split-filter per-token callback.
struct slm_split_box {
    struct slm_think_filter      filter;
    struct slm_stream_callback * cb;
};

static int slm_split_trampoline(const char * utf8, void * user) {
    struct slm_split_box * b = (struct slm_split_box *)user;
    return slm_think_filter_push(&b->filter, utf8, b->cb);
}

// Pass-through cb that ALSO accumulates content / reasoning bytes
// into local buffers. slm_generate installs this around the user's
// cb when ctrl.debug >= 9 so that, at end of turn, the full text the
// model produced can be re-emitted through the trace ring (chunked
// via slm_trace_long). Call/response/prefilled/pp chunks pass through
// untouched.
struct slm_capture_box {
    struct slm_stream_callback   base;
    struct slm_stream_callback * forward;   // user's real cb (may be NULL)
    struct chars                 content;
    struct chars                 reasoning;
};

static int slm_capture_cb_fn(const struct slm_stream_callback * cb) {
    struct slm_capture_box * me = (struct slm_capture_box *)cb;
    if (cb->content != NULL) {
        chars_puts(&me->content, cb->content);
    }
    if (cb->reasoning != NULL) {
        chars_puts(&me->reasoning, cb->reasoning);
    }
    int rc = 0;
    if (me->forward != NULL && me->forward->callback != NULL) {
        rc = slm_cb_emit(me->forward,
                         cb->content, cb->reasoning,
                         cb->call, cb->response,
                         cb->prefilled, cb->pp_pos, cb->pp_total);
    }
    return rc;
}

// ---------------------------------------------------------------------------
// Think-filter self-test
// ---------------------------------------------------------------------------

struct slm_think_test_capture {
    char content  [512];
    int  content_n;
    char reasoning[512];
    int  reasoning_n;
};

struct slm_think_test_cb_box {
    struct slm_stream_callback base;
    struct slm_think_test_capture * cap;
};

static int slm_think_test_cb(const struct slm_stream_callback * cb) {
    struct slm_think_test_cb_box * box =
        (struct slm_think_test_cb_box *)cb;
    struct slm_think_test_capture * cap = box->cap;
    if (cb->content != NULL) {
        int n     = (int)strlen(cb->content);
        int avail = (int)sizeof(cap->content) - cap->content_n - 1;
        int copy  = n < avail ? n : avail;
        if (copy > 0) {
            memcpy(cap->content + cap->content_n,
                   cb->content, (size_t)copy);
            cap->content_n += copy;
            cap->content[cap->content_n] = '\0';
        }
    }
    if (cb->reasoning != NULL) {
        int n     = (int)strlen(cb->reasoning);
        int avail = (int)sizeof(cap->reasoning) - cap->reasoning_n - 1;
        int copy  = n < avail ? n : avail;
        if (copy > 0) {
            memcpy(cap->reasoning + cap->reasoning_n,
                   cb->reasoning, (size_t)copy);
            cap->reasoning_n += copy;
            cap->reasoning[cap->reasoning_n] = '\0';
        }
    }
    return 0;
}

static int slm_think_test_case(const char * name,
                               const char ** chunks, int n_chunks,
                               const char * want_content,
                               const char * want_reasoning) {
    int failed = 0;
    struct slm_think_filter f = {0};
    slm_think_filter_init(&f);
    struct slm_think_test_capture cap;
    cap.content_n   = 0;
    cap.content[0]  = '\0';
    cap.reasoning_n = 0;
    cap.reasoning[0] = '\0';
    struct slm_think_test_cb_box box = {0};
    box.base.callback = slm_think_test_cb;
    box.cap           = &cap;
    for (int i = 0; i < n_chunks; i++) {
        slm_think_filter_push(&f, chunks[i], &box.base);
    }
    slm_think_filter_finish(&f, &box.base);
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

static int32_t slm_think_test(void) {
    int failures = 0;
    {
        const char * chunks[] = { "Hello, world." };
        failures += slm_think_test_case("A", chunks, 1,
                                        "Hello, world.", "");
    }
    {
        const char * chunks[] = {
            "</think>\n\nHello."
        };
        failures += slm_think_test_case("B", chunks, 1,
                                        "\n\nHello.", "");
    }
    {
        const char * chunks[] = {
            "<think>let me think</think>\n\nAnswer."
        };
        failures += slm_think_test_case("C", chunks, 1,
                                        "\n\nAnswer.",
                                        "let me think");
    }
    {
        const char * chunks[] = {
            "abc<thi", "nk>secret</thi", "nk>end"
        };
        failures += slm_think_test_case("D", chunks, 3,
                                        "abcend", "secret");
    }
    {
        const char * chunks[] = {
            "x<y<not_a_marker>z"
        };
        failures += slm_think_test_case("E", chunks, 1,
                                        "x<y<not_a_marker>z", "");
    }
    {
        const char * src = "a<think>b</think>c";
        const char * chunks[32];
        int n = (int)strlen(src);
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

int  slm_tokenize(struct slm_ctx * c, const char * text,
                  int32_t ** out_ids) {
    struct slm_tokens tmp = {0};
    tokenizer_encode(&c->model->tok, text, &tmp);
    *out_ids = tmp.data;     // transfer ownership; caller frees
    return (int)tmp.count;
}

int  slm_vocab_size(const struct slm_ctx * c) { return c->model->cfg.vocab_size; }
int  slm_eos_id    (const struct slm_ctx * c) { return c->model->cfg.eos_id; }
int  slm_bos_id    (const struct slm_ctx * c) { return c->model->cfg.bos_id; }

// ---------------------------------------------------------------------------

// Agent helpers (parser + tool dispatcher) — `#include`-d here so
// slm_generate's embedded agent loop can reach them.
#include "agent.c"

// ---------------------------------------------------------------------------
// slm_ctx_system_prompt - prefill the system framing into the model.
// Must be called once before any slm_generate call. Formats the
// canonical Qwen3 system block, tokenizes, appends to ctx->ids, and
// runs the prefill loop. Fires pp_pos/pp_total progress chunks via
// cb during prefill; fires a single prefilled=true chunk on success.
// Sets ctx->system_committed = true on success.
// ---------------------------------------------------------------------------
bool slm_ctx_system_prompt(struct slm_ctx * ctx,
                           const char * text,
                           struct slm_stream_callback * cb) {
    bool committed = false;
    if (ctx != NULL && !ctx->system_committed) {
        const char * sys_text = (text != NULL) ? text : "";
        // Build tools array when ctrl.tools is set.
        struct jinja_tool tools[3];
        int n_tools = 0;
        if (ctx->ctrl.tools) {
            tools[0].json = AGENT_TOOL_WEBSEARCH;
            tools[1].json = AGENT_TOOL_FETCH;
            tools[2].json = AGENT_TOOL_DISTILL;
            n_tools = 3;
        }
        // Format the canonical system block via jinja helper.
        char * block = jinja_format_system_block(sys_text, &ctx->ctrl,
                                                 tools, n_tools);
        if (block != NULL) {
            if (ctx->ctrl.debug >= 9) {
                slm_trace_long("system block",
                               block, strlen(block));
            }
            int32_t * ids  = NULL;
            int       n    = slm_tokenize(ctx, block, &ids);
            free(block);
            if (n > 0 && ids != NULL) {
                slm_ctx_ids_append(ctx, ids, (size_t)n);
                // Push the raw system text (not the framed bytes) into
                // messages so the UI can render it if it wants.
                text_puts(&ctx->messages, sys_text);
                // Run prefill loop. Fire pp_pos/pp_total per token.
                bool aborted = false;
                for (int32_t i = 0; i < n && !aborted; i++) {
                    struct tensor * logits =
                        slm_forward_step(ctx, ids[i], ctx->pos);
                    (void)logits;
                    ctx->pos++;
                    if (cb != NULL) {
                        int r = slm_cb_emit(cb, NULL, NULL, NULL, NULL,
                                            false, i + 1, n);
                        if (r != 0) { aborted = true; }
                    }
                }
                if (!aborted) {
                    slm_cb_emit(cb, NULL, NULL, NULL, NULL,
                                /*prefilled=*/true, 0, 0);
                    ctx->system_committed = true;
                    committed = true;
                }
            }
            free(ids);
        }
    }
    return committed;
}

// ---------------------------------------------------------------------------
// slm_generate - one user-turn chat round. Frames the delta as
// canonical Qwen3 form, tokenizes, prefills, decodes. Streams via cb.
// Returns the number of decode tokens emitted.
//
// With ctrl.tools the runtime takes a snapshot of the pre-turn ctx
// before any prefill. If the model emits a <tool_call>, we dispatch
// it internally, ROLL BACK the ctx (KV/SSM/pos/ids), and replay the
// turn with the curated result spliced in as preamble — so the
// model continues its assistant reply with the search content baked
// into in-context history instead of round-tripping through a
// synthetic <tool_response> user turn. The 0.8B is much steadier on
// "answer using this preamble" than on "interpret a tool_response
// then continue".
//
// Cap of 2 iterations is a hard ceiling: one pass that may emit a
// tool_call, one pass that consumes the preamble and writes the
// final answer. `snap != NULL` doubles as the "tool round still
// available" flag — after dispatch we free and NULL it.
// ---------------------------------------------------------------------------
#define SLM_GENERATE_TOOL_ITER_CAP 2
int slm_generate(struct slm_ctx * c,
                 const char * prompt,
                 int max_new, int min_new,
                 const struct slm_sampler * sampler_in,
                 uint64_t seed,
                 struct slm_stream_callback * cb) {
    assert(c != NULL && c->system_committed);
    bool    with_tools = c->ctrl.tools;
    bool    with_debug = (c->ctrl.debug > 0);
    int32_t debug_lv   = c->ctrl.debug;
    int     total_gen  = 0;
    // Push user prompt into messages — not part of the rollback set.
    const char * user_text = (prompt != NULL) ? prompt : "";
    text_puts(&c->messages, user_text);
    if (debug_lv >= 1) {
        size_t qn = strlen(user_text);
        trace("turn: prompt %zu chars, tools=%d, think=%d\n",
              qn, with_tools ? 1 : 0, c->ctrl.think ? 1 : 0);
    }
    if (debug_lv >= 7 && debug_lv < 9) {
        size_t qn = strlen(user_text);
        size_t cut = qn > 240 ? 240 : qn;
        trace("turn: prompt[0..%zu]: %.*s%s\n",
              cut, (int)cut, user_text, cut < qn ? "..." : "");
    }
    if (debug_lv >= 9) {
        slm_trace_long("turn: user prompt",
                       user_text, strlen(user_text));
    }
    // At debug >= 9 wrap the user's cb so we can re-emit the full
    // content + reasoning streams via slm_trace_long at end of turn.
    struct slm_capture_box capture = {0};
    capture.base.callback          = slm_capture_cb_fn;
    capture.forward                = cb;
    struct slm_stream_callback * effective_cb =
        (debug_lv >= 9) ? &capture.base : cb;
    // Snapshot taken at the model's "decision point" — after the user
    // delta is prefilled into KV but BEFORE the decode loop starts.
    // Set by slm_snap_capture_fn via slm_generate_raw's after_prefill
    // hook. Only filled on the first iter; restored + freed on
    // tool_call dispatch.
    struct slm_snapshot *    snap       = NULL;
    bool                     tool_round = with_tools;
    struct slm_snap_capture  snap_hook  = {
        .base = { .on_prefilled = slm_snap_capture_fn },
        .slot = &snap,
    };
    // Format the user turn delta and tokenize it into a growing token
    // buffer that we reuse across iters (each iter resets count=0
    // before refilling).
    char *            delta = jinja_format_turn_delta(user_text,
                                                      c->ctrl.think);
    struct slm_tokens cur   = {0};
    if (delta != NULL) {
        tokenizer_encode(&c->model->tok, delta, &cur);
        slm_ctx_ids_append(c, cur.data, cur.count);
        free(delta);
    }
    bool done = false;
    int  iter = 0;
    while (!done && iter < SLM_GENERATE_TOOL_ITER_CAP &&
           total_gen < max_new) {
        struct slm_split_box box = {0};
        slm_think_filter_init(&box.filter);
        box.filter.recognize_tool_calls = tool_round;
        box.filter.emit_visibility      = with_debug;
        box.cb = effective_cb;
        int budget = max_new - total_gen;
        if (debug_lv >= 3) {
            trace("agent: iter=%d pos=%d prefill_n=%zu budget=%d"
                  " tool_round=%d\n",
                  iter, c->pos, cur.count, budget, tool_round ? 1 : 0);
        }
        // On the tool-eligible iter, the snap_hook functor fires
        // after prefill (KV holds user delta + assistant prefix; pos
        // is at the decision point) and writes the captured snapshot
        // into `snap`.
        const struct slm_after_prefill * hook =
            (tool_round && snap == NULL) ? &snap_hook.base : NULL;
        int n = slm_generate_raw(c, cur.data, (int32_t)cur.count,
                                 budget, min_new,
                                 sampler_in, seed,
                                 slm_split_trampoline, &box,
                                 effective_cb, hook);
        if (debug_lv >= 1 && snap != NULL && iter == 0) {
            trace("snapshot: post-prefill pos=%d ids=%zu\n",
                  snap->pos, snap->ids_count);
        }
        total_gen += n;
        slm_think_filter_finish(&box.filter, effective_cb);
        cur.count = 0;       // reuse buffer; freed at function exit
        if (snap != NULL && box.filter.tool_call_ready &&
            total_gen < max_new) {
            // Parse + dispatch the tool synchronously.
            struct chars wrapped = {0};
            chars_puts(&wrapped, "<tool_call>");
            if (box.filter.tool_call.count > 0) {
                chars_put(&wrapped, box.filter.tool_call.data,
                          box.filter.tool_call.count);
            }
            chars_puts(&wrapped, "</tool_call>");
            chars_put(&wrapped, "", 0);
            struct agent_call calls[4];
            int nc = agent_parse_tool_calls(wrapped.data, calls, 4);
            if (debug_lv >= 1) {
                trace("tool_call: %d call(s) parsed\n", nc);
            }
            if (debug_lv >= 3) {
                for (int i = 0; i < nc; i++) {
                    trace("tool_call[%d]: %s\n", i,
                          calls[i].name != NULL ? calls[i].name : "?");
                    for (int p = 0; p < calls[i].n_params; p++) {
                        const char * v = calls[i].params[p].value;
                        size_t vlen = v != NULL ? strlen(v) : 0;
                        size_t vcut = vlen > 160 ? 160 : vlen;
                        trace("  %s = %.*s%s\n",
                              calls[i].params[p].name,
                              (int)vcut, v != NULL ? v : "",
                              vcut < vlen ? "..." : "");
                    }
                }
            }
            chars_free(&wrapped);
            // Propagate debug level into the tools.c primitives for
            // the duration of this dispatch round; reset to silent
            // immediately after so unrelated callers don't inherit it.
            tools_set_debug(debug_lv);
            struct chars result = {0};
            for (int i = 0; i < nc; i++) {
                struct tool_result r = {0};
                agent_dispatch(&calls[i], &r);
                const char * payload =
                    r.ok && r.body != NULL ? r.body
                  : r.error != NULL        ? r.error
                                           : "(no result)";
                if (debug_lv >= 1) {
                    trace("tool_call[%d]: %s -> %s (%zu bytes)\n",
                          i, calls[i].name != NULL ? calls[i].name : "?",
                          r.ok ? "ok" : "ERR",
                          payload != NULL ? strlen(payload) : 0);
                }
                if (i > 0) { chars_put(&result, "\n\n", 2); }
                chars_put(&result, payload, strlen(payload));
                tools_result_free(&r);
            }
            tools_set_debug(0);
            chars_put(&result, "", 0);
            agent_free_calls(calls, nc);
            if (result.data != NULL && with_debug) {
                text_puts(&c->messages, result.data);
                slm_cb_emit(effective_cb, NULL, NULL, NULL, result.data,
                            false, 0, 0);
            }
            // Atomic rollback: tool_call emission is wiped from KV/
            // SSM/pos/ids back to the post-prefill snapshot point.
            // The user delta + assistant prefix REMAIN in KV (they
            // were prefilled before the snapshot fired), so we only
            // need to append the curated preamble — no re-tokenize
            // of the original turn delta.
            if (debug_lv >= 1) {
                trace("restore: rolling ctx back to pos=%d\n",
                      snap->pos);
            }
            slm_ctx_restore(c, snap);
            slm_snapshot_free(snap);
            snap       = NULL;
            tool_round = false;     // one tool round per turn
            // Preamble: the curated search results followed by a
            // direct request to summarize. Continues the assistant
            // turn (KV ends at <think></think>\n\n after restore).
            struct chars inject = {0};
            chars_printf(&inject,
                "I searched the web. Here are the results:\n\n%s\n\n"
                "Based on the above, here is a concise answer to the"
                " user's question:\n\n",
                result.data ? result.data : "");
            chars_put(&inject, "", 0);
            if (debug_lv >= 1) {
                trace("preamble: %zu chars, replaying turn with"
                      " curated content\n", inject.count);
            }
            if (debug_lv >= 7 && debug_lv < 9) {
                size_t pcut = inject.count > 240 ? 240 : inject.count;
                trace("preamble[0..%zu]: %.*s%s\n",
                      pcut, (int)pcut, inject.data,
                      pcut < inject.count ? "..." : "");
            }
            if (debug_lv >= 9) {
                slm_trace_long("preamble (full)",
                               inject.data, inject.count);
            }
            tokenizer_encode(&c->model->tok, inject.data, &cur);
            slm_ctx_ids_append(c, cur.data, cur.count);
            chars_free(&inject);
            chars_free(&result);
            // Reset the visible-tokens counter: the aborted tool_call
            // emission doesn't count toward max_new. The second pass
            // gets the full budget for the actual answer.
            total_gen = 0;
        } else {
            if (debug_lv >= 1 && snap != NULL) {
                // Diagnose the "tools on but nothing happened" case:
                // the model finished without emitting a tool_call.
                // Common cause: sampler temperature > 0 on the 0.8B
                // makes <tool_call> JSON shatter — the cli auto-
                // clamps to T=0; UI / library callers should too.
                trace("agent: iter=%d ended without tool_call"
                      " (n=%d, max_new=%d, T=%.2f)\n",
                      iter, n, max_new,
                      sampler_in != NULL ? sampler_in->temperature
                                         : 0.0f);
            }
            done = true;
        }
        chars_free(&box.filter.tool_call);
        iter++;
    }
    slm_tokens_free(&cur);
    slm_snapshot_free(snap);
    if (debug_lv >= 9) {
        slm_trace_long("turn: model reasoning",
                       capture.reasoning.data, capture.reasoning.count);
        slm_trace_long("turn: model content",
                       capture.content.data,   capture.content.count);
        chars_free(&capture.reasoning);
        chars_free(&capture.content);
    }
    return total_gen;
}

// ---------------------------------------------------------------------------
// Self-test battery — public entry point. Lives outside the LLM_CLI
// gate so Swift / library consumers (Debug tab "Run Tests" button)
// can call it. Each test function logs its progress via stderr + the
// trace ring; we just sum up the failure counts and return.
// ---------------------------------------------------------------------------
int slm_run_all_tests(void) {
    trace("running C self-test battery\n");
    tools_global_init();
    int failures = 0;
    failures += (qwen_self_test()     != 0) ? 1 : 0;
    failures += (chunked_self_test()  != 0) ? 1 : 0;
    failures += (jinja_self_test()    != 0) ? 1 : 0;
    failures += (agent_parser_test()  != 0) ? 1 : 0;
    failures += (tools_self_test()    != 0) ? 1 : 0;
    failures += (slm_think_test()     != 0) ? 1 : 0;
    if (failures == 0) {
        trace("self-test battery: PASS (6 suites)\n");
    } else {
        trace("self-test battery: %d / 6 suites FAILED\n", failures);
    }
    tools_global_cleanup();
    return failures;
}

#ifdef LLM_CLI

// ---------------------------------------------------------------------------
// CLI - main() and friends.
// ---------------------------------------------------------------------------

static const char * slm_cli_gguf_path(void) {
    const char * env = getenv("QWEN_GGUF");
    if (env != NULL && env[0] != '\0') { return env; }
    return SLM_GGUF_PATH_DEFAULT;
}

// print_cb: handles all five chunk types so the `--tools` cascade and
// `--repl` agent loop don't silently swallow tool dialogue. Construct
// with `pbox.verbose = 1` to flip every channel to stdout instead of
// stderr (so a piped consumer sees everything).
struct print_cb_box {
    struct slm_stream_callback base;
    int                        verbose;
};

static int repl_cb_fn(const struct slm_stream_callback * cb) {
    const struct print_cb_box * box = (const struct print_cb_box *)cb;
    FILE * side = box->verbose ? stdout : stderr;
    if (cb->content != NULL) {
        // Content always goes to stdout (it's the visible reply).
        fputs(cb->content, stdout);
        fflush(stdout);
    }
    if (cb->reasoning != NULL) {
        fprintf(side, "[think] %s", cb->reasoning);
        fflush(side);
    }
    if (cb->call != NULL) {
        fprintf(side, "\n[tool: %s]\n", cb->call);
        fflush(side);
    }
    if (cb->response != NULL) {
        // Cap visible payload at 400 chars so a 200 KB fetch body
        // doesn't drown the terminal.
        size_t n = strlen(cb->response);
        size_t show = n > 400 ? 400 : n;
        fprintf(side, "\n[result: %.*s%s]\n",
                (int)show, cb->response,
                n > show ? " …(truncated)" : "");
        fflush(side);
    }
    return 0;
}

// capture_cb: accumulate content into `chars` via that pointer.
struct capture_cb_box {
    struct slm_stream_callback base;
    struct chars * out;
};

static int capture_cb_fn(const struct slm_stream_callback * cb) {
    struct capture_cb_box * box = (struct capture_cb_box *)cb;
    if (cb->content != NULL) {
        chars_put(box->out, cb->content, strlen(cb->content));
    }
    return 0;
}

// Strip reasoning content from a captured assistant reply before
// feeding it back into the framed history. Mirrors the Qwen3 Jinja:
//     content = content.split('</think>')[-1].lstrip('\n')
static void strip_reasoning_for_history(struct chars * s) {
    if (s->data == NULL || s->count == 0) { return; }
    const char close[] = "</think>";
    size_t close_len   = sizeof(close) - 1;
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

// Scripted multi-turn chat (-p "Q1" -p "Q2" ...). One ctx, all turns
// appended (growing-KV). ctrl.tools is set from with_tools — when on,
// slm_generate's embedded snapshot/restore/preamble flow handles any
// <tool_call> the model emits atomically; the caller-visible answer
// is the post-tool-dispatch final answer for that turn.
static int32_t run_chat(const char ** prompts, int32_t n_prompts,
                        const char * system_prompt,
                        const struct slm_sampler * sp, uint64_t seed,
                        int32_t max_new, int32_t min_new,
                        int32_t dump_layer, int32_t verbose,
                        int32_t debug_level,
                        bool with_tools, bool with_think,
                        bool trace_tokens) {
    (void)verbose;
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t r = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        struct slm_ctx * c = slm_ctx_create(m);
        if (c == NULL) {
            fprintf(stderr, "slm: ctx create failed\n");
            r = 1;
        } else {
            const char * sys = (system_prompt != NULL) ? system_prompt : "";
            slm_ctx_ctrl(c)->tools        = with_tools;
            slm_ctx_ctrl(c)->think        = with_think;
            slm_ctx_ctrl(c)->trace_tokens = trace_tokens;
            slm_ctx_ctrl(c)->dump_layer   = dump_layer;
            slm_ctx_ctrl(c)->debug        = debug_level;
            slm_ctx_system_prompt(c, sys, NULL);
            struct capture_cb_box capbox = {0};
            capbox.base.callback = capture_cb_fn;
            for (int32_t t = 0; t < n_prompts && r == 0; t++) {
                printf("\n--- turn %d/%d ---\n",
                       (int)(t + 1), (int)n_prompts);
                if (t == 0 && system_prompt != NULL &&
                    system_prompt[0] != '\0') {
                    printf("[system] %s\n", system_prompt);
                }
                printf("[user] %s\n[assistant] ", prompts[t]);
                fflush(stdout);
                struct chars reply = {0};
                capbox.out = &reply;
                slm_generate(c, prompts[t], max_new, min_new,
                             sp, seed, &capbox.base);
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
                    "pp: %.2f tok/s (%d tok)  tg: %.2f tok/s (%d tok)"
                    "  kv pos=%d\n",
                    slm_pp_per_sec(c), (int)slm_n_prefill(c),
                    slm_tg_per_sec(c), (int)slm_n_generated(c),
                    (int)c->pos);
                chars_free(&reply);
            }
            slm_ctx_destroy(c);
        }
    }
    slm_model_unload(m);
    return r;
}

// chat_test_cb: functor for run_chat_test.
struct chat_test_capture {
    struct chars text;
    uint64_t     hash;
};

struct chat_test_cb_box {
    struct slm_stream_callback base;
    struct chat_test_capture * cap;
};

static int chat_test_cb_fn(const struct slm_stream_callback * cb) {
    struct chat_test_cb_box * box = (struct chat_test_cb_box *)cb;
    struct chat_test_capture * cap = box->cap;
    const char * piece = (cb->content != NULL)   ? cb->content
                       : (cb->reasoning != NULL)  ? cb->reasoning
                                                  : NULL;
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
// twice — once on a fresh context, then again on a second fresh
// context — and verifies the reply hashes match across the two
// passes. This exercises the chat-template state machine, the
// persistent KV cache, and the slm_ctx_system_prompt + slm_generate
// lifecycle.
//
// The hashes WILL differ from the old slm_reset-based gate because
// the system framing changed. Print them; the new set is the gate.
static int32_t run_chat_test(int32_t max_new, int32_t min_new,
                             int32_t dump_layer, bool trace_tokens) {
    static const char * const turns[] = {
        "Hi! Just say hello back.",
        "What did I just ask?",
        "Thanks!"
    };
    enum { N_TURNS = 3 };
    struct slm_sampler sp = slm_sampler_defaults();
    uint64_t seed = 42;
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t r = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        struct chat_test_capture caps_a[N_TURNS];
        struct chat_test_capture caps_b[N_TURNS];
        memset(caps_a, 0, sizeof(caps_a));
        memset(caps_b, 0, sizeof(caps_b));
        struct chat_test_cb_box cbbox = {0};
        cbbox.base.callback = chat_test_cb_fn;
        for (int32_t pass = 0; pass < 2 && r == 0; pass++) {
            struct chat_test_capture * caps = (pass == 0) ? caps_a : caps_b;
            // Create a fresh ctx for each pass (replaces slm_reset).
            struct slm_ctx * c = slm_ctx_create(m);
            if (c == NULL) {
                fprintf(stderr, "slm: ctx create failed\n");
                r = 1;
            } else {
                slm_ctx_system_prompt(c, "", NULL);
                slm_ctx_ctrl(c)->trace_tokens = trace_tokens;
                slm_ctx_ctrl(c)->dump_layer   = dump_layer;
                for (int32_t t = 0; t < N_TURNS && r == 0; t++) {
                    struct chat_test_capture * cap = &caps[t];
                    cap->hash = 0xcbf29ce484222325ULL;
                    cbbox.cap = cap;
                    int32_t gen = slm_generate(c, turns[t], max_new,
                                               min_new, &sp, seed,
                                               &cbbox.base);
                    chars_put(&cap->text, "", 0);
                    fprintf(stderr,
                            "chat-test pass %d turn %d: gen=%d "
                            "hash=%016llx pos=%d\n",
                            (int)pass, (int)(t + 1), (int)gen,
                            (unsigned long long)cap->hash,
                            (int)c->pos);
                }
                slm_ctx_destroy(c);
            }
        }
        int32_t mismatch = -1;
        for (int32_t t = 0; t < N_TURNS && mismatch < 0; t++) {
            if (caps_a[t].hash != caps_b[t].hash) { mismatch = t; }
        }
        if (mismatch < 0) {
            printf("chat-test: PASS (3 turns x 2 passes,"
                   " hashes match across ctx recreate)\n");
            printf("NEW_GATE_HASHES = %016llx / %016llx / %016llx\n",
                   (unsigned long long)caps_a[0].hash,
                   (unsigned long long)caps_a[1].hash,
                   (unsigned long long)caps_a[2].hash);
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
    slm_model_unload(m);
    return r;
}

static int32_t run_ask(const char * question,
                       const struct slm_sampler * sp, uint64_t seed,
                       int32_t max_iters, int32_t max_new,
                       int32_t trace) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t rc = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        rc = 1;
    } else {
        printf("[user] %s\n", question);
        fflush(stdout);
        char * answer = agent_run(m, question, sp, seed,
                                  max_iters, max_new, trace);
        printf("\n[assistant] %s\n", answer != NULL ? answer : "");
        free(answer);
    }
    slm_model_unload(m);
    tools_global_cleanup();
    return rc;
}

static int32_t run_single(const char * prompt, int32_t max_new,
                          int32_t min_new, int32_t dump_layer,
                          int32_t verbose,
                          int32_t debug_level,
                          const struct slm_sampler * sp, uint64_t seed,
                          bool with_tools, bool with_think,
                          bool trace_tokens) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t r = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        struct slm_ctx * c = slm_ctx_create(m);
        if (c == NULL) {
            fprintf(stderr, "slm: ctx create failed\n");
            r = 1;
        } else {
            printf("model: %d layers, %d heads (%d kv), head_dim=%d, "
                   "hidden=%d, ffn=%d, vocab=%d\n",
                   (int)c->model->cfg.n_layers,
                   (int)c->model->cfg.n_heads,
                   (int)c->model->cfg.n_kv_heads,
                   (int)c->model->cfg.head_dim,
                   (int)c->model->cfg.hidden_dim,
                   (int)c->model->cfg.ffn_dim,
                   (int)c->model->cfg.vocab_size);
            printf("rope: dim=%d base=%.1f sections=[%d,%d,%d,%d]\n",
                   (int)c->model->cfg.rope_dim,
                   c->model->cfg.rope_theta,
                   (int)c->model->cfg.rope_sections[0],
                   (int)c->model->cfg.rope_sections[1],
                   (int)c->model->cfg.rope_sections[2],
                   (int)c->model->cfg.rope_sections[3]);
            slm_ctx_ctrl(c)->tools        = with_tools;
            slm_ctx_ctrl(c)->think        = with_think;
            slm_ctx_ctrl(c)->trace_tokens = trace_tokens;
            slm_ctx_ctrl(c)->dump_layer   = dump_layer;
            slm_ctx_ctrl(c)->debug        = debug_level;
            slm_ctx_system_prompt(c, "", NULL);
            struct print_cb_box pbox = {0};
            pbox.base.callback = repl_cb_fn;
            pbox.verbose       = verbose;
            printf("---\n%s", prompt);
            fflush(stdout);
            slm_generate(c, prompt, max_new, min_new, sp, seed,
                         &pbox.base);
            printf("\n");
            fprintf(stderr,
                    "pp: %.2f tok/s (%d tok)  tg: %.2f tok/s (%d tok)\n",
                    slm_pp_per_sec(c), (int)slm_n_prefill(c),
                    slm_tg_per_sec(c), (int)slm_n_generated(c));
            slm_ctx_destroy(c);
        }
    }
    slm_model_unload(m);
    return r;
}

static int32_t run_bench(int32_t dump_layer, bool trace_tokens) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t r = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        const char * prompt =
            "Explain in five short sentences why benchmarking large "
            "language models across diverse hardware matters. Avoid "
            "vague generalities; be concrete.";
        const int32_t bench_max_new = 64;
        struct slm_sampler sp = slm_sampler_defaults();
        sp.temperature = 0.0f;
        const int N = 3;
        float pps[N], tgs[N];
        int32_t n_pp = 0, n_tg = 0;
        for (int run = 0; run < N; run++) {
            // Create a fresh ctx for each run (replaces slm_reset).
            struct slm_ctx * c = slm_ctx_create(m);
            if (c != NULL) {
                slm_ctx_system_prompt(c, "", NULL);
                slm_ctx_ctrl(c)->trace_tokens = trace_tokens;
                slm_ctx_ctrl(c)->dump_layer   = dump_layer;
                slm_generate(c, prompt, bench_max_new, 0, &sp,
                             /*seed=*/1u, /*cb=*/NULL);
                pps[run] = (float)slm_pp_per_sec(c);
                tgs[run] = (float)slm_tg_per_sec(c);
                n_pp     = (int32_t)slm_n_prefill(c);
                n_tg     = (int32_t)slm_n_generated(c);
                fprintf(stderr, "  run %d: pp=%.2f tg=%.2f\n",
                        run + 1, pps[run], tgs[run]);
                slm_ctx_destroy(c);
            }
        }
        #define SORT3(a) do {                                           \
            if ((a)[0] > (a)[1]) {                                      \
                float t = (a)[0]; (a)[0] = (a)[1]; (a)[1] = t;          \
            }                                                           \
            if ((a)[1] > (a)[2]) {                                      \
                float t = (a)[1]; (a)[1] = (a)[2]; (a)[2] = t;          \
            }                                                           \
            if ((a)[0] > (a)[1]) {                                      \
                float t = (a)[0]; (a)[0] = (a)[1]; (a)[1] = t;          \
            }                                                           \
        } while (0)
        SORT3(pps);
        SORT3(tgs);
        #undef SORT3
        printf("bench: pp=%.2f tg=%.2f n_pp=%d n_tg=%d dispatch=%s\n",
               pps[1], tgs[1], n_pp, n_tg, simd_dispatch_label());
    }
    slm_model_unload(m);
    return r;
}

// REPL: growing-KV chat via slm_generate. Sys prompt = "You are a
// helpful assistant." With ctrl.tools the model can emit a single
// <tool_call> per turn that slm_generate dispatches internally
// (snapshot/restore/preamble) before writing the visible answer —
// from the user's perspective each turn is just "ask, get reply".
static int32_t run_repl(const struct slm_sampler * sp,
                        uint64_t seed, int32_t max_new, int32_t min_new,
                        int32_t dump_layer, int32_t verbose,
                        int32_t debug_level,
                        bool with_tools, bool with_think,
                        bool trace_tokens) {
    struct slm_model * m = slm_model_load(slm_cli_gguf_path());
    int32_t r = 0;
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "slm: load failed: %s\n", slm_model_error(m));
        r = 1;
    } else {
        struct slm_ctx * c = slm_ctx_create(m);
        if (c == NULL) {
            fprintf(stderr, "slm: ctx create failed\n");
            r = 1;
        } else {
            slm_ctx_ctrl(c)->tools        = with_tools;
            slm_ctx_ctrl(c)->think        = with_think;
            slm_ctx_ctrl(c)->trace_tokens = trace_tokens;
            slm_ctx_ctrl(c)->dump_layer   = dump_layer;
            slm_ctx_ctrl(c)->debug        = debug_level;
            slm_ctx_system_prompt(c, "You are a helpful assistant.",
                                  NULL);
            struct print_cb_box pbox = {0};
            pbox.base.callback = repl_cb_fn;
            pbox.verbose       = verbose;
            char line[4096];
            printf("repl: type a message, Enter to send,"
                   " Ctrl-D to exit.\n> ");
            fflush(stdout);
            while (fgets(line, sizeof(line), stdin) != NULL) {
                size_t n = strlen(line);
                while (n > 0 &&
                       (line[n-1] == '\n' || line[n-1] == '\r')) {
                    line[--n] = '\0';
                }
                if (n > 0) {
                    printf("\nassistant: ");
                    fflush(stdout);
                    slm_generate(c, line, max_new, min_new, sp, seed,
                                 &pbox.base);
                    printf("\n\n> ");
                    fflush(stdout);
                }
            }
            slm_ctx_destroy(c);
        }
    }
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
        CLI_ALL_TESTS       = 14,
    };
    enum cli_mode mode = CLI_HELP;
    const char * prompt         = "Hello, my name is";
    const char * system_prompt  = NULL;
    const char * chat_prompts[LLM_CLI_MAX_TURNS];
    int32_t      chat_n         = 0;
    struct slm_sampler sp = slm_sampler_defaults();
    uint64_t seed         = 0;
    int32_t  max_new      = 64;
    int32_t  min_new      = 0;         // --min-new N (replaces g_min_new)
    int32_t  max_iters    = 4;
    int32_t  trace_agent  = 0;
    int32_t  dump_layer   = -1;        // --dump-layer L (replaces g_dump_layer)
    bool     with_tools   = false;     // CLI --tools / --no-tools
    bool     with_think   = false;     // CLI --think / --no-think
    int32_t  verbose      = 0;         // CLI --verbose
    int32_t  debug_level  = 1;         // CLI --debug N (0..9)
    bool     repl_max_new_set = false; // user set --max-new explicitly
    for (int32_t i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            mode = CLI_SELF_TEST;
        } else if (strcmp(argv[i], "--chunked-test") == 0) {
            mode = CLI_CHUNKED_TEST;
        } else if (strcmp(argv[i], "--single") == 0) {
            mode = CLI_SINGLE;
            // Only consume the next argv as the prompt if it isn't
            // itself a flag — lets the user write
            // `--single --verbose "Q?"` or `--verbose --single "Q?"`
            // interchangeably without losing the prompt to flag-eating.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                prompt = argv[++i];
            }
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
        } else if (strcmp(argv[i], "--all-tests") == 0) {
            mode = CLI_ALL_TESTS;
        } else if (strcmp(argv[i], "--ask") == 0 && i + 1 < argc) {
            mode = CLI_ASK;
            if (argv[i + 1][0] != '-') { prompt = argv[++i]; }
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
            repl_max_new_set = true;
        } else if (strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
            max_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace_agent = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--debug") == 0 && i + 1 < argc) {
            int v = atoi(argv[++i]);
            if (v < 0) { v = 0; }
            if (v > 9) { v = 9; }
            debug_level = (int32_t)v;
        } else if (strcmp(argv[i], "--tools") == 0) {
            with_tools = true;
        } else if (strcmp(argv[i], "--no-tools") == 0) {
            with_tools = false;
        } else if (strcmp(argv[i], "--think") == 0) {
            with_think = true;
        } else if (strcmp(argv[i], "--no-think") == 0) {
            with_think = false;
        } else if (strcmp(argv[i], "--min-new") == 0 && i + 1 < argc) {
            min_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dump-layer") == 0 && i + 1 < argc) {
            dump_layer = atoi(argv[++i]);
        } else if (argv[i][0] != '-' &&
                   (mode == CLI_SINGLE || mode == CLI_ASK)) {
            // Bare positional argument — treat as the prompt for
            // --single / --ask regardless of flag order. Lets the user
            // write `--single --verbose --tools "Q?"` or any other
            // permutation without losing the prompt.
            prompt = argv[i];
        }
    }
    // REPL default max_new bumped from 64 to 256 so one tool round-
    // trip + a real reply fits. User --max-new still overrides.
    if (mode == CLI_REPL && !repl_max_new_set) { max_new = 256; }
    // When --tools is on, auto-clamp the sampler per Tool-Calling.md
    // §3 — structural <tool_call> JSON shatters at T=0.7. User can
    // still override via --temperature / --top-p / --rep-penalty
    // AFTER --tools on the same line.
    if (with_tools) {
        if (sp.temperature == 0.7f) { sp.temperature = 0.0f;  }
        if (sp.top_p       == 0.9f) { sp.top_p       = 1.0f;  }
        if (sp.repetition_penalty == 1.25f) {
            sp.repetition_penalty = 1.0f;
        }
    }
    g_no_q8k_rt       = getenv("NO_Q8K_RT")       != NULL;
    bool trace_tokens = getenv("LLM_TRACE_TOKENS") != NULL;
    int rc = 0;
    if (mode == CLI_SELF_TEST) {
        rc = qwen_self_test();
    } else if (mode == CLI_CHUNKED_TEST) {
        rc = chunked_self_test();
    } else if (mode == CLI_SINGLE) {
        rc = run_single(prompt, max_new, min_new, dump_layer, verbose,
                        debug_level,
                        &sp, seed, with_tools, with_think, trace_tokens);
    } else if (mode == CLI_REPL) {
        rc = run_repl(&sp, seed, max_new, min_new, dump_layer, verbose,
                      debug_level,
                      with_tools, with_think, trace_tokens);
    } else if (mode == CLI_CHAT) {
        if (chat_n == 0) {
            fprintf(stderr, "slm: --chat needs at least one -p/--prompt\n");
            rc = 1;
        } else {
            rc = run_chat(chat_prompts, chat_n, system_prompt,
                          &sp, seed, max_new, min_new, dump_layer,
                          verbose, debug_level,
                          with_tools, with_think, trace_tokens);
        }
    } else if (mode == CLI_CHAT_TEST) {
        rc = run_chat_test(max_new, min_new, dump_layer, trace_tokens);
    } else if (mode == CLI_PRINT_TEMPLATE) {
        struct slm_model * m = slm_model_load(slm_cli_gguf_path());
        struct slm_ctx * c = slm_model_loaded(m) ? slm_ctx_create(m) : NULL;
        if (c == NULL || !c->model->loaded) {
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
        rc = run_bench(dump_layer, trace_tokens);
    } else if (mode == CLI_ALL_TESTS) {
        rc = slm_run_all_tests();
    } else {
        printf("usage (set QWEN_GGUF=/path/to/model.gguf to override default):\n"
               "  slm --self-test\n"
               "  slm --all-tests          (qwen_self_test + chunked +\n"
               "                            jinja + agent_parser + think)\n"
               "  slm --bench\n"
               "  slm --single \"prompt\" [--max-new N] [flags]\n"
               "  slm --repl [--max-new N] [flags]\n"
               "  slm --chat -p \"turn1\" [-p \"turn2\" ...] "
                       "[-sys \"system prompt\"] [flags]\n"
               "  slm --chat-test [--max-new N]\n"
               "  slm --jinja-test\n"
               "  slm --tools-test          (live: needs network for DDG)\n"
               "  slm --agent-test          (parser fixtures, no network)\n"
               "  slm --ask \"question\" [--max-iters N] [--max-new N]"
                       " [--trace] [flags]\n"
               "  slm --print-chat-template\n"
               "\n"
               "behavior flags (apply to --single / --repl):\n"
               "  --tools / --no-tools              enable atomic"
                       " websearch tool (auto-clamps T=0, top_p=1, rep=1)\n"
               "  --think / --no-think              enable <think> mode"
                       " (must stay off for --tools per Tool-Calling.md)\n"
               "  --verbose                         stream every signal"
                       " (content + reasoning + tool dialogue) to stdout\n"
               "  --debug N                         trace verbosity 0..9"
                       " (1=events, 3=tool args, 5=hits, 7=prompts,"
                       " 9=raw bodies)\n"
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
