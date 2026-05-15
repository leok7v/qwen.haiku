// llm.c -- single-file CPU runner for Qwen3.5-0.8B-Q4_K_M.gguf.
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
// public API (llm_create / llm_generate / ...), the sampler
// chain, the <think>/<tool_call> state-machine filter that splits
// the raw token stream into content/reasoning/tool_call/
// tool_response chunks, the embedded agent loop, and the CLI
// driver. Model-specific code (GGUF parser, byte-level BPE
// tokenizer, Q4_K_M weight loader, forward pass, RNG primitives)
// lives in qwen.c, `#include`d below.

#include "tensor.c"
#include "llm.h"

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

#include "qwen.c"
#include "jinja-template.c"
#include "tools.c"


// Sampler chain order (sample_with below): penalties -> temperature
// (softmax-with-T on top-K logits) -> top-P -> min-P -> distribution.
// Matches im.ai's `Sampler.swift` ordering (see comment block at
// im.ai/src/model/Sampler.swift:19-32 for the rationale). One subtle
// difference: im.ai normalizes the softmax over the FULL vocabulary
// before top-K filtering, while we normalize over the top-K survivors
// (which represent ~99% of probability mass for typical Qwen logits,
// so the top-P cutoff shifts by <1% in practice).
struct llm_sampler llm_sampler(void) {
    struct llm_sampler s;
    s.temperature        = 0.7f;
    s.top_k              = 40;
    s.top_p              = 0.9f;
    s.min_p              = 0.05f;
    s.repetition_penalty = 1.25f;
    s.repetition_window  = 64;
    s.tools              = true;   // agent dispatch on by default
    s.think              = false;  // reasoning off (gen prompt skips
                                   // the <think> block)
    s.debug              = true;   // surface tool_call /
                                   // tool_response chunks to UI
    return s;
}

// Apply repetition penalty in-place to `logits`: any token id that
// appears in the recent-history window is scaled by /penalty
// (positive logits become less likely) or *penalty (negative logits
// become MORE negative). Standard llama.cpp convention.
static void apply_rep_penalty(struct tensor * logits,
                              const int32_t * history, int32_t hist_n,
                              float penalty, int32_t window) {
    if (penalty == 1.0f || hist_n == 0) { return; }
    int32_t start = (window > 0 && window < hist_n) ? hist_n - window : 0;
    int64_t vlen  = tensor_nelements(logits);
    for (int32_t i = start; i < hist_n; i++) {
        int32_t t = history[i];
        if (t >= 0 && (int64_t)t < vlen) {
            float lv = logits->data[t];
            logits->data[t] = (lv > 0.0f) ? (lv / penalty) : (lv * penalty);
        }
    }
}

// Top-k filter into parallel arrays (idx, val) of length filled.
// Linear scan; k is capped to LLM_SAMPLE_TOPK_MAX so the working set
// fits in a stack buffer.
#define LLM_SAMPLE_TOPK_MAX 256

static int32_t topk_collect(const struct tensor * logits, int32_t k,
                            int32_t * idx, float * val) {
    int64_t n      = tensor_nelements(logits);
    if (k <= 0 || k > LLM_SAMPLE_TOPK_MAX) { k = LLM_SAMPLE_TOPK_MAX; }
    int32_t filled = 0;
    for (int64_t i = 0; i < n; i++) {
        float lv = logits->data[i];
        if (filled < k) {
            idx[filled] = (int32_t)i;
            val[filled] = lv;
            filled++;
        } else {
            int32_t worst = 0;
            for (int32_t j = 1; j < k; j++) {
                if (val[j] < val[worst]) { worst = j; }
            }
            if (lv > val[worst]) {
                idx[worst] = (int32_t)i;
                val[worst] = lv;
            }
        }
    }
    return filled;
}

// Softmax over `filled` candidates with temperature; writes the
// normalized probability into `val` (replacing logits).
static void topk_softmax(float * val, int32_t filled, float temperature) {
    float m = val[0];
    for (int32_t j = 1; j < filled; j++) {
        if (val[j] > m) { m = val[j]; }
    }
    float sum = 0.0f;
    for (int32_t j = 0; j < filled; j++) {
        val[j] = expf((val[j] - m) / temperature);
        sum   += val[j];
    }
    if (sum > 0.0f) {
        for (int32_t j = 0; j < filled; j++) { val[j] /= sum; }
    }
}

// Sort (idx, val) pairs by val descending using insertion sort
// (filled <= 256 in practice; cheaper than qsort overhead).
static void topk_sort_desc(int32_t * idx, float * val, int32_t filled) {
    for (int32_t i = 1; i < filled; i++) {
        float   v = val[i];
        int32_t k = idx[i];
        int32_t j = i - 1;
        while (j >= 0 && val[j] < v) {
            val[j + 1] = val[j];
            idx[j + 1] = idx[j];
            j--;
        }
        val[j + 1] = v;
        idx[j + 1] = k;
    }
}

static int32_t sample_with(struct tensor * logits,
                           const struct llm_sampler * sp,
                           struct rng * rng,
                           const int32_t * history, int32_t hist_n) {
    apply_rep_penalty(logits, history, hist_n,
                      sp->repetition_penalty, sp->repetition_window);
    if (sp->temperature <= 0.0f) {
        return sample_argmax(logits);
    }
    int32_t idx[LLM_SAMPLE_TOPK_MAX];
    float   val[LLM_SAMPLE_TOPK_MAX];
    int32_t k = sp->top_k > 0 ? sp->top_k : LLM_SAMPLE_TOPK_MAX;
    if (k > LLM_SAMPLE_TOPK_MAX) { k = LLM_SAMPLE_TOPK_MAX; }
    int32_t filled = topk_collect(logits, k, idx, val);
    topk_softmax(val, filled, sp->temperature);
    topk_sort_desc(idx, val, filled);
    // Top-p (nucleus): keep the smallest prefix whose cumulative
    // probability >= top_p. Effective only when 0 < top_p < 1.
    int32_t cutoff = filled;
    if (sp->top_p > 0.0f && sp->top_p < 1.0f) {
        float acc = 0.0f;
        for (int32_t j = 0; j < filled; j++) {
            acc += val[j];
            if (acc >= sp->top_p) { cutoff = j + 1; j = filled; }
        }
    }
    // Min-p: drop tokens whose probability < min_p * top_prob.
    if (sp->min_p > 0.0f) {
        float thresh = sp->min_p * val[0];
        int32_t j2   = 1;
        while (j2 < cutoff && val[j2] >= thresh) { j2++; }
        cutoff = j2;
    }
    // Re-normalize and roulette-wheel sample from the surviving set.
    float sum = 0.0f;
    for (int32_t j = 0; j < cutoff; j++) { sum += val[j]; }
    float u = rng_uniform(rng) * sum;
    float c = 0.0f;
    int32_t picked = 0;
    for (int32_t j = 0; j < cutoff; j++) {
        c += val[j];
        if (u <= c) { picked = j; j = cutoff; }
    }
    return idx[picked];
}

// ---------------------------------------------------------------------------
// Public-ish API used by Swift bridge AND by main()
// ---------------------------------------------------------------------------

struct llm_ctx * llm_create(const char * path) {
    struct llm_ctx * c =
        (struct llm_ctx *)llm_oom(calloc(1, sizeof(struct llm_ctx)));
    if (gguf_open(&c->gguf, path) != 0) {
        snprintf(c->err, sizeof(c->err), "gguf open failed");
        return c;
    }
    if (llm_load_config(&c->gguf, &c->cfg) != 0) {
        snprintf(c->err, sizeof(c->err), "config load failed");
        return c;
    }
    if (llm_load_weights(&c->gguf, &c->cfg, &c->W) != 0) {
        snprintf(c->err, sizeof(c->err), "weights resolve failed");
        return c;
    }
    if (tok_load(&c->tok, &c->gguf, &c->cfg) != 0) {
        snprintf(c->err, sizeof(c->err), "tokenizer load failed");
        return c;
    }
    // Chat-turn stop token: look up `<|im_end|>` in the vocab now
    // that the tokenizer's string->id map is built. Stays -1 when the
    // model isn't a chat-tuned vocab.
    c->cfg.eot_id = s2i_get(&c->tok.vocab_to_id,
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
    c->cfg.n_stop_ids = 0;
    for (int32_t i = 0; stop_strs[i] != NULL; i++) {
        int32_t id = s2i_get(&c->tok.vocab_to_id,
                             stop_strs[i],
                             (int32_t)strlen(stop_strs[i]), -1);
        if (id >= 0 && c->cfg.n_stop_ids <
            (int32_t)(sizeof(c->cfg.stop_ids) / sizeof(c->cfg.stop_ids[0]))) {
            c->cfg.stop_ids[c->cfg.n_stop_ids++] = id;
        }
    }
    kv_init(&c->kv, c->cfg.n_layers, c->cfg.n_kv_heads,
            c->cfg.head_dim, c->cfg.max_position);
    ssm_cache_init(&c->ssm, &c->cfg);
    c->arena = arena_new(64 * 1024 * 1024);
    // Stash the Jinja chat template from `tokenizer.chat_template`
    // as a null-terminated copy. Optional KV - base completion
    // models won't have one; instruct/chat models do.
    const struct gguf_kv * ct = gguf_find_kv(&c->gguf,
                                             "tokenizer.chat_template");
    if (ct != NULL && ct->v.type == GGUF_VT_STR) {
        size_t cc = 0;
        uint64_t n = gr_u64(ct->v.raw, &cc);
        c->chat_template = (char *)malloc(n + 1);
        if (c->chat_template != NULL) {
            memcpy(c->chat_template, ct->v.raw + cc, n);
            c->chat_template[n] = '\0';
        }
    }
    c->loaded = 1;
    c->dump_layer = g_dump_layer;
    return c;
}

void llm_destroy(struct llm_ctx * c) {
    if (c != NULL) {
        ssm_cache_free(&c->ssm);
        kv_free(&c->kv);
        tok_free(&c->tok);
        llm_free_weights(&c->W);
        if (c->chat_template) { free(c->chat_template); }
        if (c->arena)         { arena_free(c->arena); c->arena = NULL; }
        gguf_close(&c->gguf);
        free(c);
    }
}

void llm_reset(struct llm_ctx * c) {
    if (c != NULL) {
        ssm_cache_reset(&c->ssm, &c->cfg);
        c->pos = 0;
    }
}

const char * llm_get_error(const struct llm_ctx * c) {
    return c == NULL ? "no ctx" : c->err;
}

int llm_loaded(const struct llm_ctx * c) {
    return (c != NULL && c->loaded) ? 1 : 0;
}

const char * llm_chat_template(const struct llm_ctx * c) {
    return (c == NULL) ? NULL : c->chat_template;
}

// Thin wrappers exposing jinja-template.c's chat-formatting through
// the public llm_chat_format / llm_chat_format_delta API. We copy
// the public llm_chat_message[] into the internal jinja_message
// layout so the public ABI stays minimal (no tool_calls /
// reasoning_content surface — text-only chat covers the iOS/macOS
// app's needs). Tool support, when wired up later, can either grow
// the public struct or take a richer "advanced" entry point.
char * llm_chat_format(const struct llm_chat_message * messages,
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

// llm_chat_format_delta is defined further down — its full body
// needs the AGENT_TOOL_* tool-spec JSON strings (declared in
// agent.c), so the definition lives below the `#include "agent.c"`
// site. See the matching block right after the agent.c include.

// File-internal token-stream callback. llm_generate_raw emits one
// utf8 piece per generated token to this; the public llm_generate
// trampolines into the <think>-filter on top of this (see further
// down in this file).
typedef int (*llm_token_cb)(const char * utf8, void * user);

// PREFILL_PROGRESS_EVERY: emit a prefill progress chunk every N
// tokens. 16 is small enough that a 2048-token prompt fires ~128
// progress events (smooth bar) but big enough that the overhead is
// invisible next to a forward pass at ~5-15 ms each.
#define PREFILL_PROGRESS_EVERY 16

static int llm_generate_raw(struct llm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct llm_sampler * sampler_in,
                 uint64_t seed,
                 llm_token_cb cb, void * user,
                 llm_stream_cb progress_cb, void * progress_user) {
    struct llm_sampler sp = {0};
    if (sampler_in != NULL) { sp = *sampler_in; }
    struct rng rng;
    rng_seed(&rng, seed != 0 ? seed : (uint64_t)time(NULL));
    qh_trace_open();
    int32_t   generated = 0;
    int32_t   pos       = c->pos;        // resume after previous call
    int32_t   stop      = 0;
    double    t0        = llm_monotonic_seconds();
    // Repetition-penalty history: include the full prompt so the
    // model is discouraged from immediately echoing the input back,
    // plus everything it generates in this call. Capacity grows on
    // demand (single realloc up front sized for the worst case).
    int32_t * history = (int32_t *)llm_oom(
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
        // llm_forward_ssm_batch has a multi-chunk loop, so any
        // prompt_n is allowed (CHUNK_SIZE no longer caps the call).
        struct tensor * logits = llm_forward_batch(c, prompt_ids,
                                                   prompt_n, pos);
        (void)logits;
        pos += prompt_n;
        qh_trace_close();
    } else {
        for (int32_t i = 0; i < prompt_n && !stop; i++) {
            struct tensor * logits = llm_forward_step(c, prompt_ids[i], pos);
            // Trace facility is for parity diagnostics on the FIRST
            // forward pass only - one prompt token -> ~1850 JSONL
            // lines matches the llama-qwen-haiku reference dump
            // shape. Close after step 0 to keep file size bounded.
            if (i == 0) { qh_trace_close(); }
            (void)logits;
            pos++;
            // Periodically emit prefill progress so the UI can show
            // a bar. Returning non-zero from the progress callback
            // aborts prefill via the same `stop` flag as decode.
            if (progress_cb != NULL && prompt_n > 1 &&
                ((i + 1) % PREFILL_PROGRESS_EVERY == 0 ||
                 i + 1 == prompt_n)) {
                struct llm_stream_chunk ch = {0};
                ch.prefill_done  = i + 1;
                ch.prefill_total = prompt_n;
                if (progress_cb(&ch, progress_user) != 0) { stop = 1; }
            }
        }
    }
    if (progress_cb != NULL && prompt_n > 0 && use_forward_batch) {
        // Batched prefill ran in one shot — emit one terminal
        // progress chunk so the UI can hide its bar.
        struct llm_stream_chunk ch = {0};
        ch.prefill_done  = prompt_n;
        ch.prefill_total = prompt_n;
        (void)progress_cb(&ch, progress_user);
    }
    double t1 = llm_monotonic_seconds();
    c->t_prefill_s = t1 - t0;
    c->n_prefill   = prompt_n;
    int32_t last = prompt_n > 0 ? prompt_ids[prompt_n - 1] : c->cfg.bos_id;
    // Decode loop.
    while (!stop && generated < max_new && pos < c->cfg.max_position) {
        struct tensor * logits = llm_forward_step(c, last, pos);
        // While below min_new, force the model to keep producing
        // content by zeroing out the eos / eot logits (effectively
        // -infinity once normalized). Restored automatically once
        // generated >= min_new.
        float saved_eos = 0.0f;
        float saved_eot = 0.0f;
        int32_t mask    = (generated < min_new);
        if (mask) {
            if (c->cfg.eos_id >= 0 && c->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                saved_eos = logits->data[c->cfg.eos_id];
                logits->data[c->cfg.eos_id] = -INFINITY;
            }
            if (c->cfg.eot_id >= 0 && c->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                saved_eot = logits->data[c->cfg.eot_id];
                logits->data[c->cfg.eot_id] = -INFINITY;
            }
        }
        int32_t next = sample_with(logits, &sp, &rng, history, hist_n);
        history[hist_n++] = next;
        if (mask) {
            if (c->cfg.eos_id >= 0 && c->cfg.eos_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->cfg.eos_id] = saved_eos;
            }
            if (c->cfg.eot_id >= 0 && c->cfg.eot_id < (int32_t)tensor_nelements(logits)) {
                logits->data[c->cfg.eot_id] = saved_eot;
            }
        }
        pos++;
        generated++;
        if (g_trace_tokens) { fprintf(stderr, "[tok] %d\n", (int)next); }
        int32_t is_stop = (next == c->cfg.eos_id || next == c->cfg.eot_id);
        for (int32_t si = 0; !is_stop && si < c->cfg.n_stop_ids; si++) {
            if (next == c->cfg.stop_ids[si]) { is_stop = 1; }
        }
        if (is_stop) {
            stop = 1;
        } else {
            struct chars piece = {0};
            tok_decode_one(&c->tok, next, &piece);
            chars_put(&piece, "", 0);  // ensure null-term
            if (cb != NULL && piece.data != NULL) {
                if (cb(piece.data, user) != 0) { stop = 1; }
            }
            chars_free(&piece);
            last = next;
        }
    }
    double t2 = llm_monotonic_seconds();
    c->t_gen_s     = t2 - t1;
    c->n_generated = generated;
    c->pos         = pos;                // persist across calls
    free(history);
    qh_trace_close();
    return generated;
}

// ---------------------------------------------------------------------------
// <think> stream filter. State machine that splits llm_generate's
// single output stream into two: content (outside `<think>`...
// `</think>` blocks) and reasoning (inside). Marker bytes are NOT
// emitted; both streams are routed through ONE llm_stream_cb
// callback with a struct llm_stream_chunk that carries exactly one
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
// Public surface (llm.h) is just `llm_generate_split`. The filter
// struct + push/finish helpers are file-static here: clients should
// not need the streaming pieces directly because llm_generate_split
// already drives one filter end-to-end per generate call.
// ---------------------------------------------------------------------------

// Filter state-machine modes. The filter walks through the model's
// raw byte stream and assigns each byte to one of these modes based
// on the marker tags it has seen. THINK_MODE_NONE is a sentinel
// used in the marker table for "this marker triggers an action but
// doesn't change mode" (no live state ever holds it).
enum llm_think_mode {
    THINK_MODE_CONTENT   = 0,  // visible reply text
    THINK_MODE_REASONING = 1,  // inside <think>...</think>
    THINK_MODE_TOOL_CALL = 2,  // inside <tool_call>...</tool_call>
    THINK_MODE_NONE      = -1, // sentinel for marker.mode (no-op)
};

// Marker-table row indices. Used to identify a specific marker
// from its position in THINK_MARKERS without comparing tag strings.
enum llm_think_marker_row {
    LLM_MARKER_THINK_OPEN  = 0,
    LLM_MARKER_THINK_CLOSE = 1,
    LLM_MARKER_TOOL_OPEN   = 2,
    LLM_MARKER_TOOL_CLOSE  = 3,
};

struct llm_think_filter {
    enum llm_think_mode mode;
    char                hold_data[64];   // partial-marker holdback
    int                 hold_n;
    char                tool_call_data[4096];  // body of <tool_call>
    int                 tool_call_n;
    // Set to 1 by the filter the moment a </tool_call> marker is
    // consumed. The public llm_generate loop reads this after the
    // generate_raw call returns to dispatch the tool + re-feed the
    // model. Reset by init().
    int                 tool_call_ready;
    // Per-call configuration (set by llm_generate from sampler
    // flags before kicking the trampoline). recognize_tool_calls=0
    // makes find_marker ignore <tool_call> / </tool_call> entries
    // so the body streams as plain content. emit_visibility=0
    // suppresses chunk->tool_call emission (tool dispatch still
    // happens; the UI just doesn't see the trace).
    int                 recognize_tool_calls;
    int                 emit_visibility;
};

struct llm_think_marker {
    const char *        tag;   // null-terminated literal
    int                 len;   // strlen(tag), cached
    enum llm_think_mode mode;  // mode to switch INTO on match
                               // (THINK_MODE_NONE = no-op)
};

static const struct llm_think_marker THINK_MARKERS[] = {
    [LLM_MARKER_THINK_OPEN]  = { "<think>",      7, THINK_MODE_REASONING },
    [LLM_MARKER_THINK_CLOSE] = { "</think>",     8, THINK_MODE_CONTENT   },
    [LLM_MARKER_TOOL_OPEN]   = { "<tool_call>", 11, THINK_MODE_TOOL_CALL },
    [LLM_MARKER_TOOL_CLOSE]  = { "</tool_call>",12, THINK_MODE_CONTENT   },
};
#define LLM_THINK_N_MARKERS \
    ((int)(sizeof(THINK_MARKERS) / sizeof(THINK_MARKERS[0])))
#define LLM_THINK_MAX_MARKER 12  // strlen("</tool_call>")

static void llm_think_filter_init(struct llm_think_filter * f) {
    f->mode = THINK_MODE_CONTENT;
    f->hold_n = 0;
    f->hold_data[0] = '\0';
    f->tool_call_n = 0;
    f->tool_call_data[0] = '\0';
    f->tool_call_ready = 0;
    f->recognize_tool_calls = 1;  // default on; llm_generate overrides
    f->emit_visibility      = 1;
}

// Emit `[start, end)` bytes through `cb`, tagging the chunk's text
// per the filter's current mode. Mode 2 (tool_call) routes bytes
// into the filter's internal tool_call_data buffer instead of
// firing `cb` — the buffered call is dispatched as one atomic
// chunk when </tool_call> arrives. Returns the callback's return
// value, or 0 when there's no work / no callback / mode is 2.
static int llm_think_emit(struct llm_think_filter * f,
                          const char * start, int n,
                          llm_stream_cb cb, void * user) {
    int rc = 0;
    if (n > 0) {
        if (f->mode == THINK_MODE_TOOL_CALL) {
            // Buffering inside a tool_call block — don't emit yet.
            int avail = (int)sizeof(f->tool_call_data) - 1
                      - f->tool_call_n;
            int copy = n < avail ? n : avail;
            if (copy > 0) {
                memcpy(f->tool_call_data + f->tool_call_n,
                       start, (size_t)copy);
                f->tool_call_n += copy;
                f->tool_call_data[f->tool_call_n] = '\0';
            }
            // If the call body overflows the fixed buffer the
            // overflow is dropped — small models in practice emit
            // tool_call blocks <1 KB; 4 KB is comfortable.
        } else if (cb != NULL) {
            char tmp[80];
            int  copy = n < (int)sizeof(tmp) - 1
                      ? n : (int)sizeof(tmp) - 1;
            memcpy(tmp, start, (size_t)copy);
            tmp[copy] = '\0';
            struct llm_stream_chunk chunk = {0};
            if (f->mode == THINK_MODE_REASONING) { chunk.reasoning = tmp; }
            else                                 { chunk.content   = tmp; }
            rc = cb(&chunk, user);
            if (n > copy && rc == 0) {
                int rc2 = llm_think_emit(f, start + copy, n - copy,
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
static int llm_think_find_marker(const struct llm_think_filter * f,
                                 int * pos) {
    int best  = -1;
    int best_pos = f->hold_n;
    for (int m = 0; m < LLM_THINK_N_MARKERS; m++) {
        // Skip the tool_call markers when the sampler turned tools
        // off — their bytes will stream as plain content instead.
        int is_tool_call = (m == LLM_MARKER_TOOL_OPEN) ||
                           (m == LLM_MARKER_TOOL_CLOSE);
        int skip = (!f->recognize_tool_calls && is_tool_call);
        if (!skip) {
            const char * tag = THINK_MARKERS[m].tag;
            int tlen = THINK_MARKERS[m].len;
            if (tlen <= f->hold_n) {
                int last_start = f->hold_n - tlen;
                for (int i = 0; i <= last_start; i++) {
                    int match = 1;
                    for (int k = 0; k < tlen && match; k++) {
                        if (f->hold_data[i + k] != tag[k]) {
                            match = 0;
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
static int llm_think_safe_utf8(const char * buf, int hold_n, int safe) {
    while (safe > 0 && safe < hold_n &&
           ((unsigned char)buf[safe] & 0xC0) == 0x80) {
        safe--;
    }
    return safe;
}

// Returns 1 if hold[]'s last `tail_n` bytes could be a strict prefix
// of any marker (i.e. we should hold them back rather than emit).
static int llm_think_could_be_prefix(const struct llm_think_filter * f,
                                     int tail_start) {
    int possible = 0;
    int tail_n = f->hold_n - tail_start;
    if (tail_n > 0) {
        for (int m = 0; m < LLM_THINK_N_MARKERS && !possible; m++) {
            const char * tag = THINK_MARKERS[m].tag;
            int tlen = THINK_MARKERS[m].len;
            if (tail_n < tlen) {
                int match = 1;
                for (int k = 0; k < tail_n && match; k++) {
                    if (f->hold_data[tail_start + k] != tag[k]) {
                        match = 0;
                    }
                }
                if (match) { possible = 1; }
            }
        }
    }
    return possible;
}

int llm_think_filter_push(struct llm_think_filter * f,
                          const char * utf8,
                          llm_stream_cb cb,
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
                // LLM_THINK_MAX_MARKER-1 bytes as potential prefix).
                // Same UTF-8 safety as the per-iter safe-emit below.
                int keep = LLM_THINK_MAX_MARKER - 1;
                int emit_n = f->hold_n - keep;
                emit_n = llm_think_safe_utf8(f->hold_data, f->hold_n,
                                             emit_n);
                if (emit_n > 0) {
                    int r2 = llm_think_emit(f, f->hold_data,
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
            int more = 1;
            while (more && rc == 0) {
                int pos = 0;
                int m   = llm_think_find_marker(f, &pos);
                if (m < 0) {
                    more = 0;
                } else {
                    // Emit prefix in current mode.
                    int r2 = llm_think_emit(f, f->hold_data, pos,
                                            cb, user);
                    if (r2 != 0) { rc = r2; }
                    enum llm_think_mode prev_mode = f->mode;
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
                        if (cb != NULL && f->tool_call_n > 0 &&
                            f->emit_visibility) {
                            struct llm_stream_chunk chunk = {0};
                            chunk.tool_call = f->tool_call_data;
                            int r3 = cb(&chunk, user);
                            if (r3 != 0) { rc = r3; }
                        }
                        f->tool_call_ready = 1;
                        // Force llm_generate_raw's decode loop to
                        // exit so the public llm_generate can run
                        // the dispatch + inject step before any
                        // further sampling.
                        rc = 1;
                        more = 0;
                    }
                    // other -> tool_call: entering a tool_call block;
                    // reset the accumulator.
                    if (prev_mode != THINK_MODE_TOOL_CALL &&
                        f->mode  == THINK_MODE_TOOL_CALL) {
                        f->tool_call_n = 0;
                        f->tool_call_data[0] = '\0';
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
            // marker. The trailing (LLM_THINK_MAX_MARKER-1) bytes
            // stay as potential prefix.
            int safe = f->hold_n - (LLM_THINK_MAX_MARKER - 1);
            if (safe > 0 && rc == 0) {
                // Trim the safe region further: don't emit bytes
                // that could still extend into a marker tail (i.e.
                // if the LAST byte we'd emit is a '<', keep it).
                while (safe > 0 &&
                       llm_think_could_be_prefix(f, safe)) {
                    safe--;
                }
                // Also don't split a multibyte UTF-8 codepoint —
                // hold continuation bytes back until the codepoint
                // is complete (next push() will bring the tail).
                safe = llm_think_safe_utf8(f->hold_data, f->hold_n,
                                           safe);
                if (safe > 0) {
                    int r2 = llm_think_emit(f, f->hold_data, safe,
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

int llm_think_filter_finish(struct llm_think_filter * f,
                            llm_stream_cb cb,
                            void * user) {
    int rc = 0;
    // No-more-data flush: any pending markers won't get longer, so
    // emit one last pass through the marker scan, then dump the
    // residue in current mode.
    int more = 1;
    while (more && rc == 0) {
        int pos = 0;
        int m   = llm_think_find_marker(f, &pos);
        if (m < 0) {
            more = 0;
        } else {
            int r2 = llm_think_emit(f, f->hold_data, pos, cb, user);
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
        rc = llm_think_emit(f, f->hold_data, f->hold_n, cb, user);
        f->hold_n = 0;
        f->hold_data[0] = '\0';
    }
    return rc;
}

// Trampoline box for llm_generate_split.
struct llm_split_box {
    struct llm_think_filter filter;
    llm_stream_cb           cb;
    void *                  user;
};

static int llm_split_trampoline(const char * utf8, void * user) {
    struct llm_split_box * b = (struct llm_split_box *)user;
    return llm_think_filter_push(&b->filter, utf8, b->cb, b->user);
}

// The public `llm_generate` definition lives AFTER `#include
// "agent.c"` so the tools-enabled build can reach
// agent_parse_tool_calls + agent_dispatch (defined in agent.c).
// See the matching definition further down in this file.

// Self-test for the think filter. Feeds known streams through it
// piece-by-piece (mimicking llm_generate's per-token callback) and
// asserts the routed content / reasoning outputs match hand-traced
// goldens. Returns 0 on PASS.
struct llm_think_test_capture {
    char content  [512];
    int  content_n;
    char reasoning[512];
    int  reasoning_n;
};

static int llm_think_test_cb(const struct llm_stream_chunk * chunk,
                             void * user) {
    struct llm_think_test_capture * cap =
        (struct llm_think_test_capture *)user;
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
static int llm_think_test_case(const char * name,
                               const char ** chunks, int n_chunks,
                               const char * want_content,
                               const char * want_reasoning) {
    int failed = 0;
    struct llm_think_filter f;
    llm_think_filter_init(&f);
    struct llm_think_test_capture cap;
    cap.content_n = 0;
    cap.content[0] = '\0';
    cap.reasoning_n = 0;
    cap.reasoning[0] = '\0';
    for (int i = 0; i < n_chunks; i++) {
        llm_think_filter_push(&f, chunks[i],
                              llm_think_test_cb, &cap);
    }
    llm_think_filter_finish(&f, llm_think_test_cb, &cap);
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
    return failed;
}

__attribute__((unused))
static int32_t llm_think_test(void) {
    int failures = 0;
    // A: plain content, no markers.
    {
        const char * chunks[] = { "Hello, world." };
        failures += llm_think_test_case("A", chunks, 1,
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
        failures += llm_think_test_case("B", chunks, 1,
                                        "\n\nHello.", "");
    }
    // C: real reasoning block then content.
    {
        const char * chunks[] = {
            "<think>let me think</think>\n\nAnswer."
        };
        failures += llm_think_test_case("C", chunks, 1,
                                        "\n\nAnswer.",
                                        "let me think");
    }
    // D: marker split across two chunks (the holdback case).
    {
        const char * chunks[] = {
            "abc<thi", "nk>secret</thi", "nk>end"
        };
        failures += llm_think_test_case("D", chunks, 3,
                                        "abcend", "secret");
    }
    // E: stray `<` that doesn't lead anywhere (must emit, not eat).
    {
        const char * chunks[] = {
            "x<y<not_a_marker>z"
        };
        failures += llm_think_test_case("E", chunks, 1,
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
        failures += llm_think_test_case("F", chunks, n, "ac", "b");
    }
    if (failures == 0) {
        printf("think-test: PASS (6 fixtures)\n");
    } else {
        printf("think-test: FAIL (%d fixture(s) failed)\n",
               (int)failures);
    }
    return failures;
}

double llm_pp_per_sec(const struct llm_ctx * c) {
    double r = 0.0;
    if (c != NULL && c->t_prefill_s > 0.0) {
        r = (double)c->n_prefill / c->t_prefill_s;
    }
    return r;
}

double llm_tg_per_sec(const struct llm_ctx * c) {
    double r = 0.0;
    if (c != NULL && c->t_gen_s > 0.0) {
        r = (double)c->n_generated / c->t_gen_s;
    }
    return r;
}

int32_t llm_n_prefill(const struct llm_ctx * c) {
    return (c == NULL) ? 0 : c->n_prefill;
}

int32_t llm_n_generated(const struct llm_ctx * c) {
    return (c == NULL) ? 0 : c->n_generated;
}

// Convenience wrappers exported for the Swift bridge.
int  llm_tokenize(struct llm_ctx * c, const char * text,
                  int32_t * out_ids, int max_ids) {
    return tok_encode(&c->tok, text, out_ids, max_ids);
}

int  llm_vocab_size(const struct llm_ctx * c) { return c->cfg.vocab_size; }
int  llm_eos_id    (const struct llm_ctx * c) { return c->cfg.eos_id; }
int  llm_bos_id    (const struct llm_ctx * c) { return c->cfg.bos_id; }

// ---------------------------------------------------------------------------
// --chunked-test: run the chunked SSM kernel on a 1-real-token chunk
// (padded with 63 zero-tokens) and compare to the autoregressive
// math direct evaluation. Per the degeneracy proof in chunked.c,
// they MUST agree (mathematically identical operations applied to
// the same fp32 inputs). A 1-2 ULP discrepancy on a handful of
// elements is acceptable - that is intra-kernel reduction order
// drift. A larger discrepancy means our chunked port is wrong.
#define CHUNKED_TEST_KHD 128
#define CHUNKED_TEST_VHD 128
// N=8 validates the chunked recurrence against an autoregressive
// reference implementation. Both should agree to within fp32
// accumulation noise (sub-1e-5 relative). The chunked path uses
// matmul-style reductions across all N tokens at once; the
// autoregressive ref applies the per-token recurrence step-by-step.
#define CHUNKED_TEST_NTOK 8

// Run the autoregressive recurrence (qwen3-next gated delta net) for
// `n` tokens in sequence, single head. Mirrors llm_forward_ssm step 9
// math directly without arena / tensor scaffolding. State starts at
// zero and updates in place.
static void autoregressive_ref(int n, int k_hd, int v_hd,
                               const float * Q,       // [n, k_hd]
                               const float * K,       // [n, k_hd]
                               const float * V,       // [n, v_hd]
                               const float * g_log,   // [n]
                               const float * beta,    // [n]
                               float * state,         // [k_hd, v_hd], zeroed by caller
                               float * out) {         // [n, v_hd]
    for (int t = 0; t < n; t++) {
        float g = expf(g_log[t]);
        float b = beta[t];
        // state *= g
        for (int d = 0; d < k_hd; d++) {
            for (int e = 0; e < v_hd; e++) {
                state[d * v_hd + e] *= g;
            }
        }
        // kv_mem[v] = sum_k state[k, v] * K[t, k]
        float kv_mem[CHUNKED_TEST_VHD];
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * K[t * k_hd + d];
            }
            kv_mem[e] = s;
        }
        // delta = (V - kv_mem) * beta
        float delta[CHUNKED_TEST_VHD];
        for (int e = 0; e < v_hd; e++) {
            delta[e] = (V[t * v_hd + e] - kv_mem[e]) * b;
        }
        // state[k, v] += K[t, k] * delta[v]
        for (int d = 0; d < k_hd; d++) {
            float kd = K[t * k_hd + d];
            for (int e = 0; e < v_hd; e++) {
                state[d * v_hd + e] += kd * delta[e];
            }
        }
        // out[t, v] = sum_k state[k, v] * Q[t, k]
        for (int e = 0; e < v_hd; e++) {
            float s = 0.0f;
            for (int d = 0; d < k_hd; d++) {
                s += state[d * v_hd + e] * Q[t * k_hd + d];
            }
            out[t * v_hd + e] = s;
        }
    }
}

__attribute__((unused))
static int32_t chunked_self_test(void) {
    enum { k_hd = CHUNKED_TEST_KHD, v_hd = CHUNKED_TEST_VHD,
           N   = CHUNKED_TEST_NTOK };
    // Deterministic multi-token inputs (4 distinct tokens).
    float Q[N * k_hd], K[N * k_hd], V[N * v_hd];
    float g_log[N], beta_arr_in[N];
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < k_hd; i++) {
            Q[t * k_hd + i] = sinf((float)((t + 1) * (i + 1)) * 0.0173f);
            K[t * k_hd + i] = cosf((float)((t + 1) * (i + 1)) * 0.0211f);
        }
        for (int i = 0; i < v_hd; i++) {
            V[t * v_hd + i] = sinf((float)((t + 1) * (i + 1)) * 0.0149f) * 0.5f;
        }
        g_log[t]       = -0.25f - 0.05f * (float)t;
        beta_arr_in[t] = 0.55f + 0.03f * (float)t;
    }
    // Autoregressive reference (N sequential steps).
    static float auto_state[k_hd * v_hd];
    static float auto_out  [N * v_hd];
    memset(auto_state, 0, sizeof(auto_state));
    memset(auto_out,   0, sizeof(auto_out));
    autoregressive_ref(N, k_hd, v_hd, Q, K, V, g_log, beta_arr_in,
                       auto_state, auto_out);
    // Chunked path: N real tokens padded to CHUNK_SIZE.
    static float q_pad   [CHUNK_SIZE * k_hd];
    static float k_pad   [CHUNK_SIZE * k_hd];
    static float v_pad   [CHUNK_SIZE * v_hd];
    static float g_log_p [CHUNK_SIZE];
    static float beta_p  [CHUNK_SIZE];
    static float state   [k_hd * v_hd];
    static float out     [CHUNK_SIZE * v_hd];
    memset(q_pad,   0, sizeof(q_pad));
    memset(k_pad,   0, sizeof(k_pad));
    memset(v_pad,   0, sizeof(v_pad));
    memset(g_log_p, 0, sizeof(g_log_p));
    memset(beta_p,  0, sizeof(beta_p));
    memset(state,   0, sizeof(state));
    memset(out,     0, sizeof(out));
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < k_hd; i++) {
            q_pad[t * k_hd + i] = Q[t * k_hd + i];
            k_pad[t * k_hd + i] = K[t * k_hd + i];
        }
        for (int i = 0; i < v_hd; i++) {
            v_pad[t * v_hd + i] = V[t * v_hd + i];
        }
        g_log_p[t] = g_log[t];
        beta_p [t] = beta_arr_in[t];
    }
    // Scratch (heap; one-shot test, no perf concern).
    float * sc_gcs        = (float *)llm_oom(calloc(CHUNK_SIZE, sizeof(float)));
    float * sc_gexp       = (float *)llm_oom(calloc(CHUNK_SIZE, sizeof(float)));
    float * sc_decay_mask = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_k_beta     = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_v_beta     = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_kk_dot     = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_lhs        = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_attn       = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_v_eff      = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_kbeta_gexp = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_k_cumdecay = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_attn_kq    = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * CHUNK_SIZE, sizeof(float)));
    float * sc_q_g_exp    = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_attn_inter = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_prime    = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_new      = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_v_attn     = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * v_hd,       sizeof(float)));
    float * sc_key_gdiff  = (float *)llm_oom(calloc((size_t)CHUNK_SIZE * k_hd,       sizeof(float)));
    float * sc_kgd_vnew   = (float *)llm_oom(calloc((size_t)k_hd * v_hd,             sizeof(float)));
    // Run the kernel with the actual N (dynamic chunk size — exercises
    // the same path that llm_forward_ssm_batch takes for partial tail
    // chunks).
    chunked_ssm_step_f32(N, k_hd, v_hd,
                         q_pad, k_pad, v_pad, g_log_p, beta_p,
                         state, out,
                         sc_gcs, sc_gexp, sc_decay_mask,
                         sc_k_beta, sc_v_beta, sc_kk_dot, sc_lhs,
                         sc_attn, sc_v_eff, sc_kbeta_gexp, sc_k_cumdecay,
                         sc_attn_kq, sc_q_g_exp, sc_attn_inter,
                         sc_v_prime, sc_v_new, sc_v_attn,
                         sc_key_gdiff, sc_kgd_vnew);
    // Compare per-token outputs.
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int   max_t = -1, max_e = -1;
    for (int t = 0; t < N; t++) {
        for (int e = 0; e < v_hd; e++) {
            float ae = auto_out[t * v_hd + e];
            float ce = out     [t * v_hd + e];
            float d  = fabsf(ae - ce);
            float r  = (fabsf(ae) > 1e-6f) ? d / fabsf(ae) : d;
            if (d > max_abs_diff) {
                max_abs_diff = d;
                max_rel_diff = r;
                max_t = t;
                max_e = e;
            }
        }
    }
    // Compare final states.
    float max_state_diff = 0.0f;
    int   max_state_d = -1, max_state_e = -1;
    for (int d = 0; d < k_hd; d++) {
        for (int e = 0; e < v_hd; e++) {
            float ae = auto_state[d * v_hd + e];
            float ce = state     [d * v_hd + e];
            float diff = fabsf(ae - ce);
            if (diff > max_state_diff) {
                max_state_diff = diff;
                max_state_d = d;
                max_state_e = e;
            }
        }
    }
    for (int t = 0; t < N; t++) {
        printf("chunked-test t=%d: auto[0..3] = %.7g %.7g %.7g %.7g\n",
               t,
               auto_out[t * v_hd + 0], auto_out[t * v_hd + 1],
               auto_out[t * v_hd + 2], auto_out[t * v_hd + 3]);
        printf("chunked-test t=%d: chunk[0..3] = %.7g %.7g %.7g %.7g\n",
               t,
               out[t * v_hd + 0], out[t * v_hd + 1],
               out[t * v_hd + 2], out[t * v_hd + 3]);
    }
    printf("chunked-test: max |Δ_out| = %.4g at t=%d e=%d (rel %.4g)\n",
           max_abs_diff, max_t, max_e, max_rel_diff);
    printf("chunked-test: max |Δ_state| = %.4g at d=%d e=%d\n",
           max_state_diff, max_state_d, max_state_e);
    // Free scratch.
    free(sc_gcs); free(sc_gexp); free(sc_decay_mask);
    free(sc_k_beta); free(sc_v_beta); free(sc_kk_dot); free(sc_lhs);
    free(sc_attn); free(sc_v_eff); free(sc_kbeta_gexp);
    free(sc_k_cumdecay); free(sc_attn_kq); free(sc_q_g_exp);
    free(sc_attn_inter); free(sc_v_prime); free(sc_v_new);
    free(sc_v_attn); free(sc_key_gdiff); free(sc_kgd_vnew);
    // Pass if rel <= 1e-5 (a few ULPs at fp32 precision).
    return (max_rel_diff <= 1e-5f) ? 0 : 1;
}

// Agent helpers (parser + tool dispatcher) — `#include`-d here so
// llm_generate's embedded agent loop can reach them. agent.c uses
// llm_ctx + tok_encode + chars / chars_put + tools_*, all of which
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
// model — the embedded agent loop in llm_generate fires only when
// the model emits `<tool_call>` markers, which only happens when
// it's been told tools exist.
char * llm_chat_format_delta(const char * user_message,
                             const char * system_prefix,
                             int enable_thinking,
                             int enable_tools) {
    char * result = NULL;
    if (user_message != NULL) {
        if (system_prefix != NULL && system_prefix[0] != '\0') {
            // First turn: full frame, with tool advertisements only
            // when the caller asked for them.
            struct jinja_message msgs[2];
            memset(msgs, 0, sizeof(msgs));
            msgs[0].role    = JINJA_ROLE_SYSTEM;
            msgs[0].content = system_prefix;
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
        } else {
            // Subsequent turn: bare delta (KV holds whatever was
            // advertised on turn 1, including no tools at all).
            result = jinja_apply_delta(user_message, NULL,
                                       enable_thinking);
        }
    }
    return result;
}

// Public llm_generate. The think + tool-call stream filter (set up
// via llm_split_box / llm_split_trampoline above) routes the
// model's bytes through llm_stream_cb. A completed
// <tool_call>...</tool_call> block stops llm_generate_raw, the
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
#define LLM_GENERATE_TOOL_ITER_CAP 6
int llm_generate(struct llm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct llm_sampler * sampler_in,
                 uint64_t seed,
                 llm_stream_cb cb,
                 void * user) {
    // Defaults when sampler isn't provided (rare; CLI / Swift
    // always pass one).
    int with_tools = (sampler_in != NULL) ? sampler_in->tools : 1;
    int with_debug = (sampler_in != NULL) ? sampler_in->debug : 1;
    int total_gen = 0;
    int done      = 0;
    int32_t * cur_ids =
        (int32_t *)llm_oom(calloc((size_t)(prompt_n > 0 ? prompt_n : 1),
                                  sizeof(int32_t)));
    if (prompt_n > 0) {
        memcpy(cur_ids, prompt_ids,
               (size_t)prompt_n * sizeof(int32_t));
    }
    int cur_n = prompt_n;
    int iter  = 0;
    while (!done && iter < LLM_GENERATE_TOOL_ITER_CAP &&
           total_gen < max_new) {
        struct llm_split_box box;
        llm_think_filter_init(&box.filter);
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
        int n = llm_generate_raw(c, cur_ids, cur_n,
                                 budget, min_new,
                                 sampler_in, seed,
                                 llm_split_trampoline, &box,
                                 cb, user);
        total_gen += n;
        llm_think_filter_finish(&box.filter, cb, user);
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
            if (box.filter.tool_call_n > 0) {
                chars_put(&wrapped, box.filter.tool_call_data,
                          (size_t)box.filter.tool_call_n);
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
                struct llm_stream_chunk ch = {0};
                ch.tool_response = result.data;
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
            int32_t * ids = (int32_t *)llm_oom(
                calloc(16384, sizeof(int32_t)));
            int n_ids = tok_encode(&c->tok, inject.data,
                                   ids, 16384);
            cur_ids = ids;
            cur_n   = n_ids;
            chars_free(&inject);
            chars_free(&result);
        } else {
            done = 1;
        }
        iter++;
    }
    free(cur_ids);
    return total_gen;
}

// --self-test: build a tiny synthetic model with deterministic
// weights and run one forward step. Validates the data flow without
// needing the GGUF file present. CLI-only - library callers don't
// need it.
// ---------------------------------------------------------------------------
#ifdef LLM_CLI

static int32_t llm_self_test(void) {
    // Tiny config: 2 layers, hidden=64, heads=4, head_dim=16, ffn=128,
    // vocab=256. Just enough to exercise every kernel.
    struct llm_config cfg = {
        .n_layers = 2, .n_heads = 4, .n_kv_heads = 2,
        .head_dim = 16, .hidden_dim = 64,
        .ffn_dim = 128, .vocab_size = 256,
        .max_position = 64,
        .rope_theta = 10000.0f, .norm_eps = 1e-5f,
        .bos_id = 0, .eos_id = 1,
    };
    struct arena * a = arena_new(8 * 1024 * 1024);
    // Deterministic rng for synth weights.
    uint32_t s = 1u;
    #define RNDF() ({ s = s * 1103515245u + 12345u; \
                      ((float)((s >> 16) & 0x7fff) / 16383.5f - 1.0f) * 0.05f; })
    // Allocate synthetic fp32 weights (no Q4_K to keep --self-test simple).
    int32_t hidden = cfg.hidden_dim;
    int32_t hd     = cfg.head_dim;
    int32_t nkvh   = cfg.n_kv_heads;
    int32_t kvh    = nkvh * hd;
    int32_t ffn    = cfg.ffn_dim;
    // token_embd: (hidden, vocab)
    float * tok_embd = (float *)arena_alloc(
        a, (size_t)hidden * cfg.vocab_size * sizeof(float));
    for (int32_t i = 0; i < hidden * cfg.vocab_size; i++) tok_embd[i] = RNDF();
    // output_norm: (hidden,)
    float * out_norm = (float *)arena_alloc(a, (size_t)hidden * sizeof(float));
    for (int32_t i = 0; i < hidden; i++) out_norm[i] = 1.0f + RNDF();
    // Per-layer weights:
    struct llm_layer_w * layers = (struct llm_layer_w *)llm_oom(
        calloc(cfg.n_layers, sizeof(struct llm_layer_w)));
    #define ALLOC_F32(dst, n0, n1) do {                                 \
        size_t _bytes = (size_t)(n0) * (n1) * sizeof(float);           \
        float * _p = (float *)arena_alloc(a, _bytes);                   \
        for (size_t _i = 0; _i < (size_t)(n0)*(n1); _i++) _p[_i] = RNDF(); \
        (dst).data    = _p;                                             \
        (dst).type    = GGUF_TT_F32;                                    \
        (dst).n_dims  = ((n1)==1 ? 1 : 2);                              \
        (dst).shape[0] = (n0);                                          \
        (dst).shape[1] = (n1);                                          \
        (dst).shape[2] = 1;                                             \
        (dst).shape[3] = 1;                                             \
    } while (0)

    for (int32_t L = 0; L < cfg.n_layers; L++) {
        ALLOC_F32(layers[L].attn_norm,  hidden, 1);
        ALLOC_F32(layers[L].attn_q,     hidden, hidden);
        ALLOC_F32(layers[L].attn_k,     hidden, kvh);
        ALLOC_F32(layers[L].attn_v,     hidden, kvh);
        ALLOC_F32(layers[L].attn_out,   hidden, hidden);
        ALLOC_F32(layers[L].ffn_norm,   hidden, 1);
        ALLOC_F32(layers[L].ffn_gate,   hidden, ffn);
        ALLOC_F32(layers[L].ffn_up,     hidden, ffn);
        ALLOC_F32(layers[L].ffn_down,   ffn,    hidden);
    }
    struct llm_weights W = {0};
    W.tok_embd.data = tok_embd; W.tok_embd.type = GGUF_TT_F32;
    W.tok_embd.n_dims = 2; W.tok_embd.shape[0] = hidden;
    W.tok_embd.shape[1] = cfg.vocab_size;
    W.tok_embd.shape[2] = 1; W.tok_embd.shape[3] = 1;
    W.output_norm.data = out_norm; W.output_norm.type = GGUF_TT_F32;
    W.output_norm.n_dims = 1; W.output_norm.shape[0] = hidden;
    W.output_norm.shape[1] = 1;
    W.output_norm.shape[2] = 1; W.output_norm.shape[3] = 1;
    W.output = W.tok_embd;  // tied
    W.layers = layers;
    struct llm_ctx c = {0};
    c.cfg = cfg;
    c.W = W;
    c.arena = a;
    kv_init(&c.kv, cfg.n_layers, cfg.n_kv_heads,
            cfg.head_dim, cfg.max_position);
    // Run forward on 3 dummy tokens.
    int32_t prompt[] = {7, 13, 42};
    printf("self-test: forward pass for %zu tokens...\n",
           sizeof(prompt) / sizeof(prompt[0]));
    int32_t ok = 1;
    for (size_t i = 0; i < sizeof(prompt) / sizeof(prompt[0]); i++) {
        struct tensor * logits = llm_forward_step(&c, prompt[i], (int32_t)i);
        int64_t n = tensor_nelements(logits);
        if (n != cfg.vocab_size) {
            fprintf(stderr, "self-test: logits size mismatch: %lld != %d\n",
                    (long long)n, cfg.vocab_size);
            ok = 0;
        }
        // Check no NaN/Inf.
        int32_t bad = 0;
        for (int64_t k = 0; k < n; k++) {
            float v = logits->data[k];
            if (!(v == v) || v > 1e30f || v < -1e30f) { bad++; }
        }
        if (bad > 0) {
            fprintf(stderr, "self-test: %d non-finite logits at pos %zu\n",
                    (int)bad, i);
            ok = 0;
        } else {
            int32_t am = sample_argmax(logits);
            printf("  pos=%zu tok=%d argmax=%d logit=%.4f\n",
                   i, prompt[i], (int)am, logits->data[am]);
        }
    }

    kv_free(&c.kv);
    free(layers);
    arena_free(a);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// CLI - main() and friends. Compiled only when LLM_CLI is defined
// (the standalone `llm` binary built by Makefile). Xcode/iOS builds
// compile this file as a library without LLM_CLI, so this whole
// block is elided to keep the symbol surface clean.
// ---------------------------------------------------------------------------

static const char * llm_cli_gguf_path(void) {
    const char * env = getenv("QWEN_GGUF");
    if (env != NULL && env[0] != '\0') { return env; }
    return LLM_GGUF_PATH_DEFAULT;
}

static int32_t print_cb(const struct llm_stream_chunk * chunk,
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
static int32_t capture_cb(const struct llm_stream_chunk * chunk,
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
// SSM cache cleared via llm_reset() between turns). The optional
// --system string is prepended to the FIRST user turn's body (no
// `<|im_start|>system` block), matching im.ai's observed framing.
static int32_t run_chat(const char ** prompts, int32_t n_prompts,
                        const char * system_prompt,
                        const struct llm_sampler * sp, uint64_t seed,
                        int32_t max_new) {
    struct llm_ctx * c = llm_create(llm_cli_gguf_path());
    int32_t r = 0;
    if (!c->loaded) {
        fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
        r = 1;
    } else {
        // Persistent KV: only the DELTA gets tokenized each turn.
        // Turn 0 carries the system prefix inline with the first
        // user message (matches im.ai's framing). Subsequent turns
        // tokenize a bare `<|im_start|>user\n{Q}<|im_end|>\n` +
        // assistant gen header. llm_generate advances c->pos so the
        // next call's prefill writes into KV at the correct offset.
        int32_t * ids = (int32_t *)llm_oom(calloc(16384, sizeof(int32_t)));
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
            int32_t nids = tok_encode(&c->tok, delta_str, ids, 16384);
            struct chars reply = {0};
            llm_generate(c, ids, nids, max_new, g_min_new, sp, seed,
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
                    llm_pp_per_sec(c), (int)llm_n_prefill(c),
                    llm_tg_per_sec(c), (int)llm_n_generated(c),
                    (int)c->pos);
            chars_free(&reply);
            free(delta_str);
        }
        free(ids);
    }
    llm_destroy(c);
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
static int chat_test_cb(const struct llm_stream_chunk * chunk,
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
// twice — once on a fresh context, then again after llm_reset() — and
// verifies the reply hashes match across the two passes. This exercises
// the chat-template state machine, the persistent KV cache, the
// chunked-SSM prefill, AND the llm_reset() contract that state is
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
    struct llm_sampler sp = llm_sampler();
    uint64_t seed = 42;
    struct llm_ctx * c = llm_create(llm_cli_gguf_path());
    int32_t r = 0;
    if (!c->loaded) {
        fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
        r = 1;
    } else {
        struct chat_test_capture caps_a[N_TURNS] = {0};
        struct chat_test_capture caps_b[N_TURNS] = {0};
        int32_t * ids = (int32_t *)llm_oom(calloc(16384,
                                                  sizeof(int32_t)));
        for (int32_t pass = 0; pass < 2 && r == 0; pass++) {
            struct chat_test_capture * caps = (pass == 0) ? caps_a : caps_b;
            // Reset before pass 1; pass 0 starts from a fresh ctx so
            // the SSM state and pos are already zeroed by llm_create.
            if (pass == 1) { llm_reset(c); }
            for (int32_t t = 0; t < N_TURNS && r == 0; t++) {
                struct chat_test_capture * cap = &caps[t];
                cap->hash = 0xcbf29ce484222325ULL;
                char * delta_str = jinja_apply_delta(turns[t], NULL, 0);
                int32_t nids = tok_encode(&c->tok, delta_str,
                                          ids, 16384);
                int32_t gen = llm_generate(c, ids, nids, max_new,
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
                   " hashes match across llm_reset)\n");
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
    llm_destroy(c);
    return r;
}

// One-shot agent run: feed `question` to the model with websearch /
// fetch / distill tools advertised, let it iterate up to max_iters
// tool calls, print the final answer. --trace dumps per-iteration
// trace to stderr so the user can watch what the agent does.
static int32_t run_ask(const char * question,
                       const struct llm_sampler * sp, uint64_t seed,
                       int32_t max_iters, int32_t max_new,
                       int32_t trace) {
    struct llm_ctx * c = llm_create(llm_cli_gguf_path());
    int32_t rc = 0;
    if (!c->loaded) {
        fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
        rc = 1;
    } else {
        printf("[user] %s\n", question);
        fflush(stdout);
        char * answer = agent_run(c, question, sp, seed,
                                  max_iters, max_new, trace);
        printf("\n[assistant] %s\n", answer != NULL ? answer : "");
        free(answer);
    }
    llm_destroy(c);
    tools_global_cleanup();
    return rc;
}

static int32_t run_single(const char * prompt, int32_t max_new,
                          const struct llm_sampler * sp, uint64_t seed) {
    struct llm_ctx * c = llm_create(llm_cli_gguf_path());
    int32_t r = 0;
    if (!c->loaded) {
        fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
        r = 1;
    } else {
        printf("model: %d layers, %d heads (%d kv), head_dim=%d, "
               "hidden=%d, ffn=%d, vocab=%d\n",
               (int)c->cfg.n_layers, (int)c->cfg.n_heads,
               (int)c->cfg.n_kv_heads, (int)c->cfg.head_dim,
               (int)c->cfg.hidden_dim, (int)c->cfg.ffn_dim,
               (int)c->cfg.vocab_size);
        printf("rope: dim=%d base=%.1f sections=[%d,%d,%d,%d]\n",
               (int)c->cfg.rope_dim, c->cfg.rope_theta,
               (int)c->cfg.rope_sections[0], (int)c->cfg.rope_sections[1],
               (int)c->cfg.rope_sections[2], (int)c->cfg.rope_sections[3]);
        int32_t ids[2048];
        int32_t n = tok_encode(&c->tok, prompt, ids, 2048);
        printf("prompt tokens (%d): ", (int)n);
        for (int32_t i = 0; i < n; i++) printf("%d ", (int)ids[i]);
        printf("\n---\n%s", prompt);
        fflush(stdout);
        llm_generate(c, ids, n, max_new, g_min_new, sp, seed,
                     print_cb, NULL);
        printf("\n");
        fprintf(stderr,
                "pp: %.2f tok/s (%d tok)  tg: %.2f tok/s (%d tok)\n",
                llm_pp_per_sec(c), (int)llm_n_prefill(c),
                llm_tg_per_sec(c), (int)llm_n_generated(c));
    }
    llm_destroy(c);
    return r;
}

static int32_t run_repl(const struct llm_sampler * sp,
                        uint64_t seed, int32_t max_new) {
    struct llm_ctx * c = llm_create(llm_cli_gguf_path());
    int32_t r = 0;
    if (!c->loaded) {
        fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
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
                int32_t nids = tok_encode(&c->tok, framed, ids, 2048);
                printf("\nassistant: ");
                fflush(stdout);
                llm_generate(c, ids, nids, max_new, g_min_new, sp, seed,
                             print_cb, NULL);
                printf("\n\n> ");
                fflush(stdout);
            }
        }
    }
    llm_destroy(c);
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
    struct llm_sampler sp = llm_sampler();
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
        rc = llm_self_test();
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
        struct llm_ctx * c = llm_create(llm_cli_gguf_path());
        if (!c->loaded) {
            fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
            rc = 1;
        } else {
            const char * tpl = llm_chat_template(c);
            if (tpl != NULL) { fputs(tpl, stdout); }
        }
        llm_destroy(c);
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
        rc = llm_think_test();
    } else {
        printf("usage (set QWEN_GGUF=/path/to/model.gguf to override default):\n"
               "  llm --self-test\n"
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
