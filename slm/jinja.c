// SPDX-License-Identifier: Apache-2.0
//
// utils/jinja.c - hand-translated Qwen3.5 chat template (full).
// (Renamed from llm/jinja-template.c on 2026-05-15 and moved under
// utils/; the internal `jt_buf` was replaced with `struct chars`
// from utils/chars.c. All symbols now share the `jinja_` prefix.)
//
// Source of truth: the Jinja string stored in the GGUF under
// `tokenizer.chat_template`, dumped to tmp/jinja.txt (~155 lines).
// This file is a BEHAVIORAL port of every Jinja branch to C, hand-
// coded as a sequence of fixed string emissions over a typed
// message array. No Jinja interpreter required, no runtime parse.
//
// Coverage map (Jinja line -> here):
//
//   1-2    multimodal counters         (out of scope, not needed
//                                        for the text path; caller
//                                        with vision passes already-
//                                        formatted content strings)
//   3-41   render_content macro        only the string branch is
//                                        implemented; list-of-dicts
//                                        multimodal content is the
//                                        caller's responsibility
//   42-43  empty-messages exception    jinja_apply returns NULL
//   45-60  tools system prelude        jinja_emit_tools_prelude
//   62-66  no-tools system prelude     jinja_emit_simple_system
//   67-80  last_query_index scan       jinja_find_last_query_index
//   78-80  multi_step_tool exception   no-op (we don't fail; the
//                                        caller's responsibility to
//                                        not send a tool-only
//                                        conversation)
//   83-86  system must be at the
//          beginning                   we silently no-op when a
//                                        non-first message is system
//   87-88  user message render         jinja_emit_message (USER)
//   89-131 assistant message render    jinja_emit_message (ASSISTANT)
//          + tool_calls block          jinja_emit_tool_calls
//   132-143 tool message render        jinja_emit_message (TOOL)
//   148-154 generation prompt          jinja_emit_gen_prompt
//
// Behavioral notes faithful to the Jinja:
//
//   - assistant.content split on the LAST </think>: post-think
//     body is the visible content; pre-think (between <think> and
//     </think>) is the reasoning. Mirrors Jinja lines 94-97.
//   - Assistant turns AFTER ns.last_query_index get the
//     `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{body}`
//     wrapper; turns at or before get the bare `<|im_start|>assistant\n{body}`.
//     (Jinja lines 100-104). In a no-tools conversation
//     last_query_index is the index of the last user message, so
//     no historical assistant turn gets the wrapper.
//   - tool_calls.first iteration prepends "\n\n" when content is
//     non-blank-after-trim, else nothing. Subsequent iterations
//     prepend a single "\n". (Jinja lines 110-118).
//   - Adjacent tool messages share one outer `<|im_start|>user`
//     frame: opener is emitted only when the previous message was
//     NOT tool (or there is no previous), and the trailing
//     `<|im_end|>\n` is emitted only when the next message is NOT
//     tool (or this is the last). (Jinja lines 133-143).
//   - Generation prompt: `<|im_start|>assistant\n` plus either
//     `<think>\n` if enable_thinking or `<think>\n\n</think>\n\n`
//     otherwise (Jinja 148-154).
//
// Caller responsibilities (NOT enforced here):
//
//   - Provide content as already-rendered NUL-terminated UTF-8
//     strings. Multimodal callers (vision) pre-render
//     `<|vision_start|><|image_pad|><|vision_end|>` markers into the
//     content themselves; the source Jinja's render_content macro
//     wouldn't gain anything from being executed here.
//   - For tools, provide pre-serialized JSON strings for each tool
//     spec via `jinja_tool.json`. The Jinja's `tool | tojson` filter
//     would otherwise require a JSON serializer in C, which is out
//     of scope for this single-file port.
//   - For tool_call parameters with complex (non-scalar) values,
//     pre-serialize to JSON in `jinja_tool_call_param.value`. The
//     Jinja's `args_value | tojson | safe` filter is similarly the
//     caller's job.
//
// Distribution: this file is `#include`-d from slm.c (single-file
// lib idiom, matching neon.c / chunked.c). Listed in
// QwenHaiku.xcodeproj's membershipExceptions so Xcode does not
// compile it as an independent TU. Functions are static; each
// including TU gets its own copy.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    JINJA_ROLE_SYSTEM    = 0,
    JINJA_ROLE_USER      = 1,
    JINJA_ROLE_ASSISTANT = 2,
    JINJA_ROLE_TOOL      = 3,
};

struct jinja_tool_call_param {
    const char * name;
    // Pre-serialized as the model should see it. For scalar string
    // arguments that's the raw string; for complex args (lists,
    // mappings) it's the caller's `value | tojson` output.
    const char * value;
};

struct jinja_tool_call {
    const char * name;
    const struct jinja_tool_call_param * params;
    int                                  n_params;
};

struct jinja_message {
    int          role;
    // NUL-terminated UTF-8. May be NULL (treated as empty per the
    // Jinja's `content is none` branch on line 36).
    const char * content;
    // Optional explicit reasoning text. If NULL, jinja_apply derives
    // it from `content` by splitting on the LAST `</think>` marker
    // (mirrors Jinja lines 91-97). Callers building history from the
    // model's raw output stream typically leave this NULL and let
    // the split logic do its job; callers with structured history
    // (separate text + reasoning fields) populate this directly.
    const char * reasoning_content;
    // Optional tool_call array attached to an assistant message.
    // NULL / n_tool_calls = 0 means no tool calls.
    const struct jinja_tool_call * tool_calls;
    int                            n_tool_calls;
};

struct jinja_tool {
    // Pre-serialized JSON spec for one tool, including outer braces.
    // E.g. {"name":"get_weather","description":"...","parameters":{...}}
    const char * json;
};

// Grow-as-needed string buffer comes from utils/chars.c: `struct
// chars` + `chars_put` (data, count, capacity, NUL-terminated).
// We use it directly here; the file used to ship its own jt_buf /
// chars_put / chars_puts reimplementation.

// `|trim` filter: emit `s` with leading and trailing ASCII whitespace
// stripped (Jinja applies this to content and to the inline system
// prefix). Empty / all-whitespace input emits nothing. Whitespace
// set: space, \n, \r, \t.
static int jinja_is_ws(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static void jinja_puts_trimmed(struct chars * b, const char * s) {
    if (s != NULL) {
        const char * start = s;
        while (*start != '\0' && jinja_is_ws(*start)) { start++; }
        const char * end = s + strlen(s);
        while (end > start && jinja_is_ws(*(end - 1))) { end--; }
        if (end > start) { chars_put(b, start, (size_t)(end - start)); }
    }
}

// Returns true iff `content` (after `|trim`) is non-empty. Used by
// the tool_calls block to decide whether to prepend "\n\n" before
// the first <tool_call> (Jinja lines 110-115).
static bool jinja_content_visible(const char * content) {
    bool visible = false;
    if (content != NULL) {
        const char * p = content;
        while (*p != '\0' && !visible) {
            if (!jinja_is_ws(*p)) { visible = true; }
            p++;
        }
    }
    return visible;
}

// `content.split('</think>')[-1].lstrip('\n')` from Jinja line 96.
// Returns a pointer into `content` (no allocation).
static const char * jinja_content_after_think(const char * content) {
    const char * r = content;
    if (content != NULL) {
        const char * p = content;
        const char * last = NULL;
        while (*p != '\0') {
            if (p[0] == '<' && p[1] == '/' &&
                p[2] == 't' && p[3] == 'h' &&
                p[4] == 'i' && p[5] == 'n' &&
                p[6] == 'k' && p[7] == '>') {
                last = p + 8;
                p += 8;
            } else {
                p++;
            }
        }
        if (last != NULL) {
            r = last;
            while (*r == '\n') { r++; }
        }
    }
    return r;
}

// Emit the reasoning substring of an assistant content. Mirrors the
// derivation on Jinja line 95:
//   reasoning = content.split('</think>')[0]
//                       .rstrip('\n')
//                       .split('<think>')[-1]
//                       .lstrip('\n')
// Then |trim'd (line 99). Returns a heap-allocated string (caller
// frees) so we can do the multi-step processing without aliasing
// the input. Returns NULL if no </think> appears in content (the
// Jinja's branch on line 94 is `if '</think>' in content`).
static char * jinja_extract_reasoning(const char * content) {
    char * result = NULL;
    if (content != NULL) {
        const char * end_marker = strstr(content, "</think>");
        if (end_marker != NULL) {
            // Find the open marker; take everything between the
            // LAST <think> opener (before end_marker) and end_marker.
            const char * scan = content;
            const char * last_open = NULL;
            while (scan < end_marker) {
                if (scan[0] == '<' && scan[1] == 't' && scan[2] == 'h' &&
                    scan[3] == 'i' && scan[4] == 'n' && scan[5] == 'k' &&
                    scan[6] == '>') {
                    last_open = scan + 7;
                    scan += 7;
                } else {
                    scan++;
                }
            }
            const char * start = (last_open != NULL) ? last_open : content;
            // lstrip('\n') the start (Jinja line 95's final lstrip).
            while (start < end_marker && *start == '\n') { start++; }
            // rstrip('\n') the end (Jinja line 95's rstrip between
            // split and second split — applied to the head before
            // the inner split on <think>; head is `content.split('</think>')[0]`).
            const char * tail = end_marker;
            while (tail > start && *(tail - 1) == '\n') { tail--; }
            // Then |trim (line 99): strip outer whitespace.
            while (start < tail && jinja_is_ws(*start))       { start++; }
            while (tail > start && jinja_is_ws(*(tail - 1)))  { tail--; }
            size_t n = (size_t)(tail - start);
            result = (char *)malloc(n + 1);
            assert(result != NULL);
            memcpy(result, start, n);
            result[n] = '\0';
        }
    }
    return result;
}

// Test whether `content` (after trim) is wholly wrapped in
// `<tool_response>...</tool_response>` — used by the
// last_query_index reverse-scan (Jinja lines 70-75) to decide
// whether a USER message is actually a synthesized tool response
// (which should NOT count as the user's most recent query).
static int jinja_is_tool_response(const char * content) {
    int wrapped = 0;
    if (content != NULL) {
        const char * start = content;
        while (*start != '\0' && jinja_is_ws(*start)) { start++; }
        const char * end = content + strlen(content);
        while (end > start && jinja_is_ws(*(end - 1))) { end--; }
        const char * pre  = "<tool_response>";
        const char * post = "</tool_response>";
        size_t plen = strlen(pre);
        size_t qlen = strlen(post);
        size_t n    = (size_t)(end - start);
        if (n >= plen + qlen &&
            memcmp(start, pre, plen) == 0 &&
            memcmp(end - qlen, post, qlen) == 0) {
            wrapped = 1;
        }
    }
    return wrapped;
}

// Reverse-scan to find the index of the last user message that is
// NOT a tool_response. Mirrors Jinja lines 67-77. Returns
// n_msgs - 1 as a safe default if no such message exists (the
// Jinja would `raise_exception` per line 79; we let the caller
// handle weird conversations gracefully).
static int jinja_find_last_query_index(const struct jinja_message * msgs,
                                       int n_msgs) {
    int  last  = n_msgs - 1;
    bool found = false;
    int  i     = n_msgs - 1;
    while (i >= 0 && !found) {
        if (msgs[i].role == JINJA_ROLE_USER &&
            !jinja_is_tool_response(msgs[i].content)) {
            last = i;
            found = true;
        }
        i--;
    }
    return last;
}

// Literal block from Jinja line 53: the fixed instruction text that
// follows the <tools>...</tools> JSON list in the tools system frame.
// Mostly verbatim from the official Qwen3.5 template — one bullet was
// dropped (see HISTORY below) and the HARD RULES nudge appended in
// jinja_emit_tools_prelude tightens what the official text leaves
// permissive. Drift here changes what the model sees and may affect
// tool-using fine-tunes, but the 0.8B's anchoring behavior on the
// dropped line was strong enough to justify the edit.
//
// HISTORY (2026-05-17): removed the bullet
//   "- You may provide optional reasoning for your function call in
//    natural language BEFORE the function call, but NOT after"
// because it directly conflicted with the HARD RULES nudge "Do NOT
// output any reasoning, natural-language commentary, or filler text
// before or after the <tool_call> block" and left the model picking
// the more permissive interpretation. Observed in --repl with the IP-
// location prompt: model emitted a chatty preamble before the call.
static const char K_TOOL_INSTRUCTIONS[] =
    "\n\nIf you choose to call a function ONLY reply in the following format"
    " with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner"
    " <function=...></function> block must be nested within"
    " <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- If there is no function call available, answer the question like normal"
    " with your current knowledge and do not tell the user about function"
    " calls\n</IMPORTANT>";

// Tools system prelude (Jinja lines 45-60).
static void jinja_emit_tools_prelude(struct chars * b,
                                     const struct jinja_tool * tools,
                                     int n_tools,
                                     const struct jinja_message * msgs,
                                     int n_msgs) {
    chars_puts(b, "<|im_start|>system\n");
    chars_puts(b, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (int i = 0; i < n_tools; i++) {
        chars_puts(b, "\n");
        chars_puts(b, tools[i].json);
    }
    chars_puts(b, "\n</tools>");
    chars_puts(b, K_TOOL_INSTRUCTIONS);
    // Behavioral nudge appended AFTER the structural rules. The
    // 0.8B often commits to "I don't have access to real-time
    // information" instead of calling websearch, and once that
    // pattern is in KV the model anchors to it across the rest of
    // a --repl session. Explicit "do not refuse, call websearch"
    // language tips the next-token distribution toward the tool
    // marker. Kept outside K_TOOL_INSTRUCTIONS so the official
    // jinja text stays bit-stable.
    //
    // Wording aligned with the community "qwen3.5-enhanced.jinja"
    // (allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) which
    // reports 50% -> 90% reliability for 3.5 tool-calling on the
    // back of two anchors: (1) "use the tool instead of answering
    // from memory", (2) "put any reasoning BEFORE the first
    // <tool_call>, never after". We keep the official Qwen
    // K_TOOL_INSTRUCTIONS verbatim above (training-distribution
    // alignment) and override the relaxed "may provide optional
    // reasoning BEFORE ... but NOT after" with a HARD RULE here so
    // the 0.8B emits the <tool_call> directly without a chatty
    // preamble. See reference-qwen35-tool-calling.md.
    chars_puts(b,
        "\n\n"
        "HARD RULES for tool use:\n"
        "- If a suitable tool exists for the user's request, use it"
        " instead of answering from memory or training data.\n"
        "- Use the websearch function for any question that depends"
        " on current or real-time information: time, dates, prices,"
        " news, weather, schedules, traffic, scores, exchange rates,"
        " or anything that may have changed since your training.\n"
        "- Do NOT tell the user you lack access to real-time data —"
        " call websearch instead.\n"
        "- Do NOT output any reasoning, natural-language commentary,"
        " or filler text before or after the <tool_call> block."
        " Emit the <tool_call> directly.");
    if (n_msgs > 0 && msgs[0].role == JINJA_ROLE_SYSTEM) {
        // Inline the (trimmed) system content after the instructions.
        // The Jinja's `if content` check on line 56 only emits the
        // separator + content if the trimmed content is non-empty.
        if (jinja_content_visible(msgs[0].content)) {
            chars_puts(b, "\n\n");
            jinja_puts_trimmed(b, msgs[0].content);
        }
    }
    chars_puts(b, "<|im_end|>\n");
}

// No-tools system prelude (Jinja lines 62-66). Only emits when the
// first message is a system message.
static void jinja_emit_simple_system(struct chars * b,
                                     const struct jinja_message * msgs,
                                     int n_msgs) {
    if (n_msgs > 0 && msgs[0].role == JINJA_ROLE_SYSTEM) {
        chars_puts(b, "<|im_start|>system\n");
        jinja_puts_trimmed(b, msgs[0].content);
        chars_puts(b, "<|im_end|>\n");
    }
}

// Per-message tool-calls render (Jinja lines 105-130). Called from
// within the assistant message body, AFTER the assistant role frame
// has been opened and the body content emitted.
static void jinja_emit_tool_calls(struct chars * b,
                                  const struct jinja_tool_call * tcs,
                                  int n_tcs,
                                  int body_visible) {
    for (int j = 0; j < n_tcs; j++) {
        const struct jinja_tool_call * tc = &tcs[j];
        if (j == 0) {
            if (body_visible) {
                chars_puts(b, "\n\n<tool_call>\n<function=");
            } else {
                chars_puts(b, "<tool_call>\n<function=");
            }
        } else {
            chars_puts(b, "\n<tool_call>\n<function=");
        }
        chars_puts(b, tc->name);
        chars_puts(b, ">\n");
        for (int k = 0; k < tc->n_params; k++) {
            chars_puts(b, "<parameter=");
            chars_puts(b, tc->params[k].name);
            chars_puts(b, ">\n");
            chars_puts(b, tc->params[k].value);
            chars_puts(b, "\n</parameter>\n");
        }
        chars_puts(b, "</function>\n</tool_call>");
    }
}

// Emit a single message inside the main render loop (Jinja lines
// 81-147). Takes the surrounding context (i, n_msgs, msgs,
// last_query_index) because some branches consult previtem /
// nextitem and the last_query_index.
static void jinja_emit_message(struct chars * b,
                               const struct jinja_message * msgs,
                               int i, int n_msgs,
                               int last_query_index) {
    const struct jinja_message * m = &msgs[i];
    if (m->role == JINJA_ROLE_SYSTEM) {
        // Already emitted by the prelude; non-first system is an
        // error per the Jinja (line 85). No-op here.
    } else if (m->role == JINJA_ROLE_USER) {
        chars_puts(b, "<|im_start|>user\n");
        jinja_puts_trimmed(b, m->content);
        chars_puts(b, "<|im_end|>\n");
    } else if (m->role == JINJA_ROLE_ASSISTANT) {
        // Derive reasoning / body. If caller supplied
        // reasoning_content explicitly, use the content as-is.
        // Else split on </think> and extract both halves.
        const char * raw_content = m->content;
        char *       derived_reason = NULL;
        const char * body   = raw_content;
        const char * reason = m->reasoning_content;
        if (reason == NULL && raw_content != NULL) {
            derived_reason = jinja_extract_reasoning(raw_content);
            reason = derived_reason;
            body   = jinja_content_after_think(raw_content);
        }
        // |trim the body content for the body_visible check
        // (Jinja's `content|trim` on line 111).
        int body_visible = jinja_content_visible(body);
        if (i > last_query_index) {
            chars_puts(b, "<|im_start|>assistant\n<think>\n");
            jinja_puts_trimmed(b, reason);
            chars_puts(b, "\n</think>\n\n");
            jinja_puts_trimmed(b, body);
        } else {
            chars_puts(b, "<|im_start|>assistant\n");
            jinja_puts_trimmed(b, body);
        }
        if (m->tool_calls != NULL && m->n_tool_calls > 0) {
            jinja_emit_tool_calls(b, m->tool_calls,
                                  m->n_tool_calls, body_visible);
        }
        chars_puts(b, "<|im_end|>\n");
        free(derived_reason);
    } else if (m->role == JINJA_ROLE_TOOL) {
        int prev_is_tool = (i > 0) &&
                           (msgs[i - 1].role == JINJA_ROLE_TOOL);
        int next_is_tool = (i + 1 < n_msgs) &&
                           (msgs[i + 1].role == JINJA_ROLE_TOOL);
        if (!prev_is_tool) {
            chars_puts(b, "<|im_start|>user");
        }
        chars_puts(b, "\n<tool_response>\n");
        jinja_puts_trimmed(b, m->content);
        chars_puts(b, "\n</tool_response>");
        if (!next_is_tool) {
            chars_puts(b, "<|im_end|>\n");
        }
    }
    // No `else` raise — the caller is trusted not to pass garbage
    // roles. The Jinja would raise_exception (line 145); we silently
    // skip.
}

// Emit the generation prompt. Mirrors Jinja lines 148-154.
static void jinja_emit_gen_prompt(struct chars * b, int enable_thinking) {
    chars_puts(b, "<|im_start|>assistant\n");
    if (enable_thinking) {
        chars_puts(b, "<think>\n");
    } else {
        chars_puts(b, "<think>\n\n</think>\n\n");
    }
}

// Render the whole conversation. Returns a heap-allocated NUL-
// terminated string the caller frees with free(). Returns NULL if
// `msgs` is NULL or `n_msgs <= 0` (matches the Jinja's `if not
// messages` exception on line 42).
//
// Pass `tools=NULL, n_tools=0` for a plain chat. Pass a non-empty
// tools array to emit the tool-advertisement system prelude in
// place of (or alongside) a system message.
static char * jinja_apply(const struct jinja_message * msgs, int n_msgs,
                          const struct jinja_tool * tools, int n_tools,
                          int add_generation_prompt,
                          int enable_thinking) {
    char * result = NULL;
    if (msgs != NULL && n_msgs > 0) {
        struct chars b = {0};
        if (tools != NULL && n_tools > 0) {
            jinja_emit_tools_prelude(&b, tools, n_tools, msgs, n_msgs);
        } else {
            jinja_emit_simple_system(&b, msgs, n_msgs);
        }
        int last_query_index = jinja_find_last_query_index(msgs, n_msgs);
        for (int i = 0; i < n_msgs; i++) {
            jinja_emit_message(&b, msgs, i, n_msgs, last_query_index);
        }
        if (add_generation_prompt) {
            jinja_emit_gen_prompt(&b, enable_thinking);
        }
        result = b.data;
    }
    return result;
}

// Format the canonical Qwen3 system block for slm_ctx_system_prompt.
// Emits:
//   <|im_start|>system\n
//   <effort_prefix><sys_text>\n
//   <tools block if tools array non-empty>\n
//   <|im_end|>\n
// `tools` / `n_tools` carry the serialized JSON tool specs. When
// n_tools == 0 no tools block is emitted. Caller frees the result.
static char * jinja_format_system_block(const char * sys_text,
                                        const struct slm_ctrl * ctrl,
                                        const struct jinja_tool * tools,
                                        int n_tools) {
    struct chars b = {0};
    chars_puts(&b, "<|im_start|>system\n");
    // Effort prefix per ctrl->effort.
    if (ctrl != NULL && ctrl->effort != NULL) {
        if (strcmp(ctrl->effort, "low") == 0) {
            chars_puts(&b, "(brief: 1-2 sentences)\n\n");
        } else if (strcmp(ctrl->effort, "high") == 0) {
            chars_puts(&b, "(think carefully step-by-step before "
                           "answering)\n\n");
        }
    }
    // System text (may be empty string).
    if (sys_text != NULL && sys_text[0] != '\0') {
        chars_puts(&b, sys_text);
        chars_puts(&b, "\n");
    }
    // Tools block if any tools provided.
    if (tools != NULL && n_tools > 0) {
        chars_puts(&b, "# Tools\n\nYou have access to the following"
                       " functions:\n\n<tools>");
        for (int i = 0; i < n_tools; i++) {
            chars_puts(&b, "\n");
            chars_puts(&b, tools[i].json);
        }
        chars_puts(&b, "\n</tools>");
        chars_puts(&b, K_TOOL_INSTRUCTIONS);
        chars_puts(&b, "\n");
    }
    chars_puts(&b, "<|im_end|>\n");
    return b.data;
}

// Format a single user-turn delta for slm_generate. Emits:
//   <|im_start|>user\n<user_msg><|im_end|>\n
//   <|im_start|>assistant\n<think prefix>
// where think prefix is `<think>\n` (think=true) or
// `<think>\n\n</think>\n\n` (think=false).
// Caller frees the returned buffer.
static char * jinja_format_turn_delta(const char * user_msg,
                                      bool think) {
    char * result = NULL;
    if (user_msg != NULL) {
        struct chars b = {0};
        chars_puts(&b, "<|im_start|>user\n");
        chars_puts(&b, user_msg);
        chars_puts(&b, "<|im_end|>\n");
        jinja_emit_gen_prompt(&b, think ? 1 : 0);
        result = b.data;
    }
    return result;
}

// Self-test: render conversations covering every branch and assert
// against hand-traced golden strings. Returns 0 on PASS.

static int32_t jinja_self_test(void) {
    int32_t failures = 0;
    // Fixture A: bare user + assistant, no system, no gen prompt.
    // The Jinja's last_query_index ends up at the (only) user
    // index (0); the assistant at index 1 is `> last_query_index`
    // so the <think>...</think> wrapper applies. Empty reasoning
    // because the content has no </think> marker.
    {
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER,      "Hi",    NULL, NULL, 0 },
            { JINJA_ROLE_ASSISTANT, "Hello", NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 2, NULL, 0, 0, 0);
        const char * gold =
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "Hello<|im_end|>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test A FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture B: with system, gen prompt on, thinking off.
    {
        struct jinja_message msgs[] = {
            { JINJA_ROLE_SYSTEM,    "Be brief.", NULL, NULL, 0 },
            { JINJA_ROLE_USER,      "Hi",        NULL, NULL, 0 },
            { JINJA_ROLE_ASSISTANT, "Hello",     NULL, NULL, 0 },
            { JINJA_ROLE_USER,      "Bye",       NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 4, NULL, 0, 1, 0);
        const char * gold =
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello<|im_end|>\n"
            "<|im_start|>user\nBye<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test B FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture C: thinking-enabled gen prompt.
    {
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER, "Hi", NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 1, NULL, 0, 1, 1);
        const char * gold =
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test C FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture D: assistant body with reasoning to extract. As in
    // fixture A this assistant is `> last_query_index`, so the
    // wrapper applies and reasoning_content is derived from the
    // <think>...</think> block in the content.
    {
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER,      "Hi", NULL, NULL, 0 },
            { JINJA_ROLE_ASSISTANT,
              "<think>\nlet me think\n</think>\n\nHello there",
              NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 2, NULL, 0, 0, 0);
        const char * gold =
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nlet me think\n</think>"
            "\n\nHello there<|im_end|>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test D FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture E: jinja_format_system_block, no tools, no effort.
    {
        struct slm_ctrl ctrl = {0};
        ctrl.tools  = false;
        ctrl.think  = false;
        ctrl.effort = NULL;
        char * out = jinja_format_system_block("Be helpful.", &ctrl,
                                               NULL, 0);
        const char * gold =
            "<|im_start|>system\nBe helpful.\n<|im_end|>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test E FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture F: jinja_format_turn_delta, thinking off.
    {
        char * out = jinja_format_turn_delta("Hello!", false);
        const char * gold =
            "<|im_start|>user\nHello!<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test F FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture G: tools prelude with system message inline.
    {
        struct jinja_tool tools[] = {
            { "{\"name\":\"echo\",\"parameters\":{}}" },
        };
        struct jinja_message msgs[] = {
            { JINJA_ROLE_SYSTEM, "Be helpful.",     NULL, NULL, 0 },
            { JINJA_ROLE_USER,   "Please echo Hi.", NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 2, tools, 1, 0, 0);
        // Spot-check: opens with system frame, contains the tool
        // JSON, ends with user frame (no gen prompt). The full
        // instruction text is in K_TOOL_INSTRUCTIONS; we check
        // for its sentinel substrings rather than the entire body
        // (the body is ~600 bytes, fixed verbatim from Jinja line 53).
        bool ok = (out != NULL) &&
                  (strstr(out, "<|im_start|>system\n# Tools") != NULL) &&
                 (strstr(out, "<tools>\n"
                              "{\"name\":\"echo\",\"parameters\":{}}\n"
                              "</tools>") != NULL) &&
                 (strstr(out, "<IMPORTANT>") != NULL) &&
                 (strstr(out, "Be helpful.<|im_end|>") != NULL) &&
                 (strstr(out, "<|im_start|>user\nPlease echo Hi.<|im_end|>")
                  != NULL);
        if (!ok) {
            fprintf(stderr,
                    "jinja-test G FAIL\n  got:  %s\n",
                    out == NULL ? "(null)" : out);
            failures++;
        }
        free(out);
    }
    // Fixture H: assistant with tool_calls (no preceding body content).
    {
        struct jinja_tool_call_param params[] = {
            { "city", "Paris" },
        };
        struct jinja_tool_call tcs[] = {
            { "get_weather", params, 1 },
        };
        // last_query_index logic: msgs[0] is the only user, so
        // last_query_index=0. msgs[1] (assistant) has index 1 > 0,
        // so the <think>...</think> wrapper applies.
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER,      "Weather in Paris?", NULL, NULL, 0 },
            { JINJA_ROLE_ASSISTANT, "", NULL, tcs, 1 },
        };
        char * out = jinja_apply(msgs, 2, NULL, 0, 0, 0);
        const char * gold =
            "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test H FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture I: multi-step tool sequence (user, assistant w/
    // tool_call, tool response, gen prompt).
    {
        struct jinja_tool_call_param params[] = {
            { "city", "Paris" },
        };
        struct jinja_tool_call tcs[] = {
            { "get_weather", params, 1 },
        };
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER,      "Weather in Paris?", NULL, NULL, 0 },
            { JINJA_ROLE_ASSISTANT, "", NULL, tcs, 1 },
            { JINJA_ROLE_TOOL,      "Sunny, 22C", NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 3, NULL, 0, 1, 0);
        const char * gold =
            "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nSunny, 22C\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test I FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    // Fixture J: adjacent tool responses share one outer user
    // frame (no <|im_end|> between them).
    {
        struct jinja_message msgs[] = {
            { JINJA_ROLE_USER, "Question?",      NULL, NULL, 0 },
            { JINJA_ROLE_TOOL, "First result",   NULL, NULL, 0 },
            { JINJA_ROLE_TOOL, "Second result",  NULL, NULL, 0 },
        };
        char * out = jinja_apply(msgs, 3, NULL, 0, 0, 0);
        // last_query_index reverse-scan: msgs[2]=TOOL skip,
        // msgs[1]=TOOL skip, msgs[0]=USER not tool_response →
        // last_query_index=0. msgs[1] and msgs[2] have indices > 0,
        // so the tool branch applies (not assistant branch).
        const char * gold =
            "<|im_start|>user\nQuestion?<|im_end|>\n"
            "<|im_start|>user"
            "\n<tool_response>\nFirst result\n</tool_response>"
            "\n<tool_response>\nSecond result\n</tool_response>"
            "<|im_end|>\n";
        if (out == NULL || strcmp(out, gold) != 0) {
            fprintf(stderr,
                    "jinja-test J FAIL\n  got:  %s\n  want: %s\n",
                    out == NULL ? "(null)" : out, gold);
            failures++;
        }
        free(out);
    }
    if (failures == 0) {
        printf("jinja-test: PASS (10 fixtures)\n");
    } else {
        printf("jinja-test: FAIL (%d/10 fixtures failed)\n",
               (int)failures);
    }
    return failures;
}
