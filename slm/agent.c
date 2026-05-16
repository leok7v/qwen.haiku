// SPDX-License-Identifier: Apache-2.0
//
// agent.c - tool-calling agent loop. Single-file lib `#include`-d
// from slm.c under `#ifdef LLM_WITH_TOOLS` (same gate as tools.c —
// the agent loop is only useful if the tool primitives are linked).
//
// The pieces this glues together:
//
//   - jinja-template.c      frames the conversation (system prompt
//                            with tool advertisement, user / assistant
//                            / tool messages, generation prompt)
//   - tools.c               does the actual work (websearch / fetch
//                            / distill via libcurl)
//   - slm_generate          drives the model
//
// What this file owns: parsing `<tool_call>` blocks out of the
// model's streamed output, dispatching the named tool with the
// parsed parameters, building the tool_response frame for the next
// turn, and looping until the model produces final content with no
// more tool calls (or until max_iters is hit).
//
// Tool-call syntax (matches the Qwen3.5 Jinja's template, see
// tmp/jinja.txt lines 105-128):
//
//   <tool_call>
//   <function=websearch>
//   <parameter=query>
//   What rock band recorded HelloWorld
//   </parameter>
//   <parameter=max_results>
//   5
//   </parameter>
//   </function>
//   </tool_call>
//
// Multiple tool_call blocks may appear in one assistant turn; this
// parser extracts all of them and dispatches each before continuing
// the loop. Parameter values are NUL-terminated UTF-8; the
// per-parameter `\n` immediately after `<parameter=NAME>` and the
// `\n` immediately before `</parameter>` are stripped (mirrors the
// Jinja's value emission).

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct agent_param {
    char * name;
    char * value;
};

struct agent_call {
    char *               name;
    struct agent_param * params;
    int                  n_params;
};

// JSON tool specs the model sees in its system prompt. Hand-written
// (not generated from a schema lib — keeps the dep surface zero).
// The Jinja's `tool | tojson` would pretty-print these; we ship the
// already-serialized string and the Jinja just embeds it verbatim.
static const char * AGENT_TOOL_WEBSEARCH =
    "{\"type\":\"function\",\"function\":{"
    "\"name\":\"websearch\","
    "\"description\":\"Search the web with DuckDuckGo."
    " Returns a plain-text list of result titles and snippets."
    " Use this to find information not in your training data.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"query\":{\"type\":\"string\","
    "\"description\":\"The search query (a natural-language question"
    " or keywords).\"},"
    "\"max_results\":{\"type\":\"integer\","
    "\"description\":\"Maximum number of results to return."
    " Default 5, max 16.\"}},"
    "\"required\":[\"query\"]}}}";

static const char * AGENT_TOOL_FETCH =
    "{\"type\":\"function\",\"function\":{"
    "\"name\":\"fetch\","
    "\"description\":\"HTTP GET a URL and return the raw response"
    " body (HTML, JSON, plain text). Body is truncated at 200 KB.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"url\":{\"type\":\"string\","
    "\"description\":\"The full URL to fetch (must include http(s)://).\"},"
    "\"timeout\":{\"type\":\"integer\","
    "\"description\":\"Seconds to wait before giving up. Default 30.\"}},"
    "\"required\":[\"url\"]}}}";

static const char * AGENT_TOOL_DISTILL =
    "{\"type\":\"function\",\"function\":{"
    "\"name\":\"distill\","
    "\"description\":\"Strip HTML tags and collapse whitespace,"
    " returning the meaningful text content of a page. Compose with"
    " fetch when you want the body of a URL as plain text.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"html\":{\"type\":\"string\","
    "\"description\":\"The HTML string to distill.\"}},"
    "\"required\":[\"html\"]}}}";

// Look up a parameter by name in a parsed call. Returns NULL if not
// present. Linear scan — call->n_params is small (rarely > 4).
static const char * agent_call_param(const struct agent_call * call,
                                     const char * key) {
    const char * v = NULL;
    for (int i = 0; i < call->n_params && v == NULL; i++) {
        if (strcmp(call->params[i].name, key) == 0) {
            v = call->params[i].value;
        }
    }
    return v;
}

static void agent_free_calls(struct agent_call * calls, int n) {
    for (int i = 0; i < n; i++) {
        free(calls[i].name);
        for (int j = 0; j < calls[i].n_params; j++) {
            free(calls[i].params[j].name);
            free(calls[i].params[j].value);
        }
        free(calls[i].params);
    }
}

// Carve [start, end) out of `src` into a freshly malloc'd
// NUL-terminated string. Returns NULL if start >= end. Caller frees.
static char * agent_carve(const char * start, const char * end) {
    char * r = NULL;
    if (end > start) {
        size_t n = (size_t)(end - start);
        r = (char *)malloc(n + 1);
        if (r != NULL) {
            memcpy(r, start, n);
            r[n] = '\0';
        }
    }
    return r;
}

// Read a JSON string literal starting at `*pp` (which must point at
// the opening `"`). Advances `*pp` past the closing `"`. Returns a
// heap-allocated, NUL-terminated, unescaped UTF-8 string (caller
// frees), or NULL on malformed input. Handles \" \\ \/ \n \r \t \b \f
// and \uXXXX (single BMP code point, encoded as UTF-8). No surrogate-
// pair support — small models rarely emit them in tool args.
static char * agent_json_string(const char ** pp) {
    char * result = NULL;
    const char * p = *pp;
    if (*p == '"') {
        p++;
        struct ts_buf buf = {0};
        bool closed = false;
        bool err = false;
        while (!closed && !err && *p != '\0') {
            char c = *p;
            if (c == '"') {
                closed = true;
                p++;
            } else if (c == '\\') {
                p++;
                char e = *p;
                if (e == 'n')      { ts_put(&buf, "\n", 1); p++; }
                else if (e == 'r') { ts_put(&buf, "\r", 1); p++; }
                else if (e == 't') { ts_put(&buf, "\t", 1); p++; }
                else if (e == 'b') { ts_put(&buf, "\b", 1); p++; }
                else if (e == 'f') { ts_put(&buf, "\f", 1); p++; }
                else if (e == '"' || e == '\\' || e == '/') {
                    ts_put(&buf, &e, 1); p++;
                } else if (e == 'u') {
                    p++;
                    unsigned cp = 0;
                    bool ok = true;
                    for (int k = 0; ok && k < 4; k++) {
                        char h = *p;
                        int d = -1;
                        if (h >= '0' && h <= '9') { d = h - '0'; }
                        else if (h >= 'a' && h <= 'f') { d = h - 'a' + 10; }
                        else if (h >= 'A' && h <= 'F') { d = h - 'A' + 10; }
                        else { ok = false; }
                        if (ok) { cp = cp * 16 + (unsigned)d; p++; }
                    }
                    if (!ok) { err = true; }
                    else if (cp < 0x80) {
                        char ch = (char)cp; ts_put(&buf, &ch, 1);
                    } else if (cp < 0x800) {
                        char two[2];
                        two[0] = (char)(0xC0 | (cp >> 6));
                        two[1] = (char)(0x80 | (cp & 0x3F));
                        ts_put(&buf, two, 2);
                    } else {
                        char three[3];
                        three[0] = (char)(0xE0 | (cp >> 12));
                        three[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        three[2] = (char)(0x80 | (cp & 0x3F));
                        ts_put(&buf, three, 3);
                    }
                } else {
                    err = true;
                }
            } else {
                ts_put(&buf, &c, 1);
                p++;
            }
        }
        if (closed && !err) {
            result = buf.data != NULL ? buf.data : strdup("");
            *pp = p;
        } else {
            free(buf.data);
        }
    }
    return result;
}

// Skip JSON whitespace (space, \n, \r, \t) at *pp.
static void agent_json_skip_ws(const char ** pp) {
    const char * p = *pp;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
        p++;
    }
    *pp = p;
}

// Read a JSON value (string|number|bool|null|object|array) as raw
// text starting at *pp; advances *pp past it. Returns a heap-
// allocated string of the raw JSON span (for objects / arrays this
// is the literal JSON text, which is what the model wants the tool
// to see for complex args). Returns NULL on malformed input.
static char * agent_json_value(const char ** pp) {
    char * result = NULL;
    agent_json_skip_ws(pp);
    const char * p = *pp;
    if (*p == '"') {
        result = agent_json_string(pp);
    } else if (*p == '{' || *p == '[') {
        // Track nesting; copy literal span. Strings inside the span
        // are walked so braces inside strings don't fool the depth.
        char open  = *p;
        char close = (open == '{') ? '}' : ']';
        int  depth = 0;
        const char * start = p;
        bool closed = false;
        while (!closed && *p != '\0') {
            char c = *p;
            if (c == open)  { depth++; p++; }
            else if (c == close) {
                depth--;
                p++;
                if (depth == 0) { closed = true; }
            } else if (c == '"') {
                char * dummy = agent_json_string(&p);
                free(dummy);  // we don't need the value, just to skip
            } else {
                p++;
            }
        }
        if (closed) {
            result = agent_carve(start, p);
            *pp = p;
        }
    } else {
        // number / true / false / null — read until separator.
        const char * start = p;
        while (*p != '\0' && *p != ',' && *p != '}' && *p != ']'
                          && *p != ' ' && *p != '\n' && *p != '\r'
                          && *p != '\t') {
            p++;
        }
        if (p > start) {
            result = agent_carve(start, p);
            *pp = p;
        }
    }
    return result;
}

// Try parsing the OpenAI-style JSON tool-call shape:
//   <tool_call>{"name":"<fn>","arguments":{"k1":"v1","k2":"v2",...}}</tool_call>
// Returns true on success (call populated), false if the body
// doesn't look like JSON or the parse failed. Used as a fallback by
// agent_parse_tool_calls when small models emit the OpenAI shape
// instead of the Qwen <function=...><parameter=...> verbose shape.
static bool agent_parse_json_call(const char * body, const char * end,
                                  struct agent_call * call) {
    bool ok = false;
    const char * p = body;
    agent_json_skip_ws(&p);
    if (p < end && *p == '{') {
        // Walk object members.
        p++;
        char * fn_name = NULL;
        struct agent_param tmp[16];
        int np = 0;
        agent_json_skip_ws(&p);
        bool err = false;
        bool finished = false;
        while (!finished && !err && p < end) {
            char c = *p;
            if (c == '}') { finished = true; p++; }
            else if (c == '"') {
                char * key = agent_json_string(&p);
                if (key == NULL) { err = true; }
                else {
                    agent_json_skip_ws(&p);
                    if (*p != ':') { err = true; }
                    else {
                        p++;
                        if (strcmp(key, "name") == 0 ||
                            strcmp(key, "function") == 0) {
                            free(fn_name);
                            fn_name = agent_json_value(&p);
                        } else if (strcmp(key, "arguments") == 0 ||
                                   strcmp(key, "parameters") == 0) {
                            agent_json_skip_ws(&p);
                            if (*p != '{') {
                                // Skip whatever it is.
                                char * dummy = agent_json_value(&p);
                                free(dummy);
                            } else {
                                p++;
                                agent_json_skip_ws(&p);
                                bool args_done = false;
                                while (!args_done && !err &&
                                       np < 16 && p < end) {
                                    if (*p == '}') {
                                        args_done = true; p++;
                                    } else if (*p == '"') {
                                        char * ak = agent_json_string(&p);
                                        if (ak == NULL) { err = true; }
                                        else {
                                            agent_json_skip_ws(&p);
                                            if (*p != ':') { err = true; }
                                            else {
                                                p++;
                                                char * av =
                                                    agent_json_value(&p);
                                                tmp[np].name  = ak;
                                                tmp[np].value =
                                                    av != NULL ? av
                                                               : strdup("");
                                                np++;
                                                agent_json_skip_ws(&p);
                                                if (*p == ',') { p++; }
                                                agent_json_skip_ws(&p);
                                            }
                                            if (err) { free(ak); }
                                        }
                                    } else {
                                        err = true;
                                    }
                                }
                            }
                        } else {
                            // Unknown key — skip its value.
                            char * dummy = agent_json_value(&p);
                            free(dummy);
                        }
                        agent_json_skip_ws(&p);
                        if (*p == ',') { p++; }
                        agent_json_skip_ws(&p);
                    }
                    free(key);
                }
            } else {
                err = true;
            }
        }
        if (!err && fn_name != NULL) {
            call->name     = fn_name;
            call->n_params = np;
            if (np > 0) {
                call->params = (struct agent_param *)malloc(
                    (size_t)np * sizeof(struct agent_param));
                memcpy(call->params, tmp,
                       (size_t)np * sizeof(struct agent_param));
            } else {
                call->params = NULL;
            }
            ok = true;
        } else {
            free(fn_name);
            for (int k = 0; k < np; k++) {
                free(tmp[k].name); free(tmp[k].value);
            }
        }
    }
    return ok;
}

// Parse all <tool_call>...</tool_call> blocks out of `text` (NUL-
// terminated). Stores up to `max_out` parsed calls in `out`; returns
// the number actually parsed. Caller frees with agent_free_calls.
//
// Accepts BOTH formats — small models like Qwen3.5-0.8B don't
// reliably emit the verbose Qwen syntax, so we fall back to the
// OpenAI JSON shape:
//
//   Qwen verbose:
//     <tool_call>
//     <function=NAME>
//     <parameter=KEY>
//     VALUE
//     </parameter>
//     </function>
//     </tool_call>
//
//   OpenAI JSON:
//     <tool_call>
//     {"name": "NAME", "arguments": {"KEY": "VALUE", ...}}
//     </tool_call>
//
// Tolerant of a stray space inside the opening tag (`<tool_call >`).
// Skips malformed blocks silently — the model occasionally emits
// stray text adjacent to a real call, and a strict reject would
// lose the recoverable cases.
static int agent_parse_tool_calls(const char * text,
                                  struct agent_call * out,
                                  int max_out) {
    int n = 0;
    const char * p = text;
    while (n < max_out && p != NULL) {
        // Find next opening tag — tolerate "<tool_call>" or
        // "<tool_call >" (small models sometimes emit a stray space).
        const char * start = strstr(p, "<tool_call");
        if (start == NULL) {
            p = NULL;
        } else {
            const char * gt = strchr(start, '>');
            const char * end = strstr(start, "</tool_call>");
            if (gt == NULL || end == NULL || gt >= end) {
                p = NULL;
            } else {
                const char * body = gt + 1;
                const char * fn = strstr(body, "<function=");
                bool parsed_ok = false;
                struct agent_call call = {0};
                if (fn != NULL && fn < end) {
                    // Qwen verbose shape.
                    const char * name_start = fn + 10;
                    const char * name_end =
                        (const char *)memchr(name_start, '>',
                                             (size_t)(end - name_start));
                    if (name_end != NULL) {
                        call.name = agent_carve(name_start, name_end);
                        struct agent_param tmp[16];
                        int np = 0;
                        const char * pp = name_end + 1;
                        bool more = true;
                        while (more && np < 16) {
                            const char * pm = strstr(pp, "<parameter=");
                            if (pm == NULL || pm >= end) {
                                more = false;
                            } else {
                                const char * ks = pm + 11;
                                const char * ke = (const char *)memchr(
                                    ks, '>', (size_t)(end - ks));
                                if (ke == NULL) {
                                    more = false;
                                } else {
                                    const char * vs = ke + 1;
                                    if (*vs == '\n') { vs++; }
                                    const char * ve =
                                        strstr(vs, "</parameter>");
                                    if (ve == NULL || ve >= end) {
                                        more = false;
                                    } else {
                                        const char * v_trim = ve;
                                        while (v_trim > vs &&
                                               *(v_trim - 1) == '\n') {
                                            v_trim--;
                                        }
                                        tmp[np].name  =
                                            agent_carve(ks, ke);
                                        tmp[np].value =
                                            agent_carve(vs, v_trim);
                                        np++;
                                        pp = ve + 12;
                                    }
                                }
                            }
                        }
                        call.params = (np > 0)
                            ? (struct agent_param *)malloc(
                                  (size_t)np *
                                  sizeof(struct agent_param))
                            : NULL;
                        if (call.params != NULL && np > 0) {
                            memcpy(call.params, tmp,
                                   (size_t)np *
                                   sizeof(struct agent_param));
                        }
                        call.n_params = np;
                        parsed_ok = (call.name != NULL);
                    }
                } else {
                    // Fall back to OpenAI JSON shape.
                    parsed_ok =
                        agent_parse_json_call(body, end, &call);
                }
                if (parsed_ok) {
                    out[n] = call;
                    n++;
                } else {
                    free(call.name);
                    for (int j = 0; j < call.n_params; j++) {
                        free(call.params[j].name);
                        free(call.params[j].value);
                    }
                    free(call.params);
                }
                p = end + 12;  // strlen("</tool_call>")
            }
        }
    }
    return n;
}

// Invoke one parsed call against the tools.c primitives. Maps the
// model's named function to the matching tools_* entry. Unknown
// function names produce an error result so the model sees its
// mistake in the next turn's <tool_response>.
static void agent_dispatch(const struct agent_call * call,
                           struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    if (call->name == NULL) {
        out->error = strdup("agent: tool call missing function name");
    } else if (strcmp(call->name, "websearch") == 0) {
        const char * q = agent_call_param(call, "query");
        const char * mr_s = agent_call_param(call, "max_results");
        int mr = (mr_s != NULL) ? atoi(mr_s) : 5;
        tools_websearch(q != NULL ? q : "", mr, out);
    } else if (strcmp(call->name, "fetch") == 0) {
        const char * url = agent_call_param(call, "url");
        const char * to_s = agent_call_param(call, "timeout");
        int to = (to_s != NULL) ? atoi(to_s) : 30;
        tools_fetch(url != NULL ? url : "", to, out);
    } else if (strcmp(call->name, "distill") == 0) {
        const char * html = agent_call_param(call, "html");
        if (html == NULL) {
            out->error = strdup("distill: html parameter required");
        } else {
            tools_distill(html, strlen(html), out);
        }
    } else {
        char tmp[160];
        snprintf(tmp, sizeof(tmp),
                 "agent: unknown function '%s' (available:"
                 " websearch, fetch, distill)", call->name);
        out->error = strdup(tmp);
    }
}

// Streaming-capture functor for the agent's slm_generate_raw calls.
// We only care about CONTENT bytes for tool-call parsing —
// <tool_call> blocks emitted by the model live in content, never
// inside <think>...</think>. The C-side filter already routes
// reasoning to chunk->reasoning, which the agent drops.
struct agent_capture_box {
    struct slm_stream_callback base;
    struct chars *             out;
};

static int agent_capture_cb_fn(const struct slm_stream_callback * cb) {
    struct agent_capture_box * box = (struct agent_capture_box *)cb;
    if (cb->content != NULL) {
        chars_put(box->out, cb->content, strlen(cb->content));
    }
    return 0;
}

// One agent run: question goes in, final assistant content comes
// out. The loop stops when the model emits an assistant turn with
// no <tool_call> blocks, or when max_iters is exhausted.
//
// Creates and destroys a fresh ctx per iteration so the full
// conversation history (rendered via jinja_apply) is re-prefilled
// each time — the persistent-KV delta path is for slm_generate;
// the agent's full-history replay needs a clean KV each round.
//
// `print_trace` non-zero echoes per-iteration tool calls +
// tool_response sizes to stderr.
//
// Returns a heap-allocated string (caller frees).
__attribute__((unused))
static char * agent_run(struct slm_model * model,
                        const char * question,
                        const struct slm_sampler * sp, uint64_t seed,
                        int max_iters, int max_new,
                        int print_trace) {
    struct jinja_tool tools[] = {
        { AGENT_TOOL_WEBSEARCH },
        { AGENT_TOOL_FETCH },
        { AGENT_TOOL_DISTILL },
    };
    int n_tools = (int)(sizeof(tools) / sizeof(tools[0]));
    enum { MAX_MSGS = 64 };
    struct jinja_message msgs[MAX_MSGS];
    int n_msgs = 0;
    memset(msgs, 0, sizeof(msgs));
    msgs[n_msgs].role    = JINJA_ROLE_USER;
    msgs[n_msgs].content = question;
    n_msgs++;
    char * owned[MAX_MSGS];
    int n_owned = 0;
    memset(owned, 0, sizeof(owned));
    char * final_text = NULL;
    int    iter       = 0;
    int32_t * ids =
        (int32_t *)oom(calloc(16384, sizeof(int32_t)));
    // Loop post-condition: `final_text != NULL` is "we converged",
    // `iter == max_iters` is "we ran out of attempts". A failed
    // jinja_apply forces iter = max_iters (no progress possible).
    while (final_text == NULL && iter < max_iters) {
        iter++;
        char * prompt =
            jinja_apply((const struct jinja_message *)msgs,
                        n_msgs, tools, n_tools,
                        /*add_generation_prompt=*/1,
                        /*enable_thinking=*/0);
        if (prompt == NULL) {
            iter = max_iters;
        } else {
            // Fresh ctx per iteration: full conversation re-prefilled.
            struct slm_ctx * ctx = slm_ctx_create(model);
            int32_t n_ids =
                tokenizer_encode(&ctx->model->tok, prompt, ids, 16384);
            struct chars reply = {0};
            if (print_trace) {
                fprintf(stderr,
                        "agent: iter %d/%d prompt_tokens=%d\n",
                        iter, max_iters, (int)n_ids);
            }
            // Use slm_generate_raw directly (internal to TU) with
            // the slm_split_trampoline so tool-call markers are
            // handled correctly.
            struct slm_split_box sbox = {0};
            slm_think_filter_init(&sbox.filter);
            sbox.filter.recognize_tool_calls = true;
            sbox.filter.emit_visibility      = false;
            struct agent_capture_box acbox = {0};
            acbox.base.callback = agent_capture_cb_fn;
            acbox.out           = &reply;
            sbox.cb = &acbox.base;
            slm_generate_raw(ctx, ids, n_ids, max_new, 0, sp, seed,
                             slm_split_trampoline, &sbox,
                             NULL);
            slm_think_filter_finish(&sbox.filter, &acbox.base);
            chars_free(&sbox.filter.tool_call);
            slm_ctx_destroy(ctx);
            chars_put(&reply, "", 0);
            free(prompt);
            // Trim trailing <|im_end|> if present so the assistant
            // history echo is clean for the next iteration.
            const char eot[] = "<|im_end|>";
            size_t et = sizeof(eot) - 1;
            if (reply.count >= et &&
                memcmp(reply.data + reply.count - et,
                       eot, et) == 0) {
                reply.count -= et;
                reply.data[reply.count] = '\0';
            }
            // Parse any <tool_call> blocks the model emitted.
            struct agent_call calls[8];
            int n_calls =
                agent_parse_tool_calls(reply.data, calls, 8);
            if (print_trace) {
                fprintf(stderr,
                        "agent: iter %d produced %d tool_call(s),"
                        " %zu bytes of output\n",
                        iter, n_calls, reply.count);
            }
            if (n_calls == 0) {
                // No tool calls - this is the final answer. The
                // model's content may still contain <tool_call>-
                // shaped fragments (rare) but the well-formed parser
                // saw none, so just hand the bytes back. The loop
                // exits because final_text is now non-NULL.
                final_text = reply.data;
                reply.data = NULL;
            } else {
                // Echo the assistant turn into history (verbatim;
                // the Jinja's content-after-</think> rule strips
                // any reasoning, and the <tool_call> blocks are
                // visible to the next-turn rendering as required).
                owned[n_owned] = reply.data;
                reply.data = NULL;
                msgs[n_msgs].role    = JINJA_ROLE_ASSISTANT;
                msgs[n_msgs].content = owned[n_owned];
                n_msgs++;
                n_owned++;
                // Dispatch each call, append a tool message with
                // the result content (or error message).
                for (int c = 0; c < n_calls && n_msgs < MAX_MSGS; c++) {
                    struct tool_result r = {0};
                    if (print_trace) {
                        fprintf(stderr,
                                "agent: dispatching %s(",
                                calls[c].name);
                        for (int k = 0; k < calls[c].n_params; k++) {
                            fprintf(stderr, "%s%s=%.80s",
                                    k > 0 ? ", " : "",
                                    calls[c].params[k].name,
                                    calls[c].params[k].value);
                        }
                        fprintf(stderr, ")\n");
                    }
                    agent_dispatch(&calls[c], &r);
                    const char * payload = r.ok && r.body != NULL
                                         ? r.body
                                         : (r.error != NULL
                                             ? r.error
                                             : "(no result)");
                    owned[n_owned] = strdup(payload);
                    if (print_trace) {
                        fprintf(stderr,
                                "agent: tool returned %s,"
                                " %zu bytes\n",
                                r.ok ? "OK" : "ERROR",
                                strlen(payload));
                    }
                    msgs[n_msgs].role    = JINJA_ROLE_TOOL;
                    msgs[n_msgs].content = owned[n_owned];
                    n_msgs++;
                    n_owned++;
                    tools_result_free(&r);
                }
                agent_free_calls(calls, n_calls);
            }
            chars_free(&reply);
        }
    }
    if (final_text == NULL) {
        // Hit max_iters without convergence; return whatever the
        // last assistant message was, or a placeholder.
        if (n_owned > 0) {
            final_text = strdup(owned[n_owned - 1]);
        } else {
            final_text = strdup("(agent: no output produced)");
        }
    }
    for (int i = 0; i < n_owned; i++) { free(owned[i]); }
    free(ids);
    return final_text;
}

// ---------------------------------------------------------------------------
// Decomposed multi-context cascade (Tool-Calling.md §4-5).
//
// Three fresh ctxs sharing one model, each with a tailored system
// prompt. Replaces the single-ctx growing-history agent loop for the
// `--tools` CLI path because Qwen3.5-0.8B "degrades into token-wasting
// loops or leaks thought tokens into structural parameter definitions"
// when driven recursively (Tool-Calling.md §1).
//
//   Router     tools=on, think=off   -> emits <tool_call> for websearch
//   Evaluator  tools=off, think=off  -> picks one URL from snippets
//   Synthesizer tools=off, think=off -> answers from distilled body
//
// Each phase is a fresh slm_ctx_create + manual prompt build via
// jinja_format_router_prompt / jinja_format_passive_prompt +
// slm_generate_raw. State is dropped between phases (no growing KV).
// ---------------------------------------------------------------------------

static const char * PROMPT_ROUTER =
    "You are a routing agent. Your only job is to call the websearch"
    " tool to find information answering the user's request. Output a"
    " clean <tool_call> block.";

static const char * PROMPT_EVALUATOR =
    "You are a link selector. Given the user request and search"
    " results, output exactly the single best absolute URL string to"
    " fetch. Do not output markdown, reasoning, or tags. Only output"
    " the raw URL.";

static const char * PROMPT_SYNTHESIZER =
    "You are a research synthesis engine. Answer the user's original"
    " query using only the provided distilled text content from the"
    " sourced webpage.";

// Pick the first http(s) URL out of `s`. Returns a heap-allocated
// NUL-terminated copy (caller frees), or NULL if no URL is found.
// Greedy: consumes until the first whitespace, '<', '>', '"', or end
// of string. Used to clean up the Evaluator's output before tools_fetch
// (the 0.8B often emits the URL inside backticks, markdown brackets,
// or with stray newlines).
static char * agent_extract_url(const char * s) {
    char * out = NULL;
    if (s != NULL) {
        const char * start = NULL;
        const char * p = s;
        while (*p != '\0' && start == NULL) {
            if ((p[0] == 'h' || p[0] == 'H') &&
                (p[1] == 't' || p[1] == 'T') &&
                (p[2] == 't' || p[2] == 'T') &&
                (p[3] == 'p' || p[3] == 'P') &&
                (p[4] == 's' || p[4] == 'S' || p[4] == ':')) {
                start = p;
            } else {
                p++;
            }
        }
        if (start != NULL) {
            const char * end = start;
            while (*end != '\0' && *end != ' ' && *end != '\n' &&
                   *end != '\r' && *end != '\t' && *end != '<' &&
                   *end != '>' && *end != '"' && *end != '`' &&
                   *end != ')' && *end != ']') {
                end++;
            }
            out = agent_carve(start, end);
        }
    }
    return out;
}

// Multiplexer callback: forwards each chunk to a user-supplied `cb`
// AND accumulates content bytes into `capture`. Used by Phase 4 so the
// synthesizer streams chunk-by-chunk through the user's cb while we
// also assemble the return string for the caller.
struct agent_mux_cb {
    struct slm_stream_callback   base;
    struct slm_stream_callback * forward;   // may be NULL
    struct chars *               capture;   // may be NULL
};

static int agent_mux_cb_fn(const struct slm_stream_callback * cb) {
    struct agent_mux_cb * mux = (struct agent_mux_cb *)cb;
    int rc = 0;
    if (cb->content != NULL && mux->capture != NULL) {
        chars_put(mux->capture, cb->content, strlen(cb->content));
    }
    if (mux->forward != NULL && mux->forward->callback != NULL) {
        rc = slm_cb_emit(mux->forward,
                         cb->content, cb->reasoning,
                         cb->call, cb->response,
                         cb->prefilled, cb->pp_pos, cb->pp_total);
    }
    return rc;
}

// Run one phase of the cascade: tokenize `prompt`, generate up to
// `max_new` tokens, capture content (think/tool filter applied) into
// `out`. Fires nothing through `cb` itself — the caller is responsible
// for emitting visibility chunks around each phase.
static void agent_cascade_phase(struct slm_model * model,
                                const char * prompt,
                                int max_new,
                                const struct slm_sampler * sp,
                                uint64_t seed,
                                struct chars * out,
                                bool recognize_tool_calls) {
    int32_t * ids = (int32_t *)oom(calloc(16384, sizeof(int32_t)));
    struct slm_ctx * ctx = slm_ctx_create(model);
    if (ctx != NULL) {
        int32_t n_ids = tokenizer_encode(&ctx->model->tok, prompt,
                                         ids, 16384);
        struct agent_capture_box acbox = {0};
        acbox.base.callback = agent_capture_cb_fn;
        acbox.out           = out;
        struct slm_split_box sbox = {0};
        slm_think_filter_init(&sbox.filter);
        sbox.filter.recognize_tool_calls = recognize_tool_calls;
        sbox.filter.emit_visibility      = false;
        sbox.cb = &acbox.base;
        slm_generate_raw(ctx, ids, n_ids, max_new, 0, sp, seed,
                         slm_split_trampoline, &sbox,
                         NULL);
        slm_think_filter_finish(&sbox.filter, &acbox.base);
        // Recover the captured tool_call body from the filter (Router
        // phase only — the filter buffers <tool_call>...</tool_call>
        // bytes internally and exposes them via sbox.filter.tool_call).
        if (recognize_tool_calls && sbox.filter.tool_call_ready &&
            sbox.filter.tool_call.count > 0) {
            chars_puts(out, "<tool_call>");
            chars_put(out, sbox.filter.tool_call.data,
                      sbox.filter.tool_call.count);
            chars_puts(out, "</tool_call>");
        }
        chars_free(&sbox.filter.tool_call);
        slm_ctx_destroy(ctx);
    }
    free(ids);
    chars_put(out, "", 0);
}

// Drive the decomposed cascade. Streams visibility through `cb`
// (call / response / content chunks); the final synthesized answer
// is both returned as a heap string AND streamed as content chunks.
// `verbose` controls the noise level — even when 0, the final answer
// streams; when 1, the Router's tool call / search result / chosen
// URL / fetch status all stream too.
__attribute__((unused))
static char * agent_cascade_run(struct slm_model * model,
                                const char * question,
                                const struct slm_sampler * sp,
                                uint64_t seed,
                                int max_new,
                                int verbose,
                                struct slm_stream_callback * cb) {
    char * final_text = NULL;
    if (model != NULL && question != NULL) {
        struct jinja_tool tools[1];
        tools[0].json = AGENT_TOOL_WEBSEARCH;
        // -------------------------------------------------------------------
        // PHASE 1: Routing & initial web search.
        // -------------------------------------------------------------------
        char * router_prompt = jinja_format_router_prompt(
            PROMPT_ROUTER, question, tools, 1);
        struct chars router_out = {0};
        struct tool_result search_res = {0};
        if (router_prompt != NULL) {
            agent_cascade_phase(model, router_prompt,
                                /*max_new=*/128, sp, seed,
                                &router_out,
                                /*recognize_tool_calls=*/true);
            free(router_prompt);
            struct agent_call calls[2];
            int n_calls = agent_parse_tool_calls(router_out.data,
                                                 calls, 2);
            if (n_calls > 0) {
                if (verbose && cb != NULL) {
                    char trace[256];
                    const char * q = agent_call_param(&calls[0], "query");
                    snprintf(trace, sizeof(trace),
                             "{\"name\":\"%s\",\"arguments\":"
                             "{\"query\":\"%.180s\"}}",
                             calls[0].name != NULL ? calls[0].name : "?",
                             q != NULL ? q : "");
                    slm_cb_emit(cb, NULL, NULL, trace, NULL,
                                false, 0, 0);
                }
                agent_dispatch(&calls[0], &search_res);
                if (verbose && cb != NULL &&
                    search_res.ok && search_res.body != NULL) {
                    slm_cb_emit(cb, NULL, NULL, NULL,
                                search_res.body, false, 0, 0);
                }
                agent_free_calls(calls, n_calls);
            }
        }
        chars_free(&router_out);
        // -------------------------------------------------------------------
        // PHASE 2: Evaluator picks one URL.
        // -------------------------------------------------------------------
        struct chars eval_out = {0};
        if (search_res.ok && search_res.body != NULL) {
            struct chars eval_input = {0};
            chars_puts(&eval_input, "User Query: ");
            chars_puts(&eval_input, question);
            chars_puts(&eval_input, "\n\nSearch Results:\n");
            chars_puts(&eval_input, search_res.body);
            char * eval_prompt = jinja_format_passive_prompt(
                PROMPT_EVALUATOR, eval_input.data);
            chars_free(&eval_input);
            if (eval_prompt != NULL) {
                agent_cascade_phase(model, eval_prompt,
                                    /*max_new=*/64, sp, seed,
                                    &eval_out,
                                    /*recognize_tool_calls=*/false);
                free(eval_prompt);
            }
        }
        // -------------------------------------------------------------------
        // PHASE 3: fetch + distill the chosen URL.
        // -------------------------------------------------------------------
        struct tool_result fetch_res   = {0};
        struct tool_result distill_res = {0};
        char * url = (eval_out.data != NULL)
                   ? agent_extract_url(eval_out.data) : NULL;
        chars_free(&eval_out);
        if (url != NULL) {
            if (verbose && cb != NULL) {
                slm_cb_emit(cb, NULL, NULL, NULL, url, false, 0, 0);
            }
            tools_fetch(url, 15, &fetch_res);
            if (fetch_res.ok && fetch_res.body != NULL) {
                tools_distill(fetch_res.body, strlen(fetch_res.body),
                              &distill_res);
            }
            free(url);
        }
        // -------------------------------------------------------------------
        // PHASE 4: synthesize the final answer.
        //
        // Falls back to feeding the Router's search snippets directly
        // when fetch/distill produced nothing (e.g. URL gated, body too
        // small). Better to answer from snippets than to bail.
        // -------------------------------------------------------------------
        const char * synth_body = NULL;
        if (distill_res.ok && distill_res.body != NULL &&
            strlen(distill_res.body) > 64) {
            synth_body = distill_res.body;
        } else if (search_res.ok && search_res.body != NULL) {
            synth_body = search_res.body;
        }
        if (synth_body != NULL) {
            struct chars synth_input = {0};
            chars_puts(&synth_input, "Original Query: ");
            chars_puts(&synth_input, question);
            chars_puts(&synth_input, "\n\nSourced Content:\n");
            chars_puts(&synth_input, synth_body);
            char * synth_prompt = jinja_format_passive_prompt(
                PROMPT_SYNTHESIZER, synth_input.data);
            chars_free(&synth_input);
            if (synth_prompt != NULL) {
                int32_t * ids = (int32_t *)oom(
                    calloc(16384, sizeof(int32_t)));
                struct slm_ctx * ctx = slm_ctx_create(model);
                if (ctx != NULL) {
                    int32_t n_ids = tokenizer_encode(&ctx->model->tok,
                                                     synth_prompt,
                                                     ids, 16384);
                    struct chars captured = {0};
                    struct agent_mux_cb mux = {0};
                    mux.base.callback = agent_mux_cb_fn;
                    mux.forward       = cb;
                    mux.capture       = &captured;
                    struct slm_split_box sbox = {0};
                    slm_think_filter_init(&sbox.filter);
                    sbox.filter.recognize_tool_calls = false;
                    sbox.filter.emit_visibility      = false;
                    sbox.cb = &mux.base;
                    slm_generate_raw(ctx, ids, n_ids, max_new, 0,
                                     sp, seed,
                                     slm_split_trampoline, &sbox,
                                     NULL);
                    slm_think_filter_finish(&sbox.filter, &mux.base);
                    chars_free(&sbox.filter.tool_call);
                    chars_put(&captured, "", 0);
                    if (captured.count > 0) {
                        final_text = captured.data;
                        captured.data = NULL;
                    }
                    chars_free(&captured);
                    slm_ctx_destroy(ctx);
                }
                free(ids);
                free(synth_prompt);
            }
        }
        if (final_text == NULL) {
            final_text = strdup(
                "(agent: synthesis pipeline failed to converge)");
        }
        tools_result_free(&search_res);
        tools_result_free(&fetch_res);
        tools_result_free(&distill_res);
    }
    if (final_text == NULL) {
        final_text = strdup("(agent: input invalid)");
    }
    return final_text;
}

// Self-test for the parser only (no network, no model). The dispatch
// + live agent loop are exercised by --ask (which IS the live test).
__attribute__((unused))
static int32_t agent_parser_test(void) {
    int failures = 0;
    const char * sample =
        "I'll search for that.\n"
        "<tool_call>\n"
        "<function=websearch>\n"
        "<parameter=query>\n"
        "What is the bitcoin price today?\n"
        "</parameter>\n"
        "<parameter=max_results>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n";
    struct agent_call calls[4];
    int n = agent_parse_tool_calls(sample, calls, 4);
    if (n != 1) {
        fprintf(stderr,
                "agent-test: parser expected 1 call, got %d\n", n);
        failures++;
    } else {
        if (calls[0].name == NULL ||
            strcmp(calls[0].name, "websearch") != 0) {
            fprintf(stderr,
                    "agent-test: parser wrong function name: %s\n",
                    calls[0].name ? calls[0].name : "(null)");
            failures++;
        }
        if (calls[0].n_params != 2) {
            fprintf(stderr,
                    "agent-test: parser wrong n_params: %d\n",
                    calls[0].n_params);
            failures++;
        }
        const char * q = agent_call_param(&calls[0], "query");
        if (q == NULL ||
            strcmp(q, "What is the bitcoin price today?") != 0) {
            fprintf(stderr,
                    "agent-test: parser wrong query value: %s\n",
                    q ? q : "(null)");
            failures++;
        }
        const char * mr = agent_call_param(&calls[0], "max_results");
        if (mr == NULL || strcmp(mr, "3") != 0) {
            fprintf(stderr,
                    "agent-test: parser wrong max_results: %s\n",
                    mr ? mr : "(null)");
            failures++;
        }
    }
    agent_free_calls(calls, n);
    // Double-call sample.
    const char * dual =
        "<tool_call><function=a><parameter=x>1</parameter>"
        "</function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter>"
        "</function></tool_call>";
    struct agent_call dual_calls[4];
    int dn = agent_parse_tool_calls(dual, dual_calls, 4);
    if (dn != 2) {
        fprintf(stderr,
                "agent-test: dual parser expected 2, got %d\n", dn);
        failures++;
    }
    agent_free_calls(dual_calls, dn);
    // OpenAI JSON shape (what the 0.8B Qwen model actually emits).
    const char * json_shape =
        "I'll search for that.\n"
        "<tool_call>\n"
        "{\"name\":\"websearch\","
        "\"arguments\":{\"query\":\"HelloWorld album rock band\","
        "\"max_results\":5}}\n"
        "</tool_call>\n";
    struct agent_call json_calls[4];
    int jn = agent_parse_tool_calls(json_shape, json_calls, 4);
    if (jn != 1) {
        fprintf(stderr,
                "agent-test: JSON-shape parser expected 1, got %d\n",
                jn);
        failures++;
    } else {
        if (strcmp(json_calls[0].name, "websearch") != 0) {
            fprintf(stderr,
                    "agent-test: JSON-shape wrong name: %s\n",
                    json_calls[0].name);
            failures++;
        }
        const char * q =
            agent_call_param(&json_calls[0], "query");
        if (q == NULL ||
            strcmp(q, "HelloWorld album rock band") != 0) {
            fprintf(stderr,
                    "agent-test: JSON-shape wrong query: %s\n",
                    q ? q : "(null)");
            failures++;
        }
    }
    agent_free_calls(json_calls, jn);
    // Tolerate "<tool_call >" (stray space) — observed in real
    // 0.8B output.
    const char * spaced =
        "<tool_call >{\"name\":\"x\",\"arguments\":{}}</tool_call>";
    struct agent_call spaced_calls[2];
    int sn = agent_parse_tool_calls(spaced, spaced_calls, 2);
    if (sn != 1) {
        fprintf(stderr,
                "agent-test: spaced opener expected 1, got %d\n", sn);
        failures++;
    }
    agent_free_calls(spaced_calls, sn);
    if (failures == 0) {
        printf("agent-test: PASS (parser fixtures)\n");
    } else {
        printf("agent-test: FAIL (%d sub-tests)\n", failures);
    }
    return (failures > 0) ? 1 : 0;
}
