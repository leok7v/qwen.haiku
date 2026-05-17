// SPDX-License-Identifier: Apache-2.0
//
// tools.c - agent tool primitives, post-DDG-cleanup.
//
// Six narrow, intent-routed tools the agent dispatches by name. Each
// returns a small structured-data answer (typically 100-500 bytes),
// not a page to scrape. The pre-2026-05-17 design had one generic
// websearch tool that did DDG HTML scrape → URL pick → fetch+distill.
// That pipeline was unreliable: DDG returned ad-redirect URLs as top
// hits; the HTML stripper produced 55 chars from 200 KB SPAs; the URL-
// pick model invocation always rubber-stamped hit[0]. See
// `rnd/WEBSEARCH.md` for the R&D notes that led to this rewrite.
//
//   wikipedia(query)
//       OpenSearch → REST summary. Returns the article's `extract`
//       paragraph (typically 200-500 chars) or NOT_FOUND.
//   time_now(timezone)
//       timeapi.io. Returns "Mon 2026-05-18 08:11:22 in Asia/Tokyo"
//       formatted from the JSON response.
//   weather(latitude, longitude)
//       open-meteo. Returns current temp + 2-day high/low. No
//       geocoding here; caller supplies coordinates.
//   crypto_price(symbol, vs)
//       coingecko simple/price. Returns "1 BTC = 77772 USD" style line.
//   ip_geo()
//       ip-api.com (HTTP, no key). Returns the caller IP's city,
//       region, country, timezone, ISP.
//   websearch(query)
//       mwmbl.org open-source search. Returns top-3 hits formatted
//       as "1. Title — extract\n2. ..." — pre-segmented snippets, no
//       HTML strip required.
//
// All API responses are tiny structured JSON; we use a small JSON
// value extractor (`tools_json_str` / `tools_json_num`) rather than
// pulling in a full JSON library. Each tool returns a `struct
// tool_result { ok, body, error, status }`. body is what gets
// injected into the model's context as the tool response.
//
// Single-file lib. `#include "tools.c"` from slm.c. The LLM_NO_TOOLS
// branch at the bottom keeps the file compilable on targets without
// libcurl (Android NDK etc.) and emits stub "tool unavailable"
// results.
//
// Style: SESE (single exit per function, no goto/break/continue
// except switch grammar, no `bool ok` flags). Matches the rest of
// the codebase per the project coding-discipline notes.
//
// Caller responsibilities:
//   - tools_global_init() once before any tool call; tools_global_cleanup()
//     at shutdown.
//   - tool_result lifetime: free body + error via tools_result_free.

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LLM_NO_TOOLS

#include <ctype.h>
#include <curl/curl.h>

#ifndef TOOLS_FETCH_CAP
#define TOOLS_FETCH_CAP (200 * 1024)
#endif

// User-Agent string for API calls. Wikipedia's WMF API policy requires
// a contact-identifying UA; other APIs (mwmbl, open-meteo, coingecko,
// ip-api, timeapi) accept anything but a clean identifying UA is
// good citizenship.
#define TOOLS_API_UA \
    "qwen.haiku/0.1 (https://github.com/leok7v/qwen.haiku;" \
    " leo.kuznetsov@gmail.com)"

// Debug verbosity for tool internals. slm_generate sets this from
// c->ctrl.debug right before dispatching a tool. File-scoped (not
// per-ctx) because tools_* are pure C functions with no ctx param.
static int32_t g_tools_debug = 0;

static void tools_set_debug(int32_t level) {
    g_tools_debug = level;
}

struct tool_result {
    int    ok;       // 1 on success, 0 on error
    char * body;     // heap-allocated UTF-8 (caller frees), may be NULL
    char * error;    // heap-allocated UTF-8 (caller frees), may be NULL
    long   status;   // HTTP status code (0 if no HTTP call was made)
};

static void tools_result_free(struct tool_result * r) {
    if (r != NULL) {
        free(r->body);
        free(r->error);
        r->body   = NULL;
        r->error  = NULL;
        r->ok     = 0;
        r->status = 0;
    }
}

// libcurl write-callback. Caps at TOOLS_FETCH_CAP so a runaway server
// can't make us allocate gigabytes — extra bytes silently dropped.
static size_t tools_curl_write(void * ptr, size_t size, size_t nmemb,
                               void * user) {
    struct chars * b = (struct chars *)user;
    size_t total = size * nmemb;
    size_t cap   = TOOLS_FETCH_CAP;
    size_t avail = (b->count < cap) ? (cap - b->count) : 0;
    size_t take  = (total < avail) ? total : avail;
    if (take > 0) {
        chars_put(b, (const char *)ptr, take);
    }
    return total;
}

// One-shot libcurl global init / cleanup. Idempotent.
static bool g_tools_curl_initialized;

static void tools_global_init(void) {
    if (!g_tools_curl_initialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        g_tools_curl_initialized = true;
    }
}

static void tools_global_cleanup(void) {
    if (g_tools_curl_initialized) {
        curl_global_cleanup();
        g_tools_curl_initialized = false;
    }
}

// Percent-encode `q` for use as a URL query value. ASCII alnum +
// `-._~` pass through; everything else becomes %XX. Used to build
// query strings for the API tools.
static void tools_url_encode(const char * q, struct chars * out) {
    if (q != NULL) {
        size_t n = strlen(q);
        for (size_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)q[i];
            bool safe = (c >= 'A' && c <= 'Z') ||
                        (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') ||
                        c == '-' || c == '_' ||
                        c == '.' || c == '~';
            if (safe) {
                chars_put(out, (const char *)&c, 1);
            } else {
                char hex[4];
                snprintf(hex, sizeof(hex), "%%%02X", (int)c);
                chars_put(out, hex, 3);
            }
        }
    }
}

// Clean API GET: simple identifying User-Agent, accepts JSON, follows
// redirects, fixed timeout. No browser fingerprinting (that lived in
// the old DDG-scrape path; APIs reject impersonation anyway).
// Returns 0 on success and writes status code through `status`. Caller
// owns `body` allocations.
static int tools_api_get(const char * url, long timeout_ms,
                         struct chars * body, long * status) {
    int rc = -1;
    *status = 0;
    if (g_tools_debug >= 5) {
        trace("api_get: GET %s\n", url);
    }
    tools_global_init();
    CURL * h = curl_easy_init();
    if (h != NULL) {
        struct curl_slist * hdrs = NULL;
        hdrs = curl_slist_append(hdrs, "User-Agent: " TOOLS_API_UA);
        hdrs = curl_slist_append(hdrs, "Accept: application/json, */*");
        hdrs = curl_slist_append(hdrs, "Accept-Language: en-US,en;q=0.9");
        char errbuf[CURL_ERROR_SIZE];
        errbuf[0] = '\0';
        curl_easy_setopt(h, CURLOPT_URL,               url);
        curl_easy_setopt(h, CURLOPT_HTTPHEADER,        hdrs);
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION,    1L);
        curl_easy_setopt(h, CURLOPT_MAXREDIRS,         5L);
        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS,        timeout_ms);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
        curl_easy_setopt(h, CURLOPT_NOPROGRESS,        1L);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION,     tools_curl_write);
        curl_easy_setopt(h, CURLOPT_WRITEDATA,         body);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER,       errbuf);
        curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING,   "");
        CURLcode cc = curl_easy_perform(h);
        if (cc == CURLE_OK) {
            rc = 0;
            curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, status);
            if (g_tools_debug >= 5) {
                trace("api_get: status=%ld body=%zu bytes\n",
                      *status, body->count);
            }
        } else if (g_tools_debug >= 1) {
            trace("api_get: curl_easy_perform: %s\n",
                  errbuf[0] != '\0' ? errbuf : curl_easy_strerror(cc));
        }
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(h);
    }
    return rc;
}

// ---------------------------------------------------------------------------
// Tiny JSON value extractor — handles the response shapes we actually
// see from the APIs we call. Not a full parser; designed to be small,
// readable, and bombproof against the specific JSON we accept.
//
// `tools_json_find_key` locates `"key":` in `s[0..n)` (skipping over
// quoted-string regions so the key match isn't fooled by a key-shaped
// substring inside another value). Returns a pointer just AFTER the
// colon, or NULL if the key isn't found. The pointer is suitable to
// pass into the extract helpers below.
//
// `tools_json_extract_string` reads a JSON string value at `p`,
// unescaping \" \\ \n \t \uXXXX (BMP only) into `out`. Returns
// pointer past the closing quote, or NULL on parse error.
//
// `tools_json_extract_number` reads a JSON number at `p` into `out`
// (double). Returns pointer past the number, or NULL on parse error.
//
// All three skip leading whitespace before parsing.
// ---------------------------------------------------------------------------

static void tools_json_skip_ws(const char ** pp, const char * end) {
    const char * p = *pp;
    while (p < end && (*p == ' ' || *p == '\t' ||
                       *p == '\n' || *p == '\r')) {
        p++;
    }
    *pp = p;
}

// Skip a JSON string in `s[i..n)` starting at the opening quote.
// Returns the index just past the closing quote, or `n` if the
// string is malformed.
static size_t tools_json_skip_string(const char * s, size_t n, size_t i) {
    size_t r = n;
    if (i < n && s[i] == '"') {
        size_t j = i + 1;
        bool   done = false;
        while (j < n && !done) {
            if (s[j] == '\\') {
                j += 2;
            } else if (s[j] == '"') {
                done = true;
                r    = j + 1;
            } else {
                j++;
            }
        }
        if (!done) { r = n; }
    }
    return r;
}

static const char * tools_json_find_key(const char * s, size_t n,
                                        const char * key) {
    const char * result = NULL;
    size_t       klen   = strlen(key);
    size_t       i      = 0;
    bool         found  = false;
    while (i < n && !found) {
        if (s[i] == '"') {
            // Check if this opening quote starts our key.
            if (i + klen + 1 < n &&
                memcmp(s + i + 1, key, klen) == 0 &&
                s[i + 1 + klen] == '"') {
                // Move past closing quote of key, then optional
                // whitespace, then expect ':'.
                size_t j = i + 2 + klen;
                while (j < n && (s[j] == ' ' || s[j] == '\t' ||
                                 s[j] == '\n' || s[j] == '\r')) {
                    j++;
                }
                if (j < n && s[j] == ':') {
                    result = s + j + 1;
                    found  = true;
                } else {
                    i = j;
                }
            } else {
                // Skip past this whole string so we don't match a
                // key-shaped substring inside a value.
                i = tools_json_skip_string(s, n, i);
            }
        } else {
            i++;
        }
    }
    return result;
}

// Append code point `cp` as UTF-8 to `out`.
static void tools_json_emit_utf8(struct chars * out, uint32_t cp) {
    char buf[4];
    int  n = 0;
    if (cp < 0x80) {
        buf[0] = (char)cp;
        n = 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 |  (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 |  (cp       & 0x3F));
        n = 3;
    } else {
        buf[0] = (char)(0xF0 |  (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >>  6) & 0x3F));
        buf[3] = (char)(0x80 |  (cp        & 0x3F));
        n = 4;
    }
    chars_put(out, buf, (size_t)n);
}

// Parse hex digit; returns -1 if not hex.
static int tools_hex_digit(char c) {
    int r = -1;
    if      (c >= '0' && c <= '9') { r = c - '0'; }
    else if (c >= 'a' && c <= 'f') { r = c - 'a' + 10; }
    else if (c >= 'A' && c <= 'F') { r = c - 'A' + 10; }
    return r;
}

// Parse a JSON string value beginning at *pp (which must point at the
// opening quote). On success, *pp advances past the closing quote and
// the unescaped UTF-8 contents are appended to `out`. Returns 1 on
// success, 0 on malformed input (in which case `out` may have partial
// content and *pp is unspecified).
static int tools_json_str(const char ** pp, const char * end,
                          struct chars * out) {
    int rc = 0;
    tools_json_skip_ws(pp, end);
    const char * p = *pp;
    if (p < end && *p == '"') {
        p++;
        bool done = false;
        bool err  = false;
        while (p < end && !done && !err) {
            char c = *p;
            if (c == '"') {
                done = true;
                p++;
            } else if (c == '\\' && p + 1 < end) {
                char e = p[1];
                switch (e) {
                    case '"':  chars_put(out, "\"", 1); p += 2; break;
                    case '\\': chars_put(out, "\\", 1); p += 2; break;
                    case '/':  chars_put(out, "/",  1); p += 2; break;
                    case 'b':  chars_put(out, "\b", 1); p += 2; break;
                    case 'f':  chars_put(out, "\f", 1); p += 2; break;
                    case 'n':  chars_put(out, "\n", 1); p += 2; break;
                    case 'r':  chars_put(out, "\r", 1); p += 2; break;
                    case 't':  chars_put(out, "\t", 1); p += 2; break;
                    case 'u':
                        if (p + 5 < end) {
                            int h0 = tools_hex_digit(p[2]);
                            int h1 = tools_hex_digit(p[3]);
                            int h2 = tools_hex_digit(p[4]);
                            int h3 = tools_hex_digit(p[5]);
                            if (h0 >= 0 && h1 >= 0 &&
                                h2 >= 0 && h3 >= 0) {
                                uint32_t cp = ((uint32_t)h0 << 12) |
                                              ((uint32_t)h1 <<  8) |
                                              ((uint32_t)h2 <<  4) |
                                              ((uint32_t)h3);
                                tools_json_emit_utf8(out, cp);
                                p += 6;
                            } else {
                                err = true;
                            }
                        } else {
                            err = true;
                        }
                        break;
                    default:
                        err = true;
                        break;
                }
            } else {
                chars_put(out, &c, 1);
                p++;
            }
        }
        if (done) {
            rc = 1;
            *pp = p;
        }
    }
    return rc;
}

// Parse a JSON number (integer or floating) at *pp. On success, *pp
// advances past the number and `*out` holds the parsed value.
// Returns 1 on success, 0 on parse error.
static int tools_json_num(const char ** pp, const char * end,
                          double * out) {
    int rc = 0;
    tools_json_skip_ws(pp, end);
    const char * p     = *pp;
    const char * start = p;
    if (p < end && (*p == '-' || *p == '+')) { p++; }
    bool have_digit = false;
    while (p < end && *p >= '0' && *p <= '9') { p++; have_digit = true; }
    if (p < end && *p == '.') {
        p++;
        while (p < end && *p >= '0' && *p <= '9') { p++; }
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < end && (*p == '+' || *p == '-')) { p++; }
        while (p < end && *p >= '0' && *p <= '9') { p++; }
    }
    if (have_digit) {
        char buf[64];
        size_t n = (size_t)(p - start);
        if (n >= sizeof(buf)) { n = sizeof(buf) - 1; }
        memcpy(buf, start, n);
        buf[n] = '\0';
        *out = strtod(buf, NULL);
        *pp  = p;
        rc   = 1;
    }
    return rc;
}

// Convenience: find `"key":` and parse the string value into `out`.
// Returns 1 on success.
static int tools_json_get_str(const char * s, size_t n,
                              const char * key, struct chars * out) {
    int rc = 0;
    const char * p = tools_json_find_key(s, n, key);
    if (p != NULL) {
        const char * end = s + n;
        rc = tools_json_str(&p, end, out);
    }
    return rc;
}

// Convenience: find `"key":` and parse the numeric value into `*out`.
// Returns 1 on success.
static int tools_json_get_num(const char * s, size_t n,
                              const char * key, double * out) {
    int rc = 0;
    const char * p = tools_json_find_key(s, n, key);
    if (p != NULL) {
        const char * end = s + n;
        rc = tools_json_num(&p, end, out);
    }
    return rc;
}

// Find the object value of `parent_key` and return a pointer to the
// opening `{` and the position just past the matching `}`. Returns 1
// on success (writes *child_start and *child_end), 0 otherwise.
//
// Used by the weather tool to disambiguate `temperature_2m` (which
// appears in BOTH `current_units` and `current` of the open-meteo
// response — the former as a unit string, the latter as the actual
// number).
static int tools_json_scope(const char * s, size_t n,
                            const char * parent_key,
                            const char ** child_start,
                            const char ** child_end) {
    int rc = 0;
    const char * p = tools_json_find_key(s, n, parent_key);
    if (p != NULL) {
        const char * end = s + n;
        tools_json_skip_ws(&p, end);
        if (p < end && *p == '{') {
            const char * scan  = p + 1;
            int          depth = 1;
            while (scan < end && depth > 0) {
                if (*scan == '"') {
                    size_t skip = tools_json_skip_string(
                        scan, (size_t)(end - scan), 0);
                    scan += skip;
                } else if (*scan == '{') {
                    depth++; scan++;
                } else if (*scan == '}') {
                    depth--; scan++;
                } else {
                    scan++;
                }
            }
            if (depth == 0) {
                *child_start = p;
                *child_end   = scan;
                rc           = 1;
            }
        }
    }
    return rc;
}

// ---------------------------------------------------------------------------
// Tool 1 — wikipedia(query)
//
// Two-step chain: OpenSearch (titles for the query) → REST summary
// (the `extract` paragraph of the first matching title). The
// Wikipedia API is the highest-leverage replacement for the DDG
// scrape: 200-500 char clean prose extracts, redirects resolve
// automatically, no nav chrome, no HTML strip needed.
//
// Returns `out->body` = "Title: <t>\n\n<extract>" on success.
// ---------------------------------------------------------------------------

// Pick the first non-empty title from an OpenSearch reply of the
// shape `["query",["Title 1","Title 2",...],...]`. We don't need full
// JSON parsing — find the SECOND `[` (the titles array), then read
// the first quoted string inside it.
static int tools_wiki_first_title(const char * body, size_t n,
                                  struct chars * out) {
    int rc = 0;
    // Find the position of the first '[' that's NOT the outer one.
    size_t i = 0;
    int    depth = 0;
    bool   found_inner = false;
    while (i < n && !found_inner) {
        if (body[i] == '[') {
            depth++;
            if (depth == 2) { found_inner = true; }
        } else if (body[i] == ']') {
            depth--;
        } else if (body[i] == '"') {
            i = tools_json_skip_string(body, n, i);
            continue;
        }
        i++;
    }
    if (found_inner) {
        // Find the first `"` inside the inner array and parse the
        // string value.
        size_t j = i;
        while (j < n && body[j] != '"' && body[j] != ']') { j++; }
        if (j < n && body[j] == '"') {
            const char * p   = body + j;
            const char * end = body + n;
            rc = tools_json_str(&p, end, out);
        }
    }
    return rc;
}

// URL-encode a Wikipedia title for the REST summary path: spaces
// become underscores; the rest follows tools_url_encode rules.
static void tools_wiki_title_to_path(const char * title,
                                     struct chars * out) {
    size_t n = strlen(title);
    struct chars buf = {0};
    for (size_t i = 0; i < n; i++) {
        char c = title[i];
        if (c == ' ') { c = '_'; }
        chars_put(&buf, &c, 1);
    }
    chars_put(&buf, "", 0);
    tools_url_encode(buf.data != NULL ? buf.data : "", out);
    free(buf.data);
}

static void tools_wikipedia(const char * query, struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    if (query == NULL || query[0] == '\0') {
        out->error = strdup("wikipedia: query parameter required");
    } else {
        // Step 1: OpenSearch for the query.
        struct chars eq = {0};
        tools_url_encode(query, &eq);
        struct chars open_url = {0};
        chars_puts(&open_url,
            "https://en.wikipedia.org/w/api.php"
            "?action=opensearch&format=json&limit=5&search=");
        chars_puts(&open_url, eq.data != NULL ? eq.data : "");
        chars_put(&open_url, "", 0);
        free(eq.data);
        struct chars open_body = {0};
        long open_status = 0;
        int  open_rc = tools_api_get(open_url.data, 8000,
                                     &open_body, &open_status);
        free(open_url.data);
        struct chars title = {0};
        bool   have_title = false;
        if (open_rc == 0 && open_status == 200 && open_body.count > 0) {
            have_title = tools_wiki_first_title(
                open_body.data, open_body.count, &title) != 0;
        }
        free(open_body.data);
        if (!have_title) {
            chars_free(&title);
            struct chars msg = {0};
            chars_printf(&msg,
                "wikipedia: no Wikipedia article matches \"%s\""
                " (try a shorter noun-phrase query: \"Tom Cruise\""
                " not \"Tom Cruise age\")", query);
            out->error  = msg.data;
            out->status = open_status;
        } else {
            chars_put(&title, "", 0);
            // Step 2: REST summary for the title.
            struct chars title_path = {0};
            tools_wiki_title_to_path(title.data, &title_path);
            chars_put(&title_path, "", 0);
            struct chars sum_url = {0};
            chars_puts(&sum_url,
                "https://en.wikipedia.org/api/rest_v1/page/summary/");
            chars_puts(&sum_url,
                       title_path.data != NULL ? title_path.data : "");
            chars_put(&sum_url, "", 0);
            free(title_path.data);
            struct chars sum_body = {0};
            long sum_status = 0;
            int  sum_rc = tools_api_get(sum_url.data, 8000,
                                        &sum_body, &sum_status);
            free(sum_url.data);
            if (sum_rc == 0 && sum_status == 200 && sum_body.count > 0) {
                struct chars t_field = {0};
                struct chars d_field = {0};
                struct chars x_field = {0};
                tools_json_get_str(sum_body.data, sum_body.count,
                                   "title", &t_field);
                tools_json_get_str(sum_body.data, sum_body.count,
                                   "description", &d_field);
                tools_json_get_str(sum_body.data, sum_body.count,
                                   "extract", &x_field);
                struct chars body = {0};
                chars_puts(&body, "Wikipedia: ");
                chars_puts(&body,
                           t_field.data != NULL ? t_field.data : "?");
                if (d_field.data != NULL && d_field.count > 0) {
                    chars_puts(&body, " — ");
                    chars_puts(&body, d_field.data);
                }
                chars_puts(&body, "\n\n");
                chars_puts(&body,
                           x_field.data != NULL ? x_field.data
                                                : "(no extract)");
                chars_put(&body, "", 0);
                out->ok     = 1;
                out->body   = body.data;
                out->status = sum_status;
                free(t_field.data);
                free(d_field.data);
                free(x_field.data);
            } else {
                struct chars msg = {0};
                chars_printf(&msg,
                    "wikipedia: REST summary failed for \"%s\""
                    " (HTTP %ld)",
                    title.data, sum_status);
                out->error  = msg.data;
                out->status = sum_status;
            }
            free(sum_body.data);
        }
        free(title.data);
    }
}

// ---------------------------------------------------------------------------
// Tool 2 — time_now(timezone)
//
// timeapi.io returns JSON with year/month/day/hour/minute/seconds plus
// dayOfWeek and dstActive. We extract `dateTime` (ISO 8601) and
// `dayOfWeek` and format a one-line response.
// ---------------------------------------------------------------------------

static void tools_time_now(const char * timezone,
                           struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    const char * tz = (timezone != NULL && timezone[0] != '\0')
                    ? timezone : "UTC";
    struct chars etz = {0};
    tools_url_encode(tz, &etz);
    struct chars url = {0};
    chars_puts(&url,
        "https://timeapi.io/api/Time/current/zone?timeZone=");
    chars_puts(&url, etz.data != NULL ? etz.data : "UTC");
    chars_put(&url, "", 0);
    free(etz.data);
    struct chars body = {0};
    long status = 0;
    int  rc = tools_api_get(url.data, 8000, &body, &status);
    free(url.data);
    if (rc != 0) {
        out->error = strdup(
            "time_now: network error reaching timeapi.io");
    } else if (status != 200 || body.count == 0) {
        struct chars msg = {0};
        chars_printf(&msg,
            "time_now: timeapi.io returned HTTP %ld for \"%s\""
            " (invalid timezone?)", status, tz);
        out->error  = msg.data;
        out->status = status;
    } else {
        struct chars dt  = {0};
        struct chars dow = {0};
        struct chars tzn = {0};
        tools_json_get_str(body.data, body.count, "dateTime",  &dt);
        tools_json_get_str(body.data, body.count, "dayOfWeek", &dow);
        tools_json_get_str(body.data, body.count, "timeZone",  &tzn);
        struct chars o = {0};
        chars_printf(&o,
            "Current time in %s: %s%s%s%s",
            tzn.data != NULL ? tzn.data : tz,
            dow.data != NULL ? dow.data : "",
            dow.data != NULL ? ", "     : "",
            dt.data  != NULL ? dt.data  : "(no dateTime field)",
            (dt.data != NULL) ? "" : "");
        chars_put(&o, "", 0);
        out->ok     = 1;
        out->body   = o.data;
        out->status = status;
        free(dt.data);
        free(dow.data);
        free(tzn.data);
    }
    free(body.data);
}

// ---------------------------------------------------------------------------
// Tool 3 — weather(latitude, longitude)
//
// open-meteo. Requires lat/lon — no geocoding step (caller's job, or
// a follow-up tool). Returns current temp + 2-day high/low.
// ---------------------------------------------------------------------------

// WMO weather code → short English description. Subset covering the
// codes we'll see most often; unknown codes fall back to "code N".
static const char * tools_wmo_desc(int code) {
    const char * r = NULL;
    switch (code) {
        case  0: r = "clear sky";                       break;
        case  1: r = "mainly clear";                    break;
        case  2: r = "partly cloudy";                   break;
        case  3: r = "overcast";                        break;
        case 45: r = "fog";                             break;
        case 48: r = "depositing rime fog";             break;
        case 51: r = "light drizzle";                   break;
        case 53: r = "moderate drizzle";                break;
        case 55: r = "dense drizzle";                   break;
        case 61: r = "slight rain";                     break;
        case 63: r = "moderate rain";                   break;
        case 65: r = "heavy rain";                      break;
        case 71: r = "slight snow";                     break;
        case 73: r = "moderate snow";                   break;
        case 75: r = "heavy snow";                      break;
        case 80: r = "rain showers";                    break;
        case 81: r = "moderate rain showers";           break;
        case 82: r = "violent rain showers";            break;
        case 95: r = "thunderstorm";                    break;
        case 96: r = "thunderstorm with slight hail";   break;
        case 99: r = "thunderstorm with heavy hail";    break;
        default: r = NULL;                              break;
    }
    return r;
}

static void tools_weather(double lat, double lon,
                          struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    struct chars url = {0};
    chars_printf(&url,
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=%.4f&longitude=%.4f"
        "&current=temperature_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,weather_code"
        "&timezone=auto&forecast_days=2&temperature_unit=celsius",
        lat, lon);
    chars_put(&url, "", 0);
    struct chars body = {0};
    long status = 0;
    int  rc = tools_api_get(url.data, 8000, &body, &status);
    free(url.data);
    if (rc != 0) {
        out->error = strdup(
            "weather: network error reaching open-meteo.com");
    } else if (status != 200 || body.count == 0) {
        struct chars msg = {0};
        chars_printf(&msg,
            "weather: open-meteo returned HTTP %ld for (%.4f, %.4f)",
            status, lat, lon);
        out->error  = msg.data;
        out->status = status;
    } else {
        // open-meteo's response has the same key name in TWO scopes:
        // `current_units.temperature_2m = "°C"` (string) AND
        // `current.temperature_2m = 23.0` (number). The same applies
        // to `daily_units` vs `daily`. We use tools_json_scope to
        // isolate the numeric-bearing parent before scanning for
        // child keys; otherwise tools_json_find_key returns the
        // first match (the string in *_units) and number parsing
        // fails on "°C".
        double t_now = 0, w_now = 0;
        double max0 = 0, max1 = 0, min0 = 0, min1 = 0;
        bool   have_now  = false, have_code = false;
        bool   have_max  = false, have_min  = false;
        const char * cur_s = NULL;
        const char * cur_e = NULL;
        if (tools_json_scope(body.data, body.count,
                             "current", &cur_s, &cur_e) != 0) {
            size_t cn = (size_t)(cur_e - cur_s);
            have_now  = tools_json_get_num(cur_s, cn,
                                           "temperature_2m", &t_now) != 0;
            have_code = tools_json_get_num(cur_s, cn,
                                           "weather_code",   &w_now) != 0;
        }
        const char * day_s = NULL;
        const char * day_e = NULL;
        if (tools_json_scope(body.data, body.count,
                             "daily", &day_s, &day_e) != 0) {
            size_t       dn  = (size_t)(day_e - day_s);
            const char * pmax = tools_json_find_key(
                day_s, dn, "temperature_2m_max");
            if (pmax != NULL) {
                const char * end = day_s + dn;
                while (pmax < end && (*pmax == ' ' || *pmax == '['))
                    { pmax++; }
                if (tools_json_num(&pmax, end, &max0) != 0) {
                    while (pmax < end && (*pmax == ',' || *pmax == ' '))
                        { pmax++; }
                    if (tools_json_num(&pmax, end, &max1) != 0) {
                        have_max = true;
                    }
                }
            }
            const char * pmin = tools_json_find_key(
                day_s, dn, "temperature_2m_min");
            if (pmin != NULL) {
                const char * end = day_s + dn;
                while (pmin < end && (*pmin == ' ' || *pmin == '['))
                    { pmin++; }
                if (tools_json_num(&pmin, end, &min0) != 0) {
                    while (pmin < end && (*pmin == ',' || *pmin == ' '))
                        { pmin++; }
                    if (tools_json_num(&pmin, end, &min1) != 0) {
                        have_min = true;
                    }
                }
            }
        }
        if (!have_now || !have_code || !have_max || !have_min) {
            out->error = strdup(
                "weather: open-meteo response missing expected fields");
            out->status = status;
        } else {
            const char * desc = tools_wmo_desc((int)w_now);
            struct chars o = {0};
            if (desc != NULL) {
                chars_printf(&o,
                    "Weather at (%.4f, %.4f): currently %.1f°C, %s."
                    " Today: high %.1f°C, low %.1f°C."
                    " Tomorrow: high %.1f°C, low %.1f°C.",
                    lat, lon, t_now, desc,
                    max0, min0, max1, min1);
            } else {
                chars_printf(&o,
                    "Weather at (%.4f, %.4f): currently %.1f°C"
                    " (WMO code %d). Today: high %.1f°C, low %.1f°C."
                    " Tomorrow: high %.1f°C, low %.1f°C.",
                    lat, lon, t_now, (int)w_now,
                    max0, min0, max1, min1);
            }
            chars_put(&o, "", 0);
            out->ok     = 1;
            out->body   = o.data;
            out->status = status;
        }
    }
    free(body.data);
}

// ---------------------------------------------------------------------------
// Tool 4 — crypto_price(symbol, vs)
//
// CoinGecko simple/price. Symbol is the coin id (bitcoin, ethereum,
// solana, ...). vs defaults to "usd".
// ---------------------------------------------------------------------------

static void tools_crypto_price(const char * symbol, const char * vs,
                               struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    const char * sym  = (symbol != NULL && symbol[0] != '\0')
                      ? symbol : "bitcoin";
    const char * cur  = (vs     != NULL && vs[0]     != '\0')
                      ? vs : "usd";
    struct chars esym = {0};
    struct chars ecur = {0};
    tools_url_encode(sym, &esym);
    tools_url_encode(cur, &ecur);
    struct chars url = {0};
    chars_puts(&url,
        "https://api.coingecko.com/api/v3/simple/price?ids=");
    chars_puts(&url, esym.data != NULL ? esym.data : "bitcoin");
    chars_puts(&url, "&vs_currencies=");
    chars_puts(&url, ecur.data != NULL ? ecur.data : "usd");
    chars_put(&url, "", 0);
    free(esym.data);
    free(ecur.data);
    struct chars body = {0};
    long status = 0;
    int  rc = tools_api_get(url.data, 8000, &body, &status);
    free(url.data);
    if (rc != 0) {
        out->error = strdup(
            "crypto_price: network error reaching api.coingecko.com");
    } else if (status != 200 || body.count == 0) {
        struct chars msg = {0};
        chars_printf(&msg,
            "crypto_price: coingecko returned HTTP %ld for \"%s\""
            " (unknown coin id?)", status, sym);
        out->error  = msg.data;
        out->status = status;
    } else {
        // Response: {"bitcoin":{"usd":77772}} — find the currency key.
        double price = 0;
        bool have = tools_json_get_num(body.data, body.count,
                                       cur, &price) != 0;
        if (!have) {
            struct chars msg = {0};
            chars_printf(&msg,
                "crypto_price: response missing \"%s\" field"
                " (coin \"%s\" not in coingecko ids list?)",
                cur, sym);
            out->error  = msg.data;
            out->status = status;
        } else {
            struct chars o = {0};
            chars_printf(&o,
                "Current price of %s: %g %s",
                sym, price, cur);
            chars_put(&o, "", 0);
            out->ok     = 1;
            out->body   = o.data;
            out->status = status;
        }
    }
    free(body.data);
}

// ---------------------------------------------------------------------------
// Tool 5 — ip_geo()
//
// ip-api.com (HTTP-only on the free tier). No parameters — geolocates
// the caller's outbound IP.
// ---------------------------------------------------------------------------

static void tools_ip_geo(struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    const char * url =
        "http://ip-api.com/json/?fields="
        "status,country,regionName,city,lat,lon,timezone,isp,query";
    struct chars body = {0};
    long status = 0;
    int  rc = tools_api_get(url, 8000, &body, &status);
    if (rc != 0) {
        out->error = strdup(
            "ip_geo: network error reaching ip-api.com");
    } else if (status != 200 || body.count == 0) {
        struct chars msg = {0};
        chars_printf(&msg,
            "ip_geo: ip-api returned HTTP %ld", status);
        out->error  = msg.data;
        out->status = status;
    } else {
        struct chars country = {0};
        struct chars region  = {0};
        struct chars city    = {0};
        struct chars tz      = {0};
        struct chars isp     = {0};
        struct chars ip      = {0};
        double lat = 0, lon = 0;
        tools_json_get_str(body.data, body.count, "country",     &country);
        tools_json_get_str(body.data, body.count, "regionName",  &region);
        tools_json_get_str(body.data, body.count, "city",        &city);
        tools_json_get_str(body.data, body.count, "timezone",    &tz);
        tools_json_get_str(body.data, body.count, "isp",         &isp);
        tools_json_get_str(body.data, body.count, "query",       &ip);
        tools_json_get_num(body.data, body.count, "lat",         &lat);
        tools_json_get_num(body.data, body.count, "lon",         &lon);
        struct chars o = {0};
        chars_printf(&o,
            "Your approximate location by IP %s: %s, %s, %s"
            " (lat %.4f, lon %.4f, timezone %s, ISP %s).",
            ip.data      != NULL ? ip.data      : "?",
            city.data    != NULL ? city.data    : "?",
            region.data  != NULL ? region.data  : "?",
            country.data != NULL ? country.data : "?",
            lat, lon,
            tz.data      != NULL ? tz.data      : "?",
            isp.data     != NULL ? isp.data     : "?");
        chars_put(&o, "", 0);
        out->ok     = 1;
        out->body   = o.data;
        out->status = status;
        free(country.data);
        free(region.data);
        free(city.data);
        free(tz.data);
        free(isp.data);
        free(ip.data);
    }
    free(body.data);
}

// ---------------------------------------------------------------------------
// Tool 6 — websearch(query) — mwmbl fallback for the long tail.
//
// mwmbl.org returns JSON list of {url, title, extract, source} where
// title and extract are pre-segmented arrays of {value, is_bold}.
// We flatten the segments back into plain strings and format the
// top-N hits.
// ---------------------------------------------------------------------------

// Flatten an array of {value, is_bold} segments starting at *pp into
// `out`. *pp must point at the opening `[`. Returns 1 on success and
// advances *pp past the closing `]`.
static int tools_mwmbl_flatten_segments(const char ** pp, const char * end,
                                        struct chars * out) {
    int rc = 0;
    tools_json_skip_ws(pp, end);
    const char * p = *pp;
    if (p < end && *p == '[') {
        p++;
        bool done = false;
        while (p < end && !done) {
            tools_json_skip_ws(&p, end);
            if (p < end && *p == ']') {
                p++;
                done = true;
            } else if (p < end && *p == '{') {
                // Find "value":"..." inside this object. We scan
                // forward to the matching `}`; track quoted strings
                // so we don't terminate inside one.
                size_t obj_start = (size_t)(p - end + (end - p));
                (void)obj_start;
                const char * obj_p   = p + 1;
                int          depth   = 1;
                const char * obj_end = NULL;
                while (obj_p < end && obj_end == NULL) {
                    if (*obj_p == '"') {
                        size_t skip = tools_json_skip_string(
                            obj_p, (size_t)(end - obj_p), 0);
                        obj_p += skip;
                    } else if (*obj_p == '{') {
                        depth++;
                        obj_p++;
                    } else if (*obj_p == '}') {
                        depth--;
                        obj_p++;
                        if (depth == 0) { obj_end = obj_p; }
                    } else {
                        obj_p++;
                    }
                }
                if (obj_end != NULL) {
                    const char * v = tools_json_find_key(
                        p, (size_t)(obj_end - p), "value");
                    if (v != NULL) {
                        tools_json_str(&v, obj_end, out);
                    }
                    p = obj_end;
                } else {
                    p = end;
                }
            } else if (p < end && *p == ',') {
                p++;
            } else {
                p = end;
            }
        }
        if (done) {
            rc  = 1;
            *pp = p;
        }
    }
    return rc;
}

static void tools_websearch(const char * query, int max_results,
                            struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    int want = (max_results > 0 && max_results <= 10) ? max_results : 3;
    if (query == NULL || query[0] == '\0') {
        out->error = strdup("websearch: query parameter required");
    } else {
        struct chars eq = {0};
        tools_url_encode(query, &eq);
        struct chars url = {0};
        chars_puts(&url, "https://api.mwmbl.org/api/v1/search/?s=");
        chars_puts(&url, eq.data != NULL ? eq.data : "");
        chars_put(&url, "", 0);
        free(eq.data);
        struct chars body = {0};
        long status = 0;
        int  rc = tools_api_get(url.data, 8000, &body, &status);
        free(url.data);
        if (rc != 0) {
            out->error = strdup(
                "websearch: network error reaching api.mwmbl.org");
        } else if (status != 200 || body.count == 0) {
            struct chars msg = {0};
            chars_printf(&msg,
                "websearch: mwmbl returned HTTP %ld", status);
            out->error  = msg.data;
            out->status = status;
        } else {
            // Walk the top-level array; for each object, find
            // "url" (string), "title" (segments), "extract" (segments).
            const char * p   = body.data;
            const char * end = body.data + body.count;
            tools_json_skip_ws(&p, end);
            struct chars o = {0};
            int n_emitted  = 0;
            if (p < end && *p == '[') {
                p++;
                bool done = false;
                while (p < end && !done && n_emitted < want) {
                    tools_json_skip_ws(&p, end);
                    if (p < end && *p == ']') {
                        done = true;
                        p++;
                    } else if (p < end && *p == '{') {
                        // Find this object's end.
                        const char * obj_p   = p + 1;
                        int          depth   = 1;
                        const char * obj_end = NULL;
                        while (obj_p < end && obj_end == NULL) {
                            if (*obj_p == '"') {
                                size_t skip = tools_json_skip_string(
                                    obj_p, (size_t)(end - obj_p), 0);
                                obj_p += skip;
                            } else if (*obj_p == '{') {
                                depth++;
                                obj_p++;
                            } else if (*obj_p == '}') {
                                depth--;
                                obj_p++;
                                if (depth == 0) { obj_end = obj_p; }
                            } else {
                                obj_p++;
                            }
                        }
                        if (obj_end != NULL) {
                            size_t       obj_n = (size_t)(obj_end - p);
                            struct chars url_v = {0};
                            struct chars t_v   = {0};
                            struct chars x_v   = {0};
                            tools_json_get_str(p, obj_n, "url", &url_v);
                            const char * tk = tools_json_find_key(
                                p, obj_n, "title");
                            if (tk != NULL) {
                                tools_mwmbl_flatten_segments(
                                    &tk, obj_end, &t_v);
                            }
                            const char * xk = tools_json_find_key(
                                p, obj_n, "extract");
                            if (xk != NULL) {
                                tools_mwmbl_flatten_segments(
                                    &xk, obj_end, &x_v);
                            }
                            // Emit "N. Title — extract (url)"
                            chars_printf(&o, "%d. %s",
                                n_emitted + 1,
                                t_v.data != NULL && t_v.count > 0
                                    ? t_v.data : "(no title)");
                            if (x_v.data != NULL && x_v.count > 0) {
                                // Trim leading/trailing whitespace.
                                char * x = x_v.data;
                                while (*x == ' ' || *x == '\n' ||
                                       *x == '\r' || *x == '\t') { x++; }
                                size_t xn = strlen(x);
                                while (xn > 0 && (x[xn-1] == ' ' ||
                                                   x[xn-1] == '\n' ||
                                                   x[xn-1] == '\r' ||
                                                   x[xn-1] == '\t')) {
                                    xn--;
                                }
                                if (xn > 0) {
                                    chars_puts(&o, " — ");
                                    chars_put(&o, x, xn);
                                }
                            }
                            if (url_v.data != NULL && url_v.count > 0) {
                                chars_puts(&o, "\n   ");
                                chars_puts(&o, url_v.data);
                            }
                            chars_puts(&o, "\n");
                            n_emitted++;
                            free(url_v.data);
                            free(t_v.data);
                            free(x_v.data);
                            p = obj_end;
                        } else {
                            p = end;
                        }
                    } else if (p < end && *p == ',') {
                        p++;
                    } else {
                        p = end;
                    }
                }
            }
            if (n_emitted == 0) {
                out->error = strdup(
                    "websearch: mwmbl returned no usable results"
                    " for this query (the open index is sparse;"
                    " try rephrasing or use a more specific tool)");
                out->status = status;
                chars_free(&o);
            } else {
                struct chars wrapped = {0};
                chars_printf(&wrapped,
                    "Top %d web result(s) for \"%s\" (mwmbl.org):\n\n",
                    n_emitted, query);
                chars_puts(&wrapped, o.data != NULL ? o.data : "");
                chars_put(&wrapped, "", 0);
                out->ok     = 1;
                out->body   = wrapped.data;
                out->status = status;
                chars_free(&o);
            }
        }
        free(body.data);
    }
}

// ---------------------------------------------------------------------------
// tools_self_test — exercises each tool live. Skips on network failure
// so offline CI doesn't fail. Returns 1 on hard failures, 0 otherwise.
// Each tool prints a short verification line.
// ---------------------------------------------------------------------------

static int32_t tools_self_test(void) {
    int32_t failures = 0;
    tools_global_init();
    {
        struct tool_result r = {0};
        tools_wikipedia("proton", &r);
        if (r.ok && r.body != NULL && strstr(r.body, "proton") != NULL) {
            printf("tools-test: wikipedia(proton) OK"
                   " (%zu bytes)\n", strlen(r.body));
        } else {
            printf("tools-test: wikipedia(proton) SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    {
        struct tool_result r = {0};
        tools_time_now("Asia/Tokyo", &r);
        if (r.ok && r.body != NULL && strstr(r.body, "Tokyo") != NULL) {
            printf("tools-test: time_now(Asia/Tokyo) OK"
                   " (%zu bytes)\n", strlen(r.body));
        } else {
            printf("tools-test: time_now(Asia/Tokyo) SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    {
        struct tool_result r = {0};
        tools_weather(37.77, -122.42, &r);
        if (r.ok && r.body != NULL && strstr(r.body, "°C") != NULL) {
            printf("tools-test: weather(SF) OK"
                   " (%zu bytes)\n", strlen(r.body));
        } else {
            printf("tools-test: weather(SF) SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    {
        struct tool_result r = {0};
        tools_crypto_price("bitcoin", "usd", &r);
        if (r.ok && r.body != NULL && strstr(r.body, "bitcoin") != NULL) {
            printf("tools-test: crypto_price(bitcoin) OK"
                   " (%s)\n", r.body);
        } else {
            printf("tools-test: crypto_price(bitcoin) SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    {
        struct tool_result r = {0};
        tools_ip_geo(&r);
        if (r.ok && r.body != NULL && strstr(r.body, "IP") != NULL) {
            printf("tools-test: ip_geo() OK"
                   " (%zu bytes)\n", strlen(r.body));
        } else {
            printf("tools-test: ip_geo() SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    {
        struct tool_result r = {0};
        tools_websearch("best running shoes women", 3, &r);
        if (r.ok && r.body != NULL && r.body[0] != '\0') {
            printf("tools-test: websearch(mwmbl) OK"
                   " (%zu bytes)\n", strlen(r.body));
        } else {
            printf("tools-test: websearch(mwmbl) SKIPPED/FAIL"
                   " (err=%s)\n", r.error ? r.error : "(none)");
        }
        tools_result_free(&r);
    }
    if (failures == 0) {
        printf("tools-test: PASS\n");
    } else {
        printf("tools-test: FAIL (%d)\n", (int)failures);
    }
    return failures;
}

#else  // LLM_NO_TOOLS — stub implementations (no libcurl dependency)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tool_result {
    int    ok;
    char * body;
    char * error;
    long   status;
};

static char * tools_dup_msg(const char * s) {
    size_t n = strlen(s);
    char * out = (char *)malloc(n + 1);
    if (out != NULL) { memcpy(out, s, n + 1); }
    return out;
}

static void tools_result_free(struct tool_result * r) {
    if (r != NULL) {
        free(r->body);
        free(r->error);
        r->body  = NULL;
        r->error = NULL;
    }
}

static void tools_set_debug(int32_t level) { (void)level; }
static void tools_global_init(void)    { /* no-op */ }
static void tools_global_cleanup(void) { /* no-op */ }

static void tools_wikipedia(const char * q, struct tool_result * out) {
    (void)q;
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}
static void tools_time_now(const char * tz, struct tool_result * out) {
    (void)tz;
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}
static void tools_weather(double lat, double lon,
                          struct tool_result * out) {
    (void)lat; (void)lon;
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}
static void tools_crypto_price(const char * s, const char * v,
                               struct tool_result * out) {
    (void)s; (void)v;
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}
static void tools_ip_geo(struct tool_result * out) {
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}
static void tools_websearch(const char * q, int mr,
                            struct tool_result * out) {
    (void)q; (void)mr;
    out->ok = 0;
    out->body = NULL;
    out->error = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}

static int32_t tools_self_test(void) {
    printf("tools-test: SKIP (built with -DLLM_NO_TOOLS)\n");
    return 0;
}

#endif  // LLM_NO_TOOLS
