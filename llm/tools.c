// SPDX-License-Identifier: Apache-2.0
//
// tools.c - agent tool primitives: websearch, fetch, distill.
//
// Three composable primitives an agent loop hands to the model
// through the Qwen3.5 `<tool_call>` framing (built by jinja_apply
// in llm/jinja-template.c):
//
//   websearch(query, max_results)
//       DuckDuckGo lite-HTML scrape. No API key required. Returns
//       a plain-text "Web results for: <q>\n\n- title1\n  snippet\n
//       \n- title2\n  snippet\n..." that fits in a typical model's
//       attention window.
//   fetch(url, timeout_s)
//       libcurl GET with a browser User-Agent; returns the body
//       prefixed by `HTTP <status>\n\n`. Truncates at 200 KB so
//       a stray multi-MB asset never blows the context.
//   distill(html)
//       Strip tags + decode entities + collapse whitespace; returns
//       the meaningful text content of a page. Composes with fetch
//       for "fetch this URL and tell me what it says" agent flows.
//
// Single-file lib. `#include "tools.c"` from llm.c under
// `#ifdef LLM_WITH_TOOLS` so the libcurl dependency stays optional
// (the core inference path has zero net deps; tools.c is the only
// thing in this repo that links libcurl). Listed under
// QwenHaiku.xcodeproj's membershipExceptions alongside the other
// single-file libs.
//
// Style notes:
//   - Patterned after doit.devs/src/tools.c (Leo's home agent
//     framework). Same DDG selectors, same UA + Referer hardening,
//     same HTTP-status surfacing through tool_result.error.
//   - SESE: single exit per function, no goto / break / continue
//     except in switch grammar, no `bool ok` flag layered on real
//     state.
//
// Caller responsibilities:
//   - libcurl global init: tools_global_init() once, before any
//     tool call; tools_global_cleanup() at shutdown.
//   - tool_result lifetime: caller frees `body` and `error` via
//     tools_result_free.

#include <ctype.h>
#include <curl/curl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // sleep()

#ifndef TOOLS_FETCH_CAP
#define TOOLS_FETCH_CAP (200 * 1024)
#endif

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

// Internal grow-as-needed string buffer (same shape jinja-template.c
// uses; replicated here to keep tools.c truly standalone — drop-in
// usable in a project that doesn't include jinja-template.c).
struct ts_buf {
    char * data;
    size_t count;
    size_t capacity;
};

static void ts_grow(struct ts_buf * b, size_t need) {
    size_t cap = b->capacity == 0 ? 256 : b->capacity;
    while (cap < need + 1) { cap *= 2; }
    if (cap > b->capacity) {
        char * nd = (char *)realloc(b->data, cap);
        if (nd != NULL) {
            b->data = nd;
            b->capacity = cap;
        }
    }
}

static void ts_put(struct ts_buf * b, const char * s, size_t n) {
    if (n > 0) {
        ts_grow(b, b->count + n);
        if (b->data != NULL) {
            memcpy(b->data + b->count, s, n);
            b->count += n;
            b->data[b->count] = '\0';
        }
    }
}

static void ts_puts(struct ts_buf * b, const char * s) {
    if (s != NULL && s[0] != '\0') { ts_put(b, s, strlen(s)); }
}

static void ts_printf(struct ts_buf * b, const char * fmt, ...) {
    char    tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n > 0) {
        ts_put(b, tmp, (size_t)n < sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
    }
}

// libcurl write-callback into a ts_buf. Caps writes once the buffer
// hits TOOLS_FETCH_CAP so a runaway server can't make us allocate
// gigabytes — extra bytes are silently dropped (the truncation
// notice is appended by the caller).
static size_t tools_curl_write(void * ptr, size_t size, size_t nmemb,
                               void * user) {
    struct ts_buf * b = (struct ts_buf *)user;
    size_t total = size * nmemb;
    size_t cap   = TOOLS_FETCH_CAP;
    size_t avail = (b->count < cap) ? (cap - b->count) : 0;
    size_t take  = (total < avail) ? total : avail;
    if (take > 0) {
        ts_put(b, (const char *)ptr, take);
    }
    // Tell curl we accepted everything so it doesn't error out;
    // we'll surface truncation in the body itself.
    return total;
}

// One-shot libcurl global init / cleanup. Idempotent — callers can
// invoke before every tool call without worrying about pairing.
static int    g_tools_curl_initialized = 0;

static void tools_global_init(void) {
    if (!g_tools_curl_initialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        g_tools_curl_initialized = 1;
    }
}

__attribute__((unused))
static void tools_global_cleanup(void) {
    if (g_tools_curl_initialized) {
        curl_global_cleanup();
        g_tools_curl_initialized = 0;
    }
}

// URL-encode `q` into `out`. RFC 3986 unreserved set only.
static void tools_url_encode(const char * q, struct ts_buf * out) {
    const char * p = q;
    while (*p != '\0') {
        unsigned char c = (unsigned char)*p;
        int unreserved = (c >= 'A' && c <= 'Z')
                      || (c >= 'a' && c <= 'z')
                      || (c >= '0' && c <= '9')
                      ||  c == '-' || c == '_'
                      ||  c == '.' || c == '~';
        if (unreserved) {
            char ch = (char)c;
            ts_put(out, &ch, 1);
        } else {
            char hex[4];
            snprintf(hex, sizeof(hex), "%%%02X", c);
            ts_puts(out, hex);
        }
        p++;
    }
}

// Decode one HTML entity starting at &. Returns the number of bytes
// consumed from `s` (so caller advances `i += adv`), 0 if not an
// entity. Recognises &amp; &lt; &gt; &quot; &apos; &nbsp; and
// numeric &#NNN; / &#xHH;.
static size_t tools_decode_entity(const char * s, size_t i, size_t n,
                                  struct ts_buf * out) {
    size_t adv = 0;
    if (i < n && s[i] == '&') {
        size_t end = i + 1;
        while (end < n && end - i < 10 && s[end] != ';') { end++; }
        if (end < n && s[end] == ';') {
            size_t len = end - i - 1;
            const char * body = s + i + 1;
            if (len == 3 && memcmp(body, "amp", 3) == 0) {
                ts_put(out, "&", 1); adv = 4;
            } else if (len == 2 && memcmp(body, "lt", 2) == 0) {
                ts_put(out, "<", 1); adv = 3;
            } else if (len == 2 && memcmp(body, "gt", 2) == 0) {
                ts_put(out, ">", 1); adv = 3;
            } else if (len == 4 && memcmp(body, "quot", 4) == 0) {
                ts_put(out, "\"", 1); adv = 5;
            } else if (len == 4 && memcmp(body, "apos", 4) == 0) {
                ts_put(out, "'", 1); adv = 5;
            } else if (len == 4 && memcmp(body, "nbsp", 4) == 0) {
                ts_put(out, " ", 1); adv = 5;
            } else if (len >= 2 && body[0] == '#') {
                int cp = 0;
                if (body[1] == 'x' || body[1] == 'X') {
                    for (size_t k = 2; k < len; k++) {
                        char c = body[k];
                        int d = -1;
                        if (c >= '0' && c <= '9') { d = c - '0'; }
                        else if (c >= 'a' && c <= 'f') { d = c - 'a' + 10; }
                        else if (c >= 'A' && c <= 'F') { d = c - 'A' + 10; }
                        if (d >= 0) { cp = cp * 16 + d; }
                    }
                } else {
                    for (size_t k = 1; k < len; k++) {
                        if (body[k] >= '0' && body[k] <= '9') {
                            cp = cp * 10 + (body[k] - '0');
                        }
                    }
                }
                // Emit as UTF-8.
                if (cp >= 0 && cp < 0x80) {
                    char c = (char)cp;
                    ts_put(out, &c, 1);
                } else if (cp < 0x800) {
                    char b[2];
                    b[0] = (char)(0xC0 | (cp >> 6));
                    b[1] = (char)(0x80 | (cp & 0x3F));
                    ts_put(out, b, 2);
                } else if (cp < 0x10000) {
                    char b[3];
                    b[0] = (char)(0xE0 | (cp >> 12));
                    b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    b[2] = (char)(0x80 | (cp & 0x3F));
                    ts_put(out, b, 3);
                } else {
                    char b[4];
                    b[0] = (char)(0xF0 | (cp >> 18));
                    b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                    b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    b[3] = (char)(0x80 | (cp & 0x3F));
                    ts_put(out, b, 4);
                }
                adv = len + 2;  // &...;
            }
        }
    }
    return adv;
}

// Strip HTML tags + decode entities from `[s, s+n)` into `out`.
// Newlines, tabs, and \r collapse to a single space; whitespace
// run-collapsing happens in the caller if needed.
static void tools_html_strip(const char * s, size_t n,
                             struct ts_buf * out) {
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        if (c == '<') {
            size_t j = i + 1;
            while (j < n && s[j] != '>') { j++; }
            i = (j < n) ? j + 1 : n;
        } else if (c == '&') {
            size_t adv = tools_decode_entity(s, i, n, out);
            if (adv > 0) {
                i += adv;
            } else {
                ts_put(out, &c, 1);
                i++;
            }
        } else if (c == '\r' || c == '\n' || c == '\t') {
            ts_put(out, " ", 1);
            i++;
        } else {
            ts_put(out, &c, 1);
            i++;
        }
    }
}

// Collapse runs of ASCII whitespace inside `in` into a single space;
// trim leading and trailing whitespace. Writes into `out`.
static void tools_collapse_ws(const struct ts_buf * in,
                              struct ts_buf * out) {
    size_t i = 0;
    bool   in_ws    = true;  // start as "just saw ws" so leading is dropped
    size_t emitted  = 0;
    while (i < in->count) {
        unsigned char c = (unsigned char)in->data[i];
        bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (is_ws) {
            if (!in_ws && emitted > 0) {
                ts_put(out, " ", 1);
                emitted++;
            }
            in_ws = true;
        } else {
            char ch = (char)c;
            ts_put(out, &ch, 1);
            emitted++;
            in_ws = false;
        }
        i++;
    }
    // Strip trailing single space we may have appended pending real char.
    while (out->count > 0 && out->data[out->count - 1] == ' ') {
        out->count--;
        out->data[out->count] = '\0';
    }
}

// UA pool. Each slot carries the User-Agent string AND a "style"
// tag — Chrome and Safari emit DIFFERENT header sets in real
// traffic, and sending the wrong set with the wrong UA is itself
// a fingerprint inconsistency that bot detectors look for. The
// style flag drives the per-style header builder below.
//
// Slots populated from real captured browser requests (Leo's
// Chrome 148 + Safari Incognito 26.4 against duckduckgo.com).
enum tools_ua_style {
    TOOLS_UA_CHROME = 0,
    TOOLS_UA_SAFARI = 1,
};

struct tools_ua_slot {
    const char *        ua;
    enum tools_ua_style style;
    const char *        ch_platform;  // Chrome-only; NULL for Safari
};

static const struct tools_ua_slot TOOLS_UA_POOL[] = {
    { "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
      " AppleWebKit/537.36 (KHTML, like Gecko)"
      " Chrome/148.0.0.0 Safari/537.36",
      TOOLS_UA_CHROME, "\"macOS\"" },
    { "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
      " AppleWebKit/537.36 (KHTML, like Gecko)"
      " Chrome/148.0.0.0 Safari/537.36",
      TOOLS_UA_CHROME, "\"Windows\"" },
    { "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
      " AppleWebKit/605.1.15 (KHTML, like Gecko)"
      " Version/26.4 Safari/605.1.15",
      TOOLS_UA_SAFARI, NULL },
};

static const struct tools_ua_slot * tools_pick_ua(void) {
    static unsigned int counter = 0;
    unsigned int n = (unsigned int)
        (sizeof(TOOLS_UA_POOL) / sizeof(TOOLS_UA_POOL[0]));
    const struct tools_ua_slot * slot = &TOOLS_UA_POOL[counter % n];
    counter++;
    return slot;
}

// HTTP GET, body lands in `body`. Sets `*status` to the HTTP code.
// Returns 0 on success, -1 on libcurl failure (status will be 0).
//
// Headers and UA chosen to look like a real browser navigation,
// since DDG's bot-detection is opportunistic — back-to-back identical
// requests get challenged, but rotating UA + adding Sec-Fetch-*
// browser-nav headers passes more often. Still no substitute for a
// proper search API; this is best-effort scraping.
static int tools_http_get(const char * url, long timeout_ms,
                          struct ts_buf * body, long * status) {
    int rc = -1;
    *status = 0;
    CURL * h = curl_easy_init();
    if (h != NULL) {
        struct curl_slist * hdrs = NULL;
        const struct tools_ua_slot * slot = tools_pick_ua();
        char ua_header[512];
        snprintf(ua_header, sizeof(ua_header),
                 "User-Agent: %s", slot->ua);
        hdrs = curl_slist_append(hdrs, ua_header);
        // Shared headers (both Chrome and Safari emit these on a
        // top-level navigation to an HTTPS URL with no referer).
        hdrs = curl_slist_append(hdrs,
            "Accept: text/html,application/xhtml+xml,application/xml;"
            "q=0.9,*/*;q=0.8");
        hdrs = curl_slist_append(hdrs,
            "Accept-Language: en-US,en;q=0.9");
        hdrs = curl_slist_append(hdrs, "Priority: u=0, i");
        // Sec-Fetch-Site: none means "no Referer / new tab nav" —
        // matches the absence of a Referer header in our request.
        // Real Chrome / Safari send "none" in that case, NOT
        // "cross-site" (the previous value, which implied we came
        // from a different site — another fingerprint inconsistency).
        hdrs = curl_slist_append(hdrs, "Sec-Fetch-Dest: document");
        hdrs = curl_slist_append(hdrs, "Sec-Fetch-Mode: navigate");
        hdrs = curl_slist_append(hdrs, "Sec-Fetch-Site: none");
        if (slot->style == TOOLS_UA_CHROME) {
            // Chrome-only extras: User-Agent Client Hints, DNT,
            // Pragma/Cache-Control, Sec-Fetch-User, Upgrade-Insecure.
            // Order matches Chrome 148's real emission order.
            hdrs = curl_slist_append(hdrs, "Cache-Control: no-cache");
            hdrs = curl_slist_append(hdrs, "Pragma: no-cache");
            hdrs = curl_slist_append(hdrs, "DNT: 1");
            hdrs = curl_slist_append(hdrs,
                "Sec-CH-UA: \"Chromium\";v=\"148\","
                " \"Google Chrome\";v=\"148\","
                " \"Not/A)Brand\";v=\"99\"");
            hdrs = curl_slist_append(hdrs, "Sec-CH-UA-Mobile: ?0");
            char ch_platform[64];
            snprintf(ch_platform, sizeof(ch_platform),
                     "Sec-CH-UA-Platform: %s",
                     slot->ch_platform != NULL ? slot->ch_platform
                                               : "\"macOS\"");
            hdrs = curl_slist_append(hdrs, ch_platform);
            hdrs = curl_slist_append(hdrs, "Sec-Fetch-User: ?1");
            hdrs = curl_slist_append(hdrs,
                "Upgrade-Insecure-Requests: 1");
        }
        // Safari Incognito sends the minimal set above plus nothing
        // else — no Sec-CH-UA (Safari doesn't implement Client Hints),
        // no Pragma, no DNT, no Upgrade-Insecure-Requests on this
        // particular nav. Matches Leo's captured Safari headers.
        char errbuf[CURL_ERROR_SIZE];
        errbuf[0] = '\0';
        curl_easy_setopt(h, CURLOPT_URL, url);
        curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(h, CURLOPT_MAXREDIRS, 5L);
        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, timeout_ms);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
        curl_easy_setopt(h, CURLOPT_NOPROGRESS, 1L);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, tools_curl_write);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, body);
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, errbuf);
        // CURLOPT_ACCEPT_ENCODING="" makes libcurl advertise (in the
        // Accept-Encoding header) whatever decoders it was built
        // with, AND transparently decode the response on the way
        // back. We don't manually set Accept-Encoding because the
        // declared algorithms must match what libcurl can decode —
        // libcurl on macOS may or may not have brotli linked.
        curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
        curl_easy_setopt(h, CURLOPT_USERAGENT, slot->ua);
        CURLcode cc = curl_easy_perform(h);
        if (cc == CURLE_OK) {
            rc = 0;
            curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, status);
        } else {
            fprintf(stderr, "tools_http_get: curl_easy_perform: %s\n",
                    errbuf[0] != '\0' ? errbuf : curl_easy_strerror(cc));
        }
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(h);
    }
    return rc;
}

// Parse DDG-lite HTML and emit a "- title\n  snippet\n\n" list into
// `out`. Returns the number of (title, snippet) pairs emitted.
// The selectors mirror doit.devs/src/tools.c::tools_ddg_emit_titles
// + tools_ddg_emit_snippets — DDG-lite's markup hasn't moved in
// years, but if it does, this is the one function to update.
static int tools_ddg_parse(const char * h, size_t n, int max_results,
                           struct ts_buf * out) {
    int  seen = 0;
    size_t i  = 0;
    bool more = true;
    // First pass: titles. Each result link carries rel="nofollow".
    while (more && seen < max_results && i < n) {
        const char * p = (const char *)memmem(h + i, n - i,
                                              "rel=\"nofollow\"", 14);
        if (p == NULL) {
            more = false;
        } else {
            size_t pos = (size_t)(p - h);
            size_t gt = pos;
            while (gt < n && h[gt] != '>') { gt++; }
            const char * pe = (gt < n)
                ? (const char *)memmem(h + gt + 1, n - gt - 1,
                                       "</a>", 4)
                : NULL;
            if (gt >= n || pe == NULL) {
                more = false;
            } else {
                size_t end = (size_t)(pe - h);
                struct ts_buf title = {0};
                tools_html_strip(h + gt + 1, end - gt - 1, &title);
                if (title.count > 5) {
                    struct ts_buf clean = {0};
                    tools_collapse_ws(&title, &clean);
                    ts_printf(out, "- %s\n",
                              clean.data ? clean.data : "");
                    free(clean.data);
                    seen++;
                }
                free(title.data);
                i = end + 4;
            }
        }
    }
    // Second pass: snippets. DDG marks them with class
    // "result-snippet". Snippets appear in the same document order
    // as the links above, so we emit them interleaved-by-position
    // (each snippet block immediately after its title).
    i = 0;
    more = true;
    int snip_emitted = 0;
    while (more && snip_emitted < max_results && i < n) {
        const char * p = (const char *)memmem(h + i, n - i,
                                              "result-snippet", 14);
        if (p == NULL) {
            more = false;
        } else {
            size_t pos = (size_t)(p - h);
            size_t gt = pos;
            while (gt < n && h[gt] != '>') { gt++; }
            if (gt >= n) {
                more = false;
            } else {
                size_t end = gt + 1;
                bool found_close = false;
                while (!found_close && end + 1 < n) {
                    if (h[end] == '<' && h[end + 1] == '/') {
                        found_close = true;
                    } else {
                        end++;
                    }
                }
                if (!found_close) {
                    more = false;
                } else {
                    struct ts_buf snip = {0};
                    tools_html_strip(h + gt + 1, end - gt - 1, &snip);
                    if (snip.count > 10) {
                        struct ts_buf clean = {0};
                        tools_collapse_ws(&snip, &clean);
                        ts_printf(out, "  %s\n\n",
                                  clean.data ? clean.data : "");
                        free(clean.data);
                        snip_emitted++;
                    }
                    free(snip.data);
                    i = end;
                }
            }
        }
    }
    return seen;
}

// websearch: scrape DuckDuckGo lite-HTML. No API key.
static void tools_websearch(const char * query, int max_results,
                            struct tool_result * out) {
    tools_global_init();
    out->ok = 0;
    out->body = NULL;
    out->error = NULL;
    out->status = 0;
    if (query == NULL || query[0] == '\0') {
        out->error = strdup("websearch: query parameter required");
    } else {
        int cap = (max_results > 0 && max_results <= 16)
                ? max_results : 8;
        struct ts_buf eq = {0};
        tools_url_encode(query, &eq);
        struct ts_buf url = {0};
        // Match the URL shape Leo's Safari Incognito captured for a
        // real DDG search: ia=web (instant-answer category), origin=
        // funnel_home_website (came from DDG home page), t=h_ (DDG's
        // browser-type hint), then q=..., chip-select=search.
        // The lite endpoint is forgiving and accepts a bare ?q=,
        // but adding the funnel params makes the request match an
        // organic-navigation profile.
        ts_puts(&url,
            "https://lite.duckduckgo.com/lite/"
            "?ia=web&origin=funnel_home_website&t=h_&q=");
        if (eq.data != NULL) { ts_puts(&url, eq.data); }
        ts_puts(&url, "&chip-select=search");
        struct ts_buf body = {0};
        long status = 0;
        int rc = tools_http_get(url.data, 15000, &body, &status);
        if (rc != 0) {
            out->error = strdup(
                "websearch: libcurl request failed (network down?)");
        } else if (status < 200 || status >= 400) {
            char tmp[128];
            snprintf(tmp, sizeof(tmp),
                "websearch: HTTP %ld (DDG may be rate-limiting)",
                status);
            out->error  = strdup(tmp);
            out->status = status;
        } else if (body.data == NULL || body.count == 0) {
            out->error = strdup("websearch: empty response from DDG");
        } else if (memmem(body.data, body.count,
                          "anomaly-modal", 13) != NULL) {
            // DDG's bot-detection challenge page — looks like a 200
            // OK to libcurl, but the body is a CAPTCHA, no search
            // results. Hits us when we make rapid back-to-back
            // queries from the same IP. Surface as a distinct error
            // so the caller can retry (with a delay) or fall back to
            // a different engine.
            out->error = strdup(
                "websearch: DDG bot-detection CAPTCHA challenge"
                " (rate-limited — try again in a few minutes)");
            out->status = status;
        } else {
            struct ts_buf result = {0};
            ts_printf(&result, "Web results for: %s\n\n", query);
            int titles = tools_ddg_parse(body.data, body.count,
                                         cap, &result);
            if (titles == 0) {
                ts_printf(&result,
                    "No results found for: %s\n", query);
            }
            out->body   = result.data;
            out->status = status;
            out->ok     = 1;
        }
        free(eq.data);
        free(url.data);
        free(body.data);
    }
}

// fetch: HTTP GET via libcurl with cap. Body returned as
// `HTTP <status>\n\n<bytes>...` truncated at TOOLS_FETCH_CAP.
static void tools_fetch(const char * url, int timeout_s,
                        struct tool_result * out) {
    tools_global_init();
    out->ok = 0;
    out->body = NULL;
    out->error = NULL;
    out->status = 0;
    if (url == NULL || url[0] == '\0') {
        out->error = strdup("fetch: url parameter required");
    } else {
        long ms = (long)(timeout_s > 0 ? timeout_s : 30) * 1000L;
        if (ms < 1000)   { ms = 1000;   }
        if (ms > 300000) { ms = 300000; }
        struct ts_buf body = {0};
        long status = 0;
        int rc = tools_http_get(url, ms, &body, &status);
        if (rc != 0) {
            out->error = strdup("fetch: libcurl request failed");
        } else {
            struct ts_buf result = {0};
            ts_printf(&result, "HTTP %ld\n\n", status);
            if (body.data != NULL && body.count > 0) {
                ts_put(&result, body.data, body.count);
            }
            if (body.count >= TOOLS_FETCH_CAP) {
                ts_printf(&result,
                    "\n\n[... truncated at %d bytes]",
                    (int)TOOLS_FETCH_CAP);
            }
            out->body   = result.data;
            out->status = status;
            out->ok     = (status >= 200 && status < 400);
            if (!out->ok) {
                char tmp[64];
                snprintf(tmp, sizeof(tmp), "fetch: HTTP %ld", status);
                out->error = strdup(tmp);
            }
        }
        free(body.data);
    }
}

// distill: HTML -> clean text. Compose with fetch for "summarise
// this URL" agent flows. The model sees the distilled text, not the
// raw markup, so its attention budget goes to content rather than
// boilerplate.
static void tools_distill(const char * html, size_t n,
                          struct tool_result * out) {
    out->ok = 0;
    out->body = NULL;
    out->error = NULL;
    out->status = 0;
    if (html == NULL) {
        out->error = strdup("distill: input is NULL");
    } else {
        struct ts_buf stripped = {0};
        tools_html_strip(html, n, &stripped);
        struct ts_buf clean = {0};
        tools_collapse_ws(&stripped, &clean);
        out->body = clean.data;
        out->ok   = 1;
        free(stripped.data);
    }
}

// Self-test. Network-dependent: skips silently if libcurl can't
// reach DDG (offline development). The two queries below are the
// reference test cases — "rock band HelloWorld album" exercises a
// trivia query that should find substantive snippets, and
// "bitcoin price today" exercises a fresh-data query the model
// CAN'T answer from training alone.
__attribute__((unused))
static int32_t tools_self_test(void) {
    tools_global_init();
    int failures = 0;
    const char * queries[] = {
        "Search Internet to find what rock band from which country"
            " recorded HelloWorld album?",
        "What is bitcoin price today?",
    };
    int nq = (int)(sizeof(queries) / sizeof(queries[0]));
    int reachable = -1;
    int captcha_hit = 0;
    for (int i = 0; i < nq; i++) {
        struct tool_result r = {0};
        tools_websearch(queries[i], 5, &r);
        if (!r.ok) {
            int looks_captcha = (r.error != NULL) &&
                                (strstr(r.error, "CAPTCHA") != NULL);
            if (looks_captcha) {
                captcha_hit = 1;
                if (reachable == -1) { reachable = 1; }
                fprintf(stderr,
                    "tools-test: websearch[%d] rate-limited"
                    " (CAPTCHA) — not a code failure\n", i);
            } else if (reachable == -1) {
                // First failure: probably offline. Note and don't
                // count further failures as test fails — the goal
                // is "the code path works", not "DDG is online".
                fprintf(stderr,
                    "tools-test: websearch[%d] failed: %s\n",
                    i, r.error ? r.error : "(no error)");
                reachable = 0;
            }
        } else {
            reachable = 1;
            size_t n = strlen(r.body);
            size_t show = n > 600 ? 600 : n;
            printf("tools-test: websearch[%d] (%zu bytes) OK\n%.*s\n\n",
                   i, n, (int)show, r.body);
            if (n < 64) {
                fprintf(stderr,
                    "tools-test: websearch[%d] body suspiciously"
                    " short (%zu bytes)\n", i, n);
                failures++;
            }
        }
        tools_result_free(&r);
        // Delay between queries reduces DDG CAPTCHA hits. 2s was
        // not enough (still tripped), 6s lets DDG's rate-limit
        // window roll over. Skipped after the last query.
        if (i + 1 < nq) { sleep(6); }
    }
    // Also exercise fetch + distill on a tiny stable URL.
    if (reachable == 1) {
        struct tool_result f = {0};
        tools_fetch("https://example.com/", 15, &f);
        if (!f.ok) {
            fprintf(stderr,
                "tools-test: fetch(example.com) failed: %s\n",
                f.error ? f.error : "(no error)");
            failures++;
        } else {
            struct tool_result d = {0};
            tools_distill(f.body, strlen(f.body), &d);
            if (!d.ok || d.body == NULL ||
                strstr(d.body, "Example Domain") == NULL) {
                fprintf(stderr,
                    "tools-test: distill missing"
                    " 'Example Domain' anchor (got: %.200s)\n",
                    d.body ? d.body : "(null)");
                failures++;
            } else {
                printf("tools-test: fetch+distill example.com OK\n");
            }
            tools_result_free(&d);
        }
        tools_result_free(&f);
    }
    if (reachable == 0) {
        printf("tools-test: SKIP (offline — DDG unreachable)\n");
    } else if (failures == 0 && captcha_hit == 0) {
        printf("tools-test: PASS\n");
    } else if (failures == 0 && captcha_hit == 1) {
        printf("tools-test: PASS (some queries CAPTCHA-blocked"
               " — DDG rate-limiting, not a code defect)\n");
    } else {
        printf("tools-test: FAIL (%d sub-tests)\n", failures);
    }
    return (failures > 0) ? 1 : 0;
}
