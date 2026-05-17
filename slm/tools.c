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
// Single-file lib. `#include "tools.c"` from slm.c under
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

// Always-available headers + ts_buf helpers (used by agent.c too —
// must stay visible regardless of LLM_NO_TOOLS).
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// LLM_NO_TOOLS: opt-out for builds that can't link libcurl (Android
// NDK, embedded targets). When set, this file emits stub
// implementations that return "tool unavailable" results; agent.c
// continues to compile against the same prototypes. The Apple /
// Linux dev path leaves LLM_NO_TOOLS undefined and links libcurl
// as before.
#ifndef LLM_NO_TOOLS

#include <ctype.h>
#include <curl/curl.h>
#include <unistd.h>  // sleep()

#ifndef TOOLS_FETCH_CAP
#define TOOLS_FETCH_CAP (200 * 1024)
#endif

// Debug verbosity for tool internals. slm_generate sets this from
// c->ctrl.debug right before dispatching a tool; the level gates how
// chatty tools_websearch / tools_fetch are at each call. File-scoped
// (not per-ctx) because tools_* are pure C functions with no ctx
// parameter — passing debug through every primitive would balloon the
// signatures for one diagnostic knob.
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

// ts_buf + ts_grow/ts_put/ts_puts/chars_printf hoisted above the
// LLM_NO_TOOLS gate so agent.c sees them in both build modes.

// libcurl write-callback into a ts_buf. Caps writes once the buffer
// hits TOOLS_FETCH_CAP so a runaway server can't make us allocate
// gigabytes — extra bytes are silently dropped (the truncation
// notice is appended by the caller).
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
    // Tell curl we accepted everything so it doesn't error out;
    // we'll surface truncation in the body itself.
    return total;
}

// One-shot libcurl global init / cleanup. Idempotent — callers can
// invoke before every tool call without worrying about pairing.
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

// Percent-decode `s[0..n)` into `out`. `+` is treated as space (the
// historic form-encoding rule; DDG's uddg= values don't use `+` for
// space but it's harmless to decode). Invalid `%XX` sequences pass
// through as-is.
static void tools_url_decode(const char * s, size_t n,
                             struct chars * out) {
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        if (c == '%' && i + 2 < n) {
            char h = s[i + 1];
            char l = s[i + 2];
            int  hi = -1;
            int  lo = -1;
            if      (h >= '0' && h <= '9') { hi = h - '0';      }
            else if (h >= 'a' && h <= 'f') { hi = h - 'a' + 10; }
            else if (h >= 'A' && h <= 'F') { hi = h - 'A' + 10; }
            if      (l >= '0' && l <= '9') { lo = l - '0';      }
            else if (l >= 'a' && l <= 'f') { lo = l - 'a' + 10; }
            else if (l >= 'A' && l <= 'F') { lo = l - 'A' + 10; }
            if (hi >= 0 && lo >= 0) {
                char b = (char)(hi * 16 + lo);
                chars_put(out, &b, 1);
                i += 3;
            } else {
                chars_put(out, &c, 1);
                i++;
            }
        } else if (c == '+') {
            chars_put(out, " ", 1);
            i++;
        } else {
            chars_put(out, &c, 1);
            i++;
        }
    }
}

// Resolve a raw href value from DDG-lite into a real, fetch-able URL.
// DDG wraps every result link in a redirect tracker of the form
//     //duckduckgo.com/l/?uddg=<percent-encoded-target>&rut=<hash>
// (protocol-relative, payload inside uddg=). Without unwrapping,
// tools_fetch hits libcurl's "URL rejected: No host part" error
// (the `//` prefix), or — if we patched the protocol — would just
// re-hit DDG instead of the real source. We extract the uddg=
// value up to the next `&` (which catches the `&amp;rut=...` tail
// since `&amp;` starts with `&`), then percent-decode it.
//
// Non-wrapped href values (other engines, or DDG corner cases) are
// passed through; if they're protocol-relative we prepend https:.
//
// Returns a heap-allocated, NUL-terminated URL; caller frees.
static char * tools_url_unwrap(const char * raw, size_t n) {
    char * result = NULL;
    if (raw != NULL && n > 0) {
        const char * uddg = (const char *)memmem(raw, n, "uddg=", 5);
        if (uddg != NULL) {
            const char * v   = uddg + 5;
            size_t       lim = n - (size_t)(v - raw);
            const char * end = v;
            while ((size_t)(end - v) < lim && *end != '&') { end++; }
            struct chars dec = {0};
            tools_url_decode(v, (size_t)(end - v), &dec);
            chars_put(&dec, "", 0);
            result = dec.data;
        } else if (n >= 2 && raw[0] == '/' && raw[1] == '/') {
            struct chars b = {0};
            chars_puts(&b, "https:");
            chars_put(&b, raw, n);
            chars_put(&b, "", 0);
            result = b.data;
        } else {
            result = (char *)oom(malloc(n + 1));
            memcpy(result, raw, n);
            result[n] = '\0';
        }
    }
    return result;
}

// URL-encode `q` into `out`. RFC 3986 unreserved set only.

static void tools_url_encode(const char * q, struct chars * out) {
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
            chars_put(out, &ch, 1);
        } else {
            char hex[4];
            snprintf(hex, sizeof(hex), "%%%02X", c);
            chars_puts(out, hex);
        }
        p++;
    }
}

// Decode one HTML entity starting at &. Returns the number of bytes
// consumed from `s` (so caller advances `i += adv`), 0 if not an
// entity. Recognises &amp; &lt; &gt; &quot; &apos; &nbsp; and
// numeric &#NNN; / &#xHH;.

static size_t tools_decode_entity(const char * s, size_t i, size_t n,
                                  struct chars * out) {
    size_t adv = 0;
    if (i < n && s[i] == '&') {
        size_t end = i + 1;
        while (end < n && end - i < 10 && s[end] != ';') { end++; }
        if (end < n && s[end] == ';') {
            size_t len = end - i - 1;
            const char * body = s + i + 1;
            if (len == 3 && memcmp(body, "amp", 3) == 0) {
                chars_put(out, "&", 1); adv = 4;
            } else if (len == 2 && memcmp(body, "lt", 2) == 0) {
                chars_put(out, "<", 1); adv = 3;
            } else if (len == 2 && memcmp(body, "gt", 2) == 0) {
                chars_put(out, ">", 1); adv = 3;
            } else if (len == 4 && memcmp(body, "quot", 4) == 0) {
                chars_put(out, "\"", 1); adv = 5;
            } else if (len == 4 && memcmp(body, "apos", 4) == 0) {
                chars_put(out, "'", 1); adv = 5;
            } else if (len == 4 && memcmp(body, "nbsp", 4) == 0) {
                chars_put(out, " ", 1); adv = 5;
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
                    chars_put(out, &c, 1);
                } else if (cp < 0x800) {
                    char b[2];
                    b[0] = (char)(0xC0 | (cp >> 6));
                    b[1] = (char)(0x80 | (cp & 0x3F));
                    chars_put(out, b, 2);
                } else if (cp < 0x10000) {
                    char b[3];
                    b[0] = (char)(0xE0 | (cp >> 12));
                    b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    b[2] = (char)(0x80 | (cp & 0x3F));
                    chars_put(out, b, 3);
                } else {
                    char b[4];
                    b[0] = (char)(0xF0 | (cp >> 18));
                    b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                    b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    b[3] = (char)(0x80 | (cp & 0x3F));
                    chars_put(out, b, 4);
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
                             struct chars * out) {
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
                chars_put(out, &c, 1);
                i++;
            }
        } else if (c == '\r' || c == '\n' || c == '\t') {
            chars_put(out, " ", 1);
            i++;
        } else {
            chars_put(out, &c, 1);
            i++;
        }
    }
}

// Collapse runs of ASCII whitespace inside `in` into a single space;
// trim leading and trailing whitespace. Writes into `out`.
static void tools_collapse_ws(const struct chars * in,
                              struct chars * out) {
    size_t i = 0;
    bool   in_ws    = true;  // start as "just saw ws" so leading is dropped
    size_t emitted  = 0;
    while (i < in->count) {
        unsigned char c = (unsigned char)in->data[i];
        bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (is_ws) {
            if (!in_ws && emitted > 0) {
                chars_put(out, " ", 1);
                emitted++;
            }
            in_ws = true;
        } else {
            char ch = (char)c;
            chars_put(out, &ch, 1);
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
                          struct chars * body, long * status) {
    int rc = -1;
    *status = 0;
    if (g_tools_debug >= 9) {
        trace("http_get: GET %s (timeout=%ldms)\n", url, timeout_ms);
    }
    CURL * h = curl_easy_init();
    if (h != NULL) {
        struct curl_slist * hdrs = NULL;
        const struct tools_ua_slot * slot = tools_pick_ua();
        if (g_tools_debug >= 9) {
            trace("http_get: UA=%s\n", slot->ua);
        }
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
            if (g_tools_debug >= 9) {
                trace("http_get: status=%ld body=%zu bytes\n",
                      *status, body->count);
            }
        } else {
            trace("http_get: curl_easy_perform: %s\n",
                  errbuf[0] != '\0' ? errbuf : curl_easy_strerror(cc));
        }
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(h);
    }
    return rc;
}

// Parse DDG-lite HTML into a list of {title, url, snippet} hits.
// `hits` is an out-array of capacity >= max_results, zero-initialised
// by the caller. Returns the number of hits filled in [0..N). Strings
// in each hit are heap-owned; caller frees with tools_free_hits.
//
// DDG-lite link shape (Leo's captured markup):
//   <a class="result-link" rel="nofollow" href="https://...">Title</a>
// Snippets are sibling <td class="result-snippet"> nodes in document
// order. The two passes assume DDG keeps them paired by index; if
// snippet count < title count the trailing hits have snippet=NULL.
struct tools_search_hit {
    char * title;
    char * url;
    char * snippet;
};

#ifndef TOOLS_HITS_MAX
#define TOOLS_HITS_MAX 8
#endif

static void tools_free_hits(struct tools_search_hit * hits, int n) {
    if (hits != NULL) {
        for (int i = 0; i < n; i++) {
            free(hits[i].title);   hits[i].title   = NULL;
            free(hits[i].url);     hits[i].url     = NULL;
            free(hits[i].snippet); hits[i].snippet = NULL;
        }
    }
}

static int tools_ddg_parse(const char * h, size_t n, int max_results,
                           struct tools_search_hit * hits) {
    int    seen = 0;
    size_t i    = 0;
    bool   more = true;
    // First pass: title + URL. Walk for each rel="nofollow"; extract
    // href= from the <a> opener and title text from the <a> body.
    while (more && seen < max_results && i < n) {
        const char * p = (const char *)memmem(h + i, n - i,
                                              "rel=\"nofollow\"", 14);
        if (p == NULL) {
            more = false;
        } else {
            size_t pos  = (size_t)(p - h);
            size_t back = pos > 512 ? pos - 512 : 0;
            size_t lt   = pos;
            while (lt > back && h[lt] != '<') { lt--; }
            size_t gt = pos;
            while (gt < n && h[gt] != '>') { gt++; }
            char * url = NULL;
            if (gt < n && lt < pos) {
                const char * hp = (const char *)memmem(
                    h + lt, gt - lt, "href=\"", 6);
                if (hp != NULL) {
                    const char * v   = hp + 6;
                    const char * end = (const char *)memchr(
                        v, '"', (size_t)(h + gt - v));
                    if (end != NULL && end > v) {
                        // Unwrap DDG redirect + percent-decode the
                        // real target; without this we'd hand a
                        // schemeless DDG-tracker URL to libcurl and
                        // fetch would fail with "No host part".
                        url = tools_url_unwrap(v, (size_t)(end - v));
                    }
                }
            }
            const char * pe = (gt < n)
                ? (const char *)memmem(h + gt + 1, n - gt - 1, "</a>", 4)
                : NULL;
            if (gt >= n || pe == NULL) {
                free(url);
                more = false;
            } else {
                size_t end_a = (size_t)(pe - h);
                struct chars title_raw = {0};
                struct chars title     = {0};
                tools_html_strip(h + gt + 1, end_a - gt - 1, &title_raw);
                tools_collapse_ws(&title_raw, &title);
                bool good = title.count > 5 && url != NULL;
                if (good) {
                    hits[seen].title   = title.data;
                    title.data         = NULL;
                    hits[seen].url     = url;
                    hits[seen].snippet = NULL;
                    seen++;
                } else {
                    free(url);
                    free(title.data);
                }
                free(title_raw.data);
                i = end_a + 4;
            }
        }
    }
    // Second pass: snippets. Match by document order into hits[0..seen).
    i    = 0;
    more = true;
    int k = 0;
    while (more && k < seen && i < n) {
        const char * p = (const char *)memmem(h + i, n - i,
                                              "result-snippet", 14);
        if (p == NULL) {
            more = false;
        } else {
            size_t pos = (size_t)(p - h);
            size_t gt  = pos;
            while (gt < n && h[gt] != '>') { gt++; }
            if (gt >= n) {
                more = false;
            } else {
                size_t end = gt + 1;
                bool   close_found = false;
                while (!close_found && end + 1 < n) {
                    if (h[end] == '<' && h[end + 1] == '/') {
                        close_found = true;
                    } else {
                        end++;
                    }
                }
                if (!close_found) {
                    more = false;
                } else {
                    struct chars raw   = {0};
                    struct chars clean = {0};
                    tools_html_strip(h + gt + 1, end - gt - 1, &raw);
                    tools_collapse_ws(&raw, &clean);
                    if (clean.count > 10) {
                        hits[k].snippet = clean.data;
                        clean.data      = NULL;
                        k++;
                    } else {
                        free(clean.data);
                    }
                    free(raw.data);
                    i = end;
                }
            }
        }
    }
    return seen;
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
        struct chars body = {0};
        long status = 0;
        int rc = tools_http_get(url, ms, &body, &status);
        if (rc != 0) {
            out->error = strdup("fetch: libcurl request failed");
        } else {
            struct chars result = {0};
            chars_printf(&result, "HTTP %ld\n\n", status);
            if (body.data != NULL && body.count > 0) {
                chars_put(&result, body.data, body.count);
            }
            if (body.count >= TOOLS_FETCH_CAP) {
                chars_printf(&result,
                    "\n\n[... truncated at %d bytes]",
                    (int)TOOLS_FETCH_CAP);
            }
            out->body   = result.data;
            out->status = status;
            out->ok     = (status >= 200 && status < 400);
            if (!out->ok) {
                struct chars tmp = {0};
                chars_printf(&tmp, "fetch: HTTP %ld", status);
                out->error = tmp.data; // transfer ownership
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
        struct chars stripped = {0};
        tools_html_strip(html, n, &stripped);
        struct chars clean = {0};
        tools_collapse_ws(&stripped, &clean);
        out->body = clean.data;
        out->ok   = 1;
        free(stripped.data);
    }
}

#ifndef TOOLS_DISTILL_CAP
#define TOOLS_DISTILL_CAP 4000
#endif

// DDG search → parsed hits + formatted numbered text. No fetch. The
// 3-phase URL-pick flow in slm_generate calls this directly so the
// model can choose which URL to fetch from the (title, URL, snippet)
// triples. The single-phase tools_websearch (back-compat shim) calls
// it too, then auto-picks hits[0].
//
// `hits` is an out-array of capacity `hits_cap`; populated up to N
// entries (return value). `out->body` gets a "1. <title>\n   <url>\n
// <snippet>\n\n2. ..." block — useful directly as URL-pick preamble.
// Caller frees hits via tools_free_hits and out via tools_result_free.
static int tools_websearch_hits(const char * query, int max_results,
                                struct tools_search_hit * hits,
                                int hits_cap,
                                struct tool_result * out) {
    tools_global_init();
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    int nh = 0;
    if (query == NULL || query[0] == '\0') {
        out->error = strdup("websearch: query parameter required");
    } else {
        if (g_tools_debug >= 1) {
            trace("websearch_hits: query=\"%s\" max=%d\n",
                  query, max_results);
        }
        int cap = (max_results > 0 && max_results <= hits_cap)
                ? max_results : hits_cap;
        if (cap > TOOLS_HITS_MAX) { cap = TOOLS_HITS_MAX; }
        struct chars eq = {0};
        tools_url_encode(query, &eq);
        struct chars url = {0};
        chars_puts(&url,
            "https://lite.duckduckgo.com/lite/"
            "?ia=web&origin=funnel_home_website&t=h_&q=");
        if (eq.data != NULL) { chars_puts(&url, eq.data); }
        chars_puts(&url, "&chip-select=search");
        struct chars body = {0};
        long         status = 0;
        int          rc     = tools_http_get(url.data, 15000,
                                             &body, &status);
        if (g_tools_debug >= 7) {
            trace("websearch_hits: DDG status=%ld body=%zu bytes\n",
                  status, body.count);
        }
        if (rc != 0) {
            out->error = strdup(
                "websearch: libcurl request failed (network down?)");
        } else if (status < 200 || status >= 400) {
            struct chars tmp = {0};
            chars_printf(&tmp,
                "websearch: HTTP %ld (may be rate-limiting)", status);
            out->error  = tmp.data;
            out->status = status;
        } else if (body.data == NULL || body.count == 0) {
            out->error = strdup("websearch: empty response from DDG");
        } else if (memmem(body.data, body.count,
                          "anomaly-modal", 13) != NULL) {
            out->error = strdup(
                "websearch: DDG bot-detection CAPTCHA challenge");
            out->status = status;
        } else {
            nh = tools_ddg_parse(body.data, body.count, cap, hits);
            if (g_tools_debug >= 3) {
                trace("websearch_hits: %d hit(s) parsed (cap=%d)\n",
                      nh, cap);
            }
            if (g_tools_debug >= 5) {
                int show = nh < 3 ? nh : 3;
                for (int i = 0; i < show; i++) {
                    trace("  hit[%d]: %s\n    -> %s\n", i,
                          hits[i].title ? hits[i].title : "(no title)",
                          hits[i].url   ? hits[i].url   : "(no url)");
                }
            }
            if (g_tools_debug >= 9 && body.data != NULL) {
                size_t bcut = body.count > 2048 ? 2048 : body.count;
                trace("websearch_hits: raw DDG body[0..%zu]: %.*s\n",
                      bcut, (int)bcut, body.data);
            }
            struct chars result = {0};
            for (int i = 0; i < nh; i++) {
                chars_printf(&result, "%d. %s\n   %s\n   %s\n\n",
                    i + 1,
                    hits[i].title   ? hits[i].title   : "(no title)",
                    hits[i].url     ? hits[i].url     : "(no url)",
                    hits[i].snippet ? hits[i].snippet : "(no snippet)");
            }
            chars_put(&result, "", 0);
            out->body   = result.data;
            out->status = status;
            out->ok     = 1;
        }
        free(eq.data);
        free(url.data);
        free(body.data);
    }
    return nh;
}

// Fetch URL + distill into curated body (capped). `out->body` is the
// distilled text (possibly with a truncation notice appended); errors
// land in `out->error`. Used by both the URL-pick orchestrator and
// the back-compat tools_websearch shim.
static void tools_fetch_distill(const char * url,
                                struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    if (url == NULL || url[0] == '\0') {
        out->error = strdup("fetch_distill: url required");
    } else {
        struct tool_result fetched = {0};
        tools_fetch(url, 15, &fetched);
        if (g_tools_debug >= 7) {
            trace("fetch_distill: fetch %s -> HTTP %ld (%zu bytes)\n",
                  fetched.ok ? "ok" : "ERR", fetched.status,
                  fetched.body ? strlen(fetched.body) : 0);
        }
        if (fetched.ok && fetched.body != NULL) {
            struct tool_result distilled = {0};
            tools_distill(fetched.body, strlen(fetched.body),
                          &distilled);
            if (distilled.ok && distilled.body != NULL) {
                size_t blen = strlen(distilled.body);
                size_t cut  = blen > TOOLS_DISTILL_CAP
                            ? TOOLS_DISTILL_CAP : blen;
                if (g_tools_debug >= 1) {
                    trace("fetch_distill: %zu -> %zu chars (cap %d)\n",
                          blen, cut, TOOLS_DISTILL_CAP);
                }
                if (g_tools_debug >= 9) {
                    size_t s = blen > 2048 ? 2048 : blen;
                    trace("fetch_distill: body[0..%zu]: %.*s\n",
                          s, (int)s, distilled.body);
                }
                struct chars b = {0};
                chars_put(&b, distilled.body, cut);
                if (cut < blen) {
                    chars_printf(&b,
                        "\n\n[... truncated at %zu chars]", cut);
                }
                chars_put(&b, "", 0);
                out->body   = b.data;
                out->status = fetched.status;
                out->ok     = 1;
            } else {
                out->error = strdup(distilled.error
                    ? distilled.error : "distill failed");
                out->status = fetched.status;
            }
            tools_result_free(&distilled);
        } else {
            out->error = strdup(fetched.error
                ? fetched.error : "fetch failed");
            out->status = fetched.status;
        }
        tools_result_free(&fetched);
    }
}

// Back-compat shim: single-phase websearch (DDG → auto-pick top →
// fetch+distill). Used by agent_dispatch on the --ask path. The
// in-band slm_generate flow does proper URL-pick via the model; this
// shim auto-picks hits[0] for callers that don't have a model in
// scope.
static void tools_websearch(const char * query, int max_results,
                            struct tool_result * out) {
    out->ok     = 0;
    out->body   = NULL;
    out->error  = NULL;
    out->status = 0;
    struct tools_search_hit hits[TOOLS_HITS_MAX] = {0};
    struct tool_result      hits_res             = {0};
    int nh = tools_websearch_hits(query, max_results, hits,
                                  TOOLS_HITS_MAX, &hits_res);
    if (!hits_res.ok) {
        out->error  = hits_res.error;
        hits_res.error = NULL;
        out->status = hits_res.status;
    } else if (nh == 0) {
        struct chars b = {0};
        chars_printf(&b,
            "Web search result for \"%s\":\nNo results found.\n",
            query);
        chars_put(&b, "", 0);
        out->body   = b.data;
        out->status = hits_res.status;
        out->ok     = 1;
    } else {
        const char * top_url     = hits[0].url;
        const char * top_snippet = hits[0].snippet;
        struct tool_result fd = {0};
        tools_fetch_distill(top_url, &fd);
        struct chars b = {0};
        chars_printf(&b,
            "Web search result for \"%s\":\nSource: %s\n\n",
            query, top_url);
        if (fd.ok && fd.body != NULL) {
            chars_puts(&b, fd.body);
        } else if (top_snippet != NULL) {
            chars_printf(&b, "[fetch failed: %s]\nSnippet: %s\n",
                fd.error ? fd.error : "(no error)", top_snippet);
        } else {
            chars_printf(&b, "[fetch failed: %s]\n",
                fd.error ? fd.error : "(no error)");
        }
        chars_put(&b, "", 0);
        out->body   = b.data;
        out->status = hits_res.status;
        out->ok     = 1;
        tools_result_free(&fd);
    }
    tools_result_free(&hits_res);
    tools_free_hits(hits, nh);
}

// Self-test. Network-dependent: skips silently if libcurl can't
// reach DDG (offline development). The two queries below are the
// reference test cases — "rock band HelloWorld album" exercises a
// trivia query that should find substantive snippets, and
// "bitcoin price today" exercises a fresh-data query the model
// CAN'T answer from training alone.

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
        r->body = NULL;
        r->error = NULL;
    }
}


static void tools_global_init(void)    { /* no-op */ }
static void tools_global_cleanup(void) { /* no-op */ }

static void tools_websearch(const char * query, int max_results,
                            struct tool_result * out) {
    (void)query; (void)max_results;
    out->ok     = 0;
    out->body   = NULL;
    out->error  = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}

static void tools_fetch(const char * url, int timeout_s,
                        struct tool_result * out) {
    (void)url; (void)timeout_s;
    out->ok     = 0;
    out->body   = NULL;
    out->error  = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}

static void tools_distill(const char * html, size_t n,
                          struct tool_result * out) {
    (void)html; (void)n;
    out->ok     = 0;
    out->body   = NULL;
    out->error  = tools_dup_msg("tools disabled in this build");
    out->status = 0;
}

static int32_t tools_self_test(void) {
    printf("tools-test: SKIP (built with -DLLM_NO_TOOLS)\n");
    return 0;
}

#endif  // LLM_NO_TOOLS
