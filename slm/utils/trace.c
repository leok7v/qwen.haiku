// SPDX-License-Identifier: Apache-2.0
//
// trace.c — printf-style trace + 1024-entry process-wide ring buffer
// + single-observer functor, in ONE file (stb-style).
//
// Two usage modes:
//
//   1. Implementation owner (exactly one TU):
//          #define TRACE_IMPLEMENTATION
//          #include "utils/trace.c"
//      This TU gets the ring storage, the atomic head, and the
//      observer slot. In this codebase that TU is model.c.
//
//   2. Consumer (any other TU, including Swift via bridge.h):
//          #include "utils/trace.c"
//      Sees only the declarations + the trace() macro; the linker
//      resolves the extern functions to the implementation TU.
//
// Single-writer model (the slm runtime + CLI are single-threaded
// per ctx). The ring uses an atomic head with release/acquire
// ordering so an observer on another thread sees the entry contents
// before the head's new value. No mutexes; no wait-state sync.
//
// Usage:
//   trace("loaded %zu tensors in %.3fs", n, dt);
//
// Swift / external readers:
//   struct trace_observer obs = { .that = ctx,
//                                 .on_trace = my_callback };
//   trace_subscribe(&obs);
//   // ... events flow into my_callback as they happen ...
//   trace_subscribe(NULL);  // detach

// Two guards: TRACE_C wraps only the declarations (so they include
// safely once per TU); the implementation block below is gated on
// TRACE_IMPLEMENTATION alone, so a consumer that #include's the file
// for declarations followed by the owning TU #include with
// TRACE_IMPLEMENTATION still compiles the implementation.
#ifndef TRACE_C
#define TRACE_C

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================
// Public surface — always visible.
// =============================================================

#define TRACE_RING_CAPACITY 1024     // power of two; head % CAPACITY
#define TRACE_MESSAGE_MAX   232      // sizeof(struct trace_entry) ~ 256

// One ring slot. `file` / `function` point to literal __FILE__ and
// __func__ strings owned by the program's read-only segment, so
// they're stable without copying. `message` is a vsnprintf'd payload.
struct trace_entry {
    double       timestamp;          // seconds since first trace() call
    const char * file;               // basename (after last '/')
    const char * function;           // __func__ at the call site
    int32_t      line;
    char         message[TRACE_MESSAGE_MAX];
};

// Observer functor. on_trace fires once per trace() invocation, from
// the thread that called trace(). Keep the body short; bounce to a
// queue for any real work (SwiftUI MainActor dispatch etc.).
struct trace_observer {
    void *       that;
    void       (*on_trace)(const struct trace_observer * o,
                           const struct trace_entry *    entry);
};

// Implementation entry; use the `trace(...)` macro at call sites so
// __FILE__ / __LINE__ / __func__ get captured.
void _trace_(const char * filename, int32_t line, const char * func,
             const char * format, ...)
     __attribute__((format(printf, 4, 5)));

#define trace(fmt, ...) \
    _trace_(__FILE__, __LINE__, __func__, (fmt), ##__VA_ARGS__)

// Subscribe / replace / detach the single observer. Pass NULL to
// detach. The observer slot is updated atomically; in-flight trace()
// calls that loaded the previous observer pointer may fire the
// previous callback once more.
void trace_subscribe(const struct trace_observer * observer);

// Total trace events ever written. Increasing monotonically; wraps
// in the ring at (head - 1) % TRACE_RING_CAPACITY.
uint64_t trace_head(void);

// Look up an entry by logical index (0 <= index < trace_head()).
// NULL if the index has been overwritten by ring wrap-around
// (index < trace_head() - TRACE_RING_CAPACITY) or is in the future.
// Pointer stays valid until the ring wraps past `index` again —
// copy out anything you want to keep.
const struct trace_entry * trace_at(uint64_t index);

#ifdef __cplusplus
}
#endif

#endif // TRACE_C  (end of declaration guard)


// =============================================================
// Implementation — only compiled in the TU that defines
// TRACE_IMPLEMENTATION (model.c in this codebase). This block is
// OUTSIDE the TRACE_C guard so a previous decl-only include can't
// hide the impl from the owning TU.
// =============================================================

#ifdef TRACE_IMPLEMENTATION

#include <locale.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Wall-clock helper. File-static; slm.c has its own seconds() for
// generate-loop timing and they don't share. Keep both file-static
// so a future module move doesn't tangle linkage.
static double trace_seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Seconds elapsed since the FIRST trace() call. Stable absolute
// reference for the log; readable in the printed prefix.
static double trace_since_start(void) {
    static double start_time = -1.0;
    double now = trace_seconds_now();
    if (start_time < 0.0) { start_time = now; }
    return now - start_time;
}

// `%'d` thousand separators readable for big numbers. Idempotent;
// fires once per process.
static void trace_set_numeric_locale(void) {
    static bool done = false;
    if (!done) {
        setlocale(LC_NUMERIC, "");
        setlocale(LC_NUMERIC, "en_US.UTF-8");
        done = true;
    }
}

// Ring storage. `g_trace_head` is the count of trace events EVER
// written; the slot index is `(g_trace_head - 1) % CAPACITY` for the
// latest entry. Single-writer (the slm runtime is single-threaded
// per ctx) so we don't need a CAS loop — a plain relaxed-load +
// release-store keeps the head monotonic and makes the entry
// contents visible to readers on other threads before the head's
// new value.
static struct trace_entry g_trace_ring[TRACE_RING_CAPACITY];
static _Atomic uint64_t   g_trace_head = 0;

// Observer functor. Loaded atomically per trace() call so subscribers
// that detach mid-flight don't need any external sync. There's one
// slot — replacing while a trace is in flight may fire the old
// callback once more (benign).
static _Atomic(struct trace_observer *) g_trace_observer = NULL;

// Storage backing the atomic pointer above. trace_subscribe copies
// the supplied observer struct here, then publishes the pointer.
static struct trace_observer g_trace_observer_slot;

void _trace_(const char * filename, int32_t line, const char * func,
             const char * format, ...) {
    trace_set_numeric_locale();
    const char * file = filename;
    const char * slash = strrchr(file, '/');
    if (slash != NULL) { file = slash + 1; }

    uint64_t idx = atomic_load_explicit(&g_trace_head,
                                        memory_order_relaxed);
    struct trace_entry * e = &g_trace_ring[idx % TRACE_RING_CAPACITY];
    e->timestamp = trace_since_start();
    e->file      = file;
    e->function  = func;
    e->line      = line;

    va_list ap;
    va_start(ap, format);
    int n = vsnprintf(e->message, sizeof(e->message), format, ap);
    va_end(ap);
    if (n < 0) { e->message[0] = '\0'; n = 0; }

    // Mirror to stderr so the CLI experience stays unchanged.
    fprintf(stderr, "%10.6f %s:%d @%s %s",
            e->timestamp, file, line, func, e->message);
    if (n == 0 || e->message[n - 1] != '\n') {
        fputc('\n', stderr);
    }

    // Publish the entry: release-store the new head so readers on
    // other threads see the entry contents before the head update.
    atomic_store_explicit(&g_trace_head, idx + 1,
                          memory_order_release);

    // Notify the observer, if any.
    struct trace_observer * obs =
        atomic_load_explicit(&g_trace_observer, memory_order_acquire);
    if (obs != NULL && obs->on_trace != NULL) {
        obs->on_trace(obs, e);
    }
}

void trace_subscribe(const struct trace_observer * observer) {
    if (observer == NULL) {
        atomic_store_explicit(&g_trace_observer, NULL,
                              memory_order_release);
    } else {
        g_trace_observer_slot = *observer;
        atomic_store_explicit(&g_trace_observer,
                              &g_trace_observer_slot,
                              memory_order_release);
    }
}

uint64_t trace_head(void) {
    return atomic_load_explicit(&g_trace_head,
                                memory_order_acquire);
}

const struct trace_entry * trace_at(uint64_t index) {
    uint64_t head = atomic_load_explicit(&g_trace_head,
                                         memory_order_acquire);
    const struct trace_entry * r = NULL;
    if (index < head) {
        bool wrapped = head > TRACE_RING_CAPACITY &&
                       index < head - TRACE_RING_CAPACITY;
        if (!wrapped) {
            r = &g_trace_ring[index % TRACE_RING_CAPACITY];
        }
    }
    return r;
}

#endif // TRACE_IMPLEMENTATION
