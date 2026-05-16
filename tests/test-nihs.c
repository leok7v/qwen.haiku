// SPDX-License-Identifier: Apache-2.0
//
// test-nihs.c -- Needle-in-a-Haystack regression test.
//
// Hides a known sentinel inside a long passage at a fractional
// position, asks the model to recall the sentinel, scores binary
// PASS/FAIL by substring match in the reply. Sweeps (context length ×
// needle position) cells and prints a recall matrix.
//
// Build: `make -C slm nihs-test` -> Build/cli/nihs-test
// Run:   QWEN_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf ./Build/cli/nihs-test
//
// Single-TU shape: this file #includes slm/slm.c without LLM_CLI, so
// slm.c's CLI main() is excluded and our main() drives the test.
// All the slm_* symbols (file-scope statics in the slm.c TU) become
// file-scope statics in THIS TU, callable directly. Same pattern as
// qwen.c's -DQWEN_TESTS standalone build.

// We do NOT define LLM_CLI: slm.c's `main()` is gated by it, and we
// want our own main() here.
#include "../slm/slm.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NIHS_SENTINEL_KEY "BANANA42"
#define NIHS_SENTINEL_PHRASE \
    "Important note: the secret password is " NIHS_SENTINEL_KEY \
    ". Remember it. "

// Haystack chunks — neutral prose, ~50-60 tokens each by inspection.
// Rotated through `n_repeat` times to build the document body. Picked
// for topical diversity (history / science / mundane facts) so the
// model can't pattern-match the boilerplate as a single repeating
// motif.
static const char * NIHS_CHUNKS[] = {
    "Paper was invented in ancient China during the Han Dynasty,"
    " with the earliest fragments dating to the second century BCE."
    " Cai Lun is traditionally credited with refining pulp-based"
    " paper around 105 CE. The technique spread along the Silk Road"
    " to the Islamic world and reached Europe by the twelfth century. ",
    "Wind energy has been harvested by humans for over five thousand"
    " years. Sailing vessels carried trade across the Mediterranean"
    " and the Indian Ocean long before recorded history. Later"
    " windmills ground grain and pumped water across Persia, Spain,"
    " the Low Countries, and eventually the American Great Plains. ",
    "Bread is among humanity's oldest prepared foods. Evidence of"
    " grinding cereals and baking dough has been found at sites at"
    " least thirty thousand years old. Ancient Egyptians refined"
    " leavened bread around three thousand BCE, and the technique"
    " then spread across the Mediterranean basin and into Europe. ",
    "The bicycle as we know it took shape in the nineteenth century."
    " The first practical pedal bicycle appeared in France in 1864."
    " The safety bicycle, with two wheels of equal size and a chain"
    " drive, debuted in 1885 and rapidly displaced earlier high-wheel"
    " designs because it was far easier to mount and steer. ",
    "Coffee originates from the Ethiopian highlands, where wild"
    " Coffea arabica still grows. Cultivation spread to Yemen by the"
    " fifteenth century and from there to the rest of the Arab"
    " world. The first European coffeehouses opened in Venice and"
    " Oxford in the seventeenth century, sparking a literary salon"
    " culture. ",
    NULL,
};

static int nihs_chunk_count(void) {
    int n = 0;
    while (NIHS_CHUNKS[n] != NULL) { n++; }
    return n;
}

// Build a haystack of `n_repeat` chunks (cycling through NIHS_CHUNKS),
// with the sentinel paragraph inserted at fractional position
// `pos_frac` (0.0..1.0). Returns a heap-allocated NUL-terminated
// string (caller frees).
static char * nihs_build_haystack(int n_repeat, double pos_frac) {
    struct chars b = {0};
    int n_chunks = nihs_chunk_count();
    int needle_at = (int)floor(pos_frac * (double)n_repeat);
    if (needle_at < 0)         { needle_at = 0; }
    if (needle_at >= n_repeat) { needle_at = n_repeat - 1; }
    for (int i = 0; i < n_repeat; i++) {
        if (i == needle_at) {
            chars_puts(&b, NIHS_SENTINEL_PHRASE);
        }
        chars_puts(&b, NIHS_CHUNKS[i % n_chunks]);
    }
    // Edge case: needle at very end (pos_frac=1.0 with needle_at =
    // n_repeat-1) — placed before the LAST chunk. To support an
    // actually-last-position case, append a sentinel after all chunks
    // when pos_frac >= 0.99.
    if (pos_frac >= 0.99) {
        chars_puts(&b, NIHS_SENTINEL_PHRASE);
    }
    return b.data;
}

// Streaming sink: captures content chunks into a chars buffer.
struct nihs_capture_box {
    struct slm_stream_callback base;
    struct chars *             out;
};

static int nihs_capture_cb_fn(const struct slm_stream_callback * cb) {
    struct nihs_capture_box * box = (struct nihs_capture_box *)cb;
    if (cb->content != NULL && box->out != NULL) {
        chars_put(box->out, cb->content, strlen(cb->content));
    }
    return 0;
}

// QWEN_GGUF resolution. LLM_CLI's slm_cli_gguf_path() is hidden from
// this TU; replicate the env-var lookup with no hardcoded default
// (forces explicit QWEN_GGUF in CI / on dev boxes).
static const char * nihs_gguf_path(void) {
    const char * env = getenv("QWEN_GGUF");
    return (env != NULL && env[0] != '\0') ? env : NULL;
}

struct nihs_cell_result {
    int   found;          // 1 = sentinel echoed
    int   ctx_tokens;     // tokens prefilled (haystack + framing)
    int   reply_tokens;   // tokens generated
    char  reply[256];     // first 255 bytes of model's reply (debug)
};

// Run one cell: build haystack, prefill, ask, score.
static void nihs_run_cell(struct slm_model * m,
                          int n_repeat, double pos_frac,
                          struct nihs_cell_result * out) {
    memset(out, 0, sizeof(*out));
    char * haystack = nihs_build_haystack(n_repeat, pos_frac);
    struct chars prompt = {0};
    chars_puts(&prompt,
        "Read the following document carefully and remember its"
        " contents:\n\n");
    chars_puts(&prompt, haystack);
    chars_puts(&prompt,
        "\n\nQuestion: What is the secret password mentioned"
        " somewhere in the document above? Answer with just the"
        " password value, nothing else.");
    struct slm_ctx * ctx = slm_ctx_create(m);
    if (ctx != NULL) {
        slm_ctx_ctrl(ctx)->tools = false;
        slm_ctx_ctrl(ctx)->think = false;
        slm_ctx_system_prompt(ctx, "You are a helpful assistant.",
                              NULL);
        struct slm_sampler sp = slm_sampler_defaults();
        sp.temperature        = 0.0f;    // greedy → deterministic
        sp.repetition_penalty = 1.0f;    // don't deflect echoing
        struct chars reply = {0};
        struct nihs_capture_box box = {0};
        box.base.callback = nihs_capture_cb_fn;
        box.out           = &reply;
        slm_generate(ctx, prompt.data, /*max_new=*/48,
                     /*min_new=*/0, &sp, /*seed=*/1, &box.base);
        chars_put(&reply, "", 0);
        out->ctx_tokens   = (int)ctx->pos;
        out->reply_tokens = (int)ctx->n_generated;
        if (reply.data != NULL) {
            if (strstr(reply.data, NIHS_SENTINEL_KEY) != NULL) {
                out->found = 1;
            }
            int n = (int)reply.count;
            if (n >= (int)sizeof(out->reply)) {
                n = (int)sizeof(out->reply) - 1;
            }
            memcpy(out->reply, reply.data, (size_t)n);
            out->reply[n] = '\0';
            // Strip any trailing newlines for a one-line debug print.
            while (n > 0 && (out->reply[n - 1] == '\n' ||
                             out->reply[n - 1] == '\r')) {
                n--;
                out->reply[n] = '\0';
            }
        }
        chars_free(&reply);
        slm_ctx_destroy(ctx);
    }
    chars_free(&prompt);
    free(haystack);
}

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    const char * path = nihs_gguf_path();
    if (path == NULL) {
        fprintf(stderr,
            "nihs-test: QWEN_GGUF env var required (path to GGUF)\n");
        return 1;
    }
    struct slm_model * m = slm_model_load(path);
    if (!slm_model_loaded(m)) {
        fprintf(stderr, "nihs-test: load failed: %s\n",
                slm_model_error(m));
        slm_model_unload(m);
        return 1;
    }
    // Sweep: 4 haystack sizes × 5 needle positions = 20 cells. Sizes
    // chosen so the LARGEST haystack (~1600 tokens of chunks + ~50
    // tokens of framing + ~48 tokens of reply) stays well under the
    // 4096-token KV cap (qwen.c slm_load_config).
    static const int    REPEATS[]   = { 1, 4, 12, 32 };
    static const double POSITIONS[] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    const int n_r = (int)(sizeof(REPEATS)   / sizeof(REPEATS[0]));
    const int n_p = (int)(sizeof(POSITIONS) / sizeof(POSITIONS[0]));
    printf("nihs-test: sentinel=\"%s\", %d cells\n",
           NIHS_SENTINEL_KEY, n_r * n_p);
    printf("dispatch: %s\n\n", simd_dispatch_label());
    // Matrix header.
    printf("repeats       ");
    for (int p = 0; p < n_p; p++) {
        printf(" pos=%.2f ", POSITIONS[p]);
    }
    printf("\n");
    int total_pass = 0;
    int total_run  = 0;
    struct nihs_cell_result fails[20];
    int fails_repeats[20];
    double fails_positions[20];
    int n_fails = 0;
    for (int r = 0; r < n_r; r++) {
        printf("repeat=%-3d   ", REPEATS[r]);
        fflush(stdout);
        for (int p = 0; p < n_p; p++) {
            struct nihs_cell_result res;
            nihs_run_cell(m, REPEATS[r], POSITIONS[p], &res);
            printf("  %4s(%4d)",
                   res.found ? "PASS" : "FAIL", res.ctx_tokens);
            fflush(stdout);
            total_run++;
            if (res.found) {
                total_pass++;
            } else if (n_fails < (int)(sizeof(fails) / sizeof(fails[0]))) {
                fails[n_fails]           = res;
                fails_repeats[n_fails]   = REPEATS[r];
                fails_positions[n_fails] = POSITIONS[p];
                n_fails++;
            }
        }
        printf("\n");
    }
    printf("\nnihs-test: %d/%d cells PASS (%.0f%%)\n",
           total_pass, total_run,
           100.0 * (double)total_pass / (double)total_run);
    for (int i = 0; i < n_fails; i++) {
        printf("  FAIL repeat=%d pos=%.2f tokens=%d reply=%s\n",
               fails_repeats[i], fails_positions[i],
               fails[i].ctx_tokens, fails[i].reply);
    }
    slm_model_unload(m);
    return (total_pass == total_run) ? 0 : 1;
}
