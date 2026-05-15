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

#include "tensor.c"
#include "llm.h"
#include "jinja-template.c"

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

static double llm_monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// CLI-only fallback path. Library callers (Swift, Obj-C, etc.) pass
// their own path to llm_create(). The CLI reads QWEN_GGUF env var
// first and falls back to this constant if unset. The default points
// at the macOS sandbox cache the app uses, so a one-off CLI run on
// the dev machine works without exporting the env var. Override with
// QWEN_GGUF when running elsewhere.
#ifdef LLM_CLI
static const char * LLM_GGUF_PATH_DEFAULT =
    "/Users/leo/Library/Containers/io.github.leok7v.QwenHaiku/Data/"
    "Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf";
#endif

// ---------------------------------------------------------------------------
// chars / arr / map primitives - small dependency-free utilities.
// ---------------------------------------------------------------------------

struct arr {
    void * data;
    size_t count;
    size_t capacity;
};

static inline void * llm_oom(void * a) {
    if (a == NULL) {
        fprintf(stderr, "llm: OOM\n");
        abort();
    }
    return a;
}

static inline void arr_grow(struct arr * a, size_t esize, size_t need) {
    if (a->data == NULL) {
        a->capacity = need;
        a->data = llm_oom(malloc(need * esize));
    } else if (need > a->capacity) {
        a->capacity = need * 2;
        a->data = llm_oom(realloc(a->data, a->capacity * esize));
    }
}

struct chars {
    char * data;
    size_t count;
    size_t capacity;
};

static inline void chars_grow(struct chars * s, size_t need) {
    arr_grow((struct arr *)s, 1, need);
}

static inline void chars_put(struct chars * s, const char * d, size_t n) {
    chars_grow(s, s->count + n + 1);
    memcpy(s->data + s->count, d, n);
    s->count += n;
    s->data[s->count] = '\0';
}

static inline void chars_free(struct chars * s) {
    free(s->data);
    s->data = NULL;
    s->count = 0;
    s->capacity = 0;
}

// String hashmap (chars -> int32). Simple open-addressing.
struct s2i_entry {
    struct chars key;
    int32_t      val;
    uint8_t      live;
};

struct s2i {
    struct s2i_entry * slots;
    size_t             count;
    size_t             capacity;
};

static uint64_t s2i_hash(const char * d, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)d[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void s2i_grow(struct s2i * m, size_t new_cap) {
    struct s2i_entry * ns =
        (struct s2i_entry *)llm_oom(calloc(new_cap,
                                           sizeof(struct s2i_entry)));
    size_t mask = new_cap - 1;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].live) {
            uint64_t h = s2i_hash(m->slots[i].key.data,
                                  m->slots[i].key.count);
            size_t j = (size_t)h & mask;
            while (ns[j].live) { j = (j + 1) & mask; }
            ns[j] = m->slots[i];
        }
    }
    free(m->slots);
    m->slots = ns;
    m->capacity = new_cap;
}

static void s2i_put(struct s2i * m, const char * k, size_t kn, int32_t v) {
    if (m->capacity == 0) {
        s2i_grow(m, 1024);
    } else if ((m->count + 1) * 4 > m->capacity * 3) {
        s2i_grow(m, m->capacity * 2);
    }
    size_t mask = m->capacity - 1;
    size_t j    = (size_t)s2i_hash(k, kn) & mask;
    int done = 0;
    while (!done) {
        if (!m->slots[j].live) {
            struct chars c = {0};
            chars_put(&c, k, kn);
            m->slots[j].key  = c;
            m->slots[j].val  = v;
            m->slots[j].live = 1;
            m->count++;
            done = 1;
        } else if (m->slots[j].key.count == kn
                && memcmp(m->slots[j].key.data, k, kn) == 0) {
            m->slots[j].val = v;
            done = 1;
        } else {
            j = (j + 1) & mask;
        }
    }
}

static int32_t s2i_get(const struct s2i * m, const char * k, size_t kn,
                       int32_t miss) {
    int32_t r = miss;
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j    = (size_t)s2i_hash(k, kn) & mask;
        int done = 0;
        while (!done) {
            if (!m->slots[j].live) {
                done = 1;
            } else if (m->slots[j].key.count == kn
                    && memcmp(m->slots[j].key.data, k, kn) == 0) {
                r = m->slots[j].val;
                done = 1;
            } else {
                j = (j + 1) & mask;
            }
        }
    }
    return r;
}

static void s2i_free(struct s2i * m) {
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].live) { chars_free(&m->slots[i].key); }
    }
    free(m->slots);
    m->slots = NULL;
    m->count = 0;
    m->capacity = 0;
}

// ---------------------------------------------------------------------------
// GGUF v3 reader (subset: covers Qwen3 GGUFs from llama.cpp)
// ---------------------------------------------------------------------------

#define GGUF_MAGIC 0x46554747u  // 'G''G''U''F' little-endian
#define GGUF_VERSION 3

enum gguf_value_type {
    GGUF_VT_U8    = 0,  GGUF_VT_I8    = 1,
    GGUF_VT_U16   = 2,  GGUF_VT_I16   = 3,
    GGUF_VT_U32   = 4,  GGUF_VT_I32   = 5,
    GGUF_VT_F32   = 6,  GGUF_VT_BOOL  = 7,
    GGUF_VT_STR   = 8,  GGUF_VT_ARRAY = 9,
    GGUF_VT_U64   = 10, GGUF_VT_I64   = 11,
    GGUF_VT_F64   = 12,
};

enum gguf_tensor_type {
    GGUF_TT_F32  = 0,
    GGUF_TT_F16  = 1,
    GGUF_TT_Q8_0 = 8,
    GGUF_TT_Q4_K = 12,
    GGUF_TT_Q5_K = 13,
    GGUF_TT_Q6_K = 14,
};

struct gguf_str { uint64_t n; const char * s; };

struct gguf_value {
    uint32_t        type;
    const uint8_t * raw;       // points into mmap; type-specific layout
    uint64_t        raw_bytes; // for arrays/strings, total bytes
    // for arrays:
    uint32_t        arr_type;
    uint64_t        arr_n;
    const uint8_t * arr_data;
};

struct gguf_kv {
    struct gguf_str   key;
    struct gguf_value v;
};

struct gguf_tensor {
    struct gguf_str  name;
    uint32_t         n_dims;
    uint64_t         shape[4];
    uint32_t         type;
    uint64_t         offset;     // from tensor data section start
    const uint8_t *  data;       // resolved at finalize
};

struct gguf {
    int           fd;
    const uint8_t * base;     // mmap base
    size_t        bytes;
    uint64_t      n_kv;
    uint64_t      n_tensors;
    struct gguf_kv     * kvs;
    struct gguf_tensor * tensors;
    const uint8_t * tensor_data; // base + (aligned) end-of-info-section
    uint32_t      alignment;     // from general.alignment, default 32
};

static int gguf_open(struct gguf * g, const char * path);
static void gguf_close(struct gguf * g);
static const struct gguf_kv * gguf_find_kv(const struct gguf * g, const char * k);
static const struct gguf_tensor * gguf_find_tensor(const struct gguf * g,
                                                   const char * name);

// All "read" helpers advance a cursor by reference. The mmap base is
// expected to be 8-byte aligned (mmap pages are 16 KB aligned on
// Apple Silicon), but values within the file aren't aligned in
// general - use memcpy to read.

static uint32_t gr_u32(const uint8_t * b, size_t * c) {
    uint32_t v; memcpy(&v, b + *c, 4); *c += 4; return v;
}

static uint64_t gr_u64(const uint8_t * b, size_t * c) { uint64_t v;
    memcpy(&v, b + *c, 8); *c += 8; return v;
}

static float gr_f32(const uint8_t * b, size_t * c) {
    float v; memcpy(&v, b + *c, 4); *c += 4; return v;
}

static struct gguf_str gr_str(const uint8_t * b, size_t * c) {
    struct gguf_str s;
    s.n = gr_u64(b, c);
    s.s = (const char *)(b + *c);
    *c += s.n;
    return s;
}

static size_t gguf_value_skip(const uint8_t * b, size_t c, uint32_t type) {
    size_t end = c;
    switch (type) {
        case GGUF_VT_U8:
        case GGUF_VT_I8:
        case GGUF_VT_BOOL:
            end += 1;
            break;
        case GGUF_VT_U16:
        case GGUF_VT_I16:
            end += 2;
            break;
        case GGUF_VT_U32:
        case GGUF_VT_I32:
        case GGUF_VT_F32:
            end += 4;
            break;
        case GGUF_VT_U64:
        case GGUF_VT_I64:
        case GGUF_VT_F64:
            end += 8;
            break;
        case GGUF_VT_STR: {
            uint64_t n = gr_u64(b, &end);
            end += n;
            break;
        }
        case GGUF_VT_ARRAY: {
            uint32_t at = gr_u32(b, &end);
            uint64_t an = gr_u64(b, &end);
            for (uint64_t i = 0; i < an; i++) {
                end = gguf_value_skip(b, end, at);
            }
            break;
        }
        default:
            fprintf(stderr, "gguf: unknown value type %u\n", type);
            abort();
    }
    return end;
}

static int gguf_open(struct gguf * g, const char * path) {
    memset(g, 0, sizeof(*g));
    g->alignment = 32;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gguf: open(%s): %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "gguf: fstat: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    void * m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
        fprintf(stderr, "gguf: mmap: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    g->fd    = fd;
    g->base  = (const uint8_t *)m;
    g->bytes = (size_t)st.st_size;
    size_t c = 0;
    uint32_t magic = gr_u32(g->base, &c);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic %08x\n", magic);
        gguf_close(g);
        return -1;
    }
    uint32_t ver = gr_u32(g->base, &c);
    if (ver != GGUF_VERSION) {
        fprintf(stderr, "gguf: version %u not supported (need %u)\n",
                ver, GGUF_VERSION);
        gguf_close(g);
        return -1;
    }
    g->n_tensors = gr_u64(g->base, &c);
    g->n_kv      = gr_u64(g->base, &c);
    const size_t kv_bytes = sizeof(struct gguf_kv);
    g->kvs = (struct gguf_kv *)llm_oom(calloc((size_t)g->n_kv, kv_bytes));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        struct gguf_kv * kv = &g->kvs[i];
        kv->key = gr_str(g->base, &c);
        kv->v.type = gr_u32(g->base, &c);
        kv->v.raw  = g->base + c;
        if (kv->v.type == GGUF_VT_ARRAY) {
            size_t after_type = c;
            kv->v.arr_type = gr_u32(g->base, &c);
            kv->v.arr_n    = gr_u64(g->base, &c);
            kv->v.arr_data = g->base + c;
            c = gguf_value_skip(g->base, after_type, GGUF_VT_ARRAY);
        } else {
            c = gguf_value_skip(g->base, c, kv->v.type);
        }
        kv->v.raw_bytes = (uint64_t)((g->base + c) - kv->v.raw);
    }
    // Pick up alignment override if present.
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const struct gguf_kv * kv = &g->kvs[i];
        if (kv->key.n == strlen("general.alignment")
            && memcmp(kv->key.s, "general.alignment", kv->key.n) == 0
            && kv->v.type == GGUF_VT_U32) {
            size_t cc = 0;
            g->alignment = gr_u32(kv->v.raw, &cc);
        }
    }
    const size_t t_bytes = sizeof(struct gguf_tensor);
    g->tensors = (struct gguf_tensor *)llm_oom(calloc((size_t)g->n_tensors,
                                                       t_bytes));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        struct gguf_tensor * t = &g->tensors[i];
        t->name   = gr_str(g->base, &c);
        t->n_dims = gr_u32(g->base, &c);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->shape[d] = gr_u64(g->base, &c);
        }
        t->type   = gr_u32(g->base, &c);
        t->offset = gr_u64(g->base, &c);
    }
    // Tensor data starts at first aligned offset past tensor info section.
    size_t info_end = c;
    size_t a = g->alignment;
    size_t data_start = (info_end + a - 1) & ~(a - 1);
    g->tensor_data = g->base + data_start;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        g->tensors[i].data = g->tensor_data + g->tensors[i].offset;
    }
    return 0;
}

static void gguf_close(struct gguf * g) {
    if (g->kvs) {
        free(g->kvs);
        g->kvs = NULL;
    }
    if (g->tensors) {
        free(g->tensors);
        g->tensors = NULL;
    }
    if (g->base) {
        munmap((void *)g->base, g->bytes);
        g->base = NULL;
    }
    if (g->fd > 0) {
        close(g->fd);
        g->fd = 0;
    }
}

static const struct gguf_kv * gguf_find_kv(const struct gguf * g, const char * k) {
    const struct gguf_kv * r = NULL;
    size_t n = strlen(k);
    for (uint64_t i = 0; i < g->n_kv && r == NULL; i++) {
        if (g->kvs[i].key.n == n
            && memcmp(g->kvs[i].key.s, k, n) == 0) {
            r = &g->kvs[i];
        }
    }
    return r;
}

static const struct gguf_tensor * gguf_find_tensor(const struct gguf * g,
                                                   const char * name) {
    const struct gguf_tensor * r = NULL;
    size_t n = strlen(name);
    for (uint64_t i = 0; i < g->n_tensors && r == NULL; i++) {
        if (g->tensors[i].name.n == n
            && memcmp(g->tensors[i].name.s, name, n) == 0) {
            r = &g->tensors[i];
        }
    }
    return r;
}

static uint32_t gguf_kv_u32(const struct gguf * g, const char * k, uint32_t dflt) {
    uint32_t r = dflt;
    const struct gguf_kv * kv = gguf_find_kv(g, k);
    if (kv && kv->v.type == GGUF_VT_U32) {
        size_t c = 0;
        r = gr_u32(kv->v.raw, &c);
    }
    return r;
}

static float gguf_kv_f32(const struct gguf * g, const char * k, float dflt) {
    float r = dflt;
    const struct gguf_kv * kv = gguf_find_kv(g, k);
    if (kv && kv->v.type == GGUF_VT_F32) {
        size_t c = 0;
        r = gr_f32(kv->v.raw, &c);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Model config
// ---------------------------------------------------------------------------

struct llm_config {
    int32_t  n_layers;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  hidden_dim;
    int32_t  ffn_dim;
    int32_t  vocab_size;
    int32_t  max_position;
    float    rope_theta;
    int32_t  rope_dim;              // see footnote (1)
    float    norm_eps;
    int32_t  bos_id;
    int32_t  eos_id;
    int32_t  eot_id;                // see footnote (4)
    int32_t  stop_ids[8];           // see footnote (5); -1 in unused slots
    int32_t  n_stop_ids;
    int32_t  attn_output_gate;      // see footnote (2)
    // qwen35 hybrid-only (Gated DeltaNet linear-attention block):
    int32_t  is_hybrid;
    int32_t  full_attn_interval;
    int32_t  linear_n_heads;        // num_k_heads = num_v_heads, =16
    int32_t  linear_k_head_dim;     // =128
    int32_t  linear_v_head_dim;     // =128
    int32_t  linear_conv_kernel;    // =4
    int32_t  rope_sections[4];      // see footnote (3)
};

// llm_config footnotes:
//
// (1) rope_dim: partial rotary - rotate only the first rope_dim
//     entries of each head's vector (64 of 256 for qwen35). 0 means
//     rotate all.
//
// (2) attn_output_gate: 1 if the Q projection includes the qwen35
//     output gate (Q proj emits n_heads*head_dim*2, split into Q
//     and a sigmoid gate); 0 for plain Qwen3.
//
// (3) rope_sections: mrope T / H / W / E sections. qwen35 ships
//     [11, 11, 10, 0]. All zeros falls back to standard NEOX RoPE.
//
// (4) eot_id: secondary stop token for chat turns. eos_id is
//     `<|endoftext|>` (end-of-document); chat turns end with
//     `<|im_end|>` and the model otherwise keeps generating into a
//     hallucinated next role. Looked up by string at load and
//     defaulted to -1 when absent.
//
// (5) stop_ids[]: full set of end-of-generation tokens that
//     llama.cpp recognizes for qwen35. Populated at tokenizer load
//     by string lookup. Mirror of llama.cpp's load-time print:
//     `<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`,
//     `<|file_sep|>`. Without these we run until max_new on any
//     completion that wants to emit `<|endoftext|>` (the model card
//     uses it as the document terminator and emits it readily under
//     non-chat prompts).

static int32_t llm_load_config(const struct gguf * g, struct llm_config * c) {
    memset(c, 0, sizeof(*c));
    // Detect arch + pick KV prefix.
    const struct gguf_kv * arch_kv = gguf_find_kv(g, "general.architecture");
    const char * prefix = "qwen3";
    if (arch_kv && arch_kv->v.type == GGUF_VT_STR) {
        size_t cc = 0;
        uint64_t n = gr_u64(arch_kv->v.raw, &cc);
        const char * s = (const char *)(arch_kv->v.raw + cc);
        if (n == 6 && memcmp(s, "qwen35", 6) == 0) {
            prefix = "qwen35";
            // qwen35 is a hybrid SSM+Transformer (Mamba-2/DeltaNet
            // style SSM blocks with full attention every
            // full_attention_interval-th layer). See PLAN.md for the
            // architectural notes; SSM block math is reverse-
            // engineered from tensor shapes/names and is marked
            // "PROBABLY" where uncertain.
            fprintf(stderr,
                "llm: qwen35 hybrid (Mamba-2/DeltaNet + attention)."
                " SSM math is best-guess.\n");
        } else if (n == 5 && memcmp(s, "qwen2", 5) == 0) {
            prefix = "qwen2";
        }
    }
    char key[128];
    #define KV_U32(k, dflt) \
        (snprintf(key, sizeof(key), "%s.%s", prefix, k), \
         (int32_t)gguf_kv_u32(g, key, (dflt)))
    #define KV_F32(k, dflt) \
        (snprintf(key, sizeof(key), "%s.%s", prefix, k), \
         gguf_kv_f32(g, key, (dflt)))
    c->hidden_dim   = KV_U32("embedding_length",                0);
    c->n_layers     = KV_U32("block_count",                     0);
    c->n_heads      = KV_U32("attention.head_count",            0);
    c->n_kv_heads   = KV_U32("attention.head_count_kv",   (uint32_t)c->n_heads);
    c->ffn_dim      = KV_U32("feed_forward_length",             0);
    // qwen35's nominal context_length is 262144 but the KV cache is
    // sized linearly to it (k + v, fp16, per kv-head per layer). At
    // the full 262144 that's 12 GB of committed virtual memory — fine
    // on macOS where zero-fill-on-demand keeps RSS low, lethal on iOS
    // where jetsam counts the virtual commit. Cap at 4096 by default;
    // power users can raise via QWEN_MAX_POS env var.
    int32_t cl = KV_U32("context_length", 2048);
    const char * cap_env = getenv("QWEN_MAX_POS");
    int32_t cap = (cap_env != NULL && atoi(cap_env) > 0) ? atoi(cap_env)
                                                         : 4096;
    c->max_position = (cl < cap) ? cl : cap;
    c->rope_theta   = KV_F32("rope.freq_base",            10000.0f);
    c->norm_eps     = KV_F32("attention.layer_norm_rms_epsilon", 1e-5f);
    int32_t kl      = KV_U32("attention.key_length", 0);
    if (kl > 0) {
        c->head_dim = kl;
    } else if (c->n_heads > 0) {
        c->head_dim = c->hidden_dim / c->n_heads;
    }
    c->rope_dim = KV_U32("rope.dimension_count", 0);
    // mrope sections (qwen35 / qwen3-vl interleaved mrope).
    snprintf(key, sizeof(key), "%s.rope.dimension_sections", prefix);
    const struct gguf_kv * sec_kv = gguf_find_kv(g, key);
    for (int32_t i = 0; i < 4; i++) { c->rope_sections[i] = 0; }
    if (sec_kv && sec_kv->v.type == GGUF_VT_ARRAY
               && sec_kv->v.arr_type == GGUF_VT_I32
               && sec_kv->v.arr_n >= 1) {
        size_t cc = 0;
        int32_t n = (int32_t)sec_kv->v.arr_n;
        if (n > 4) { n = 4; }
        for (int32_t i = 0; i < n; i++) {
            c->rope_sections[i] = (int32_t)gr_u32(sec_kv->v.arr_data, &cc);
        }
    }
    // qwen35 hybrid extras (Gated DeltaNet linear attention).
    if (strcmp(prefix, "qwen35") == 0) {
        c->is_hybrid          = 1;
        c->attn_output_gate   = 1;
        c->full_attn_interval = KV_U32("full_attention_interval",  4);
        c->linear_n_heads     = KV_U32("ssm.group_count",         16);
        c->linear_k_head_dim  = KV_U32("ssm.state_size",         128);
        // GGUF's "ssm.state_size" = head_k_dim; head_v_dim is the
        // same in this model. Total inner V_dim = n_heads*head_v_dim.
        c->linear_v_head_dim  = c->linear_k_head_dim;
        c->linear_conv_kernel = KV_U32("ssm.conv_kernel",          4);
    } else {
        c->is_hybrid          = 0;
        c->attn_output_gate   = 0;
        c->full_attn_interval = 1;
    }
    #undef KV_U32
    #undef KV_F32
    const struct gguf_kv * tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (tk && tk->v.type == GGUF_VT_ARRAY) {
        c->vocab_size = (int32_t)tk->v.arr_n;
    }
    c->bos_id = (int32_t)gguf_kv_u32(g, "tokenizer.ggml.bos_token_id", 0);
    c->eos_id = (int32_t)gguf_kv_u32(g, "tokenizer.ggml.eos_token_id", 0);
    // eot_id is filled in later (after tok_load), when the tokenizer
    // vocab map is available. Initialize to -1 so the stop check
    // ignores it on models without an `<|im_end|>` token.
    c->eot_id = -1;
    return   c->hidden_dim > 0
          && c->n_layers   > 0
          && c->n_heads    > 0
          && c->ffn_dim    > 0
          && c->vocab_size > 0 ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Tokenizer (byte-level BPE, simplified)
// ---------------------------------------------------------------------------
//
// GPT-2 / Qwen3 byte-level BPE has a quirk: input UTF-8 bytes are
// first mapped through a fixed 256-entry "bytes-to-unicode" table so
// that whitespace + control bytes become printable unicode glyphs.
// Vocab and merges are stored in this remapped space.
//
// We implement: load vocab + merges from GGUF, build vocab-to-id map
// and merge-to-rank map. Encode: bytes -> remapped chars -> initial
// per-byte tokens -> greedy merge by lowest rank. Decode: id ->
// vocab string -> reverse byte map -> raw UTF-8.

// A "special" token that must bypass BPE: when its literal text
// appears in the input, the encoder emits the vocab id directly
// instead of running byte-level BPE on its bytes. Without this,
// markers like `<|im_start|>` get split into 6 byte-level pieces
// (`<`, `|`, `im`, `_start`, `|`, `>`) and the model sees garbled
// framing where its training expected a single token. Populated at
// load time by looking up known marker strings in `vocab_to_id`.
struct tok_special {
    const char * str;
    int32_t      len;
    int32_t      id;
};
#define TOK_MAX_SPECIALS 8

struct tokenizer {
    int32_t         vocab_size;
    struct chars *  vocab_strs;        // [vocab_size]
    struct s2i      vocab_to_id;
    struct s2i      merge_rank;        // "tokenA tokenB" -> rank
    int32_t         bos_id;
    int32_t         eos_id;
    int32_t         byte_to_uni[256];  // initial codepoint per raw byte
    int32_t         uni_to_byte[1024]; // reverse map; sparse, indexed by
                                       // codepoint mod 1024 (the GPT-2
                                       // set is < 1024)
    int32_t            n_specials;
    struct tok_special specials[TOK_MAX_SPECIALS];
};

// GPT-2's bytes_to_unicode table, expressed as direct codepoints.
// Reference: huggingface/tokenizers's gpt2 byte_level pre-tokenizer.
// Returns the codepoint for byte b.
static int32_t bytes_to_unicode_cp(int32_t b) {
    // 33..126, 161..172, 174..255 are passed through directly.
    // Remaining bytes (0..32, 127..160, 173) map to codepoints
    // 256, 257, ..., in order. So:
    //   0   -> 256
    //   1   -> 257
    //   ...
    //   32  -> 288
    //   33..126 -> 33..126
    //   127 -> 289
    //   ...
    //   160 -> 322
    //   161..172 -> 161..172
    //   173 -> 323
    //   174..255 -> 174..255
    int32_t r = b;
    if ((b >= 33  && b <= 126)
     || (b >= 161 && b <= 172)
     || (b >= 174 && b <= 255)) {
        r = b;
    } else {
        // assign in order
        int32_t counter = 0;
        for (int32_t i = 0; i < 256; i++) {
            int32_t is_printable = (i >= 33  && i <= 126)
                                || (i >= 161 && i <= 172)
                                || (i >= 174 && i <= 255);
            if (!is_printable) {
                if (i == b) {
                    r = 256 + counter;
                    i = 256;  // exit-via-predicate
                }
                counter++;
            }
        }
    }
    return r;
}

// Encode codepoint cp as UTF-8 into out (up to 4 bytes). Returns
// number of bytes written.
static int32_t utf8_encode_cp(int32_t cp, char * out) {
    int32_t n = 0;
    if (cp < 0x80) {
        out[0] = (char)cp;
        n = 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    return n;
}

static int32_t utf8_decode_one(const char * s, size_t n, int32_t * out_cp) {
    int32_t consumed = 0;
    if (n > 0) {
        unsigned char c0 = (unsigned char)s[0];
        if (c0 < 0x80) {
            *out_cp = c0;
            consumed = 1;
        } else if ((c0 & 0xE0) == 0xC0 && n >= 2) {
            *out_cp = ((c0 & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
            consumed = 2;
        } else if ((c0 & 0xF0) == 0xE0 && n >= 3) {
            *out_cp = ((c0 & 0x0F) << 12)
                    | (((unsigned char)s[1] & 0x3F) << 6)
                    | ((unsigned char)s[2] & 0x3F);
            consumed = 3;
        } else if ((c0 & 0xF8) == 0xF0 && n >= 4) {
            *out_cp = ((c0 & 0x07) << 18)
                    | (((unsigned char)s[1] & 0x3F) << 12)
                    | (((unsigned char)s[2] & 0x3F) << 6)
                    | ((unsigned char)s[3] & 0x3F);
            consumed = 4;
        } else {
            *out_cp = 0xFFFD;
            consumed = 1;
        }
    }
    return consumed;
}

static void tok_build_byte_maps(struct tokenizer * t) {
    for (int32_t i = 0; i < 1024; i++) { t->uni_to_byte[i] = -1; }
    for (int32_t b = 0; b < 256; b++) {
        int32_t cp = bytes_to_unicode_cp(b);
        t->byte_to_uni[b] = cp;
        if (cp >= 0 && cp < 1024) { t->uni_to_byte[cp] = b; }
    }
}

static int32_t tok_load(struct tokenizer * t, const struct gguf * g,
                        const struct llm_config * cfg) {
    memset(t, 0, sizeof(*t));
    t->vocab_size = cfg->vocab_size;
    t->bos_id     = cfg->bos_id;
    t->eos_id     = cfg->eos_id;
    tok_build_byte_maps(t);
    const struct gguf_kv * tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!tk || tk->v.type != GGUF_VT_ARRAY
            || tk->v.arr_type != GGUF_VT_STR) {
        fprintf(stderr, "tok: missing tokenizer.ggml.tokens (str array)\n");
        return -1;
    }
    t->vocab_strs = (struct chars *)llm_oom(
        calloc((size_t)t->vocab_size, sizeof(struct chars)));
    size_t c = 0;
    for (uint64_t i = 0; i < tk->v.arr_n; i++) {
        uint64_t n = gr_u64(tk->v.arr_data, &c);
        chars_put(&t->vocab_strs[i], (const char *)(tk->v.arr_data + c), n);
        c += n;
        s2i_put(&t->vocab_to_id,
                t->vocab_strs[i].data, t->vocab_strs[i].count, (int32_t)i);
    }

    const struct gguf_kv * mg = gguf_find_kv(g, "tokenizer.ggml.merges");
    if (mg && mg->v.type == GGUF_VT_ARRAY
           && mg->v.arr_type == GGUF_VT_STR) {
        size_t cc = 0;
        for (uint64_t i = 0; i < mg->v.arr_n; i++) {
            uint64_t n = gr_u64(mg->v.arr_data, &cc);
            const char * s = (const char *)(mg->v.arr_data + cc);
            s2i_put(&t->merge_rank, s, n, (int32_t)i);
            cc += n;
        }
    } else {
        fprintf(stderr, "tok: warning: no tokenizer.ggml.merges (BPE will\n"
                        "     fall back to byte-token-only encoding)\n");
    }
    // Known chat-framing specials (Qwen3 vocab). Looked up in the
    // vocab map; populated only when present, so non-chat GGUFs are
    // unaffected. Listed longest-first so the encoder's greedy match
    // picks the longest applicable token at each position.
    static const char * known[] = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        NULL,
    };
    t->n_specials = 0;
    for (int32_t i = 0; known[i] != NULL; i++) {
        int32_t slen = (int32_t)strlen(known[i]);
        int32_t id   = s2i_get(&t->vocab_to_id, known[i], slen, -1);
        if (id >= 0 && t->n_specials < TOK_MAX_SPECIALS) {
            t->specials[t->n_specials].str = known[i];
            t->specials[t->n_specials].len = slen;
            t->specials[t->n_specials].id  = id;
            t->n_specials++;
        }
    }
    return 0;
}

static void tok_free(struct tokenizer * t) {
    if (t->vocab_strs) {
        for (int32_t i = 0; i < t->vocab_size; i++) {
            chars_free(&t->vocab_strs[i]);
        }
        free(t->vocab_strs);
    }
    s2i_free(&t->vocab_to_id);
    s2i_free(&t->merge_rank);
}

// Try to match any registered special token at `text[0..tlen)`.
// Returns the special's index in `t->specials` (>=0) and writes its
// byte length to `*sp_len`, or -1 if no special matches at position
// 0. Greedy longest-first match: specials are registered in
// longest-first order so this just returns the first hit.
static int32_t tok_match_special(const struct tokenizer * t,
                                 const char * text, size_t tlen,
                                 int32_t * sp_len) {
    int32_t hit = -1;
    for (int32_t i = 0; i < t->n_specials && hit < 0; i++) {
        size_t slen = (size_t)t->specials[i].len;
        if (slen <= tlen && memcmp(text, t->specials[i].str, slen) == 0) {
            *sp_len = (int32_t)slen;
            hit     = i;
        }
    }
    return hit;
}

// BPE-encode a slice of raw text (no special-token recognition).
// `text[0..tlen)` is split on spaces with the GPT-2 convention that
// a leading space belongs to the next word, then each word is
// byte-level-remapped and greedy-BPE-merged. Returns the number of
// ids written to `out_ids` (bounded by `max_ids`).
static int32_t tok_encode_bpe(const struct tokenizer * t,
                              const char * text, size_t tlen,
                              int32_t * out_ids, int32_t max_ids) {
    int32_t n_out = 0;
    size_t  pos   = 0;
    while (pos < tlen && n_out < max_ids) {
        // Find next "word" boundary: include leading space (if any).
        size_t word_start = pos;
        if (text[pos] == ' ') { pos++; }
        while (pos < tlen && text[pos] != ' ') { pos++; }
        size_t word_len = pos - word_start;

        // Build initial token list: one entry per UTF-8 byte, each
        // mapped through bytes-to-unicode.
        struct chars * toks = (struct chars *)llm_oom(
            calloc(word_len, sizeof(struct chars)));
        int32_t n_toks = 0;
        for (size_t i = 0; i < word_len; i++) {
            int32_t cp = t->byte_to_uni[(unsigned char)text[word_start + i]];
            char    buf[4];
            int32_t nb = utf8_encode_cp(cp, buf);
            chars_put(&toks[n_toks++], buf, nb);
        }

        // Greedy BPE merge.
        int32_t changed = 1;
        while (changed && n_toks > 1) {
            changed = 0;
            int32_t best_idx  = -1;
            int32_t best_rank = INT32_MAX;
            // Scan adjacent pairs; find lowest rank.
            for (int32_t i = 0; i + 1 < n_toks; i++) {
                struct chars pair = {0};
                chars_put(&pair, toks[i].data,    toks[i].count);
                chars_put(&pair, " ",             1);
                chars_put(&pair, toks[i+1].data,  toks[i+1].count);
                int32_t rank = s2i_get(&t->merge_rank,
                                       pair.data, pair.count, INT32_MAX);
                chars_free(&pair);
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx  = i;
                }
            }
            if (best_idx >= 0) {
                // Merge toks[best_idx] + toks[best_idx+1].
                chars_put(&toks[best_idx],
                          toks[best_idx + 1].data, toks[best_idx + 1].count);
                chars_free(&toks[best_idx + 1]);
                for (int32_t j = best_idx + 1; j + 1 < n_toks; j++) {
                    toks[j] = toks[j + 1];
                }
                n_toks--;
                changed = 1;
            }
        }

        // Emit ids.
        for (int32_t i = 0; i < n_toks && n_out < max_ids; i++) {
            int32_t id = s2i_get(&t->vocab_to_id,
                                 toks[i].data, toks[i].count, -1);
            if (id < 0) {
                // Unknown sub-token. Should never happen for
                // byte-level BPE: every single byte is a known token.
                fprintf(stderr, "tok: unknown sub-token (len=%zu)\n",
                        toks[i].count);
            } else {
                out_ids[n_out++] = id;
            }
        }
        for (int32_t i = 0; i < n_toks; i++) { chars_free(&toks[i]); }
        free(toks);
    }
    return n_out;
}

// Encode: input UTF-8 text -> token ids. Returns count.
//
// Two-level scan. The outer loop walks the text looking for
// registered special-token strings (`<|im_start|>` etc.) and emits
// their vocab IDs directly. The inner spans (text between specials)
// go through `tok_encode_bpe` for the existing byte-level BPE.
// Without the outer scan a marker like `<|im_start|>` is split into
// six byte pieces, the model sees garbled framing instead of its
// trained chat envelope, and quality collapses.
static int32_t tok_encode(const struct tokenizer * t,
                          const char * text,
                          int32_t * out_ids, int32_t max_ids) {
    int32_t n_out = 0;
    size_t  tlen  = strlen(text);
    size_t  pos   = 0;
    while (pos < tlen && n_out < max_ids) {
        // Scan ahead for the next special-token occurrence.
        size_t  sp_at  = tlen;
        int32_t sp_idx = -1;
        int32_t sp_len = 0;
        for (size_t i = pos; i < tlen && sp_idx < 0; i++) {
            int32_t mlen = 0;
            int32_t hit  = tok_match_special(t, text + i, tlen - i, &mlen);
            if (hit >= 0) {
                sp_at  = i;
                sp_idx = hit;
                sp_len = mlen;
            }
        }
        // BPE-encode the text in front of the special (or all of it).
        if (sp_at > pos) {
            n_out += tok_encode_bpe(t, text + pos, sp_at - pos,
                                    out_ids + n_out, max_ids - n_out);
        }
        if (sp_idx >= 0 && n_out < max_ids) {
            out_ids[n_out++] = t->specials[sp_idx].id;
            pos = sp_at + (size_t)sp_len;
        } else {
            pos = tlen;
        }
    }
    return n_out;
}

// Decode: token id -> raw UTF-8 bytes appended to `out`. Reverses the
// byte-level remap.
static void tok_decode_one(const struct tokenizer * t,
                           int32_t id, struct chars * out) {
    if (id >= 0 && id < t->vocab_size) {
        const struct chars * s = &t->vocab_strs[id];
        size_t pos = 0;
        while (pos < s->count) {
            int32_t cp  = 0;
            int32_t adv = utf8_decode_one(s->data + pos,
                                          s->count - pos, &cp);
            if (adv == 0) { adv = 1; }
            if (cp >= 0 && cp < 1024 && t->uni_to_byte[cp] >= 0) {
                char b = (char)(unsigned char)t->uni_to_byte[cp];
                chars_put(out, &b, 1);
            } else {
                // Special token / non-byte: pass through as-is.
                chars_put(out, s->data + pos, adv);
            }
            pos += adv;
        }
    }
}

// ---------------------------------------------------------------------------
// Model weights - pointers into the mmap'd GGUF, plus type tags.
// ---------------------------------------------------------------------------

struct llm_tensor_ref {
    const void * data;
    int32_t      type;        // GGUF_TT_*
    int32_t      n_dims;
    int64_t      shape[4];
};

struct llm_layer_w {
    int32_t               is_ssm;       // 1 for qwen35 SSM layer, 0 for attention
    // Common (both attention and SSM):
    struct llm_tensor_ref attn_norm;    // RMSNorm before mixer
    struct llm_tensor_ref ffn_norm;     // RMSNorm before FFN (called
                                        // post_attention_norm in qwen35)
    struct llm_tensor_ref ffn_gate;
    struct llm_tensor_ref ffn_up;
    struct llm_tensor_ref ffn_down;
    // Attention-only:
    struct llm_tensor_ref attn_q;
    struct llm_tensor_ref attn_k;
    struct llm_tensor_ref attn_v;
    struct llm_tensor_ref attn_q_norm;
    struct llm_tensor_ref attn_k_norm;
    struct llm_tensor_ref attn_out;
    // SSM-only (qwen35 hybrid):
    struct llm_tensor_ref attn_qkv;     // 1024 -> 6144 input projection
    struct llm_tensor_ref attn_gate;    // 1024 -> 2048 output gate
    struct llm_tensor_ref ssm_a;        // (16,) per-group A_log
    struct llm_tensor_ref ssm_alpha;    // 1024 -> 16 (DeltaNet alpha)
    struct llm_tensor_ref ssm_beta;     // 1024 -> 16 (DeltaNet beta)
    struct llm_tensor_ref ssm_conv1d;   // (4, 6144) depthwise causal conv
    struct llm_tensor_ref ssm_dt_bias;  // (16,) per-group dt bias
    struct llm_tensor_ref ssm_norm;     // (128,) group-wise RMSNorm
    struct llm_tensor_ref ssm_out;      // 2048 -> 1024 output projection
};

struct llm_weights {
    struct llm_tensor_ref tok_embd;
    struct llm_tensor_ref output_norm;
    struct llm_tensor_ref output;       // may equal tok_embd (tied)
    struct llm_layer_w * layers;        // [n_layers]
};

static int32_t llm_resolve_tensor(const struct gguf * g, const char * name,
                                  struct llm_tensor_ref * out,
                                  int32_t required) {
    const struct gguf_tensor * t = gguf_find_tensor(g, name);
    int32_t r = 0;
    if (t == NULL) {
        if (required) {
            fprintf(stderr, "llm: missing required tensor: %s\n", name);
            r = -1;
        }
    } else {
        out->data   = t->data;
        out->type   = (int32_t)t->type;
        out->n_dims = (int32_t)t->n_dims;
        for (int32_t i = 0; i < 4; i++) {
            out->shape[i] = (i < (int32_t)t->n_dims) ? (int64_t)t->shape[i] : 1;
        }
    }
    return r;
}

static int32_t llm_load_weights(const struct gguf * g,
                                const struct llm_config * cfg,
                                struct llm_weights * w) {
    memset(w, 0, sizeof(*w));
    int32_t r = 0;
    r |= llm_resolve_tensor(g, "token_embd.weight",   &w->tok_embd,    1);
    r |= llm_resolve_tensor(g, "output_norm.weight",  &w->output_norm, 1);
    // output.weight is optional; absent => tied embeddings.
    llm_resolve_tensor(g, "output.weight", &w->output, 0);
    if (w->output.data == NULL) { w->output = w->tok_embd; }
    w->layers = (struct llm_layer_w *)llm_oom(
        calloc((size_t)cfg->n_layers, sizeof(struct llm_layer_w)));
    char nm[64];
    for (int32_t L = 0; L < cfg->n_layers; L++) {
        struct llm_layer_w * Lw = &w->layers[L];
        // Layer-type detection: in qwen35 the SSM layers carry an
        // ssm_a tensor; the attention layers don't. (We could also
        // use full_attn_interval modulo, but probing the GGUF is
        // the safer signal.)
        snprintf(nm, sizeof(nm), "blk.%d.ssm_a", L);
        struct llm_tensor_ref probe = {0};
        Lw->is_ssm = (cfg->is_hybrid
                      && llm_resolve_tensor(g, nm, &probe, 0) == 0
                      && probe.data != NULL) ? 1 : 0;

        // Shared norms + FFN. qwen35 calls the post-mixer norm
        // "post_attention_norm.weight"; classic Qwen uses
        // "ffn_norm.weight". Try both.
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", L);
        r |= llm_resolve_tensor(g, nm, &Lw->attn_norm, 1);
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", L);
        if (llm_resolve_tensor(g, nm, &Lw->ffn_norm, 0) != 0
            || Lw->ffn_norm.data == NULL) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", L);
            r |= llm_resolve_tensor(g, nm, &Lw->ffn_norm, 1);
        }
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", L);
        r |= llm_resolve_tensor(g, nm, &Lw->ffn_gate, 1);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",   L);
        r |= llm_resolve_tensor(g, nm, &Lw->ffn_up,   1);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", L);
        r |= llm_resolve_tensor(g, nm, &Lw->ffn_down, 1);

        if (Lw->is_ssm) {
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight",    L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_qkv,    1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight",   L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_gate,   1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a",              L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_a,       1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight",   L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_alpha,   1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight",    L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_beta,    1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight",  L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_conv1d,  1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias",        L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_dt_bias, 1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight",    L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_norm,    1);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight",     L);
            r |= llm_resolve_tensor(g, nm, &Lw->ssm_out,     1);
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_q, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_k, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_v, 1);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", L);
            llm_resolve_tensor(g, nm, &Lw->attn_q_norm, 0);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", L);
            llm_resolve_tensor(g, nm, &Lw->attn_k_norm, 0);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", L);
            r |= llm_resolve_tensor(g, nm, &Lw->attn_out, 1);
        }
    }
    return r;
}

static void llm_free_weights(struct llm_weights * w) {
    if (w->layers) {
        free(w->layers);
        w->layers = NULL;
    }
}

// Dispatch matmul based on weight tensor's quantization type. Weight
// shape in GGUF is (in_features, out_features) - first dim is the
// inner dim that contracts with x.
static struct tensor * matmul_dispatch(const struct llm_tensor_ref * w,
                                       struct tensor * x) {
    int64_t k     = w->shape[0];
    int64_t out_f = w->shape[1];
    assert(x->ne[0] == k);
    struct tensor * r = NULL;
    switch (w->type) {
        case GGUF_TT_F32: {
            // Wrap the mmap'd weights in a tensor-like view for the
            // fp32 matmul. We use a temporary stack tensor to avoid a
            // copy; matmul_f32 only reads from it.
            struct tensor wt;
            memset(&wt, 0, sizeof(wt));
            wt.ndim  = 2;
            wt.ne[0] = k; wt.ne[1] = out_f;
            for (int i = 2; i < TENSOR_MAX_DIMS; i++) wt.ne[i] = 1;
            wt.nb[0] = sizeof(float);
            wt.nb[1] = k * sizeof(float);
            wt.data  = (float *)w->data;
            wt.arena = x->arena;
            r = tensor_matmul_f32(&wt, x);
            break;
        }
        case GGUF_TT_Q4_K:
            r = tensor_matmul_q4k_f32((const q4k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q5_K:
            r = tensor_matmul_q5k_f32((const q5k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q6_K:
            r = tensor_matmul_q6k_f32((const q6k_block *)w->data,
                                      out_f, k, x);
            break;
        case GGUF_TT_Q8_0:
            r = tensor_matmul_q8_0_f32((const q8_0_block *)w->data,
                                       out_f, k, x);
            break;
        default:
            fprintf(stderr, "llm: matmul: unsupported weight type %d\n",
                    w->type);
            abort();
    }
    return r;
}

// Cast an mmap'd weights ref to a struct tensor (read-only view) for
// RMSNorm weight tensors etc., which GGUF stores as F32.
static struct tensor weights_as_f32_view(const struct llm_tensor_ref * w,
                                         struct arena * a) {
    struct tensor t;
    memset(&t, 0, sizeof(t));
    t.ndim = w->n_dims;
    for (int32_t i = 0; i < TENSOR_MAX_DIMS; i++) {
        t.ne[i] = (i < w->n_dims) ? w->shape[i] : 1;
    }
    t.nb[0] = sizeof(float);
    for (int32_t i = 1; i < TENSOR_MAX_DIMS; i++) {
        t.nb[i] = t.nb[i - 1] * t.ne[i - 1];
    }
    t.data  = (float *)w->data;
    t.arena = a;
    return t;
}

// ---------------------------------------------------------------------------
// KV cache (fp16 storage; fp32 working buffers for now)
// ---------------------------------------------------------------------------

struct llm_kv {
    int32_t    n_layers;
    int32_t    n_kv_heads;
    int32_t    head_dim;
    int32_t    max_position;
    int32_t    used;
    _Float16 * k;       // [n_layers][max_position][n_kv_heads][head_dim]
    _Float16 * v;
};

static int32_t kv_init(struct llm_kv * c, int32_t n_layers, int32_t n_kv_heads,
                       int32_t head_dim, int32_t max_position) {
    memset(c, 0, sizeof(*c));
    c->n_layers     = n_layers;
    c->n_kv_heads   = n_kv_heads;
    c->head_dim     = head_dim;
    c->max_position = max_position;
    c->used         = 0;
    size_t per_layer = (size_t)max_position * n_kv_heads * head_dim;
    c->k = (_Float16 *)llm_oom(calloc((size_t)n_layers * per_layer,
                                      sizeof(_Float16)));
    c->v = (_Float16 *)llm_oom(calloc((size_t)n_layers * per_layer,
                                      sizeof(_Float16)));
    return 0;
}

static void kv_free(struct llm_kv * c) {
    free(c->k); c->k = NULL;
    free(c->v); c->v = NULL;
}

static _Float16 * kv_row_k(struct llm_kv * c, int32_t L, int32_t pos) {
    size_t per_row = (size_t)c->n_kv_heads * c->head_dim;
    size_t row     = (size_t)L * c->max_position + pos;
    return c->k + row * per_row;
}

static _Float16 * kv_row_v(struct llm_kv * c, int32_t L, int32_t pos) {
    size_t per_row = (size_t)c->n_kv_heads * c->head_dim;
    size_t row     = (size_t)L * c->max_position + pos;
    return c->v + row * per_row;
}

// ---------------------------------------------------------------------------
// SSM state cache (qwen35 hybrid only)
//
// Per SSM layer we carry:
//   - conv_state: a (conv_kernel - 1) * n_channels rolling buffer for
//                 the depthwise causal conv1d. We use a circular index
//                 within a fixed [conv_kernel] buffer so the conv
//                 takes O(conv_kernel) per step.
//   - ssm_state:  the recurrent state per group, shape (group_count,
//                 state_size). Updated in place every step. Stored as
//                 fp32 (state values can be large in magnitude during
//                 long sequences; fp16 would saturate).
//
// Allocated once for every layer in the model; non-SSM layers leave
// their slots zero and untouched. Cheap enough to skip the bookkeeping.
// ---------------------------------------------------------------------------

struct llm_ssm_cache {
    int32_t   n_layers;
    int32_t   n_channels;     // 6144
    int32_t   conv_kernel;    // 4
    int32_t   group_count;    // 16
    int32_t   state_size;     // 128
    float *   conv_state;     // [n_layers][conv_kernel][n_channels] (ring)
    int32_t * conv_head;      // [n_layers] - circular index into conv_state
    float *   ssm_state;      // [n_layers][group_count][state_size]
};

static int32_t ssm_cache_init(struct llm_ssm_cache * s,
                              const struct llm_config * cfg) {
    memset(s, 0, sizeof(*s));
    if (cfg->is_hybrid) {
        s->n_layers    = cfg->n_layers;
        // conv_dim = 2*K_dim + V_dim = 2*(n_heads*head_k) + n_heads*head_v.
        int32_t K_dim  = cfg->linear_n_heads * cfg->linear_k_head_dim;
        int32_t V_dim  = cfg->linear_n_heads * cfg->linear_v_head_dim;
        s->n_channels  = 2 * K_dim + V_dim;
        s->conv_kernel = cfg->linear_conv_kernel;
        s->group_count = cfg->linear_n_heads;
        s->state_size  = cfg->linear_k_head_dim;
        // Recurrent state per head is (head_k_dim, head_v_dim). Total
        // state per layer = n_heads * head_k_dim * head_v_dim.
        size_t cb = (size_t)s->n_layers * s->conv_kernel *
                    s->n_channels * sizeof(float);
        size_t sb = (size_t)s->n_layers * cfg->linear_n_heads *
                    cfg->linear_k_head_dim * cfg->linear_v_head_dim *
                    sizeof(float);
        s->conv_state = (float *)llm_oom(calloc(1, cb));
        s->ssm_state  = (float *)llm_oom(calloc(1, sb));
        s->conv_head  = (int32_t *)llm_oom(
            calloc((size_t)s->n_layers, sizeof(int32_t)));
    }
    return 0;
}

static void ssm_cache_free(struct llm_ssm_cache * s) {
    free(s->conv_state); s->conv_state = NULL;
    free(s->ssm_state);  s->ssm_state  = NULL;
    free(s->conv_head);  s->conv_head  = NULL;
}

// Reset recurrent state to "fresh conversation". The KV cache is
// overwritten by the next forward pass so it does not need
// clearing here; only the SSM/conv recurrent buffers do.
static void ssm_cache_reset(struct llm_ssm_cache * s,
                            const struct llm_config * cfg) {
    if (s->conv_state != NULL) {
        size_t cb = (size_t)s->n_layers * s->conv_kernel *
                    s->n_channels * sizeof(float);
        memset(s->conv_state, 0, cb);
    }
    if (s->ssm_state != NULL) {
        size_t sb = (size_t)s->n_layers * cfg->linear_n_heads *
                    cfg->linear_k_head_dim * cfg->linear_v_head_dim *
                    sizeof(float);
        memset(s->ssm_state, 0, sb);
    }
    if (s->conv_head != NULL) {
        memset(s->conv_head, 0, (size_t)s->n_layers * sizeof(int32_t));
    }
}

// ---------------------------------------------------------------------------
// Forward pass - single token, with KV cache update.
//
// Returns logits as a (vocab_size,) tensor. Uses the per-call arena
// and dispatch matmul; KV cache pre-fill of past tokens is handled
// by feeding tokens through this same function in sequence.
// ---------------------------------------------------------------------------

struct llm_ctx {
    struct gguf            gguf;
    struct llm_config      cfg;
    struct llm_weights     W;
    struct tokenizer       tok;
    struct llm_kv          kv;
    struct llm_ssm_cache   ssm;
    struct arena *         arena;
    int32_t                loaded;
    char                   err[256];
    char *                 chat_template;   // see footnote (1)
    int32_t                dump_layer;      // see footnote (2)
    double                 t_prefill_s;     // see footnote (3)
    double                 t_gen_s;
    int32_t                n_prefill;
    int32_t                n_generated;
    int32_t                pos;             // see footnote (4)
};

// llm_ctx footnotes:
//
// (1) chat_template: Jinja string from GGUF KV
//     `tokenizer.chat_template`, copied as a NUL-terminated heap
//     string (GGUF stores it length-prefixed). NULL if the GGUF lacks
//     the KV (base completion models). Freed in llm_destroy.
//
// (2) dump_layer: --dump-layer L diagnostic. When >= 0, the forward
//     path prints the first 8 values of selected intermediate
//     tensors for layer L (and only that layer) before/after each
//     major op. Used for layer-by-layer parity diffing against
//     llama-eval-callback.
//
// (3) t_prefill_s / t_gen_s / n_prefill / n_generated: filled by
//     llm_generate's prefill and decode loops, read back via the
//     llm_pp_per_sec / llm_tg_per_sec / llm_n_* accessors. Zero
//     before any call has completed.
//
// (4) pos: next free position in the KV cache. Carries across
//     consecutive llm_generate calls so multi-turn chat does not
//     need to reformat and re-prefill prior turns - tokenize only
//     the new delta (e.g. `<|im_start|>user\nQ<|im_end|>\n` +
//     gen header) and call llm_generate again. llm_reset() zeroes
//     this back to 0 along with the SSM/conv recurrent state.

static int g_dump_layer = -1;
int g_no_q8k_rt = 0;
// Token-level trace: when set, llm_generate prints each sampled
// token ID to stderr as "[tok] %d\n". Used by tools/bench.sh's
// token-level parity mode, which survives ULP drift across
// perf rewrites that change the underlying logits but keep the
// argmax stable.
static int g_trace_tokens = 0;
static int g_min_new      = 0;

// DUMP(label, data, n) - one-line dump-when-this-layer-is-selected.
// Captures `c` (the llm_ctx) and `L` (the current layer index) from
// the calling scope; both are in scope at every dump site in
// llm_forward_ssm / llm_forward_attn / llm_forward_step's layer loop.
// Off (no fprintf, no read of `data`) when c->dump_layer != L.
#define DUMP(label, data, n) \
    do { \
        if (c->dump_layer == L) { \
            llm_dump_row((label), (data), (n)); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// qh_trace: machine-comparable per-tensor JSONL dump, enabled when the
// QH_TRACE_OUT env var is set to a writable path. The schema matches
// what llama.cpp/tools/qwen-haiku/llama-qwen-haiku emits, so a side-by-side
// proofdiff of the two files reveals the first divergence on a fixed
// prompt + greedy run. Disabled at runtime when the file pointer is
// NULL - the qh_trace_f32 call sites then degenerate to a single
// pointer comparison.
//
// Emitted fields per line:
//   name, op, type=f32, ne[4], nb[4] (synthetic dense-row strides),
//   nbytes, fnv64 (raw byte hash for binary equality), head/head_hex,
//   tail/tail_hex (first/last up-to-8 floats as both %.9g and as raw
//   fp32 hex bits), sum (drift-sensitive single scalar), n_nan, n_inf.
// ---------------------------------------------------------------------------

static FILE * g_qh_trace_fp = NULL;

static void qh_trace_open(void) {
    if (g_qh_trace_fp == NULL) {
        const char * p = getenv("QH_TRACE_OUT");
        if (p != NULL && p[0] != '\0') {
            g_qh_trace_fp = fopen(p, "w");
            if (g_qh_trace_fp != NULL) {
                fprintf(stderr, "qh_trace: writing JSONL to %s\n", p);
            }
        }
    }
}

static void qh_trace_close(void) {
    if (g_qh_trace_fp != NULL) {
        fclose(g_qh_trace_fp);
        g_qh_trace_fp = NULL;
    }
}

// FNV-1a 64-bit over raw bytes. Stable across runs and across
// implementations; identical bytes -> identical hash. We do NOT need
// cryptographic strength, just "did the bytes match?" - collision
// probability for typical tensor sizes is negligible.
static uint64_t qh_fnv1a64(const uint8_t * data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void qh_emit_floats(FILE * f, const char * key,
                           const float * v, int n) {
    fprintf(f, ",\"%s\":[", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) { fputc(',', f); }
        if (isnan(v[i])) {
            fputs("\"nan\"", f);
        } else if (isinf(v[i])) {
            fputs(v[i] < 0.0f ? "\"-inf\"" : "\"inf\"", f);
        } else {
            fprintf(f, "%.9g", v[i]);
        }
    }
    fputs("]", f);
    fprintf(f, ",\"%s_hex\":[", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) { fputc(',', f); }
        uint32_t u;
        memcpy(&u, &v[i], 4);
        fprintf(f, "\"%08x\"", u);
    }
    fputs("]", f);
}

static void qh_trace_f32(const char * name, const char * op,
                         const float * data,
                         int64_t ne0, int64_t ne1,
                         int64_t ne2, int64_t ne3) {
    FILE *  f = g_qh_trace_fp;
    int64_t n = ne0 * ne1 * ne2 * ne3;
    if (f != NULL && n > 0) {
        size_t   nbytes = (size_t)n * sizeof(float);
        uint64_t fnv    = qh_fnv1a64((const uint8_t *)data, nbytes);
        int64_t  k      = n < 8 ? n : 8;
        int64_t  tstart = n - k > 0 ? n - k : 0;
        float    head[8] = {0,0,0,0,0,0,0,0};
        float    tail[8] = {0,0,0,0,0,0,0,0};
        for (int64_t i = 0; i < k; i++) { head[i] = data[i];          }
        for (int64_t i = 0; i < k; i++) { tail[i] = data[tstart + i]; }
        float sum   = 0.0f;
        int   n_nan = 0;
        int   n_inf = 0;
        for (int64_t i = 0; i < n; i++) {
            float v = data[i];
            if (isnan(v))      { n_nan++; }
            else if (isinf(v)) { n_inf++; }
            else               { sum += v; }
        }
        // Synthetic dense-row strides: nb[0]=4, then ne0*4, ne0*ne1*4,
        // ne0*ne1*ne2*4. This matches what llama.cpp emits for
        // contiguous fp32 tensors.
        int64_t nb1 = ne0 * 4;
        int64_t nb2 = ne0 * ne1 * 4;
        int64_t nb3 = ne0 * ne1 * ne2 * 4;
        fprintf(f,
                "{\"name\":\"%s\",\"op\":\"%s\",\"type\":\"f32\","
                "\"ne\":[%lld,%lld,%lld,%lld],"
                "\"nb\":[4,%lld,%lld,%lld],"
                "\"nbytes\":%zu,\"fnv64\":\"%016llx\"",
                name, op,
                (long long)ne0, (long long)ne1,
                (long long)ne2, (long long)ne3,
                (long long)nb1, (long long)nb2, (long long)nb3,
                nbytes, (unsigned long long)fnv);
        qh_emit_floats(f, "head", head, (int)k);
        qh_emit_floats(f, "tail", tail, (int)k);
        fprintf(f, ",\"sum\":%.9g,\"n_nan\":%d,\"n_inf\":%d}\n",
                sum, n_nan, n_inf);
    }
}

// Convenience: trace a 1D row of n fp32 values.
static inline void qh_trace_row(const char * name, const char * op,
                                const float * data, int64_t n) {
    qh_trace_f32(name, op, data, n, 1, 1, 1);
}

// Convenience: trace a 2D batch tensor of shape (dim, n_tokens).
// Emits one JSONL line covering all n_tokens; the receiving side
// (llama.cpp/qwen-haiku reference dumper) uses the same shape when
// it runs chunked prefill, so byte-comparing the two files works.
static inline void qh_trace_batch(const char * name, const char * op,
                                  const float * data,
                                  int64_t dim, int64_t n_tokens) {
    qh_trace_f32(name, op, data, dim, n_tokens, 1, 1);
}

static void llm_dump_row(const char * label, const float * data, int32_t n) {
    // Mirror llama-eval-callback's format: head + tail + sum. The
    // sum across the whole tensor is the cheapest single number that
    // exposes drift in middle elements (head/tail can be identical
    // while the bulk diverges).
    double dsum = 0.0;
    for (int32_t i = 0; i < n; i++) { dsum += data[i]; }
    fprintf(stderr, "[dump] %s: ", label);
    int32_t k_head = n > 4 ? 4 : n;
    for (int32_t i = 0; i < k_head; i++) {
        fprintf(stderr, "%9.4f ", data[i]);
    }
    if (n >= 1024) {
        fprintf(stderr, " | mid: ");
        int32_t mids[] = { n / 8, n / 4, n / 2, (3 * n) / 4 };
        for (int32_t j = 0; j < 4; j++) {
            fprintf(stderr, "[%d]=%9.4f ", (int)mids[j], data[mids[j]]);
        }
    }
    if (n > 8) {
        fprintf(stderr, "...");
        for (int32_t i = n - 3; i < n; i++) {
            fprintf(stderr, "%9.4f ", data[i]);
        }
    }
    fprintf(stderr, "  sum=%.6f\n", dsum);
}

// ---------------------------------------------------------------------------
// SSM block - Gated DeltaNet (qwen35 / Qwen3-Next linear attention)
//
// Ported from transformers/models/qwen3_next/modeling_qwen3_next.py
// (Qwen3NextGatedDeltaNet.forward + torch_recurrent_gated_delta_rule).
// We use the recurrent (token-wise) kernel for both prefill and decode
// - slower but algorithmically simple. The chunked variant is a
// later optimisation.
//
// Symbol map between Python / GGUF / this C:
//
//   Python                                     GGUF tensor             local
//   ---------------------------------------    --------------------    ------
//   in_proj_qkvz first three chunks (Q,K,V)    attn_qkv.weight         qkv
//   in_proj_qkvz fourth chunk      (Z gate)    attn_gate.weight        z
//   in_proj_ba split 'b'                       ssm_beta.weight         b
//   in_proj_ba split 'a'                       ssm_alpha.weight        a
//   conv1d.weight                              ssm_conv1d.weight       conv_w
//   A_log                                      ssm_a                   a_log
//   dt_bias                                    ssm_dt.bias             dt_bias
//   norm (RMSNormGated) weight                 ssm_norm.weight         norm_w
//   out_proj                                   ssm_out.weight          out_w
//
// Math (single token, our use case):
//   1. h_norm = rms_norm(h, attn_norm)
//   2. qkv = attn_qkv · h_norm                              # (6144,)
//      z   = attn_gate · h_norm                             # (2048,)
//      b   = ssm_beta  · h_norm                             # (16,)
//      a   = ssm_alpha · h_norm                             # (16,)
//   3. conv state shift; conv_in = qkv (current step at last slot)
//   4. mixed_qkv = silu(causal_conv1d(qkv with kernel=4))   # (6144,)
//   5. Split mixed_qkv into Q (2048), K (2048), V (2048).
//   6. Reshape to per-head: Q[h, k_d] (16,128), K[h, k_d] (16,128),
//      V[h, v_d] (16,128).
//   7. L2-norm Q[h, :] and K[h, :] along inner dim (use_qk_l2norm
//      in the kernel call).
//   8. beta = sigmoid(b)                                    # (16,)
//      g    = -exp(A_log) · softplus(a + dt_bias)           # (16,)
//      scale = 1 / sqrt(k_head_dim) ; Q *= scale
//   9. Recurrence per head h (state[h] is (k_d, v_d) = (128,128)):
//        g_h   = exp(g[h])
//        state[h, :, :] *= g_h
//        kv_mem[h, v_d] = Σ_k state[h, k, v_d] · K[h, k]
//        delta[h, v_d] = (V[h, v_d] - kv_mem[h, v_d]) · beta[h]
//        state[h, k, v_d] += K[h, k] · delta[h, v_d]
//        out[h, v_d]      = Σ_k state[h, k, v_d] · Q[h, k]
//  10. core_attn_out flat = (16 * 128) = 2048. Apply gated RMSNorm
//      per-head with z as the silu-gate: for each head h,
//         out[h, :] = silu(z[h, :]) * (norm_w * rmsnorm(out[h, :]))
//      Note ssm_norm is (head_v_dim=128,), same per head.
//  11. result = ssm_out · core_attn_out                    # (1024,)
//
// Reference: NVlabs/GatedDeltaNet (ICLR'25) + qwen3_next paper.
// ---------------------------------------------------------------------------

static struct tensor * llm_forward_ssm(struct llm_ctx * c,
                                       int32_t L,
                                       struct tensor * h) {
    struct arena * a = c->arena;
    struct llm_layer_w * Lw = &c->W.layers[L];
    int32_t n_heads  = c->cfg.linear_n_heads;     // 16
    int32_t k_hd     = c->cfg.linear_k_head_dim;  // 128
    int32_t v_hd     = c->cfg.linear_v_head_dim;  // 128
    int32_t K_dim    = n_heads * k_hd;            // 2048
    int32_t V_dim    = n_heads * v_hd;            // 2048
    int32_t kK       = c->cfg.linear_conv_kernel; //    4
    int32_t conv_dim = 2 * K_dim + V_dim;         // 6144
    assert(Lw->ssm_conv1d.shape[1] == conv_dim);
    // 1. RMSNorm.
    struct tensor attn_norm_w = weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->cfg.norm_eps);
    DUMP("attn_norm", h_norm->data, c->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        qh_trace_row(nm, "RMS_NORM", h_norm->data, c->cfg.hidden_dim);
    }
    // 2. In-projections.
    struct tensor * qkv_pre = matmul_dispatch(&Lw->attn_qkv,  h_norm);
    DUMP("attn_qkv ", qkv_pre->data, conv_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_qkv-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", qkv_pre->data, conv_dim);
    }
    struct tensor * z = matmul_dispatch(&Lw->attn_gate, h_norm);
    // GGUF naming ambiguity: ssm_alpha vs ssm_beta - neither the
    // Python code nor the file name documents which is which. The
    // Python reference takes `mixed_ba` and splits into (b, a) in
    // that order, where 'b' goes through sigmoid (-> beta in the
    // recurrence) and 'a' is added to dt_bias before softplus
    // (-> g_log). Per llama.cpp's qwen3_next GGUF converter, the
    // FIRST half of mixed_ba (i.e. 'b') is named ssm_beta, and the
    // SECOND half ('a') is named ssm_alpha. So:
    //   ssm_beta  -> b -> sigmoid -> beta
    //   ssm_alpha -> a -> softplus -> g_log
    struct tensor * b_t = matmul_dispatch(&Lw->ssm_beta,  h_norm);
    struct tensor * a_t = matmul_dispatch(&Lw->ssm_alpha, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "alpha-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", a_t->data, n_heads);
        snprintf(nm, sizeof(nm), "beta-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", b_t->data, n_heads);
    }
    // shapes: qkv_pre(conv_dim,1), z(V_dim,1), b_t(n_heads,1), a_t(n_heads,1).
    // 3. Shift conv state ring; write qkv_pre into the new slot.
    {
        int32_t head    = c->ssm.conv_head[L];
        size_t  lane_off = ((size_t)L * kK + head) * conv_dim;
        float * lane    = c->ssm.conv_state + lane_off;
        for (int32_t i = 0; i < conv_dim; i++) { lane[i] = qkv_pre->data[i]; }
        c->ssm.conv_head[L] = (head + 1) % kK;
    }
    // 4. Causal conv1d step: for each channel ch, sum_k w[k, ch] *
    //    buffer_slot(k, ch), then SiLU. GGUF storage convention:
    //    ssm_conv1d.weight has shape [4, 6144] = [kernel_size,
    //    conv_dim]. shape[0] is the inner (fastest-changing) dim,
    //    so memory layout is conv_w[ch * kK + k]: per channel, kK
    //    contiguous kernel coefficients. Verified against the
    //    eval-callback dump. Position k=kK-1 = current step; k=0 =
    //    oldest in window.
    int32_t head_after = c->ssm.conv_head[L];
    const float * conv_w = (const float *)Lw->ssm_conv1d.data;
    struct tensor * mixed = tensor_new_2d(a, conv_dim, 1);
    size_t conv_lane_base = (size_t)L * kK * conv_dim;
    // Conv pass: write raw conv output into mixed->data so we can
    // trace it before the SiLU activation. Splitting conv from silu
    // does not change the conv kernel's bit output (acc lives in a
    // register; we only write the final value).
    for (int32_t ch = 0; ch < conv_dim; ch++) {
        float acc = 0.0f;
        const float * wch = conv_w + (size_t)ch * kK;
        for (int32_t k = 0; k < kK; k++) {
            int32_t slot = (head_after + k) % kK;
            float   v    = c->ssm.conv_state[conv_lane_base +
                                             (size_t)slot * conv_dim + ch];
            acc += v * wch[k];
        }
        mixed->data[ch] = acc;
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_raw-%d", (int)L);
        qh_trace_row(nm, "CONV1D", mixed->data, conv_dim);
    }
    // SiLU using ggml's NEON polynomial-approximation expf; bit-
    // identical to ggml's GGML_OP_SILU output. The libm scalar form
    // x / (1 + expf(-x)) differs by 1-2 ULPs and compounds through
    // 24 layers to visible drift, so we use ggml's vectorized variant
    // exclusively for parity.
    neon_silu_vec_f32(conv_dim, mixed->data, mixed->data);
    // 5. Split mixed into Q, K, V. Per the Python reference's
    //    `mixed_qkv = torch.cat((query, key, value), dim=-1)` after
    //    reshape, the FLAT layout in memory is block-concatenated
    //    (head-major within each block):
    //      mixed[0 .. K_dim-1]              = Q  (all heads of Q)
    //      mixed[K_dim .. 2*K_dim-1]        = K
    //      mixed[2*K_dim .. 2*K_dim+V_dim-1] = V
    //    Confirmed by the llama-eval-callback dump: v_conv/k_conv/q_conv
    //    are all VIEWS into conv_output_silu of width 2048 each.
    const float * Q_flat = mixed->data + 0;
    const float * K_flat = mixed->data + K_dim;
    const float * V_flat = mixed->data + 2 * K_dim;
    assert(K_dim == 2048 && V_dim == 2048);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_raw-%d", (int)L);
        qh_trace_row(nm, "VIEW", Q_flat, K_dim);
        snprintf(nm, sizeof(nm), "K_raw-%d", (int)L);
        qh_trace_row(nm, "VIEW", K_flat, K_dim);
        snprintf(nm, sizeof(nm), "V_raw-%d", (int)L);
        qh_trace_row(nm, "VIEW", V_flat, V_dim);
    }
    // 6/7. L2-normalise Q and K per-head along head dim. Then scale
    //      Q by 1/sqrt(k_hd) (the attention temperature).
    //
    // Accumulator shape matches ggml_compute_forward_l2_norm_f32:
    //   sum += (double)(q*q)        -- square in fp32, cast UP to fp64
    //   scale = 1/fmaxf(sqrtf(sum), eps)
    // Doing the multiply in fp32 first (then promoting to fp64 for
    // accumulation) is what ggml does, and gives bit-identical sum;
    // `(double)q * (double)q` keeps more precision but produces a
    // different bit result that compounds. eps applied via fmaxf
    // (not inside sqrt) also matters - same reason. The qwen3-next
    // l2_norm eps is 1e-6f per the GGUF default.
    const float l2_eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)k_hd);
    float Q_norm[2048];
    float K_norm[2048];
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        double qss = 0.0, kss = 0.0;
        for (int32_t i = 0; i < k_hd; i++) {
            float q = Q_flat[h2 * k_hd + i];
            float k = K_flat[h2 * k_hd + i];
            qss += (double)(q * q);
            kss += (double)(k * k);
        }
        float qrs = 1.0f / fmaxf(sqrtf((float)qss), l2_eps);
        float krs = 1.0f / fmaxf(sqrtf((float)kss), l2_eps);
        for (int32_t i = 0; i < k_hd; i++) {
            Q_norm[h2 * k_hd + i] = Q_flat[h2 * k_hd + i] * qrs * scale;
            K_norm[h2 * k_hd + i] = K_flat[h2 * k_hd + i] * krs;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_l2norm-%d", (int)L);
        qh_trace_row(nm, "L2_NORM_SCALE", Q_norm, K_dim);
        snprintf(nm, sizeof(nm), "K_l2norm-%d", (int)L);
        qh_trace_row(nm, "L2_NORM", K_norm, K_dim);
    }
    // 8. Per-head beta and g.
    //    NOTE: ssm_a in the GGUF is already pre-computed as the
    //    negative-exp-of-A_log scalar (the converter folds the
    //    -exp() into the stored tensor). Verified via
    //    llama-eval-callback dump: g_in = MUL(softplus(α+dt_bias),
    //    ssm_a) - a plain multiply, no extra exp. So we use
    //    `a_log_w[h]` (badly named here - it's actually neg-exp-A)
    //    directly without exponentiating.
    const float * ssm_a_w   = (const float *)Lw->ssm_a.data;
    const float * dt_bias_w = (const float *)Lw->ssm_dt_bias.data;
    float beta[64], g_t[64], g_log_dbg[64], a_softplus_dbg[64];
    assert(n_heads <= 64);
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        float bb = b_t->data[h2];
        beta[h2] = 1.0f / (1.0f + expf(-bb));   // sigmoid(b)
        float aa = a_t->data[h2] + dt_bias_w[h2];
        float softplus_a = aa > 20.0f ? aa : logf(1.0f + expf(aa));
        a_softplus_dbg[h2] = softplus_a;
        float g_log = ssm_a_w[h2] * softplus_a; // already neg-exp
        g_log_dbg[h2] = g_log;
        g_t[h2] = expf(g_log);
    }
    {
        // Trace names match the equivalent llama.cpp ggml node names
        // for layer 0 of qwen3-next; see tools/qwen-haiku trace map.
        char nm[48];
        snprintf(nm, sizeof(nm), "beta_in-%d", (int)L);
        qh_trace_row(nm, "SIGMOID", beta, n_heads);
        snprintf(nm, sizeof(nm), "a_softplus-%d", (int)L);
        qh_trace_row(nm, "SOFTPLUS", a_softplus_dbg, n_heads);
        // llama's `g_in-0` is the PRE-EXP g_log value (ssm_a * softplus_a),
        // not the post-exp g. Trace g_log here so byte-equality is direct;
        // g_t = expf(g_log) is the local intermediate we feed into the
        // recurrent state update below.
        snprintf(nm, sizeof(nm), "g_in-%d", (int)L);
        qh_trace_row(nm, "MUL", g_log_dbg, n_heads);
    }
    // 9. Recurrent state update per head.
    //    state[L, h, k, v] flat as ssm_state[((L*n_heads + h)*k_hd + k)*v_hd + v]
    //
    // 9.2 and 9.5 are dot reductions whose bit-output must match
    // ggml's qwen3_next chunked path: that path materialises
    // state * K (or state * Q) as a fp32 tensor via ggml_mul,
    // transposes+conts to make k contiguous, then sums each k-row
    // with `ggml_vec_sum_f32`. On Apple builds, ggml_vec_sum_f32
    // dispatches to Accelerate's `vDSP_sve`. We mirror this exactly:
    // form a contiguous k-inner products buffer per v, then call
    // vDSP_sve. Without Accelerate the fallback is fp64 accumulation
    // which matches ggml's #else branch (also fp64).
    size_t  state_off  = (size_t)L * n_heads * k_hd * v_hd;
    float * state_base = c->ssm.ssm_state + state_off;
    float out_flat[2048];   // n_heads * v_hd = 2048
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        float gh = g_t[h2];
        float bh = beta[h2];
        float * st = state_base + (size_t)h2 * k_hd * v_hd;
        const float * Q_h = Q_norm + h2 * k_hd;
        const float * K_h = K_norm + h2 * k_hd;
        const float * V_h = V_flat + h2 * v_hd;
        // 9.1 state *= g
        for (int32_t kv = 0; kv < k_hd * v_hd; kv++) { st[kv] *= gh; }
        // 9.2 kv_mem[v] = Σ_k state[k, v] * K[k]
        float kv_mem[128];
#ifdef LLM_USE_ACCELERATE
        float prod[128];  // k-contiguous products for vDSP_sve
#endif
        assert(v_hd <= 128);
        assert(k_hd <= 128);
        for (int32_t v = 0; v < v_hd; v++) {
#ifdef LLM_USE_ACCELERATE
            vDSP_vmul(st + v, v_hd, K_h, 1, prod, 1, (vDSP_Length)k_hd);
            float r;
            vDSP_sve(prod, 1, &r, (vDSP_Length)k_hd);
            kv_mem[v] = r;
#else
            double acc = 0.0;
            for (int32_t k = 0; k < k_hd; k++) {
                acc += (double)(st[k * v_hd + v] * K_h[k]);
            }
            kv_mem[v] = (float)acc;
#endif
        }
        // 9.3 delta[v] = (V[v] - kv_mem[v]) * beta
        float delta[128];
        for (int32_t v = 0; v < v_hd; v++) {
            delta[v] = (V_h[v] - kv_mem[v]) * bh;
        }
        // 9.4 state[k, v] += K[k] * delta[v]
        for (int32_t k = 0; k < k_hd; k++) {
            float kk = K_h[k];
            float * row = st + (size_t)k * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                row[v] += kk * delta[v];
            }
        }
        // 9.5 out[h, v] = Σ_k state[k, v] * Q[k]
        for (int32_t v = 0; v < v_hd; v++) {
#ifdef LLM_USE_ACCELERATE
            vDSP_vmul(st + v, v_hd, Q_h, 1, prod, 1, (vDSP_Length)k_hd);
            float r;
            vDSP_sve(prod, 1, &r, (vDSP_Length)k_hd);
            out_flat[h2 * v_hd + v] = r;
#else
            double acc = 0.0;
            for (int32_t k = 0; k < k_hd; k++) {
                acc += (double)(st[k * v_hd + v] * Q_h[k]);
            }
            out_flat[h2 * v_hd + v] = (float)acc;
#endif
        }
    }
    {
        // Recurrent-state output - matches llama's attn_output-0
        // shape (V_dim contiguous, head-major). Trace before any
        // gating / norm so we can pinpoint divergence in the
        // recurrent update separately from the gated rmsnorm.
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_output-%d", (int)L);
        qh_trace_row(nm, "RECURRENT", out_flat, V_dim);
    }
    // 10. Gated RMSNorm with z (per head, v_hd wide):
    //       y[h, :] = silu(z[h, :]) * (norm_w * rmsnorm(out[h, :]))
    //     Implementation per Qwen3NextRMSNormGated: variance over
    //     v_hd, rsqrt, multiply by norm_w, then multiply by silu(z).
    // Accumulator pattern + SiLU formulation match ggml's RMS_NORM
    // and ggml_v_silu (NEON polynomial expf). Without these the
    // step-10 gated output drifts by 1-2 ULPs/element vs llama.cpp.
    const float * norm_w = (const float *)Lw->ssm_norm.data;
    struct tensor * y_norm = tensor_new_2d(a, V_dim, 1);
    float z_silu[2048];
    neon_silu_vec_f32(V_dim, z_silu, z->data);
    for (int32_t h2 = 0; h2 < n_heads; h2++) {
        double ssq = 0.0;
        float * yh = out_flat + h2 * v_hd;
        for (int32_t v = 0; v < v_hd; v++) {
            ssq += (double)(yh[v] * yh[v]);
        }
        float mean = (float)(ssq / (double)v_hd);
        float rs = 1.0f / sqrtf(mean + c->cfg.norm_eps);
        const float * sg = z_silu + h2 * v_hd;
        float * yo = y_norm->data + h2 * v_hd;
        // ggml's RMS_NORM/MUL/MUL pipeline stores each intermediate
        // to memory, so each fp32 multiply is a separate operation.
        // Split into 3 distinct write-back stages to prevent clang
        // from fusing into a single FMA (which would change bits).
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yh[v] * rs;            // RMS_NORM scale
        }
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yo[v] * norm_w[v];     // MUL weight
        }
        for (int32_t v = 0; v < v_hd; v++) {
            yo[v] = yo[v] * sg[v];         // MUL silu(z)
        }
    }
    DUMP("conv_silu", mixed->data, conv_dim);
    DUMP("y_norm   ", y_norm->data, V_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_silu-%d", (int)L);
        qh_trace_row(nm, "SILU", mixed->data, conv_dim);
        snprintf(nm, sizeof(nm), "y_norm-%d", (int)L);
        qh_trace_row(nm, "RMS_NORM_GATED", y_norm->data, V_dim);
    }
    // 11. Output projection V_dim -> hidden.
    struct tensor * out_t = matmul_dispatch(&Lw->ssm_out, y_norm);
    DUMP("ssm_out  ", out_t->data, c->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "ssm_out-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", out_t->data, c->cfg.hidden_dim);
    }
    return out_t;
}

// Multi-token SSM block for batched prefill. Processes `n` tokens
// in one call using the chunked Gated DeltaNet kernel
// (chunked_ssm_step_f32). Returns a [hidden_dim, n] tensor of
// post-output-projection vectors, ready for the residual add.
//
// Any n >= 1 is accepted: the internal multi-chunk loop segments
// the input into successive chunked_ssm_step_f32 calls of size
// min(remaining, CHUNK_SIZE), and the per-head state carries in
// place across chunks. State is updated by the kernel itself, not
// by this wrapper.
//
// `h` is [hidden_dim, n] - the input residual stream for these n
// tokens. The SSM layer reads h as-is; the embedding lookup is
// done by the caller (llm_forward_batch).
static struct tensor * llm_forward_ssm_batch(struct llm_ctx * c,
                                              int32_t L, int32_t n,
                                              struct tensor * h) {
    struct arena * a = c->arena;
    struct llm_layer_w * Lw = &c->W.layers[L];
    int32_t n_heads  = c->cfg.linear_n_heads;
    int32_t k_hd     = c->cfg.linear_k_head_dim;
    int32_t v_hd     = c->cfg.linear_v_head_dim;
    int32_t K_dim    = n_heads * k_hd;
    int32_t V_dim    = n_heads * v_hd;
    int32_t kK       = c->cfg.linear_conv_kernel;
    int32_t conv_dim = 2 * K_dim + V_dim;
    assert(n > 0);
    // 1. attn_norm per token (rms_norm handles n via x->ne[1]).
    struct tensor attn_norm_w = weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->cfg.norm_eps);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        qh_trace_batch(nm, "RMS_NORM",
                       h_norm->data, c->cfg.hidden_dim, n);
    }
    // 2. In-projections - matmul_dispatch already iterates the n axis.
    struct tensor * qkv_pre = matmul_dispatch(&Lw->attn_qkv,  h_norm);  // [conv_dim, n]
    struct tensor * z       = matmul_dispatch(&Lw->attn_gate, h_norm);  // [V_dim, n]
    struct tensor * b_t     = matmul_dispatch(&Lw->ssm_beta,  h_norm);  // [n_heads, n]
    struct tensor * a_t     = matmul_dispatch(&Lw->ssm_alpha, h_norm);  // [n_heads, n]
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_qkv-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", qkv_pre->data, conv_dim, n);
        snprintf(nm, sizeof(nm), "alpha-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", a_t->data, n_heads, n);
        snprintf(nm, sizeof(nm), "beta-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", b_t->data, n_heads, n);
    }
    // 3. Conv1d step per token (sequential, ring buffer state).
    struct tensor * mixed = tensor_new_2d(a, conv_dim, n);
    const float * conv_w = (const float *)Lw->ssm_conv1d.data;
    size_t conv_lane_base = (size_t)L * kK * conv_dim;
    for (int32_t t = 0; t < n; t++) {
        int32_t head = c->ssm.conv_head[L];
        size_t  lane_off = conv_lane_base + (size_t)head * conv_dim;
        float * lane = c->ssm.conv_state + lane_off;
        for (int32_t i = 0; i < conv_dim; i++) {
            lane[i] = qkv_pre->data[t * conv_dim + i];
        }
        c->ssm.conv_head[L] = (head + 1) % kK;
        int32_t head_after = c->ssm.conv_head[L];
        for (int32_t ch = 0; ch < conv_dim; ch++) {
            float acc = 0.0f;
            const float * wch = conv_w + (size_t)ch * kK;
            for (int32_t k = 0; k < kK; k++) {
                int32_t slot = (head_after + k) % kK;
                float v_l = c->ssm.conv_state[
                    conv_lane_base + (size_t)slot * conv_dim + ch];
                acc += v_l * wch[k];
            }
            mixed->data[t * conv_dim + ch] = acc;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_raw-%d", (int)L);
        qh_trace_batch(nm, "CONV1D", mixed->data, conv_dim, n);
    }
    // 4. SiLU on all tokens at once.
    neon_silu_vec_f32(n * conv_dim, mixed->data, mixed->data);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "conv_silu-%d", (int)L);
        qh_trace_batch(nm, "SILU", mixed->data, conv_dim, n);
    }
    // 5. Split Q, K, V (per-token contiguous within conv_dim).
    float * Q_all = (float *)arena_alloc(a, (size_t)n * K_dim * sizeof(float));
    float * K_all = (float *)arena_alloc(a, (size_t)n * K_dim * sizeof(float));
    float * V_all = (float *)arena_alloc(a, (size_t)n * V_dim * sizeof(float));
    for (int32_t t = 0; t < n; t++) {
        const float * row = mixed->data + (size_t)t * conv_dim;
        memcpy(Q_all + (size_t)t * K_dim, row,         (size_t)K_dim * sizeof(float));
        memcpy(K_all + (size_t)t * K_dim, row + K_dim, (size_t)K_dim * sizeof(float));
        memcpy(V_all + (size_t)t * V_dim, row + 2 * K_dim, (size_t)V_dim * sizeof(float));
    }
    // 6/7. L2-norm Q, K per head, scale Q by 1/sqrt(k_hd).
    const float l2_eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)k_hd);
    for (int32_t t = 0; t < n; t++) {
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            double qss = 0.0, kss = 0.0;
            for (int32_t i = 0; i < k_hd; i++) {
                float qv = Q_all[t * K_dim + h2 * k_hd + i];
                float kv = K_all[t * K_dim + h2 * k_hd + i];
                qss += (double)(qv * qv);
                kss += (double)(kv * kv);
            }
            float qrs = 1.0f / fmaxf(sqrtf((float)qss), l2_eps);
            float krs = 1.0f / fmaxf(sqrtf((float)kss), l2_eps);
            for (int32_t i = 0; i < k_hd; i++) {
                Q_all[t * K_dim + h2 * k_hd + i] *= qrs * scale;
                K_all[t * K_dim + h2 * k_hd + i] *= krs;
            }
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Q_l2norm-%d", (int)L);
        qh_trace_batch(nm, "L2_NORM_SCALE", Q_all, K_dim, n);
        snprintf(nm, sizeof(nm), "K_l2norm-%d", (int)L);
        qh_trace_batch(nm, "L2_NORM", K_all, K_dim, n);
    }
    // 8. beta, g_log per token per head.
    const float * ssm_a_w   = (const float *)Lw->ssm_a.data;
    const float * dt_bias_w = (const float *)Lw->ssm_dt_bias.data;
    float * beta_all  = (float *)arena_alloc(a, (size_t)n * n_heads * sizeof(float));
    float * g_log_all = (float *)arena_alloc(a, (size_t)n * n_heads * sizeof(float));
    for (int32_t t = 0; t < n; t++) {
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            float bb = b_t->data[t * n_heads + h2];
            beta_all[t * n_heads + h2] = 1.0f / (1.0f + expf(-bb));
            float aa = a_t->data[t * n_heads + h2] + dt_bias_w[h2];
            float softplus_a = aa > 20.0f ? aa : logf(1.0f + expf(aa));
            g_log_all[t * n_heads + h2] = ssm_a_w[h2] * softplus_a;
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "beta_in-%d", (int)L);
        qh_trace_batch(nm, "SIGMOID", beta_all, n_heads, n);
        snprintf(nm, sizeof(nm), "g_in-%d", (int)L);
        qh_trace_batch(nm, "MUL", g_log_all, n_heads, n);
    }
    // 9. Chunked SSM per head. Per-chunk buffers (sized at CHUNK_SIZE
    //    so the same scratch handles every chunk; the kernel uses the
    //    actual chunk_n as its leading dim).
    float * q_chunk    = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * k_chunk    = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * v_chunk    = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * g_log_h    = (float *)arena_alloc(a, CHUNK_SIZE * sizeof(float));
    float * beta_h     = (float *)arena_alloc(a, CHUNK_SIZE * sizeof(float));
    float * out_chunk  = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    // Chunked scratch — reused per head.
    float * sc_gcs        = (float *)arena_alloc(a, CHUNK_SIZE * sizeof(float));
    float * sc_gexp       = (float *)arena_alloc(a, CHUNK_SIZE * sizeof(float));
    float * sc_decay_mask = (float *)arena_alloc(a, CHUNK_SIZE * CHUNK_SIZE * sizeof(float));
    float * sc_k_beta     = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * sc_v_beta     = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_kk_dot     = (float *)arena_alloc(a, CHUNK_SIZE * CHUNK_SIZE * sizeof(float));
    float * sc_lhs        = (float *)arena_alloc(a, CHUNK_SIZE * CHUNK_SIZE * sizeof(float));
    float * sc_attn       = (float *)arena_alloc(a, CHUNK_SIZE * CHUNK_SIZE * sizeof(float));
    float * sc_v_eff      = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_kbeta_gexp = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * sc_k_cumdecay = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * sc_attn_kq    = (float *)arena_alloc(a, CHUNK_SIZE * CHUNK_SIZE * sizeof(float));
    float * sc_q_g_exp    = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * sc_attn_inter = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_v_prime    = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_v_new      = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_v_attn     = (float *)arena_alloc(a, CHUNK_SIZE * v_hd * sizeof(float));
    float * sc_key_gdiff  = (float *)arena_alloc(a, CHUNK_SIZE * k_hd * sizeof(float));
    float * sc_kgd_vnew   = (float *)arena_alloc(a, k_hd * v_hd * sizeof(float));
    size_t state_off  = (size_t)L * n_heads * k_hd * v_hd;
    float * state_base = c->ssm.ssm_state + state_off;
    float * out_flat = (float *)arena_alloc(a, (size_t)n * V_dim * sizeof(float));
    // Multi-chunk loop: process min(remaining, CHUNK_SIZE) tokens
    // per call. State carries in place across chunks because
    // chunked_ssm_step_f32 updates it. The tail chunk is whatever
    // length is left (no zero-padding — the kernel takes its own
    // chunk_n parameter).
    int32_t n_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (int32_t ck = 0; ck < n_chunks; ck++) {
        int32_t chunk_start = ck * CHUNK_SIZE;
        int32_t chunk_n     = n - chunk_start;
        if (chunk_n > CHUNK_SIZE) { chunk_n = CHUNK_SIZE; }
        // Pack each head's chunk_n tokens; no zero-padding needed
        // because the kernel processes exactly chunk_n tokens this
        // call (state still carries across chunks).
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            for (int32_t t = 0; t < chunk_n; t++) {
                int32_t tg = chunk_start + t;
                memcpy(q_chunk + (size_t)t * k_hd,
                       Q_all   + (size_t)tg * K_dim + (size_t)h2 * k_hd,
                       (size_t)k_hd * sizeof(float));
                memcpy(k_chunk + (size_t)t * k_hd,
                       K_all   + (size_t)tg * K_dim + (size_t)h2 * k_hd,
                       (size_t)k_hd * sizeof(float));
                memcpy(v_chunk + (size_t)t * v_hd,
                       V_all   + (size_t)tg * V_dim + (size_t)h2 * v_hd,
                       (size_t)v_hd * sizeof(float));
                g_log_h[t] = g_log_all[tg * n_heads + h2];
                beta_h [t] = beta_all [tg * n_heads + h2];
            }
            float * state_h = state_base + (size_t)h2 * k_hd * v_hd;
            chunked_ssm_step_f32(chunk_n, k_hd, v_hd,
                                 q_chunk, k_chunk, v_chunk, g_log_h, beta_h,
                                 state_h, out_chunk,
                                 sc_gcs, sc_gexp, sc_decay_mask,
                                 sc_k_beta, sc_v_beta, sc_kk_dot, sc_lhs,
                                 sc_attn, sc_v_eff, sc_kbeta_gexp, sc_k_cumdecay,
                                 sc_attn_kq, sc_q_g_exp, sc_attn_inter,
                                 sc_v_prime, sc_v_new, sc_v_attn,
                                 sc_key_gdiff, sc_kgd_vnew);
            for (int32_t t = 0; t < chunk_n; t++) {
                int32_t tg = chunk_start + t;
                memcpy(out_flat + (size_t)tg * V_dim + (size_t)h2 * v_hd,
                       out_chunk + (size_t)t * v_hd,
                       (size_t)v_hd * sizeof(float));
            }
        }
    }
    // 10. Gated RMSNorm + z-SiLU per token. Each token's y_norm
    //     depends only on its own out_flat row + z row.
    const float * norm_w = (const float *)Lw->ssm_norm.data;
    struct tensor * y_norm = tensor_new_2d(a, V_dim, n);
    float z_silu_buf[2048];
    for (int32_t t = 0; t < n; t++) {
        neon_silu_vec_f32(V_dim, z_silu_buf, z->data + (size_t)t * V_dim);
        for (int32_t h2 = 0; h2 < n_heads; h2++) {
            double ssq = 0.0;
            float * yh = out_flat + (size_t)t * V_dim + (size_t)h2 * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                ssq += (double)(yh[v] * yh[v]);
            }
            float mean = (float)(ssq / (double)v_hd);
            float rs   = 1.0f / sqrtf(mean + c->cfg.norm_eps);
            const float * sg = z_silu_buf + (size_t)h2 * v_hd;
            float * yo = y_norm->data + (size_t)t * V_dim + (size_t)h2 * v_hd;
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yh[v] * rs;
            }
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yo[v] * norm_w[v];
            }
            for (int32_t v = 0; v < v_hd; v++) {
                yo[v] = yo[v] * sg[v];
            }
        }
    }
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "y_norm-%d", (int)L);
        qh_trace_batch(nm, "GATED_RMSNORM", y_norm->data, V_dim, n);
    }
    // 11. Output projection V_dim -> hidden_dim, per token.
    struct tensor * out_t = matmul_dispatch(&Lw->ssm_out, y_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "ssm_out-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT",
                       out_t->data, c->cfg.hidden_dim, n);
    }
    return out_t;
}

// Attention block for one transformer layer (non-SSM). Returns the
// post-output-projection tensor; the caller is responsible for the
// residual add. Mirrors llm_forward_ssm's contract.
static struct tensor * llm_forward_attn(struct llm_ctx * c,
                                        int32_t L, int32_t pos,
                                        struct tensor * h) {
    struct arena * a = c->arena;
    struct llm_layer_w * Lw = &c->W.layers[L];
    int32_t hd         = c->cfg.head_dim;
    int32_t n_h        = c->cfg.n_heads;
    int32_t n_kvh      = c->cfg.n_kv_heads;
    // attn_inner is the attention output width before attn_output.
    // For pure Qwen3 it equals hidden_dim, for qwen35 it does not
    // (head_dim=256 * n_heads=8 = 2048, hidden=1024).
    int32_t attn_inner = n_h * hd;
    int32_t kv_hidden  = n_kvh * hd;
    // RMSNorm.
    struct tensor attn_norm_w =
        weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->cfg.norm_eps);
    DUMP("[A]attn_nm", h_norm->data, c->cfg.hidden_dim);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        qh_trace_row(nm, "RMS_NORM", h_norm->data, c->cfg.hidden_dim);
    }
    // Q / K / V projections. Qwen3.5 has attn_output_gate=true: the
    // Q projection emits n_heads*head_dim*2, split into (Q, gate).
    // The gate is sigmoid'd and multiplied with the attention output
    // before attn_output. attn_q.weight shape is therefore
    // (hidden, 2*n_heads*head_dim).
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Qfull-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", q_raw->data, attn_inner * 2);
        snprintf(nm, sizeof(nm), "Kcur-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", k->data, kv_hidden);
        snprintf(nm, sizeof(nm), "Vcur-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT", v->data, kv_hidden);
    }
    if (c->dump_layer == L) {
        llm_dump_row("[A]Qfull  ", q_raw->data, attn_inner * 2);
        llm_dump_row("[A]Kcur   ", k->data, n_kvh * hd);
        llm_dump_row("[A]Vcur   ", v->data, n_kvh * hd);
        // One-shot diagnostics for attn_v block 0.
        float dqv[QK_K];
        const q6k_block * vrow0 = (const q6k_block *)Lw->attn_v.data;
        q6k_dequant_block(vrow0, dqv);
        llm_dump_row("[A]Vw[0]  ", dqv, 16);
        const uint8_t * raw = (const uint8_t *)vrow0;
        fprintf(stderr, "[dump] Vw[0] block bytes: d=0x%02x%02x"
                " scales[0..3]=%d %d %d %d  qh[0]=0x%02x  ql[0]=0x%02x\n",
                raw[209], raw[208],
                (int)((const int8_t *)raw)[192],
                (int)((const int8_t *)raw)[193],
                (int)((const int8_t *)raw)[194],
                (int)((const int8_t *)raw)[195],
                raw[128], raw[0]);
        fprintf(stderr, "[dump] sizeof q6k_block = %zu (should be 210)\n",
                sizeof(q6k_block));
        double s = 0.0;
        for (int32_t b = 0; b < 4; b++) {
            q6k_dequant_block(vrow0 + b, dqv);
            for (int32_t i = 0; i < QK_K; i++) {
                s += (double)dqv[i] * (double)h_norm->data[b * QK_K + i];
            }
        }
        fprintf(stderr, "[dump] manual Vw0 . hnorm = %.6f (v[0]=%.6f)\n",
                s, v->data[0]);
    }
    // Split q_raw (2*attn_inner) into Q and gate when qwen35's
    // attn_output_gate is on. Layout is HEAD-INTERLEAVED per
    // llama.cpp src/models/qwen35.cpp:491-518: per head h, the
    // 2*hd block holds Q[hd] then gate[hd].
    struct tensor * q = tensor_new_3d(a, hd, n_h, 1);
    struct tensor * attn_gate_v = NULL;
    if (c->cfg.attn_output_gate) {
        attn_gate_v = tensor_new_2d(a, attn_inner, 1);
        for (int32_t h2 = 0; h2 < n_h; h2++) {
            const float * src = q_raw->data + h2 * (2 * hd);
            for (int32_t i = 0; i < hd; i++) {
                q->data[h2 * hd + i]           = src[i];
                attn_gate_v->data[h2 * hd + i] = src[hd + i];
            }
        }
    } else {
        for (int32_t i = 0; i < attn_inner; i++) {
            q->data[i] = q_raw->data[i];
        }
    }
    // Reshape K / V to (head_dim, n_kv_heads, 1). q is already
    // shaped (head_dim, n_heads, 1).
    k->ndim = 3; k->ne[0] = hd; k->ne[1] = n_kvh; k->ne[2] = 1;
    k->ne[3] = 1; tensor_set_packed_strides(k);
    v->ndim = 3; v->ne[0] = hd; v->ne[1] = n_kvh; v->ne[2] = 1;
    v->ne[3] = 1; tensor_set_packed_strides(v);
    // Optional Q/K per-head RMSNorm (Qwen3 specific).
    if (Lw->attn_q_norm.data != NULL) {
        struct tensor qn_w = weights_as_f32_view(&Lw->attn_q_norm, a);
        struct tensor q2 = *q;
        q2.ndim = 2;
        q2.ne[0] = hd; q2.ne[1] = n_h;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) q2.ne[i] = 1;
        tensor_set_packed_strides(&q2);
        struct tensor * qnorm =
            tensor_rms_norm(&q2, &qn_w, c->cfg.norm_eps);
        qnorm->ndim = 3;
        qnorm->ne[0] = hd; qnorm->ne[1] = n_h; qnorm->ne[2] = 1;
        tensor_set_packed_strides(qnorm);
        q = qnorm;
    }
    if (Lw->attn_k_norm.data != NULL) {
        struct tensor kn_w = weights_as_f32_view(&Lw->attn_k_norm, a);
        struct tensor k2 = *k;
        k2.ndim = 2;
        k2.ne[0] = hd; k2.ne[1] = n_kvh;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) k2.ne[i] = 1;
        tensor_set_packed_strides(&k2);
        struct tensor * knorm =
            tensor_rms_norm(&k2, &kn_w, c->cfg.norm_eps);
        knorm->ndim = 3;
        knorm->ne[0] = hd; knorm->ne[1] = n_kvh; knorm->ne[2] = 1;
        tensor_set_packed_strides(knorm);
        k = knorm;
    }
    // Partial RoPE on q and k. For qwen35 this is the interleaved-
    // mrope variant ported from ggml-cpu/ops.cpp. For plain Qwen3
    // it falls back to standard NEOX-style RoPE when sections are
    // all zero.
    int32_t rotary_dim = c->cfg.rope_dim > 0 ? c->cfg.rope_dim : hd;
    int32_t use_imrope = c->cfg.rope_sections[0] ||
                         c->cfg.rope_sections[1] ||
                         c->cfg.rope_sections[2] ||
                         c->cfg.rope_sections[3];
    struct tensor * q_rope;
    struct tensor * k_rope;
    if (use_imrope) {
        q_rope = tensor_rope_mrope_i(q, pos, c->cfg.rope_theta,
                                     rotary_dim, c->cfg.rope_sections);
        k_rope = tensor_rope_mrope_i(k, pos, c->cfg.rope_theta,
                                     rotary_dim, c->cfg.rope_sections);
    } else {
        q_rope = tensor_rope(q, pos, c->cfg.rope_theta, rotary_dim);
        k_rope = tensor_rope(k, pos, c->cfg.rope_theta, rotary_dim);
    }
    // Write new K/V row into KV cache (cast to fp16).
    _Float16 * kdst = kv_row_k(&c->kv, L, pos);
    _Float16 * vdst = kv_row_v(&c->kv, L, pos);
    for (int32_t i = 0; i < kv_hidden; i++) {
        kdst[i] = (_Float16)k_rope->data[i];
        vdst[i] = (_Float16)v->data[i];
    }
    // Build K/V cache views for attention over positions 0..pos.
    int32_t kv_len = pos + 1;
    struct tensor * k_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    struct tensor * v_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    for (int32_t p = 0; p < kv_len; p++) {
        const _Float16 * ks = kv_row_k(&c->kv, L, p);
        const _Float16 * vs = kv_row_v(&c->kv, L, p);
        float * kd = k_all->data + p * n_kvh * hd;
        float * vd = v_all->data + p * n_kvh * hd;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kd[i] = (float)ks[i];
            vd[i] = (float)vs[i];
        }
    }
    // Attention with causal k_offset = pos so kv_max = pos+1.
    // Passing 0 here was a historical catastrophic bug.
    struct tensor * ctx_t =
        tensor_attention(q_rope, k_all, v_all, /*k_offset=*/pos);
    ctx_t->ndim = 2;
    ctx_t->ne[0] = attn_inner; ctx_t->ne[1] = 1;
    for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) ctx_t->ne[i] = 1;
    tensor_set_packed_strides(ctx_t);
    DUMP("[A]Qrope  ", q_rope->data, attn_inner);
    DUMP("[A]Krope  ", k_rope->data, n_kvh * hd);
    DUMP("[A]attn   ", ctx_t->data, attn_inner);
    // Output gate: attn_output *= sigmoid(gate). qwen35 only.
    if (c->cfg.attn_output_gate && attn_gate_v != NULL) {
        for (int32_t i = 0; i < attn_inner; i++) {
            float g_v = attn_gate_v->data[i];
            float sig = 1.0f / (1.0f + expf(-g_v));
            ctx_t->data[i] *= sig;
        }
        DUMP("[A]gated  ", ctx_t->data, attn_inner);
    }
    // Output projection.
    struct tensor * attn_out_t = matmul_dispatch(&Lw->attn_out, ctx_t);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_out-%d", (int)L);
        qh_trace_row(nm, "MUL_MAT",
                     attn_out_t->data, c->cfg.hidden_dim);
    }
    return attn_out_t;
}

// Multi-token attention block for batched prefill. Processes `n`
// tokens against the existing KV cache (rows 0..pos_start-1) plus
// the n new rows written by this call (rows pos_start..pos_start+n-1).
//
// Mirrors llm_forward_attn's contract: input is the (hidden_dim, n)
// residual stream, output is the (hidden_dim, n) post-projection
// tensor; caller does the residual add. State writes:
//   - KV cache rows pos_start..pos_start+n-1 (k_rope/v, cast to fp16)
//
// Causal masking is handled by tensor_attention, which uses
// k_offset = pos_start so each query t attends only to keys
// 0..(pos_start + t).
static struct tensor * llm_forward_attn_batch(struct llm_ctx * c,
                                              int32_t L,
                                              int32_t pos_start,
                                              int32_t n,
                                              struct tensor * h) {
    struct arena * a = c->arena;
    struct llm_layer_w * Lw = &c->W.layers[L];
    int32_t hd         = c->cfg.head_dim;
    int32_t n_h        = c->cfg.n_heads;
    int32_t n_kvh      = c->cfg.n_kv_heads;
    int32_t attn_inner = n_h * hd;
    int32_t kv_hidden  = n_kvh * hd;
    // RMSNorm over the n tokens. tensor_rms_norm iterates ne[1] = n.
    struct tensor attn_norm_w =
        weights_as_f32_view(&Lw->attn_norm, a);
    struct tensor * h_norm =
        tensor_rms_norm(h, &attn_norm_w, c->cfg.norm_eps);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_norm-%d", (int)L);
        qh_trace_batch(nm, "RMS_NORM",
                       h_norm->data, c->cfg.hidden_dim, n);
    }
    // Q/K/V projections — matmul_dispatch handles ne[1]=n natively.
    struct tensor * q_raw = matmul_dispatch(&Lw->attn_q, h_norm);
    struct tensor * k     = matmul_dispatch(&Lw->attn_k, h_norm);
    struct tensor * v     = matmul_dispatch(&Lw->attn_v, h_norm);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "Qfull-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", q_raw->data, attn_inner * 2, n);
        snprintf(nm, sizeof(nm), "Kcur-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", k->data, kv_hidden, n);
        snprintf(nm, sizeof(nm), "Vcur-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT", v->data, kv_hidden, n);
    }
    // Split q_raw into (q, gate) per token when attn_output_gate=true.
    // q_raw has shape [2*attn_inner, n] in head-interleaved layout.
    struct tensor * q = tensor_new_3d(a, hd, n_h, n);
    struct tensor * attn_gate_v = NULL;
    if (c->cfg.attn_output_gate) {
        attn_gate_v = tensor_new_2d(a, attn_inner, n);
        for (int32_t t = 0; t < n; t++) {
            const float * src = q_raw->data + (size_t)t * (2 * attn_inner);
            float * qdst   = q->data           + (size_t)t * attn_inner;
            float * gdst   = attn_gate_v->data + (size_t)t * attn_inner;
            for (int32_t h2 = 0; h2 < n_h; h2++) {
                const float * sh = src + h2 * (2 * hd);
                for (int32_t i = 0; i < hd; i++) {
                    qdst[h2 * hd + i] = sh[i];
                    gdst[h2 * hd + i] = sh[hd + i];
                }
            }
        }
    } else {
        for (int32_t t = 0; t < n; t++) {
            memcpy(q->data + (size_t)t * attn_inner,
                   q_raw->data + (size_t)t * attn_inner,
                   (size_t)attn_inner * sizeof(float));
        }
    }
    // Reshape k/v to (hd, n_kvh, n).
    k->ndim = 3; k->ne[0] = hd; k->ne[1] = n_kvh; k->ne[2] = n;
    k->ne[3] = 1; tensor_set_packed_strides(k);
    v->ndim = 3; v->ne[0] = hd; v->ne[1] = n_kvh; v->ne[2] = n;
    v->ne[3] = 1; tensor_set_packed_strides(v);
    // Per-head Q / K RMS norm (Qwen3 specific). View as 2D
    // (hd, n_h*n) / (hd, n_kvh*n) so tensor_rms_norm normalises
    // each head-vector independently.
    if (Lw->attn_q_norm.data != NULL) {
        struct tensor qn_w = weights_as_f32_view(&Lw->attn_q_norm, a);
        struct tensor q2 = *q;
        q2.ndim = 2;
        q2.ne[0] = hd; q2.ne[1] = (int64_t)n_h * n;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) q2.ne[i] = 1;
        tensor_set_packed_strides(&q2);
        struct tensor * qnorm =
            tensor_rms_norm(&q2, &qn_w, c->cfg.norm_eps);
        qnorm->ndim = 3;
        qnorm->ne[0] = hd; qnorm->ne[1] = n_h; qnorm->ne[2] = n;
        tensor_set_packed_strides(qnorm);
        q = qnorm;
    }
    if (Lw->attn_k_norm.data != NULL) {
        struct tensor kn_w = weights_as_f32_view(&Lw->attn_k_norm, a);
        struct tensor k2 = *k;
        k2.ndim = 2;
        k2.ne[0] = hd; k2.ne[1] = (int64_t)n_kvh * n;
        for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) k2.ne[i] = 1;
        tensor_set_packed_strides(&k2);
        struct tensor * knorm =
            tensor_rms_norm(&k2, &kn_w, c->cfg.norm_eps);
        knorm->ndim = 3;
        knorm->ne[0] = hd; knorm->ne[1] = n_kvh; knorm->ne[2] = n;
        tensor_set_packed_strides(knorm);
        k = knorm;
    }
    // RoPE on q [hd, n_h, n] and k [hd, n_kvh, n] starting at pos_start.
    // tensor_rope* already iterate seq axis (ne[2]) and apply
    // pos_t = pos_offset + s.
    int32_t rotary_dim = c->cfg.rope_dim > 0 ? c->cfg.rope_dim : hd;
    int32_t use_imrope = c->cfg.rope_sections[0] ||
                         c->cfg.rope_sections[1] ||
                         c->cfg.rope_sections[2] ||
                         c->cfg.rope_sections[3];
    struct tensor * q_rope;
    struct tensor * k_rope;
    if (use_imrope) {
        q_rope = tensor_rope_mrope_i(q, pos_start, c->cfg.rope_theta,
                                     rotary_dim, c->cfg.rope_sections);
        k_rope = tensor_rope_mrope_i(k, pos_start, c->cfg.rope_theta,
                                     rotary_dim, c->cfg.rope_sections);
    } else {
        q_rope = tensor_rope(q, pos_start, c->cfg.rope_theta, rotary_dim);
        k_rope = tensor_rope(k, pos_start, c->cfg.rope_theta, rotary_dim);
    }
    // Write the n new (K, V) rows into the KV cache.
    for (int32_t t = 0; t < n; t++) {
        _Float16 * kdst = kv_row_k(&c->kv, L, pos_start + t);
        _Float16 * vdst = kv_row_v(&c->kv, L, pos_start + t);
        const float * ksrc = k_rope->data + (size_t)t * kv_hidden;
        const float * vsrc = v->data      + (size_t)t * kv_hidden;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kdst[i] = (_Float16)ksrc[i];
            vdst[i] = (_Float16)vsrc[i];
        }
    }
    // Build full K/V views [hd, n_kvh, kv_len] over the whole cache
    // up to pos_start + n - 1.
    int32_t kv_len = pos_start + n;
    struct tensor * k_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    struct tensor * v_all = tensor_new_3d(a, hd, n_kvh, kv_len);
    for (int32_t p = 0; p < kv_len; p++) {
        const _Float16 * ks = kv_row_k(&c->kv, L, p);
        const _Float16 * vs = kv_row_v(&c->kv, L, p);
        float * kd = k_all->data + (size_t)p * kv_hidden;
        float * vd = v_all->data + (size_t)p * kv_hidden;
        for (int32_t i = 0; i < kv_hidden; i++) {
            kd[i] = (float)ks[i];
            vd[i] = (float)vs[i];
        }
    }
    // Batched attention: causal mask handled by tensor_attention via
    // k_offset = pos_start (query t attends keys 0..pos_start+t).
    struct tensor * ctx_t =
        tensor_attention(q_rope, k_all, v_all, /*k_offset=*/pos_start);
    // View as (attn_inner, n) for output projection.
    ctx_t->ndim = 2;
    ctx_t->ne[0] = attn_inner;
    ctx_t->ne[1] = n;
    for (int32_t i = 2; i < TENSOR_MAX_DIMS; i++) ctx_t->ne[i] = 1;
    tensor_set_packed_strides(ctx_t);
    // Output gate (per token): ctx *= sigmoid(gate).
    if (c->cfg.attn_output_gate && attn_gate_v != NULL) {
        for (int32_t t = 0; t < n; t++) {
            float * out_row = ctx_t->data       + (size_t)t * attn_inner;
            float * gat_row = attn_gate_v->data + (size_t)t * attn_inner;
            for (int32_t i = 0; i < attn_inner; i++) {
                float sig = 1.0f / (1.0f + expf(-gat_row[i]));
                out_row[i] *= sig;
            }
        }
    }
    // Output projection [hidden_dim, n].
    struct tensor * attn_out_t = matmul_dispatch(&Lw->attn_out, ctx_t);
    {
        char nm[48];
        snprintf(nm, sizeof(nm), "attn_out-%d", (int)L);
        qh_trace_batch(nm, "MUL_MAT",
                       attn_out_t->data, c->cfg.hidden_dim, n);
    }
    return attn_out_t;
}

static struct tensor * llm_forward_step(struct llm_ctx * c,
                                        int32_t tok_id, int32_t pos) {
    struct arena * a = c->arena;
    arena_reset(a);

    // 1. Embedding lookup. tok_embd in qwen35 GGUF is Q6_K-quantised
    //    (not fp32), so we can't view it as a plain fp32 row table.
    //    Dequantise the single row we need by routing through
    //    matmul_dispatch with a one-hot activation.
    int32_t hidden_dim = c->cfg.hidden_dim;
    struct tensor * h;
    if (c->W.tok_embd.type == GGUF_TT_F32) {
        struct tensor * ids = tensor_new_1d(a, 1);
        ids->data[0] = (float)tok_id;
        struct tensor tev = weights_as_f32_view(&c->W.tok_embd, a);
        h = tensor_get_rows(&tev, ids);
    } else {
        // tok_embd shape in GGUF is (hidden_dim, vocab_size). Rather
        // than build a one-hot and run a 200M-weight matmul through
        // matmul_dispatch just to extract one row, dequant the single
        // block-aligned chunk of size hidden_dim that lives at
        // tok_id*hidden_dim and copy into h directly. Q6_K is the
        // only quant we hit for tok_embd in Qwen3.5-0.8B-Q4_K_M;
        // other quants would need a corresponding dequant call here.
        h = tensor_new_2d(a, hidden_dim, 1);
        assert(hidden_dim % QK_K == 0);
        int32_t blocks_per_row = hidden_dim / QK_K;
        const q6k_block * row_blocks =
            (const q6k_block *)c->W.tok_embd.data
            + (size_t)tok_id * blocks_per_row;
        for (int32_t b = 0; b < blocks_per_row; b++) {
            q6k_dequant_block(row_blocks + b, h->data + b * QK_K);
        }
    }
    // h shape: (hidden_dim, 1)
    qh_trace_row("inp_embd", "GET_ROWS", h->data, c->cfg.hidden_dim);

    static int8_t use_ssm_batch = -1;
    if (use_ssm_batch < 0) {
        use_ssm_batch = (getenv("LLM_USE_SSM_BATCH") != NULL) ? 1 : 0;
    }
    for (int32_t L = 0; L < c->cfg.n_layers; L++) {
        struct llm_layer_w * Lw = &c->W.layers[L];
        struct tensor * mix;
        if (Lw->is_ssm) {
            // Single-token forward: route to the chunked-SSM batched
            // wrapper under LLM_USE_SSM_BATCH=1. For n=1 the chunked
            // kernel is mathematically equivalent to the recurrent
            // autoregressive path; this gate exists so we can run
            // `--single "Hello"` with both paths and diff the traces.
            mix = use_ssm_batch
                ? llm_forward_ssm_batch(c, L, 1, h)
                : llm_forward_ssm(c, L, h);
        } else {
            mix = llm_forward_attn(c, L, pos, h);
        }
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix-%d", (int)L);
            qh_trace_row(nm, Lw->is_ssm ? "SSM" : "ATTN",
                         mix->data, c->cfg.hidden_dim);
        }
        h = tensor_add(h, mix);

        // FFN: SwiGLU (shared by attention and SSM layers).
        DUMP("[F]residual", h->data, c->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix_residual-%d", (int)L);
            qh_trace_row(nm, "ADD", h->data, c->cfg.hidden_dim);
        }
        struct tensor ffn_norm_w =
            weights_as_f32_view(&Lw->ffn_norm, a);
        struct tensor * h_ffn_norm =
            tensor_rms_norm(h, &ffn_norm_w, c->cfg.norm_eps);
        DUMP("[F]post_atn", h_ffn_norm->data, c->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_norm-%d", (int)L);
            qh_trace_row(nm, "RMS_NORM",
                         h_ffn_norm->data, c->cfg.hidden_dim);
        }
        struct tensor * gate     = matmul_dispatch(&Lw->ffn_gate, h_ffn_norm);
        struct tensor * up       = matmul_dispatch(&Lw->ffn_up,   h_ffn_norm);
        // FFN SwiGLU uses NEON-poly SiLU (bit-exact with ggml's
        // GGML_OP_SILU). Scalar libm would drift 1-2 ULP per element.
        struct tensor * gate_act = tensor_new_nd(a, gate->ndim, gate->ne);
        neon_silu_vec_f32((int)tensor_nelements(gate),
                          gate_act->data, gate->data);
        struct tensor * ffn_in   = tensor_mul(gate_act, up);
        struct tensor * ffn_out  = matmul_dispatch(&Lw->ffn_down, ffn_in);
        DUMP("[F]ffn_out ", ffn_out->data, c->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_out-%d", (int)L);
            qh_trace_row(nm, "MUL_MAT",
                         ffn_out->data, c->cfg.hidden_dim);
        }
        h = tensor_add(h, ffn_out);
        DUMP("[F]post_ffn", h->data, c->cfg.hidden_dim);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "l_out-%d", (int)L);
            qh_trace_row(nm, "ADD", h->data, c->cfg.hidden_dim);
        }
    }

    // 12. Final RMSNorm + lm_head.
    struct tensor output_norm_w =
        weights_as_f32_view(&c->W.output_norm, a);
    struct tensor * h_final =
        tensor_rms_norm(h, &output_norm_w, c->cfg.norm_eps);
    qh_trace_row("result_norm", "RMS_NORM",
                 h_final->data, c->cfg.hidden_dim);
    struct tensor * logits = matmul_dispatch(&c->W.output, h_final);
    qh_trace_row("result_output", "MUL_MAT",
                 logits->data, (int64_t)tensor_nelements(logits));
    return logits;
}

// ---------------------------------------------------------------------------
// Batched (multi-token) forward for prefill.
//
// Processes `n` tokens at positions [pos_start, pos_start + n) in a
// single call, returning the logits for the LAST token. Used for
// prefill of multi-token prompts where the SSM layers benefit from
// the chunked Gated DeltaNet kernel (~10x throughput vs n calls to
// the autoregressive `llm_forward_step` for long prompts).
//
// Per-layer routing:
//   - SSM layer: llm_forward_ssm_batch (chunked, all n tokens in one
//                call; internal multi-chunk loop for n > CHUNK_SIZE).
//   - Attention layer: llm_forward_attn_batch (batched-queries-vs-
//                full-KV-cache kernel; causal mask via k_offset =
//                pos_start so query t attends only to keys
//                0..pos_start+t).
//   - FFN: shape [hidden, n] flows through tensor_rms_norm,
//          matmul_dispatch, tensor_mul, neon_silu_vec_f32 untouched -
//          all four operate over the `n` axis natively.
//
// Caller invariants:
//   - Caller is responsible for advancing `c->pos` (mirrors
//     llm_forward_step semantics; we just compute, no state pointer).
//   - SSM state (`c->ssm.ssm_state`, `c->ssm.conv_state`,
//     `c->ssm.conv_head`) and KV cache rows for
//     pos_start..pos_start+n-1 are written by the two batch helpers.
//
// Returns logits for the LAST token (shape [vocab_size, 1]). Earlier
// tokens' logits are discarded (we only need them for prefill
// state-building; the first sample-able token is pos_start + n - 1).
static struct tensor * llm_forward_batch(struct llm_ctx * c,
                                         const int32_t * tok_ids,
                                         int32_t n,
                                         int32_t pos_start) {
    assert(n > 0);
    struct arena * a = c->arena;
    arena_reset(a);
    int32_t hidden_dim = c->cfg.hidden_dim;
    // 1. Embedding lookup for n tokens. Same Q6_K / F32 split as the
    //    single-token path; we just stack n rows.
    struct tensor * h = tensor_new_2d(a, hidden_dim, n);
    for (int32_t t = 0; t < n; t++) {
        int32_t tok_id = tok_ids[t];
        float * dst = h->data + (size_t)t * hidden_dim;
        if (c->W.tok_embd.type == GGUF_TT_F32) {
            const float * src = (const float *)c->W.tok_embd.data
                              + (size_t)tok_id * hidden_dim;
            memcpy(dst, src, (size_t)hidden_dim * sizeof(float));
        } else {
            // Q6_K dequant per row, same as llm_forward_step.
            assert(hidden_dim % QK_K == 0);
            int32_t blocks_per_row = hidden_dim / QK_K;
            const q6k_block * row_blocks =
                (const q6k_block *)c->W.tok_embd.data
                + (size_t)tok_id * blocks_per_row;
            for (int32_t b = 0; b < blocks_per_row; b++) {
                q6k_dequant_block(row_blocks + b, dst + b * QK_K);
            }
        }
    }
    qh_trace_batch("inp_embd", "GET_ROWS", h->data, hidden_dim, n);
    // 2. Per-layer.
    for (int32_t L = 0; L < c->cfg.n_layers; L++) {
        struct llm_layer_w * Lw = &c->W.layers[L];
        struct tensor * mix;
        if (Lw->is_ssm) {
            // SSM: process all n tokens via chunked kernel. Internal
            // multi-chunk loop handles n > CHUNK_SIZE.
            mix = llm_forward_ssm_batch(c, L, n, h);
        } else {
            // Attention: batched-queries kernel. Processes all n
            // tokens against the KV cache in one pass; tensor_attention
            // applies the causal mask via k_offset = pos_start so
            // query t attends only to keys 0..(pos_start+t).
            mix = llm_forward_attn_batch(c, L, pos_start, n, h);
        }
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix-%d", (int)L);
            qh_trace_batch(nm, Lw->is_ssm ? "SSM" : "ATTN",
                           mix->data, hidden_dim, n);
        }
        // h = h + mix (per-token elementwise add).
        h = tensor_add(h, mix);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "mix_residual-%d", (int)L);
            qh_trace_batch(nm, "ADD", h->data, hidden_dim, n);
        }
        // FFN: rms_norm + SwiGLU + matmul, all n-axis-aware.
        struct tensor ffn_norm_w = weights_as_f32_view(&Lw->ffn_norm, a);
        struct tensor * h_ffn_norm =
            tensor_rms_norm(h, &ffn_norm_w, c->cfg.norm_eps);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_norm-%d", (int)L);
            qh_trace_batch(nm, "RMS_NORM", h_ffn_norm->data, hidden_dim, n);
        }
        struct tensor * gate     = matmul_dispatch(&Lw->ffn_gate, h_ffn_norm);
        struct tensor * up       = matmul_dispatch(&Lw->ffn_up,   h_ffn_norm);
        struct tensor * gate_act = tensor_new_2d(a, gate->ne[0], gate->ne[1]);
        neon_silu_vec_f32((int)tensor_nelements(gate),
                          gate_act->data, gate->data);
        struct tensor * ffn_in   = tensor_mul(gate_act, up);
        struct tensor * ffn_out  = matmul_dispatch(&Lw->ffn_down, ffn_in);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "ffn_out-%d", (int)L);
            qh_trace_batch(nm, "MUL_MAT", ffn_out->data, hidden_dim, n);
        }
        h = tensor_add(h, ffn_out);
        {
            char nm[48];
            snprintf(nm, sizeof(nm), "l_out-%d", (int)L);
            qh_trace_batch(nm, "ADD", h->data, hidden_dim, n);
        }
    }
    // 3. Final RMS-norm + lm_head on the LAST token only.
    //    Earlier tokens' logits aren't needed for prefill.
    struct tensor * h_last = tensor_new_2d(a, hidden_dim, 1);
    memcpy(h_last->data,
           h->data + (size_t)(n - 1) * hidden_dim,
           (size_t)hidden_dim * sizeof(float));
    struct tensor output_norm_w =
        weights_as_f32_view(&c->W.output_norm, a);
    struct tensor * h_final =
        tensor_rms_norm(h_last, &output_norm_w, c->cfg.norm_eps);
    qh_trace_row("result_norm", "RMS_NORM", h_final->data, hidden_dim);
    struct tensor * logits = matmul_dispatch(&c->W.output, h_final);
    qh_trace_row("result_output", "MUL_MAT",
                 logits->data, c->cfg.vocab_size);
    return logits;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

static int32_t sample_argmax(const struct tensor * logits) {
    int64_t n = tensor_nelements(logits);
    int32_t best = 0;
    float v = logits->data[0];
    for (int64_t i = 1; i < n; i++) {
        if (logits->data[i] > v) {
            v    = logits->data[i];
            best = (int32_t)i;
        }
    }
    return best;
}

// xoroshiro128** PRNG (Blackman/Vigna, public-domain reference).
// Cheap, seedable, much better statistical quality than libc rand().
// Returns a uniform u64; the sampler divides into [0, 1) below.
struct rng { uint64_t s0, s1; };

static inline uint64_t rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(struct rng * r) {
    uint64_t s0    = r->s0;
    uint64_t s1    = r->s1;
    uint64_t res   = rng_rotl(s0 * 5, 7) * 9;
    s1            ^= s0;
    r->s0          = rng_rotl(s0, 24) ^ s1 ^ (s1 << 16);
    r->s1          = rng_rotl(s1, 37);
    return res;
}

// SplitMix64 to expand a single 64-bit seed into the two-word state.
static void rng_seed(struct rng * r, uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s0 = z ^ (z >> 31);
    z     = (r->s0 + 0x9e3779b97f4a7c15ULL);
    z     = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z     = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s1 = z ^ (z >> 31);
    // Avoid the all-zero state, which would lock xoroshiro at zero.
    if (r->s0 == 0 && r->s1 == 0) { r->s0 = 1; }
}

static inline float rng_uniform(struct rng * r) {
    // 24-bit mantissa, divided to [0, 1).
    return (float)(rng_next(r) >> 40) / (float)(1u << 24);
}

struct llm_sampler llm_sampler_default(void) {
    struct llm_sampler s;
    s.temperature        = 1.0f;
    s.top_k              = 20;
    s.top_p              = 0.95f;
    s.min_p              = 0.0f;
    s.repetition_penalty = 1.05f;
    s.repetition_window  = 64;
    return s;
}

// Sampler chain order (sample_with below): penalties -> temperature
// (softmax-with-T on top-K logits) -> top-P -> min-P -> distribution.
// Matches im.ai's `Sampler.swift` ordering (see comment block at
// im.ai/src/model/Sampler.swift:19-32 for the rationale). One subtle
// difference: im.ai normalizes the softmax over the FULL vocabulary
// before top-K filtering, while we normalize over the top-K survivors
// (which represent ~99% of probability mass for typical Qwen logits,
// so the top-P cutoff shifts by <1% in practice).
struct llm_sampler llm_sampler_im_ai(void) {
    struct llm_sampler s;
    s.temperature        = 0.7f;
    s.top_k              = 40;
    s.top_p              = 0.9f;
    s.min_p              = 0.05f;
    s.repetition_penalty = 1.25f;
    s.repetition_window  = 64;
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

char * llm_chat_format_delta(const char * user_message,
                             const char * system_prefix,
                             int enable_thinking) {
    return jinja_apply_delta(user_message, system_prefix,
                             enable_thinking);
}

typedef int (*llm_token_cb)(const char * utf8, void * user);

int llm_generate(struct llm_ctx * c,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, int min_new,
                 const struct llm_sampler * sampler_in,
                 uint64_t seed,
                 llm_token_cb cb, void * user) {
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
        }
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

static int32_t print_cb(const char * s, void * user) {
    (void)user;
    fputs(s, stdout);
    fflush(stdout);
    return 0;
}

// Token callback for the chat CLI: capture into `chars` only. The
// turn is printed in one shot at the end (after strip_reasoning_for_history
// runs on the buffer), so the user sees a clean reply without the
// leading `</think>` leak that streaming-print would expose. Loses
// the per-token streaming feel but the chat CLI is for offline
// testing - the SwiftUI surface still streams via ChatStreamFilter.
static int32_t capture_cb(const char * s, void * user) {
    struct chars * out = (struct chars *)user;
    chars_put(out, s, (int32_t)strlen(s));
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

static int chat_test_cb(const char * utf8, void * user) {
    struct chat_test_capture * cap = (struct chat_test_capture *)user;
    size_t n = strlen(utf8);
    chars_put(&cap->text, utf8, n);
    // Same FNV-1a polynomial as qh_fnv1a64 — keep its byte loop
    // inline so the hash is computed incrementally as tokens stream
    // in (the qh_fnv1a64 helper expects a single contiguous buffer).
    for (size_t i = 0; i < n; i++) {
        cap->hash ^= (uint64_t)(uint8_t)utf8[i];
        cap->hash *= 0x100000001b3ULL;
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
    struct llm_sampler sp = llm_sampler_im_ai();
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
    int32_t mode = 0;  // 0=help, 1=self-test, 2=single, 3=repl, 4=chat
    const char * prompt = "Hello, my name is";
    const char * system_prompt   = NULL;
    const char * chat_prompts[LLM_CLI_MAX_TURNS];
    int32_t      chat_n          = 0;
    struct llm_sampler sp = {0};        // greedy by default
    sp.top_k              = 40;
    sp.repetition_penalty = 1.0f;
    sp.repetition_window  = 64;
    uint64_t seed         = 0;          // 0 = derive from wall clock
    int32_t  max_new      = 64;
    int32_t  dump_layer   = -1;
    for (int32_t i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            mode = 1;
        } else if (strcmp(argv[i], "--chunked-test") == 0) {
            mode = 5;
        } else if (strcmp(argv[i], "--single") == 0) {
            mode = 2;
            if (i + 1 < argc) { prompt = argv[++i]; }
        } else if (strcmp(argv[i], "--repl") == 0) {
            mode = 3;
        } else if (strcmp(argv[i], "--chat") == 0) {
            mode = 4;
        } else if (strcmp(argv[i], "--chat-test") == 0) {
            mode = 6;
        } else if (strcmp(argv[i], "--print-chat-template") == 0) {
            mode = 7;
        } else if (strcmp(argv[i], "--jinja-test") == 0) {
            mode = 8;
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
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            const char * pn = argv[++i];
            if (strcmp(pn, "default") == 0) {
                sp = llm_sampler_default();
            } else if (strcmp(pn, "im_ai") == 0 || strcmp(pn, "im-ai") == 0) {
                sp = llm_sampler_im_ai();
            } else if (strcmp(pn, "greedy") == 0) {
                memset(&sp, 0, sizeof(sp));
                sp.repetition_penalty = 1.0f;
                sp.repetition_window  = 64;
            } else {
                fprintf(stderr,
                        "llm: unknown --preset '%s'"
                        " (default|im_ai|greedy)\n", pn);
            }
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
    if (mode == 1) {
        rc = llm_self_test();
    } else if (mode == 5) {
        rc = chunked_self_test();
    } else if (mode == 2) {
        rc = run_single(prompt, max_new, &sp, seed);
    } else if (mode == 3) {
        rc = run_repl(&sp, seed, max_new);
    } else if (mode == 4) {
        if (chat_n == 0) {
            fprintf(stderr, "llm: --chat needs at least one -p/--prompt\n");
            rc = 1;
        } else {
            rc = run_chat(chat_prompts, chat_n, system_prompt,
                          &sp, seed, max_new);
        }
    } else if (mode == 6) {
        rc = run_chat_test(max_new);
    } else if (mode == 7) {
        struct llm_ctx * c = llm_create(llm_cli_gguf_path());
        if (!c->loaded) {
            fprintf(stderr, "llm: load failed: %s\n", llm_get_error(c));
            rc = 1;
        } else {
            const char * tpl = llm_chat_template(c);
            if (tpl != NULL) { fputs(tpl, stdout); }
        }
        llm_destroy(c);
    } else if (mode == 8) {
        rc = jinja_self_test();
    } else {
        printf("usage (set QWEN_GGUF=/path/to/model.gguf to override default):\n"
               "  llm --self-test\n"
               "  llm --single \"prompt\" [--max-new N] [sampler flags]\n"
               "  llm --repl [--max-new N] [sampler flags]\n"
               "  llm --chat -p \"turn1\" [-p \"turn2\" ...] "
                       "[-sys \"system prompt\"] [sampler flags]\n"
               "  llm --chat-test [--max-new N]\n"
               "  llm --jinja-test\n"
               "  llm --print-chat-template\n"
               "\n"
               "sampler flags:\n"
               "  --preset {default|im_ai|greedy}   load named preset\n"
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
