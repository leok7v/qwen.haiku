// gguf.c -- minimal GGUF v3 reader (subset that covers Qwen3 GGUFs
// from llama.cpp). Used by qwen.c for weights / tokenizer / config
// lookups; broken out into its own single-file lib so other model
// loaders can share it without dragging in the Qwen-specific layers.
//
// `#include`-d once from slm.c (or from any other TU root). Depends
// on POSIX (open/mmap/fstat) and on utils/chars.c being available
// (only for the `oom()` wrapper). No allocation outside `kvs` /
// `tensors` arrays; tensor and KV bodies live inside the mmap, the
// caller never owns them.
//
// Public surface: `gguf_open` / `gguf_close` plus `gguf_find_kv`
// `gguf_find_tensor` `gguf_kv_u32` `gguf_kv_f32` accessors. The
// `gr_*` low-level cursor helpers and the `gguf_value_skip` walker
// are also visible because the tokenizer reaches into raw array
// bodies that the high-level API doesn't surface yet.

#ifndef GGUF_C
#define GGUF_C

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// `oom()` allocator wrapper. Idempotent (own guard); when slm.c has
// already pulled it in, this is a no-op.
#include "utils/arrays.c"

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
    g->kvs = (struct gguf_kv *)oom(calloc((size_t)g->n_kv, kv_bytes));
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
    g->tensors = (struct gguf_tensor *)oom(calloc((size_t)g->n_tensors,
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

#endif // GGUF_C
