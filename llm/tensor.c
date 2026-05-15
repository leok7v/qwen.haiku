// tensor.c -- LLM-shaped fp32 tensor lib for the qwen.haiku runner.
//
// What a transformer decoder needs (no AdaIN, no iSTFT, no conv1d /
// conv-transpose, no bidir LSTM): RMSNorm, RoPE (including interleaved
// mrope for qwen35), causal-masked attention helper, Q4_0/Q4_K/Q5_K/
// Q6_K/Q8_0 weight block formats, and matmul kernels for each.
//
// Conventions:
//   * arena-allocated tensors, 64-byte aligned data
//   * eager evaluation, no graph builder
//   * single-file include from llm.c via `#include "tensor.c"`
//   * single-entry/single-exit functions, no break/continue
//
// Activations are fp32 in this first cut. The fp16-storage decision
// from PLAN.md applies to KV cache and weights; activations being
// fp32 keeps the parity-against-ggml-oracle step simple. A later
// pass can move activations to fp16 once the math is verified.

#ifndef TENSOR_C
#define TENSOR_C

#include <arm_neon.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

#define TENSOR_MAX_DIMS 4
#define TENSOR_ALIGN    64

struct arena;

struct tensor {
    int64_t        ne[TENSOR_MAX_DIMS];
    int64_t        nb[TENSOR_MAX_DIMS];
    int32_t        ndim;
    float *        data;
    struct arena * arena;
    char           name[32];
};

// Q4_0 block: 32 fp16 weights packed as
//   _Float16 scale + 32 x s4 indices (low nibble = even, high = odd).
// Total 2 + 16 = 18 bytes per 32 weights.
typedef struct {
    _Float16 scale;
    uint8_t  qs[16];
} q4_block;

#define Q4_BLOCK_SIZE 32

// ---------------------------------------------------------------------------
// Arena: page-aligned bump allocator with slab chaining.
// ---------------------------------------------------------------------------

typedef struct arena_slab {
    char *              data;
    size_t              capacity;
    size_t              mapped;
    size_t              used;
    struct arena_slab * next;
} arena_slab;

struct arena {
    arena_slab * head;
    arena_slab * first;
    size_t       initial_bytes;
};

static size_t arena_page_size(void) {
    static size_t cached = 0;
    if (cached == 0) {
        long ps = sysconf(_SC_PAGESIZE);
        cached = ps > 0 ? (size_t)ps : 4096;
    }
    return cached;
}

static void * arena_pages_alloc(size_t bytes, size_t * out_mapped) {
    const size_t pg = arena_page_size();
    size_t rounded = (bytes + pg - 1) & ~(pg - 1);
    void * result = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
    if (result == MAP_FAILED) {
        result  = NULL;
        rounded = 0;
    }
    *out_mapped = rounded;
    return result;
}

static void arena_pages_free(void * p, size_t mapped) {
    if (p != NULL && mapped > 0) {
        munmap(p, mapped);
    }
}

static arena_slab * arena_slab_new(size_t bytes) {
    arena_slab * s = (arena_slab *)calloc(1, sizeof(arena_slab));
    assert(s != NULL);
    size_t mapped = 0;
    s->data     = (char *)arena_pages_alloc(bytes, &mapped);
    assert(s->data != NULL);
    s->capacity = bytes;
    s->mapped   = mapped;
    s->used     = 0;
    s->next     = NULL;
    return s;
}

static void arena_slab_free(arena_slab * s) {
    if (s != NULL) {
        arena_pages_free(s->data, s->mapped);
        free(s);
    }
}

struct arena * arena_new(size_t initial_bytes) {
    struct arena * a = (struct arena *)calloc(1, sizeof(struct arena));
    assert(a != NULL);
    size_t sz = initial_bytes < 4096 ? 4096 : initial_bytes;
    a->initial_bytes = sz;
    a->first = arena_slab_new(sz);
    a->head  = a->first;
    return a;
}

void arena_free(struct arena * a) {
    if (a != NULL) {
        arena_slab * s = a->first;
        while (s != NULL) {
            arena_slab * next = s->next;
            arena_slab_free(s);
            s = next;
        }
        free(a);
    }
}

void arena_reset(struct arena * a) {
    assert(a != NULL);
    arena_slab * s = a->first->next;
    while (s != NULL) {
        arena_slab * next = s->next;
        arena_slab_free(s);
        s = next;
    }
    a->first->next = NULL;
    a->first->used = 0;
    a->head        = a->first;
}

static void * arena_alloc(struct arena * a, size_t bytes) {
    assert(a != NULL);
    size_t rounded = (bytes + (TENSOR_ALIGN - 1))
                     & ~((size_t)(TENSOR_ALIGN - 1));
    arena_slab * s = a->head;
    size_t aligned_used = (s->used + (TENSOR_ALIGN - 1))
                          & ~((size_t)(TENSOR_ALIGN - 1));
    if (aligned_used + rounded > s->capacity) {
        const size_t MIN_NEW = (size_t)1 << 20;
        size_t new_cap = rounded + TENSOR_ALIGN;
        if (new_cap < MIN_NEW) { new_cap = MIN_NEW; }
        arena_slab * ns = arena_slab_new(new_cap);
        s->next = ns;
        a->head = ns;
        s = ns;
        aligned_used = 0;
    }
    void * out = s->data + aligned_used;
    s->used = aligned_used + rounded;
    return out;
}

// ---------------------------------------------------------------------------
// Tensor construction / shape
// ---------------------------------------------------------------------------

static void tensor_set_packed_strides(struct tensor * t) {
    t->nb[0] = sizeof(float);
    for (int32_t i = 1; i < TENSOR_MAX_DIMS; i++) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
}

int64_t tensor_nelements(const struct tensor * t) {
    assert(t != NULL);
    int64_t n = 1;
    for (int32_t i = 0; i < TENSOR_MAX_DIMS; i++) { n *= t->ne[i]; }
    return n;
}

static struct tensor * tensor_new_nd(struct arena * a, int32_t ndim,
                                     const int64_t * ne) {
    assert(a != NULL);
    assert(1 <= ndim && ndim <= TENSOR_MAX_DIMS);
    struct tensor * t =
        (struct tensor *)arena_alloc(a, sizeof(struct tensor));
    memset(t, 0, sizeof(*t));
    for (int32_t i = 0; i < TENSOR_MAX_DIMS; i++) { t->ne[i] = 1; }
    for (int32_t i = 0; i < ndim; i++)            { t->ne[i] = ne[i]; }
    t->ndim  = ndim;
    t->arena = a;
    tensor_set_packed_strides(t);
    size_t bytes = (size_t)tensor_nelements(t) * sizeof(float);
    t->data = (float *)arena_alloc(a, bytes);
    return t;
}

struct tensor * tensor_new_1d(struct arena * a, int64_t n0) {
    int64_t ne[1] = { n0 };
    return tensor_new_nd(a, 1, ne);
}

struct tensor * tensor_new_2d(struct arena * a, int64_t n0, int64_t n1) {
    int64_t ne[2] = { n0, n1 };
    return tensor_new_nd(a, 2, ne);
}

struct tensor * tensor_new_3d(struct arena * a,
                              int64_t n0, int64_t n1, int64_t n2) {
    int64_t ne[3] = { n0, n1, n2 };
    return tensor_new_nd(a, 3, ne);
}

struct tensor * tensor_new_4d(struct arena * a,
                              int64_t n0, int64_t n1,
                              int64_t n2, int64_t n3) {
    int64_t ne[4] = { n0, n1, n2, n3 };
    return tensor_new_nd(a, 4, ne);
}

void tensor_set_name(struct tensor * t, const char * name) {
    assert(t != NULL);
    if (name != NULL) {
        size_t n = strlen(name);
        if (n >= sizeof(t->name)) { n = sizeof(t->name) - 1; }
        memcpy(t->name, name, n);
        t->name[n] = '\0';
    } else {
        t->name[0] = '\0';
    }
}

// ---------------------------------------------------------------------------
// Elementwise (small set; LLMs don't need many)
// ---------------------------------------------------------------------------

struct tensor * tensor_add(struct tensor * x, struct tensor * y) {
    assert(x->ndim == y->ndim);
    for (int32_t i = 0; i < x->ndim; i++) { assert(x->ne[i] == y->ne[i]); }
    struct tensor * out = tensor_new_nd(x->arena, x->ndim, x->ne);
    int64_t n = tensor_nelements(x);
    for (int64_t i = 0; i < n; i++) { out->data[i] = x->data[i] + y->data[i]; }
    return out;
}

struct tensor * tensor_mul(struct tensor * x, struct tensor * y) {
    assert(x->ndim == y->ndim);
    for (int32_t i = 0; i < x->ndim; i++) { assert(x->ne[i] == y->ne[i]); }
    struct tensor * out = tensor_new_nd(x->arena, x->ndim, x->ne);
    int64_t n = tensor_nelements(x);
    for (int64_t i = 0; i < n; i++) { out->data[i] = x->data[i] * y->data[i]; }
    return out;
}

// SiLU: x * sigmoid(x). Used in SwiGLU FFN.
struct tensor * tensor_silu(struct tensor * x) {
    struct tensor * out = tensor_new_nd(x->arena, x->ndim, x->ne);
    int64_t n = tensor_nelements(x);
    for (int64_t i = 0; i < n; i++) {
        float v = x->data[i];
        out->data[i] = v / (1.0f + expf(-v));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

// `table` is (vocab_size, dim). `ids` is a 1-d tensor of int-valued
// floats (we don't have a separate int tensor type; the caller writes
// integers as floats). Output is (n_ids, dim).
struct tensor * tensor_get_rows(struct tensor * table,
                                struct tensor * ids) {
    assert(table->ndim == 2 && ids->ndim == 1);
    int64_t dim = table->ne[0];
    int64_t n   = ids->ne[0];
    struct tensor * out = tensor_new_2d(table->arena, dim, n);
    for (int64_t i = 0; i < n; i++) {
        int64_t row = (int64_t)ids->data[i];
        assert(0 <= row && row < table->ne[1]);
        memcpy(out->data + i * dim,
               table->data + row * dim,
               (size_t)dim * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------------
// RMSNorm: y = x * rsqrt(mean(x*x) + eps) * weight
// ---------------------------------------------------------------------------

// `x` is (dim, seq). `weight` is (dim,). Norm is along inner dim.
struct tensor * tensor_rms_norm(struct tensor * x,
                                struct tensor * weight,
                                float           eps) {
    assert(x->ndim == 2 && weight->ndim == 1);
    assert(weight->ne[0] == x->ne[0]);
    int64_t dim = x->ne[0];
    int64_t n   = x->ne[1];
    struct tensor * out = tensor_new_2d(x->arena, dim, n);
    for (int64_t t = 0; t < n; t++) {
        const float * xr = x->data + t * dim;
        float * or_ = out->data + t * dim;
        double ssq = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            ssq += (double)(xr[i] * xr[i]);
        }
        float rscale = 1.0f / sqrtf((float)(ssq / (double)dim) + eps);
        for (int64_t i = 0; i < dim; i++) {
            or_[i] = xr[i] * rscale * weight->data[i];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// RoPE: rotary position embedding
// ---------------------------------------------------------------------------

// Interleaved-mrope (qwen35 / qwen3-vl convention).
// Ported from ggml-cpu/ops.cpp: ggml_mrope_cache_init + rotate_pairs.
//
// Dim-pair convention: NEOX-style; pair is (dim i, dim i + rotary_dim/2),
// NOT (dim 2i, dim 2i+1). Iterating i from 0 to rotary_dim/2 - 1.
//
// Section assignment under is_imrope: sector = i % 3.
//   sector 0 -> axis T (text position), if i < sections[0]*1, but the
//             cap test in the kernel is `sector < 3 * sections[0]`,
//             matching the cyclic count of T pairs across all sectors.
//   sector 1 -> axis H, if i (as sector value) < 3*sections[1].
//   sector 2 -> axis W, if i < 3*sections[2].
//   otherwise -> axis E (extra; 0 for text).
//
// For text-only with mrope_section [11, 11, 10, 0] and pos_text=P:
//   axis_T = P, axis_H = 0, axis_W = 0, axis_E = 0.
//   Pairs with axis position = 0 don't rotate (cos=1, sin=0).
//
// `x` shape: (head_dim, n_heads, seq).
struct tensor * tensor_rope_mrope_i(struct tensor * x,
                                    int64_t         pos_offset,
                                    float           base,
                                    int64_t         rotary_dim,
                                    const int32_t   sections[4]) {
    assert(x->ndim == 3);
    int64_t hd = x->ne[0];
    int64_t nh = x->ne[1];
    int64_t ns = x->ne[2];
    if (rotary_dim <= 0 || rotary_dim > hd) { rotary_dim = hd; }
    assert(rotary_dim % 2 == 0);
    struct tensor * out = tensor_new_3d(x->arena, hd, nh, ns);
    int64_t half = rotary_dim / 2;
    int32_t sec0 = sections[0];
    int32_t sec1 = sections[1];
    int32_t sec2 = sections[2];
    for (int64_t s = 0; s < ns; s++) {
        int64_t pos_t = pos_offset + s;
        for (int64_t h = 0; h < nh; h++) {
            const float * xr = x->data + (s * nh + h) * hd;
            float * or_      = out->data + (s * nh + h) * hd;
            for (int64_t i = rotary_dim; i < hd; i++) { or_[i] = xr[i]; }
            // Recompute powf per-pair to avoid drift from incremental
            // `scale *= theta_step`.
            for (int64_t i = 0; i < half; i++) {
                int32_t sector = (int32_t)i;
                float pos_axis;
                if      (sector % 3 == 0 && sector < 3 * sec0) { pos_axis = (float)pos_t; }
                else if (sector % 3 == 1 && sector < 3 * sec1) { pos_axis = 0.0f; }
                else if (sector % 3 == 2 && sector < 3 * sec2) { pos_axis = 0.0f; }
                else                                           { pos_axis = 0.0f; }
                float theta_scale = powf(base, -2.0f * (float)i / (float)rotary_dim);
                float theta = pos_axis * theta_scale;
                float c  = cosf(theta);
                float sn = sinf(theta);
                float x0 = xr[i];
                float x1 = xr[i + half];
                or_[i]        = x0 * c  - x1 * sn;
                or_[i + half] = x0 * sn + x1 * c;
            }
        }
    }
    return out;
}

// Legacy single-axis RoPE kept for callers that don't need mrope
// (none in the current llm.c, but useful for synthetic-test models).
struct tensor * tensor_rope(struct tensor * x,
                            int64_t         pos_offset,
                            float           base,
                            int64_t         rotary_dim) {
    assert(x->ndim == 3);
    int64_t hd = x->ne[0];
    int64_t nh = x->ne[1];
    int64_t ns = x->ne[2];
    if (rotary_dim <= 0 || rotary_dim > hd) { rotary_dim = hd; }
    assert(rotary_dim % 2 == 0);
    struct tensor * out = tensor_new_3d(x->arena, hd, nh, ns);
    int64_t half = rotary_dim / 2;
    for (int64_t s = 0; s < ns; s++) {
        int64_t pos = pos_offset + s;
        for (int64_t h = 0; h < nh; h++) {
            const float * xr = x->data + (s * nh + h) * hd;
            float * or_      = out->data + (s * nh + h) * hd;
            for (int64_t i = 0; i < half; i++) {
                float theta = (float)pos
                    * powf(base, -2.0f * (float)i / (float)rotary_dim);
                float c  = cosf(theta);
                float sn = sinf(theta);
                float x0 = xr[i];
                float x1 = xr[i + half];
                or_[i]        = x0 * c  - x1 * sn;
                or_[i + half] = x0 * sn + x1 * c;
            }
            for (int64_t i = rotary_dim; i < hd; i++) { or_[i] = xr[i]; }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Softmax (inner axis, with optional causal mask via -inf)
// ---------------------------------------------------------------------------

// `x` is (n, b...). Softmax along axis 0 in place is unsafe (arena
// owns memory), so we produce a new tensor.
struct tensor * tensor_softmax(struct tensor * x) {
    struct tensor * out = tensor_new_nd(x->arena, x->ndim, x->ne);
    int64_t n     = x->ne[0];
    int64_t outer = tensor_nelements(x) / n;
    for (int64_t b = 0; b < outer; b++) {
        const float * xr = x->data + b * n;
        float * or_ = out->data + b * n;
        float m = xr[0];
        for (int64_t i = 1; i < n; i++) { if (xr[i] > m) { m = xr[i]; } }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            float v = expf(xr[i] - m);
            or_[i]  = v;
            sum    += v;
        }
        float inv = 1.0f / sum;
        for (int64_t i = 0; i < n; i++) { or_[i] *= inv; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Matmul (fp32 baseline)
//
// `w` is (in_features, out_features), packed row-major in our layout
// (matching the project-wide mul_mat convention: nb[0]=sizeof(float),
// nb[1]=in_features*sizeof(float)). `x` is (in_features, n_tokens).
// Output is (out_features, n_tokens).
//
// Plain triple-loop; clang -O3 will SIMD-ise this for the common
// shapes (out_features and in_features multiples of 8). Good enough
// for the Phase 1 fp32 path. Phase 4 swaps this for the Q4 kernel
// for the big weight matrices; this version stays for any tensor
// product that legitimately has both operands in fp32 (lm_head
// against fp32 logits, etc.).
// ---------------------------------------------------------------------------

struct tensor * tensor_matmul_f32(struct tensor * w, struct tensor * x) {
    assert(w->ndim == 2 && x->ndim == 2);
    assert(w->ne[0] == x->ne[0]);
    int64_t k     = w->ne[0];
    int64_t out_f = w->ne[1];
    int64_t n     = x->ne[1];
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    for (int64_t t = 0; t < n; t++) {
        const float * xr = x->data + t * k;
        float * or_      = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const float * wr = w->data + j * k;
            float acc = 0.0f;
            for (int64_t i = 0; i < k; i++) { acc += wr[i] * xr[i]; }
            or_[j] = acc;
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// Matmul (Q4_0 weights * fp32 activations). Tier 1.
//
// Each output row j has (k / Q4_BLOCK_SIZE) blocks of 18 bytes each.
// Per block: read scale, unpack 32 * s4 -> fp32, multiply against the
// next 32 activations, accumulate into fp32 sum.
//
// Clang -O3 will autovec the inner block loop on NEON. Tier-3 (manual
// intrinsics + Q8_0 activations + vdot) is deferred per PLAN.md until
// measurement shows it's needed.
// ---------------------------------------------------------------------------

struct tensor * tensor_matmul_q4_f32(const q4_block * w_blocks,
                                     int64_t          out_f,
                                     int64_t          k,
                                     struct tensor *  x) {
    assert(x->ndim == 2 && x->ne[0] == k);
    assert(k % Q4_BLOCK_SIZE == 0);
    int64_t n          = x->ne[1];
    int64_t nb_per_row = k / Q4_BLOCK_SIZE;
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    for (int64_t t = 0; t < n; t++) {
        const float * xr = x->data + t * k;
        float * or_      = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const q4_block * row = w_blocks + j * nb_per_row;
            float acc = 0.0f;
            for (int64_t b = 0; b < nb_per_row; b++) {
                const q4_block * bl = row + b;
                float scale = (float)bl->scale;
                const float * xb = xr + b * Q4_BLOCK_SIZE;
                float bsum = 0.0f;
                // 16 packed bytes -> 32 weights (low nibble = even).
                for (int32_t i = 0; i < 16; i++) {
                    uint8_t byte = bl->qs[i];
                    int8_t  lo = (int8_t)((byte & 0x0f)) - 8;
                    int8_t  hi = (int8_t)((byte >> 4))   - 8;
                    bsum += (float)lo * xb[2 * i];
                    bsum += (float)hi * xb[2 * i + 1];
                }
                acc += bsum * scale;
            }
            or_[j] = acc;
        });
    }
    return out;
}

// Pack helper: produce a Q4_0 row from `k` fp32 weights into `out`,
// which must point at `k / 32` consecutive q4_block records. Used by
// --self-test (to fabricate deterministic synthetic weights). Real
// weights come from a Q4_K_M GGUF; see q4k / q6k block types below.
void q4_pack_row(const float * src, int64_t k, q4_block * out) {
    assert(k % Q4_BLOCK_SIZE == 0);
    int64_t nb = k / Q4_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const float * sb = src + b * Q4_BLOCK_SIZE;
        float amax = 0.0f;
        for (int32_t i = 0; i < Q4_BLOCK_SIZE; i++) {
            float v = fabsf(sb[i]);
            if (v > amax) { amax = v; }
        }
        float scale = amax / 7.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = (_Float16)scale;
        for (int32_t i = 0; i < 16; i++) {
            int32_t lo = (int32_t)roundf(sb[2 * i]     * inv);
            int32_t hi = (int32_t)roundf(sb[2 * i + 1] * inv);
            if (lo < -8) { lo = -8; }
            if (lo >  7) { lo =  7; }
            if (hi < -8) { hi = -8; }
            if (hi >  7) { hi =  7; }
            out[b].qs[i] = (uint8_t)(((lo + 8) & 0x0f) | (((hi + 8) & 0x0f) << 4));
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF k-quant block types (Q4_K, Q6_K) for Qwen3.5-0.8B-Q4_K_M.gguf
//
// Q4_K: 256 weights per super-block. 8 sub-blocks of 32 weights. Each
// sub-block has a 6-bit scale and 6-bit min, packed into 12 bytes. The
// super-block has fp16 d (scale-of-scales) and fp16 dmin (scale-of-mins).
//   weight(i) = d * scale[sub] * q4[i] - dmin * min[sub]
// Layout from llama.cpp ggml-quants.h:
//   struct block_q4_K {
//     ggml_half d;          // 2 bytes
//     ggml_half dmin;       // 2 bytes
//     uint8_t   scales[12]; // packed 6-bit scales + 6-bit mins
//     uint8_t   qs[128];    // 4-bit quants for 256 weights
//   };
// Total: 144 bytes per 256 weights = 4.5 bits per weight + overhead.
//
// Q6_K: 256 weights per super-block. 6-bit quants split as ql (lower 4)
// + qh (upper 2). 16 sub-blocks of 16 weights, each with an int8 scale.
//   struct block_q6_K {
//     uint8_t   ql[128];    // 4 low bits per weight (256 weights)
//     uint8_t   qh[64];     // 2 high bits per weight (256 weights)
//     int8_t    scales[16]; // per-sub-block scales (int8)
//     ggml_half d;          // super-block scale
//   };
// Total: 128 + 64 + 16 + 2 = 210 bytes per 256 weights = ~6.56 bpw.
// ---------------------------------------------------------------------------

#define QK_K 256

// Q8_K activation quantization round-trip, matching llama.cpp's
// internal MUL_MAT path. Activations are quantized to Q8_K (256-elem
// blocks of int8 + per-block fp32 scale) before the int*int dot
// product with Q4_K/Q5_K/Q6_K weights. The model was imatrix-
// calibrated to expect this quantization noise; using full fp32
// activations produces outputs that diverge from the trained model
// (verified empirically against llama-eval-callback: ~30% mismatch
// on attn_v output without round-trip, ~1% with).
//
// This helper does the round-trip in fp32: quantize, then
// dequantize, so the rest of the matmul kernel can keep operating
// in fp32 and still produce numerics equivalent to llama.cpp's int8
// dot product (modulo accumulation order, sub-1% drift).
//
// Per llama.cpp's quantize_row_q8_K_ref: iscale = -128/max where
// max is the SIGNED value of the largest-|x|. Asymmetric: dominant
// magnitude lands at -128, opposite-sign extreme at +127. Range
// [-128, 127]. We use the same convention for byte-for-byte parity.
extern int g_no_q8k_rt;
// Banker's rounding to match llama.cpp's nearest_int() exactly.
// Half-values round to even; differs from roundf which rounds half away
// from zero. The off-by-1 from roundf vs nearest_int compounds through
// matmul -> measurable drift in deeper layers.
static inline int32_t q8k_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int32_t i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}
// Q8_K activation block. Matches ggml's block_q8_K layout so the
// NEON int8 dot kernels port cleanly. Built by quantize_row_q8_K
// from an fp32 activation row; consumed by q4k_dot_q8k_neon (and
// later q5k / q6k variants).
typedef struct {
    float   d;                 // 1 / iscale where iscale = -127/xmax
    int8_t  qs[QK_K];          // 256 int8 weights
    int16_t bsums[QK_K / 16];  // 16 pre-computed sums of 16-element chunks
} q8k_block;

// Quantize an fp32 row into Q8_K blocks (int8 + per-block fp32 scale
// + pre-computed 16-wide sub-block sums). Mirrors llama.cpp's
// quantize_row_q8_K_ref. Output is the NEON-friendly format consumed
// by q4k_dot_q8k_neon / q5k_dot_q8k_neon / q6k_dot_q8k_neon.
static void quantize_row_q8_K(const float * src, q8k_block * dst,
                              int64_t n) {
    int64_t nb = n / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const float * sb = src + b * QK_K;
        q8k_block * y = dst + b;
        float amax = 0.0f;
        float xmax = 0.0f;
        for (int32_t i = 0; i < QK_K; i++) {
            float ax = fabsf(sb[i]);
            if (ax > amax) {
                amax = ax;
                xmax = sb[i];
            }
        }
        if (amax == 0.0f) {
            y->d = 0.0f;
            memset(y->qs,    0, sizeof(y->qs));
            memset(y->bsums, 0, sizeof(y->bsums));
        } else {
            float iscale = -127.0f / xmax;
            y->d = 1.0f / iscale;
            for (int32_t i = 0; i < QK_K; i++) {
                int32_t v = q8k_nearest_int(iscale * sb[i]);
                if (v > 127)  { v = 127;  }
                if (v < -128) { v = -128; }
                y->qs[i] = (int8_t)v;
            }
            for (int32_t j = 0; j < QK_K / 16; j++) {
                int32_t sum = 0;
                for (int32_t l = 0; l < 16; l++) {
                    sum += y->qs[j * 16 + l];
                }
                y->bsums[j] = (int16_t)sum;
            }
        }
    }
}

static inline void q8k_round_trip(const float * src, float * dst,
                                  int64_t n) {
    if (g_no_q8k_rt) {
        for (int64_t i = 0; i < n; i++) { dst[i] = src[i]; }
        return;
    }
    int64_t nb = n / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const float * sb = src + b * QK_K;
        float * db = dst + b * QK_K;
        float amax = 0.0f;
        float xmax = 0.0f;
        for (int32_t i = 0; i < QK_K; i++) {
            float ax = fabsf(sb[i]);
            if (ax > amax) {
                amax = ax;
                xmax = sb[i];
            }
        }
        if (amax == 0.0f) {
            for (int32_t i = 0; i < QK_K; i++) { db[i] = 0.0f; }
        } else {
            // -127.f/max per llama.cpp/ggml-quants.c (the -128.f form
            // was commented out for IQ2_XXS AVX compatibility; the
            // released version uses -127).
            float iscale = -127.0f / xmax;
            float d      = 1.0f / iscale;
            for (int32_t i = 0; i < QK_K; i++) {
                int32_t v = q8k_nearest_int(iscale * sb[i]);
                if (v > 127)  { v = 127;  }
                if (v < -128) { v = -128; }
                db[i] = (float)v * d;
            }
        }
    }
}

typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} q4k_block;

typedef struct {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    _Float16 d;
} q6k_block;

typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} q5k_block;

typedef struct {
    _Float16 d;
    int8_t   qs[32];
} q8_0_block;

#define Q8_0_BLOCK_SIZE 32

// NEON int8 dot kernels (and ggml-ported elementwise helpers like
// silu/exp) live in neon.c, included here as a single-file library
// so all block typedefs above are in scope. neon.c is the ONLY place
// in this codebase that includes <arm_neon.h>; tensor.c and llm.c
// stay scalar / portable. To add another SIMD variant (avx2.c
// later), add a sibling file and switch the include with #if.
#include "neon.c"

// Unpack 8 * (6-bit scale, 6-bit min) from the 12-byte `scales` field.
// Layout per llama.cpp dequantize_row_q4_K reference:
//   For sub-block j in [0,4):  sc = scales[j]   & 63
//                              m  = scales[j+4] & 63
//   For sub-block j in [4,8):  sc = (scales[j+4] & 0x0f)
//                                  | ((scales[j-4] >> 6) << 4)
//                              m  = (scales[j+4] >> 4)
//                                  | ((scales[j]   >> 6) << 4)
static inline void q4k_get_scale_min(const uint8_t * sc, int32_t j,
                                     uint8_t * out_d, uint8_t * out_m) {
    if (j < 4) {
        *out_d = sc[j]     & 63;
        *out_m = sc[j + 4] & 63;
    } else {
        *out_d = (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4);
        *out_m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

// Q4_K dequant: port of ggml-quants.c dequantize_row_q4_K. Per
// super-block (256 weights) iterate 4 chunks of 64. Each chunk uses
// TWO scales (is+0, is+1): low nibbles of qs[0..31] use scale is+0,
// high nibbles use scale is+1. qs advances 32 and is advances 2.
static inline void q4k_dequant_block(const q4k_block * bl, float * y) {
    const float     d   = (float)bl->d;
    const float     min = (float)bl->dmin;
    const uint8_t * qs  = bl->qs;
    int32_t         is  = 0;
    for (int32_t j = 0; j < QK_K; j += 64) {
        uint8_t sc_q, m_q;
        q4k_get_scale_min(bl->scales, is + 0, &sc_q, &m_q);
        float d1 = d   * (float)sc_q;
        float m1 = min * (float)m_q;
        q4k_get_scale_min(bl->scales, is + 1, &sc_q, &m_q);
        float d2 = d   * (float)sc_q;
        float m2 = min * (float)m_q;
        for (int32_t l = 0; l < 32; l++) {
            y[j + l]      = d1 * (float)(qs[l] & 0x0F) - m1;
            y[j + l + 32] = d2 * (float)(qs[l] >> 4)   - m2;
        }
        qs += 32;
        is += 2;
    }
}

// q4k_dot_q8k_neon, q5k_dot_q8k_neon, q6k_dot_q8k_neon - bodies live
// in neon.c (included near the top of this file). Comments and
// licensing context for each kernel are at the function definition
// in neon.c.

struct tensor * tensor_matmul_q4k_f32(const q4k_block * w_blocks,
                                      int64_t           out_f,
                                      int64_t           k,
                                      struct tensor *   x) {
    assert(x->ndim == 2 && x->ne[0] == k);
    assert(k % QK_K == 0);
    int64_t n          = x->ne[1];
    int64_t nb_per_row = k / QK_K;
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    // Per-call Q8_K activation buffer. Arena-allocated rather than
    // stack so the dispatch_apply block can refer to it via pointer
    // (clang forbids array references inside blocks).
    q8k_block * xq8 = (q8k_block *)arena_alloc(
        x->arena, (size_t)nb_per_row * sizeof(q8k_block));
    for (int64_t t = 0; t < n; t++) {
        quantize_row_q8_K(x->data + t * k, xq8, k);
        float * or_ = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const q4k_block * row = w_blocks + j * nb_per_row;
            float acc = 0.0f;
            for (int64_t b = 0; b < nb_per_row; b++) {
                acc += q4k_dot_q8k_neon(row + b, xq8 + b);
            }
            or_[j] = acc;
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// Q5_K dequant (block typedefs for Q5_K and Q8_0 live near q4k_block
// at the top of the quant section so neon.c sees them; see comment
// there for the layout reference).
//
// Q5_K dequant; port of ggml-quants.c dequantize_row_q5_K.
// Per super-block (256 weights), iterate 4 chunks of 64. Each chunk:
//   sub-block 'is':  d1 = d*sc[is], m1 = dmin*m[is].
//   sub-block 'is+1': d2 = d*sc[is+1], m2 = dmin*m[is+1].
//   First 32 weights of chunk: y = d1 * (qs[l] & 0xF + (qh[l] & u1 ? 16:0)) - m1
//   Second 32:                 y = d2 * (qs[l] >> 4  + (qh[l] & u2 ? 16:0)) - m2
//   qs += 32, is += 2, u1 <<= 2, u2 <<= 2.
static inline void q5k_dequant_block(const q5k_block * bl, float * y) {
    const float     d   = (float)bl->d;
    const float     min = (float)bl->dmin;
    const uint8_t * qs  = bl->qs;
    const uint8_t * qh  = bl->qh;
    int32_t         is  = 0;
    uint8_t         u1  = 1;
    uint8_t         u2  = 2;
    for (int32_t j = 0; j < QK_K; j += 64) {
        uint8_t sc_q, m_q;
        q4k_get_scale_min(bl->scales, is + 0, &sc_q, &m_q);
        float d1 = d   * (float)sc_q;
        float m1 = min * (float)m_q;
        q4k_get_scale_min(bl->scales, is + 1, &sc_q, &m_q);
        float d2 = d   * (float)sc_q;
        float m2 = min * (float)m_q;
        for (int32_t l = 0; l < 32; l++) {
            int32_t q = (qs[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            y[j + l] = d1 * (float)q - m1;
        }
        for (int32_t l = 0; l < 32; l++) {
            int32_t q = (qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            y[j + l + 32] = d2 * (float)q - m2;
        }
        qs += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

struct tensor * tensor_matmul_q5k_f32(const q5k_block * w_blocks,
                                      int64_t           out_f,
                                      int64_t           k,
                                      struct tensor *   x) {
    assert(x->ndim == 2 && x->ne[0] == k);
    assert(k % QK_K == 0);
    int64_t n          = x->ne[1];
    int64_t nb_per_row = k / QK_K;
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    q8k_block * xq8 = (q8k_block *)arena_alloc(
        x->arena, (size_t)nb_per_row * sizeof(q8k_block));
    for (int64_t t = 0; t < n; t++) {
        quantize_row_q8_K(x->data + t * k, xq8, k);
        float * or_ = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const q5k_block * row = w_blocks + j * nb_per_row;
            float acc = 0.0f;
            for (int64_t b = 0; b < nb_per_row; b++) {
                acc += q5k_dot_q8k_neon(row + b, xq8 + b);
            }
            or_[j] = acc;
        });
    }
    return out;
}

struct tensor * tensor_matmul_q8_0_f32(const q8_0_block * w_blocks,
                                       int64_t            out_f,
                                       int64_t            k,
                                       struct tensor *    x) {
    assert(x->ndim == 2 && x->ne[0] == k);
    assert(k % Q8_0_BLOCK_SIZE == 0);
    // Quantise input row to Q8_0 once per token then dot int8*int8
    // per output row - mirrors ggml's GGML_OP_MUL_MAT dispatch for
    // Q8_0 weights x F32 activations (activation is quantised to its
    // matching `vec_dot_type`, which for Q8_0 is also Q8_0). The
    // previous scalar fp64 path multiplied weights' int8 directly by
    // un-quantised fp32 activations - mathematically different from
    // what the model was trained for and what llama.cpp computes;
    // produced ~50-100 ULPs of drift on ssm_alpha / ssm_beta outputs.
    int64_t n          = x->ne[1];
    int64_t nb_per_row = k / Q8_0_BLOCK_SIZE;
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    q8_0_block * xq = (q8_0_block *)arena_alloc(
        x->arena, (size_t)nb_per_row * sizeof(q8_0_block));
    for (int64_t t = 0; t < n; t++) {
        quantize_row_q8_0(x->data + t * k, xq, k);
        float * or_ = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const q8_0_block * row = w_blocks + j * nb_per_row;
            or_[j] = q8_0_dot_q8_0_neon(row, xq, k);
        });
    }
    return out;
}

// Q6_K dequant; port of ggml-quants.c dequantize_row_q6_K.
// Per super-block (256 weights): 128-element half processed at a time,
// with l in [0..32) producing q1,q2,q3,q4 written to y[l], y[l+32],
// y[l+64], y[l+96]. Scales picked as sc[is], sc[is+2], sc[is+4],
// sc[is+6] with is = l/16.
static inline void q6k_dequant_block(const q6k_block * bl, float * y) {
    const float     d  = (float)bl->d;
    const uint8_t * ql = bl->ql;     // 128 bytes
    const uint8_t * qh = bl->qh;     //  64 bytes
    const int8_t  * sc = bl->scales; //  16 ints
    for (int32_t half = 0; half < 2; half++) {
        for (int32_t l = 0; l < 32; l++) {
            int32_t is = l / 16;
            int32_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int32_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int32_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int32_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        ql += 64;
        qh += 32;
        sc +=  8;
        y  += 128;
    }
}

struct tensor * tensor_matmul_q6k_f32(const q6k_block * w_blocks,
                                      int64_t           out_f,
                                      int64_t           k,
                                      struct tensor *   x) {
    assert(x->ndim == 2 && x->ne[0] == k);
    assert(k % QK_K == 0);
    int64_t n          = x->ne[1];
    int64_t nb_per_row = k / QK_K;
    struct tensor * out = tensor_new_2d(x->arena, out_f, n);
    q8k_block * xq8 = (q8k_block *)arena_alloc(
        x->arena, (size_t)nb_per_row * sizeof(q8k_block));
    for (int64_t t = 0; t < n; t++) {
        quantize_row_q8_K(x->data + t * k, xq8, k);
        float * or_ = out->data + t * out_f;
        dispatch_apply((size_t)out_f, DISPATCH_APPLY_AUTO, ^(size_t jj) {
            int64_t j = (int64_t)jj;
            const q6k_block * row = w_blocks + j * nb_per_row;
            float acc = 0.0f;
            for (int64_t b = 0; b < nb_per_row; b++) {
                acc += q6k_dot_q8k_neon(row + b, xq8 + b);
            }
            or_[j] = acc;
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// Attention helper: scaled dot-product, causal mask, GQA-aware
//
// q   : (head_dim, n_heads,    n_q)
// k   : (head_dim, n_kv_heads, n_kv)
// v   : (head_dim, n_kv_heads, n_kv)
//
// returns: (head_dim, n_heads, n_q)
//
// Causal: for the q'th query token (q in [0..n_q)), it can attend to
// the first (k_offset + q + 1) key tokens. `k_offset` is how many
// tokens are in the KV cache before this call (0 on prefill).
// ---------------------------------------------------------------------------

struct tensor * tensor_attention(struct tensor * q,
                                 struct tensor * k,
                                 struct tensor * v,
                                 int64_t         k_offset) {
    assert(q->ndim == 3 && k->ndim == 3 && v->ndim == 3);
    int64_t hd       = q->ne[0];
    int64_t n_heads  = q->ne[1];
    int64_t n_q      = q->ne[2];
    int64_t n_kvh    = k->ne[1];
    int64_t n_kv     = k->ne[2];
    assert(v->ne[0] == hd && v->ne[1] == n_kvh && v->ne[2] == n_kv);
    assert(n_heads % n_kvh == 0);
    int64_t group = n_heads / n_kvh;
    float   scale = 1.0f / sqrtf((float)hd);
    struct tensor * out = tensor_new_3d(q->arena, hd, n_heads, n_q);
    // Scratch for one row of scores per (head, q_token).
    float * scores = (float *)arena_alloc(q->arena,
                                          (size_t)n_kv * sizeof(float));
    for (int64_t t = 0; t < n_q; t++) {
        int64_t kv_max = k_offset + t + 1;
        for (int64_t h = 0; h < n_heads; h++) {
            int64_t kh = h / group;
            const float * qv = q->data + (t * n_heads + h) * hd;
            // scores[i] = <q, k_i>
            for (int64_t i = 0; i < kv_max; i++) {
                const float * kv = k->data + (i * n_kvh + kh) * hd;
                float dot = 0.0f;
                for (int64_t d = 0; d < hd; d++) { dot += qv[d] * kv[d]; }
                scores[i] = dot * scale;
            }
            // softmax(scores[0..kv_max])
            float m = scores[0];
            for (int64_t i = 1; i < kv_max; i++) {
                if (scores[i] > m) { m = scores[i]; }
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < kv_max; i++) {
                scores[i] = expf(scores[i] - m);
                sum      += scores[i];
            }
            float inv = 1.0f / sum;
            for (int64_t i = 0; i < kv_max; i++) { scores[i] *= inv; }
            // out = sum_i scores[i] * v_i
            float * o = out->data + (t * n_heads + h) * hd;
            for (int64_t d = 0; d < hd; d++) { o[d] = 0.0f; }
            for (int64_t i = 0; i < kv_max; i++) {
                const float * vv = v->data + (i * n_kvh + kh) * hd;
                float w = scores[i];
                for (int64_t d = 0; d < hd; d++) { o[d] += w * vv[d]; }
            }
        }
    }
    return out;
}

#endif // TENSOR_C
