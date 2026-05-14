// q6k_diff.c -- diff our Q6_K dequant against ggml's.
//
// Reads attn_v.weight (blk.3) from the qwen35 GGUF, dequantizes the
// first row with BOTH my dequant and ggml's, and reports element-wise
// max absolute diff plus the first few values from each.

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// llama.cpp's ggml header (just for type forward-decls).
typedef uint16_t ggml_fp16_t;

// ggml-quants.h block layout (verbatim).
#define QK_K 256
typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    ggml_fp16_t d;
} block_q6_K;

// ggml's dequant - link via static lib (C linkage).
extern "C" void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);

// Our dequant - extracted from llm/tensor.c.
typedef struct {
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    _Float16 d;
} q6k_block;

static inline void q6k_dequant_block(const q6k_block * bl, float * y) {
    const float d = (float)bl->d;
    const uint8_t * ql = bl->ql;
    const uint8_t * qh = bl->qh;
    const int8_t  * sc = bl->scales;
    for (int half = 0; half < 2; half++) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        ql += 64; qh += 32; sc += 8; y += 128;
    }
}

// Minimal GGUF reader - just enough to find a tensor by name and
// return its data pointer.
static uint32_t rd_u32(const uint8_t * b, size_t * c)
    { uint32_t v; memcpy(&v, b+*c, 4); *c += 4; return v; }
static uint64_t rd_u64(const uint8_t * b, size_t * c)
    { uint64_t v; memcpy(&v, b+*c, 8); *c += 8; return v; }

static size_t skip_v(const uint8_t * b, size_t c, uint32_t t) {
    size_t e = c;
    switch (t) {
        case 0:
        case 1:
        case 7:
            e += 1;
            break;
        case 2:
        case 3:
            e += 2;
            break;
        case 4:
        case 5:
        case 6:
            e += 4;
            break;
        case 10:
        case 11:
        case 12:
            e += 8;
            break;
        case 8: {
            uint64_t n = rd_u64(b, &e);
            e += n;
            break;
        }
        case 9: {
            uint32_t at = rd_u32(b, &e);
            uint64_t an = rd_u64(b, &e);
            for (uint64_t i = 0; i < an; i++) {
                e = skip_v(b, e, at);
            }
            break;
        }
    }
    return e;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s file.gguf tensor_name\n", argv[0]);
        return 1;
    }
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    const uint8_t * b = (const uint8_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size_t c = 0;
    rd_u32(b, &c);  // magic
    rd_u32(b, &c);  // ver
    uint64_t nt = rd_u64(b, &c);
    uint64_t nk = rd_u64(b, &c);
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < nk; i++) {
        uint64_t nm_n = rd_u64(b, &c);
        const char * nm = (const char *)(b + c); c += nm_n;
        uint32_t ty = rd_u32(b, &c);
        if (nm_n == strlen("general.alignment") &&
            memcmp(nm, "general.alignment", nm_n) == 0 && ty == 4) {
            size_t cc = c; alignment = rd_u32(b, &cc);
        }
        c = skip_v(b, c, ty);
    }
    const char * target = argv[2];
    size_t target_len = strlen(target);
    uint32_t target_type = 0;
    uint64_t target_offset = (uint64_t)-1;
    uint64_t target_shape0 = 0, target_shape1 = 0;
    for (uint64_t i = 0; i < nt; i++) {
        uint64_t nm_n = rd_u64(b, &c);
        const char * nm = (const char *)(b + c); c += nm_n;
        uint32_t nd = rd_u32(b, &c);
        uint64_t shape[8] = {0};
        for (uint32_t d = 0; d < nd && d < 8; d++) shape[d] = rd_u64(b, &c);
        uint32_t ty = rd_u32(b, &c);
        uint64_t off = rd_u64(b, &c);
        if (nm_n == target_len && memcmp(nm, target, nm_n) == 0) {
            target_type   = ty;
            target_offset = off;
            target_shape0 = shape[0];
            target_shape1 = shape[1];
        }
    }
    if (target_offset == (uint64_t)-1) {
        fprintf(stderr, "tensor not found: %s\n", target);
        return 1;
    }
    size_t data_start = (c + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t * tensor_data = b + data_start + target_offset;
    printf("tensor: %s  shape=[%llu,%llu]  type=%u  abs_offset=%zu\n",
           target, (unsigned long long)target_shape0,
           (unsigned long long)target_shape1, target_type,
           data_start + target_offset);

    if (target_type != 14) {
        fprintf(stderr, "expected Q6_K (type 14), got %u\n", target_type);
        return 1;
    }

    // Dequant first super-block with both.
    float y_mine[QK_K];
    float y_ggml[QK_K];
    q6k_dequant_block((const q6k_block *)tensor_data, y_mine);
    dequantize_row_q6_K((const block_q6_K *)tensor_data, y_ggml, QK_K);

    double max_abs_diff = 0.0;
    int    max_idx = 0;
    for (int i = 0; i < QK_K; i++) {
        double d = (double)y_ggml[i] - (double)y_mine[i];
        if (d < 0) d = -d;
        if (d > max_abs_diff) {
            max_abs_diff = d;
            max_idx      = i;
        }
    }
    printf("first 8 ggml: ");
    for (int i = 0; i < 8; i++) printf("%9.6f ", y_ggml[i]);
    printf("\nfirst 8 mine: ");
    for (int i = 0; i < 8; i++) printf("%9.6f ", y_mine[i]);
    printf("\nmax abs diff = %g (at idx %d: ggml=%g mine=%g)\n",
           max_abs_diff, max_idx, y_ggml[max_idx], y_mine[max_idx]);

    return 0;
}
