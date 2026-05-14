// q4k_diff.cpp - diff our Q4_K dequant block against ggml's reference.
// Reads ffn_gate.weight from the Qwen3.5 GGUF, picks the first
// super-block of row 0, dequants with BOTH paths, reports max abs diff.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define QK_K 256

typedef uint16_t ggml_fp16_t;

typedef struct {
    union {
        struct {
            ggml_fp16_t d;
            ggml_fp16_t dmin;
        } a;
        ggml_fp16_t pair[2];
    } u;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K;

extern "C" void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);

typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} q4k_block;

static inline void q4k_get_scale_min(const uint8_t * sc, int j,
                                     uint8_t * out_d, uint8_t * out_m) {
    if (j < 4) {
        *out_d = sc[j]     & 63;
        *out_m = sc[j + 4] & 63;
    } else {
        *out_d = (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4);
        *out_m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

static inline void q4k_dequant_block_mine(const q4k_block * bl, float * y) {
    const float    d   = (float)bl->d;
    const float    min = (float)bl->dmin;
    const uint8_t * qs = bl->qs;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc_q, m_q;
        q4k_get_scale_min(bl->scales, is + 0, &sc_q, &m_q);
        float d1 = d * (float)sc_q;
        float m1 = min * (float)m_q;
        q4k_get_scale_min(bl->scales, is + 1, &sc_q, &m_q);
        float d2 = d * (float)sc_q;
        float m2 = min * (float)m_q;
        for (int l = 0; l < 32; l++) {
            y[j + l]      = d1 * (float)(qs[l] & 0x0F) - m1;
            y[j + l + 32] = d2 * (float)(qs[l] >> 4)   - m2;
        }
        qs += 32;
        is += 2;
    }
}

// Minimal GGUF read.
static uint32_t rd_u32(const uint8_t * b, size_t * c) {
    uint32_t v; memcpy(&v, b+*c, 4); *c += 4; return v;
}
static uint64_t rd_u64(const uint8_t * b, size_t * c) {
    uint64_t v; memcpy(&v, b+*c, 8); *c += 8; return v;
}
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
    rd_u32(b, &c); rd_u32(b, &c);
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
    for (uint64_t i = 0; i < nt; i++) {
        uint64_t nm_n = rd_u64(b, &c);
        const char * nm = (const char *)(b + c); c += nm_n;
        uint32_t nd = rd_u32(b, &c);
        for (uint32_t d = 0; d < nd && d < 8; d++) rd_u64(b, &c);
        uint32_t ty = rd_u32(b, &c);
        uint64_t off = rd_u64(b, &c);
        if (nm_n == target_len && memcmp(nm, target, nm_n) == 0) {
            target_type = ty; target_offset = off;
        }
    }
    if (target_offset == (uint64_t)-1) {
        fprintf(stderr, "tensor not found\n"); return 1;
    }
    size_t data_start = (c + alignment - 1) & ~(size_t)(alignment - 1);
    const uint8_t * tensor_data = b + data_start + target_offset;
    if (target_type != 12) {
        fprintf(stderr, "expected Q4_K (type 12), got %u\n", target_type); return 1;
    }
    fprintf(stderr, "sizeof q4k_block (mine) = %zu  block_q4_K (ggml) = %zu\n",
            sizeof(q4k_block), sizeof(block_q4_K));

    int n_super = 4;  // diff first 4 super-blocks (= 1024 elements)
    double max_abs_diff = 0;
    int max_idx = 0;
    int max_blk = 0;
    for (int sb = 0; sb < n_super; sb++) {
        float y_mine[QK_K], y_ggml[QK_K];
        q4k_dequant_block_mine((const q4k_block *)tensor_data + sb, y_mine);
        dequantize_row_q4_K((const block_q4_K *)tensor_data + sb, y_ggml, QK_K);
        for (int i = 0; i < QK_K; i++) {
            double d = (double)y_ggml[i] - (double)y_mine[i];
            if (d < 0) d = -d;
            if (d > max_abs_diff) {
                max_abs_diff = d;
                max_idx      = i;
                max_blk      = sb;
            }
        }
    }
    fprintf(stderr, "Q4_K diff over %d super-blocks: max_abs = %g at blk=%d idx=%d\n",
            n_super, max_abs_diff, max_blk, max_idx);

    // Show first 8 of first block from both:
    float y_mine[QK_K], y_ggml[QK_K];
    q4k_dequant_block_mine((const q4k_block *)tensor_data, y_mine);
    dequantize_row_q4_K((const block_q4_K *)tensor_data, y_ggml, QK_K);
    fprintf(stderr, "first 8 ggml: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%9.6f ", y_ggml[i]);
    fprintf(stderr, "\nfirst 8 mine: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%9.6f ", y_mine[i]);
    fprintf(stderr, "\n");
    return 0;
}
