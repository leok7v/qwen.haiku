// Compare my Q8_K round-trip against llama.cpp's quantize_row_q8_K + dequant.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define QK_K 256
typedef uint16_t ggml_fp16_t;

typedef struct {
    float    d;
    int8_t   qs[QK_K];
    int16_t  bsums[QK_K/16];
} block_q8_K;

extern "C" void quantize_row_q8_K(const float * x, void * y, int64_t k);

static void q8k_round_trip_mine(const float * src, float * dst, int64_t n) {
    int64_t nb = n / QK_K;
    for (int64_t b = 0; b < nb; b++) {
        const float * sb = src + b * QK_K;
        float * db = dst + b * QK_K;
        float amax = 0.0f;
        float xmax = 0.0f;
        for (int i = 0; i < QK_K; i++) {
            float ax = fabsf(sb[i]);
            if (ax > amax) {
                amax = ax;
                xmax = sb[i];
            }
        }
        if (amax == 0.0f) {
            for (int i = 0; i < QK_K; i++) db[i] = 0.0f;
        } else {
            float iscale = -128.0f / xmax;
            float d      = 1.0f / iscale;
            for (int i = 0; i < QK_K; i++) {
                int v = (int)roundf(iscale * sb[i]);
                if (v > 127) v = 127;
                db[i] = (float)v * d;
            }
        }
    }
}

int main(void) {
    // Build a synthetic 256-element activation that resembles attn_norm-3:
    // one huge value, mixed small values.
    float x[QK_K];
    x[0] = -23.86f;
    x[1] =  3.34f;
    x[2] = -3.20f;
    for (int i = 3; i < QK_K; i++) {
        x[i] = ((float)(i * 173 % 1000) / 1000.0f - 0.5f) * 5.0f;
    }
    float y_mine[QK_K];
    q8k_round_trip_mine(x, y_mine, QK_K);

    block_q8_K y_llama;
    quantize_row_q8_K(x, &y_llama, QK_K);
    float y_llama_recon[QK_K];
    for (int i = 0; i < QK_K; i++) {
        y_llama_recon[i] = (float)y_llama.qs[i] * y_llama.d;
    }

    double max_diff = 0;
    int max_idx = 0;
    for (int i = 0; i < QK_K; i++) {
        double d = (double)y_mine[i] - (double)y_llama_recon[i];
        if (d < 0) d = -d;
        if (d > max_diff) {
            max_diff = d;
            max_idx  = i;
        }
    }
    printf("llama.cpp d=%.6f  mine d=(see)\n", y_llama.d);
    printf("first 8 ggml:  ");
    for (int i = 0; i < 8; i++) printf("%9.4f ", y_llama_recon[i]);
    printf("\nfirst 8 mine:  ");
    for (int i = 0; i < 8; i++) printf("%9.4f ", y_mine[i]);
    printf("\nmax abs diff = %g (at idx %d: ggml=%g mine=%g)\n",
           max_diff, max_idx, y_llama_recon[max_idx], y_mine[max_idx]);
    return 0;
}
