// quick & dirty: dump all KV keys from the GGUF for diagnostic.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t gr_u32(const uint8_t * b, size_t * c)
    { uint32_t v; memcpy(&v, b + *c, 4); *c += 4; return v; }
static uint64_t gr_u64(const uint8_t * b, size_t * c)
    { uint64_t v; memcpy(&v, b + *c, 8); *c += 8; return v; }

static size_t skip(const uint8_t * b, size_t c, uint32_t t) {
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
            uint64_t n = gr_u64(b, &e);
            e += n;
            break;
        }
        case 9: {
            uint32_t at = gr_u32(b, &e);
            uint64_t an = gr_u64(b, &e);
            for (uint64_t i = 0; i < an; i++) {
                e = skip(b, e, at);
            }
            break;
        }
    }
    return e;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s file.gguf [--tensors]\n", argv[0]);
        return 1;
    }
    int show_tensors = (argc > 2 && strcmp(argv[2], "--tensors") == 0);
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    const uint8_t * b = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size_t c = 0;
    uint32_t magic = gr_u32(b, &c);
    uint32_t ver   = gr_u32(b, &c);
    uint64_t nt    = gr_u64(b, &c);
    uint64_t nk    = gr_u64(b, &c);
    printf("magic=%08x ver=%u tensors=%llu kvs=%llu\n",
           magic, ver, (unsigned long long)nt, (unsigned long long)nk);
    for (uint64_t i = 0; i < nk; i++) {
        uint64_t n = gr_u64(b, &c);
        printf("  %.*s", (int)n, (const char *)(b + c));
        c += n;
        uint32_t t = gr_u32(b, &c);
        printf("  type=%u", t);
        if (t == 8) {
            // string value
            uint64_t sn = gr_u64(b, &c);
            printf("  = \"%.*s\"", (int)(sn < 100 ? sn : 100),
                   (const char *)(b + c));
            if (sn > 100) printf("...");
            c += sn;
        } else if (t == 4) {
            uint32_t v = gr_u32(b, &c);
            printf("  = %u", v);
        } else if (t == 5) {
            int32_t v = (int32_t)gr_u32(b, &c);
            printf("  = %d", v);
        } else if (t == 6) {
            float v; memcpy(&v, b + c, 4); c += 4;
            printf("  = %.6f", v);
        } else if (t == 9) {
            // array
            uint32_t at = gr_u32(b, &c);
            uint64_t an = gr_u64(b, &c);
            printf("  = array(type=%u, n=%llu)",
                   at, (unsigned long long)an);
            // need to skip the array body
            size_t after = c;
            for (uint64_t j = 0; j < an; j++) {
                after = skip(b, after, at);
            }
            c = after;
        } else {
            c = skip(b, c - 4, t);
            c = c + 0;  // c is now at end of value
            // (this branch shouldn't fire for the common cases)
        }
        printf("\n");
    }
    if (show_tensors) {
        printf("\n=== tensors (%llu) ===\n", (unsigned long long)nt);
        for (uint64_t i = 0; i < nt; i++) {
            uint64_t nn = gr_u64(b, &c);
            const char * nm = (const char *)(b + c); c += nn;
            uint32_t nd = gr_u32(b, &c);
            uint64_t shape[8] = {0};
            for (uint32_t d = 0; d < nd && d < 8; d++) shape[d] = gr_u64(b, &c);
            uint32_t ty = gr_u32(b, &c);
            uint64_t off = gr_u64(b, &c);
            printf("  %.*s  ndim=%u shape=[", (int)nn, nm, nd);
            for (uint32_t d = 0; d < nd; d++)
                printf("%llu%s", (unsigned long long)shape[d],
                       d+1<nd ? "," : "");
            printf("]  type=%u offset=%llu\n",
                   ty, (unsigned long long)off);
        }
    }
    return 0;
}
