#ifndef CHARS_C
#define CHARS_C

#include "arrays.c"

#include <stdarg.h>

struct chars { // always zero terminated array of bytes
    char * data;
    size_t count;
    size_t capacity;
};

static inline void chars_grow(struct chars * s, size_t need) {
    arr_grow((struct arr *)s, 1, need);
}

static inline void chars_put(struct chars * s, const char * d, size_t count) {
    chars_grow(s, s->count + count + 1);
    if (s->data) {
        memcpy(s->data + s->count, d, count);
        s->count += count;
        s->data[s->count] = '\0'; // always zero terminated
    }
}

static inline void chars_free(struct chars * s) {
    free(s->data);
    s->data = NULL;
    s->count = 0;
    s->capacity = 0;
}

static inline void chars_puts(struct chars * s, const char * a) {
    chars_put(s, a, strlen(a));
}

static inline void chars_vprintf(struct chars * s, const char * f,
                                 va_list vl) {
    va_list cp;
    va_copy(cp, vl);
    int r = vsnprintf(NULL, 0, f, cp);
    va_end(cp);
    if (r > 0) {
        size_t n = (size_t)r;
        chars_grow(s, s->count + n + 1);
        vsnprintf(s->data + s->count, n + 1, f, vl);
        s->count += n;
    }
}

static inline void chars_printf(struct chars * s, const char * f, ...) {
    va_list vl;
    va_start(vl, f);
    chars_vprintf(s, f, vl);
    va_end(vl);
}

#ifdef CHARS_TESTS

static void test_chars(void) {
    struct chars s = {0};
    assert(s.data == NULL && s.count == 0 && s.capacity == 0);
    chars_put(&s, "abc", 3);
    assert(s.count == 3);
    assert(s.capacity >= 4);
    assert(strcmp(s.data, "abc") == 0);
    chars_put(&s, "def", 3);
    assert(s.count == 6);
    assert(strcmp(s.data, "abcdef") == 0);
    size_t n = 1000;
    for (size_t i = 0; i < n; i++) { chars_put(&s, "x", 1); }
    assert(s.count == 6 + n);
    assert(s.data[s.count] == '\0');
    assert(strncmp(s.data, "abcdef", 6) == 0);
    for (size_t i = 0; i < n; i++) { assert(s.data[6 + i] == 'x'); }
    chars_put(&s, "", 0);
    assert(s.count == 6 + n);
    assert(s.data[s.count] == '\0');
    chars_free(&s);
    assert(s.data == NULL && s.count == 0 && s.capacity == 0);
    struct chars empty = {0};
    chars_free(&empty); // free of zero-init must be a no-op
    struct chars p = {0};
    chars_puts(&p, "hello");
    chars_puts(&p, ", ");
    chars_puts(&p, "world");
    assert(p.count == 12);
    assert(strcmp(p.data, "hello, world") == 0);
    chars_free(&p);
    struct chars q = {0};
    chars_printf(&q, "%d+%d=%d", 2, 3, 5);
    assert(strcmp(q.data, "2+3=5") == 0);
    assert(q.count == 5);
    chars_printf(&q, " [%s]", "ok");
    assert(strcmp(q.data, "2+3=5 [ok]") == 0);
    chars_free(&q);
    struct chars big = {0};
    for (int i = 0; i < 100; i++) { chars_printf(&big, "%04d,", i); }
    assert(big.count == 500);
    assert(big.data[big.count] == '\0');
    assert(strncmp(big.data, "0000,0001,", 10) == 0);
    assert(strncmp(big.data + big.count - 5, "0099,", 5) == 0);
    chars_free(&big);
}

int main(int argc, char * argv[], char * env[]) {
    (void)argc; (void)argv; (void)env;
    test_chars();
    printf("OK\n");
    return 0;
}

#endif /* CHARS_TESTS */

#endif /* CHARS_C */
