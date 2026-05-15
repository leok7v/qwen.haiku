#ifndef ARRAYS_C
#define ARRAYS_C

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// we observe that most of C++ code we see uses std::vector in
// simple append pattern.
// we want K.I.S.S. and D.R.Y. replacement for std::vector with minimalistic
// probably #define based approach with almost none preconsived conviniences
// assuming that code that uses struct chars, struct ints (also castable to void*)
// and struct nums usually works somewhere in http: json: domains and
// we can implement swift like extension local functions for specific
// minimalistic operation later on as needed basis but not from get-go
// Match code style to the letter and max width < 80
// No empty lines inside statement flow, no comments explaing why.
// No early returns, break (except switch), continue or any othe control
// flow breaks. Single entry single exit in functions code
// (to allow pre post conditions for reasoning about code).
// 64-bit virtual memory high performance architectures only.

struct arr { // type-erased shape shared by chars/ints/nums/ptrs/...
    void * data;
    size_t count;
    size_t capacity;
};

static inline void* oom(void* a) {
    if (!a) {fprintf(stderr, "OOM"); abort(); } // no atexit() call!
    return a;
}

static inline void arr_grow(struct arr * a, size_t esize, size_t need) {
    if (!a->data) { // because realloc(NULL) is not well defined
        a->capacity = need;
        a->data = oom(malloc(need * esize));
    } else if (need > a->capacity) {
        a->capacity = need * 2;
        a->data = oom(realloc(a->data, a->capacity * esize));
    }
}

#define define_array(T, name) \
struct name { T * data; size_t count; size_t capacity; }; \
static inline void name##_grow(struct name * a, size_t need) { \
    arr_grow((struct arr *)a, sizeof(T), need); \
} \
static inline void name##_put(struct name * a, T v) { \
    name##_grow(a, a->count + 1); \
    a->data[a->count++] = v; \
} \
static inline void name##_free(struct name * a) { \
    free(a->data); \
    a->data = NULL; \
    a->count = 0; \
    a->capacity = 0; \
} \
struct name##_swallow_semicolon

#ifdef ARRAYS_TESTS

define_array(int64_t, ints);
define_array(double,  nums);
define_array(void*,   ptrs);

static void test_ints(void) {
    struct ints a = {0};
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    ints_put(&a, 1);
    ints_put(&a, 2);
    ints_put(&a, 3);
    assert(a.count == 3);
    assert(a.data[0] == 1 && a.data[1] == 2 && a.data[2] == 3);
    int64_t n = 1000;
    for (int64_t i = 0; i < n; i++) { ints_put(&a, i * i); }
    assert(a.count == 3 + (size_t)n);
    for (int64_t i = 0; i < n; i++) { assert(a.data[3 + i] == i * i); }
    ints_put(&a, INT64_MIN);
    ints_put(&a, INT64_MAX);
    assert(a.data[a.count - 2] == INT64_MIN);
    assert(a.data[a.count - 1] == INT64_MAX);
    ints_free(&a);
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    struct ints empty = {0};
    ints_free(&empty);
}

static void test_nums(void) {
    struct nums a = {0};
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    nums_put(&a, 1.5);
    nums_put(&a, 2.5);
    nums_put(&a, 3.5);
    assert(a.count == 3);
    assert(a.data[0] == 1.5 && a.data[1] == 2.5 && a.data[2] == 3.5);
    size_t n = 1000;
    for (size_t i = 0; i < n; i++) { nums_put(&a, (double)i / 4.0); }
    assert(a.count == 3 + n);
    for (size_t i = 0; i < n; i++) {
        assert(a.data[3 + i] == (double)i / 4.0);
    }
    nums_free(&a);
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    struct nums empty = {0};
    nums_free(&empty);
}

static void test_ptrs(void) {
    struct ptrs a = {0};
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    int x = 10, y = 20, z = 30;
    ptrs_put(&a, &x);
    ptrs_put(&a, &y);
    ptrs_put(&a, &z);
    ptrs_put(&a, NULL);
    assert(a.count == 4);
    assert(a.data[0] == &x);
    assert(a.data[1] == &y);
    assert(a.data[2] == &z);
    assert(a.data[3] == NULL);
    size_t n = 1000;
    int sentinel = 0;
    for (size_t i = 0; i < n; i++) { ptrs_put(&a, &sentinel); }
    assert(a.count == 4 + n);
    for (size_t i = 0; i < n; i++) { assert(a.data[4 + i] == &sentinel); }
    ptrs_free(&a);
    assert(a.data == NULL && a.count == 0 && a.capacity == 0);
    struct ptrs empty = {0};
    ptrs_free(&empty);
}

static void test(void) {
    test_ints();
    test_nums();
    test_ptrs();
}

int main(int argc, char * argv[], char * env[]) {
    (void)argc; (void)argv; (void)env;
    test();
    printf("OK\n");
    return 0;
}

#endif /* ARRAYS_TESTS */

#endif /* ARRAYS_C */
