// text.c: append only vector of owned C strings.
// Layout matches the arrays family: { T * data; size_t count; size_t capacity }.
// Each string is its own small malloc so t.data[i] is usable directly
// and t.count is the element count, no accessor helpers needed.

#ifndef TEXT_C
#define TEXT_C

#include "arrays.c"

struct text { // vector of zero terminated owned strings
    char ** data;
    size_t count;
    size_t capacity;
};

static inline void text_put(struct text * t, const char * s, size_t n) {
    arr_grow((struct arr *)t, sizeof(char *), t->count + 1);
    char * copy = oom(malloc(n + 1));
    memcpy(copy, s, n);
    copy[n] = '\0';
    t->data[t->count++] = copy;
}

static inline void text_puts(struct text * t, const char * s) {
    text_put(t, s, strlen(s));
}

static inline void text_free(struct text * t) {
    for (size_t i = 0; i < t->count; i++) { free(t->data[i]); }
    free(t->data);
    t->data = NULL;
    t->count = 0;
    t->capacity = 0;
}

#ifdef TEXT_TESTS

static void test_text(void) {
    struct text t = {0};
    assert(t.data == NULL && t.count == 0 && t.capacity == 0);
    text_puts(&t, "hello");
    text_puts(&t, "");
    text_puts(&t, "world");
    assert(t.count == 3);
    assert(strcmp(t.data[0], "hello") == 0);
    assert(strcmp(t.data[1], "") == 0);
    assert(strcmp(t.data[2], "world") == 0);
    size_t n = 1000;
    for (size_t i = 0; i < n; i++) {
        char buf[32];
        int w = snprintf(buf, sizeof buf, "item_%zu", i);
        text_put(&t, buf, (size_t)w);
    }
    assert(t.count == 3 + n);
    assert(strcmp(t.data[3], "item_0") == 0);
    assert(strcmp(t.data[3 + n - 1], "item_999") == 0);
    text_put(&t, "abc\0def", 7);
    assert(t.data[t.count - 1][0] == 'a');
    assert(t.data[t.count - 1][3] == '\0');
    assert(t.data[t.count - 1][4] == 'd');
    text_free(&t);
    assert(t.data == NULL && t.count == 0 && t.capacity == 0);
    struct text empty = {0};
    text_free(&empty);
}

static void test(void) {
    test_text();
}

int main(int argc, char * argv[], char * env[]) {
    (void)argc; (void)argv; (void)env;
    test();
    printf("OK\n");
    return 0;
}

#endif /* TEXT_TESTS */

#endif /* TEXT_C */
