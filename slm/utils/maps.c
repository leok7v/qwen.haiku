// maps.c: minimal hash map. Two supported key kinds:
// MAP_KEY_INT (int64_t) and MAP_KEY_CHARS (struct chars).
// Values are arbitrary fixed-size bytes (sizeof(V)). If V owns heap
// resources, caller registers value_free at init time.
// Keys are deep-copied on put. Values are memcpy'd on put: for value
// types that own heap (struct chars) this is a move, and the caller
// is expected to zero-init the source after put.
// Open addressing, linear probing, tombstones, 0.75 load, grow 2x.
// Capacity is always power of two starting from 16 on first put.

#ifndef MAPS_C
#define MAPS_C

#include "chars.c"

enum map_key { MAP_KEY_INT, MAP_KEY_CHARS };

#define MAP_EMPTY     0
#define MAP_LIVE      1
#define MAP_TOMBSTONE 2

struct map {
    uint8_t * states;
    void * keys;
    void * values;
    size_t count;
    size_t capacity;
    size_t key_size;
    size_t value_size;
    enum map_key key_kind;
    void (*value_free)(void *);
};

__attribute__((unused)) static inline uint64_t map_hash_i(int64_t k) {
    uint64_t x = (uint64_t)k;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__attribute__((unused)) static inline uint64_t map_hash_s(const struct chars * k) {
    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset
    for (size_t i = 0; i < k->count; i++) {
        h ^= (uint8_t)k->data[i];
        h *= 0x100000001b3ULL; // FNV prime
    }
    return h;
}

__attribute__((unused)) static inline uint64_t map_hash(const struct map * m, const void * k) {
    uint64_t h = 0;
    if (m->key_kind == MAP_KEY_INT) {
        h = map_hash_i(*(const int64_t *)k);
    } else {
        h = map_hash_s((const struct chars *)k);
    }
    return h;
}

__attribute__((unused)) static inline bool map_eq(const struct map * m, const void * a,
                          const void * b) {
    bool r = false;
    if (m->key_kind == MAP_KEY_INT) {
        r = *(const int64_t *)a == *(const int64_t *)b;
    } else {
        const struct chars * x = a;
        const struct chars * y = b;
        r = x->count == y->count
            && memcmp(x->data, y->data, x->count) == 0;
    }
    return r;
}

__attribute__((unused)) static inline void * map_k(struct map * m, size_t i) {
    return (char *)m->keys + i * m->key_size;
}

__attribute__((unused)) static inline void * map_v(struct map * m, size_t i) {
    return (char *)m->values + i * m->value_size;
}

__attribute__((unused)) static inline void map_key_copy(struct map * m, void * dst,
                                const void * src) {
    if (m->key_kind == MAP_KEY_INT) {
        memcpy(dst, src, sizeof(int64_t));
    } else {
        const struct chars * s = src;
        struct chars * d = dst;
        *d = (struct chars){0};
        chars_put(d, s->data, s->count);
    }
}

__attribute__((unused)) static inline void map_key_free(struct map * m, void * k) {
    if (m->key_kind == MAP_KEY_CHARS) {
        chars_free(k);
    }
}

__attribute__((unused)) static void map_init(struct map * m, enum map_key kk,
                     size_t ks, size_t vs, void (*vf)(void *)) {
    m->states = NULL;
    m->keys = NULL;
    m->values = NULL;
    m->count = 0;
    m->capacity = 0;
    m->key_size = ks;
    m->value_size = vs;
    m->key_kind = kk;
    m->value_free = vf;
}

__attribute__((unused)) static void map_grow(struct map * m, size_t new_cap) {
    uint8_t * ns = oom(calloc(new_cap, 1));
    void * nk = oom(malloc(new_cap * m->key_size));
    void * nv = oom(malloc(new_cap * m->value_size));
    size_t mask = new_cap - 1;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->states[i] == MAP_LIVE) {
            void * ok = map_k(m, i);
            size_t j = (size_t)map_hash(m, ok) & mask;
            while (ns[j] != MAP_EMPTY) { j = (j + 1) & mask; }
            ns[j] = MAP_LIVE;
            memcpy((char *)nk + j * m->key_size, ok, m->key_size);
            memcpy((char *)nv + j * m->value_size,
                   map_v(m, i), m->value_size);
        }
    }
    free(m->states);
    free(m->keys);
    free(m->values);
    m->states = ns;
    m->keys = nk;
    m->values = nv;
    m->capacity = new_cap;
}

__attribute__((unused)) static void * map_put(struct map * m, const void * k, const void * v) {
    void * r = NULL;
    if (m->capacity == 0) {
        map_grow(m, 16);
    } else if ((m->count + 1) * 4 > m->capacity * 3) {
        map_grow(m, m->capacity * 2);
    }
    size_t mask = m->capacity - 1;
    size_t j = (size_t)map_hash(m, k) & mask;
    size_t tomb = (size_t)-1;
    bool done = false;
    while (!done && m->states[j] != MAP_EMPTY) {
        if (m->states[j] == MAP_TOMBSTONE) {
            if (tomb == (size_t)-1) { tomb = j; }
            j = (j + 1) & mask;
        } else if (map_eq(m, map_k(m, j), k)) {
            void * vs = map_v(m, j);
            if (m->value_free) { m->value_free(vs); }
            memcpy(vs, v, m->value_size);
            r = vs;
            done = true;
        } else {
            j = (j + 1) & mask;
        }
    }
    if (!done) {
        if (tomb != (size_t)-1) { j = tomb; }
        m->states[j] = MAP_LIVE;
        map_key_copy(m, map_k(m, j), k);
        memcpy(map_v(m, j), v, m->value_size);
        m->count++;
        r = map_v(m, j);
    }
    return r;
}

__attribute__((unused)) static void * map_get(struct map * m, const void * k) {
    void * r = NULL;
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j = (size_t)map_hash(m, k) & mask;
        bool done = false;
        while (!done && m->states[j] != MAP_EMPTY) {
            if (m->states[j] == MAP_LIVE
                && map_eq(m, map_k(m, j), k)) {
                r = map_v(m, j);
                done = true;
            } else {
                j = (j + 1) & mask;
            }
        }
    }
    return r;
}

__attribute__((unused)) static void map_remove(struct map * m, const void * k) {
    if (m->capacity > 0) {
        size_t mask = m->capacity - 1;
        size_t j = (size_t)map_hash(m, k) & mask;
        bool done = false;
        while (!done && m->states[j] != MAP_EMPTY) {
            if (m->states[j] == MAP_LIVE
                && map_eq(m, map_k(m, j), k)) {
                if (m->value_free) { m->value_free(map_v(m, j)); }
                map_key_free(m, map_k(m, j));
                m->states[j] = MAP_TOMBSTONE;
                m->count--;
                done = true;
            } else {
                j = (j + 1) & mask;
            }
        }
    }
}

__attribute__((unused)) static void map_free(struct map * m) {
    if (m->capacity > 0) {
        for (size_t i = 0; i < m->capacity; i++) {
            if (m->states[i] == MAP_LIVE) {
                if (m->value_free) { m->value_free(map_v(m, i)); }
                map_key_free(m, map_k(m, i));
            }
        }
        free(m->states);
        free(m->keys);
        free(m->values);
    }
    m->states = NULL;
    m->keys = NULL;
    m->values = NULL;
    m->count = 0;
    m->capacity = 0;
}

__attribute__((unused))
__attribute__((unused)) static void map_for_each(struct map * m,
                         void (*fn)(const void * k, void * v, void * ctx),
                         void * ctx) {
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->states[i] == MAP_LIVE) {
            fn(map_k(m, i), map_v(m, i), ctx);
        }
    }
}

__attribute__((unused)) static inline void * map_puti(struct map * m, int64_t k, const void * v) {
    return map_put(m, &k, v);
}

__attribute__((unused)) static inline void * map_geti(struct map * m, int64_t k) {
    return map_get(m, &k);
}

__attribute__((unused)) static inline void map_removei(struct map * m, int64_t k) {
    map_remove(m, &k);
}

__attribute__((unused)) static inline void * map_puts(struct map * m, const char * k,
                              const void * v) {
    struct chars tmp = {0};
    tmp.data = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    return map_put(m, &tmp, v);
}

__attribute__((unused)) static inline void * map_gets(struct map * m, const char * k) {
    struct chars tmp = {0};
    tmp.data = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    return map_get(m, &tmp);
}

__attribute__((unused)) static inline void map_removes(struct map * m, const char * k) {
    struct chars tmp = {0};
    tmp.data = (char *)(uintptr_t)k;
    tmp.count = strlen(k);
    map_remove(m, &tmp);
}

__attribute__((unused)) static inline void chars_free_v(void * s) {
    chars_free((struct chars *)s);
}

#ifdef MAP_TESTS

__attribute__((unused)) static void sum_ints(const void * k, void * v, void * ctx) {
    (void)k;
    int64_t * sum = ctx;
    *sum += *(int64_t *)v;
}

__attribute__((unused)) static void count_entries(const void * k, void * v, void * ctx) {
    (void)k; (void)v;
    size_t * n = ctx;
    (*n)++;
}

__attribute__((unused)) static void test_map_ii(void) {
    struct map m;
    map_init(&m, MAP_KEY_INT, sizeof(int64_t), sizeof(int64_t), NULL);
    assert(m.count == 0 && m.capacity == 0);
    int64_t v1 = 100, v2 = 200, v3 = 300;
    map_puti(&m, 1, &v1);
    map_puti(&m, 2, &v2);
    map_puti(&m, 3, &v3);
    assert(m.count == 3);
    assert(*(int64_t *)map_geti(&m, 1) == 100);
    assert(*(int64_t *)map_geti(&m, 2) == 200);
    assert(*(int64_t *)map_geti(&m, 3) == 300);
    assert(map_geti(&m, 999) == NULL);
    int64_t v1b = 111;
    map_puti(&m, 1, &v1b);
    assert(m.count == 3);
    assert(*(int64_t *)map_geti(&m, 1) == 111);
    map_removei(&m, 2);
    assert(m.count == 2);
    assert(map_geti(&m, 2) == NULL);
    map_removei(&m, 9999);
    assert(m.count == 2);
    for (int64_t i = 0; i < 1000; i++) {
        int64_t val = i + 10000;
        map_puti(&m, 100 + i, &val);
    }
    assert(m.count == 2 + 1000);
    for (int64_t i = 0; i < 1000; i++) {
        int64_t * got = map_geti(&m, 100 + i);
        assert(got != NULL);
        assert(*got == i + 10000);
    }
    int64_t sum = 0;
    map_for_each(&m, sum_ints, &sum);
    int64_t expected = 111 + 300;
    for (int64_t i = 0; i < 1000; i++) { expected += i + 10000; }
    assert(sum == expected);
    size_t n = 0;
    map_for_each(&m, count_entries, &n);
    assert(n == m.count);
    map_free(&m);
    assert(m.count == 0 && m.capacity == 0);
    struct map empty;
    map_init(&empty, MAP_KEY_INT, sizeof(int64_t), sizeof(int64_t), NULL);
    map_free(&empty);
}

__attribute__((unused)) static void test_map_si(void) {
    struct map m;
    map_init(&m, MAP_KEY_CHARS, sizeof(struct chars),
             sizeof(int64_t), NULL);
    int64_t v1 = 1, v2 = 2, v3 = 3;
    map_puts(&m, "alpha", &v1);
    map_puts(&m, "beta", &v2);
    map_puts(&m, "gamma", &v3);
    assert(m.count == 3);
    assert(*(int64_t *)map_gets(&m, "alpha") == 1);
    assert(*(int64_t *)map_gets(&m, "beta") == 2);
    assert(*(int64_t *)map_gets(&m, "gamma") == 3);
    assert(map_gets(&m, "delta") == NULL);
    int64_t v1b = 10;
    map_puts(&m, "alpha", &v1b);
    assert(m.count == 3);
    assert(*(int64_t *)map_gets(&m, "alpha") == 10);
    map_removes(&m, "beta");
    assert(m.count == 2);
    assert(map_gets(&m, "beta") == NULL);
    for (int64_t i = 0; i < 500; i++) {
        char buf[32];
        snprintf(buf, sizeof buf, "key_%zu", (size_t)i);
        map_puts(&m, buf, &i);
    }
    assert(m.count == 2 + 500);
    for (int64_t i = 0; i < 500; i++) {
        char buf[32];
        snprintf(buf, sizeof buf, "key_%zu", (size_t)i);
        int64_t * got = map_gets(&m, buf);
        assert(got != NULL);
        assert(*got == i);
    }
    map_free(&m);
}

__attribute__((unused)) static void test_map_sv(void) {
    struct map m;
    map_init(&m, MAP_KEY_CHARS, sizeof(struct chars),
             sizeof(struct chars), chars_free_v);
    struct chars v1 = {0};
    chars_puts(&v1, "hello");
    map_puts(&m, "greeting", &v1);
    v1 = (struct chars){0};
    struct chars v2 = {0};
    chars_puts(&v2, "world");
    map_puts(&m, "subject", &v2);
    v2 = (struct chars){0};
    assert(m.count == 2);
    struct chars * got = map_gets(&m, "greeting");
    assert(got != NULL);
    assert(strcmp(got->data, "hello") == 0);
    got = map_gets(&m, "subject");
    assert(got != NULL);
    assert(strcmp(got->data, "world") == 0);
    struct chars v3 = {0};
    chars_puts(&v3, "hi");
    map_puts(&m, "greeting", &v3);
    v3 = (struct chars){0};
    assert(m.count == 2);
    got = map_gets(&m, "greeting");
    assert(strcmp(got->data, "hi") == 0);
    map_removes(&m, "subject");
    assert(m.count == 1);
    assert(map_gets(&m, "subject") == NULL);
    map_free(&m);
}

__attribute__((unused)) static void test(void) {
    test_map_ii();
    test_map_si();
    test_map_sv();
}

int main(int argc, char * argv[], char * env[]) {
    (void)argc; (void)argv; (void)env;
    test();
    printf("OK\n");
    return 0;
}

#endif /* MAP_TESTS */

#endif /* MAPS_C */
