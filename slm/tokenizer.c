// tokenizer.c -- byte-level BPE tokenizer (GPT-2 / Qwen3 style).
//
// `#include`-d once from qwen.c, after `struct slm_config` is in
// scope (the tokenizer's `tokenizer_load` takes a config pointer
// and dereferences vocab_size / bos_id / eos_id from it). Holds
// the vocab + merges tables as two `struct map` instances from
// utils/maps.c. Independent of the model forward pass.
//
// Public surface (all `tokenizer_*`):
//
//   tokenizer_load      - load vocab + merges from GGUF KV
//   tokenizer_free      - release maps and vocab string storage
//   tokenizer_encode    - utf-8 string → token id sequence
//   tokenizer_decode_one- token id → utf-8 piece (single token)
//
// Plus `struct tokenizer` itself, which is embedded by value in
// `struct slm_ctx`. The `tokenizer_match_special` and
// `tokenizer_encode_bpe` helpers are public-ish too (they're called
// from `tokenizer_encode` and could be from elsewhere within the
// TU); `tokenizer_build_byte_maps` is an init helper.
//
// File-internal (`static`): the byte-to-unicode codepoint mapping
// (`bytes_to_unicode_cp` / `utf8_*`) and a small `s2i_*` adapter
// that wraps `struct map` so call sites can pass `(data, length)`
// pairs instead of building a `struct chars` each time. Keys can in
// principle contain 0x00 bytes (after byte-to-unicode remapping they
// don't, but the adapter doesn't rely on null termination).

#ifndef TOKENIZER_C
#define TOKENIZER_C

// ---------------------------------------------------------------------------
// s2i: thin (data, length, int32) adapter over utils/maps.c.
// File-internal — only the tokenizer needs explicit-length string
// keys. Auto-inits on first put.
// ---------------------------------------------------------------------------

static inline void s2i_init_if_needed(struct map * m) {
    if (m->key_size == 0) {
        map_init(m, MAP_KEY_CHARS,
                 sizeof(struct chars), sizeof(int32_t), NULL);
    }
}

static inline void s2i_put(struct map * m,
                           const char * k, size_t kn, int32_t v) {
    s2i_init_if_needed(m);
    struct chars key = { .data = (char *)k, .count = kn, .capacity = 0 };
    map_put(m, &key, &v);
}

static inline int32_t s2i_get(const struct map * m,
                              const char * k, size_t kn, int32_t miss) {
    int32_t r = miss;
    struct chars key = { .data = (char *)k, .count = kn, .capacity = 0 };
    int32_t * vp = (int32_t *)map_get((struct map *)m, &key);
    if (vp != NULL) { r = *vp; }
    return r;
}

static inline void s2i_free(struct map * m) {
    map_free(m);
}

// ---------------------------------------------------------------------------
// Tokenizer (byte-level BPE, simplified)
// ---------------------------------------------------------------------------
//
// GPT-2 / Qwen3 byte-level BPE has a quirk: input UTF-8 bytes are
// first mapped through a fixed 256-entry "bytes-to-unicode" table so
// that whitespace + control bytes become printable unicode glyphs.
// Vocab and merges are stored in this remapped space.
//
// We implement: load vocab + merges from GGUF, build vocab-to-id map
// and merge-to-rank map. Encode: bytes -> remapped chars -> initial
// per-byte tokens -> greedy merge by lowest rank. Decode: id ->
// vocab string -> reverse byte map -> raw UTF-8.

// A "special" token that must bypass BPE: when its literal text
// appears in the input, the encoder emits the vocab id directly
// instead of running byte-level BPE on its bytes. Without this,
// markers like `<|im_start|>` get split into 6 byte-level pieces
// (`<`, `|`, `im`, `_start`, `|`, `>`) and the model sees garbled
// framing where its training expected a single token. Populated at
// load time by looking up known marker strings in `vocab_to_id`.
struct tokenizer_special {
    const char * str;
    int32_t      len;
    int32_t      id;
};
#define TOK_MAX_SPECIALS 8

struct tokenizer {
    int32_t        vocab_size;
    struct chars * vocab_strs;        // [vocab_size]
    struct map     vocab_to_id;       // chars → int32 token id
    struct map     merge_rank;        // "tokenA tokenB" → rank
    int32_t        bos_id;
    int32_t        eos_id;
    int32_t        byte_to_uni[256];  // initial codepoint per raw byte
    int32_t        uni_to_byte[1024]; // reverse map; see footnote
    int32_t        n_specials;
    struct tokenizer_special specials[TOK_MAX_SPECIALS];
};

// footnote: .uni_to_byte reverse map; sparse, indexed by
// codepoint mod 1024 (the GPT-2 set is < 1024)

// GPT-2's bytes_to_unicode table, expressed as direct codepoints.
// Reference: huggingface/tokenizers's gpt2 byte_level pre-tokenizer.
// Returns the codepoint for byte b.
static int32_t bytes_to_unicode_cp(int32_t b) {
    // 33..126, 161..172, 174..255 are passed through directly.
    // Remaining bytes (0..32, 127..160, 173) map to codepoints
    // 256, 257, ..., in order. So:
    //   0   -> 256
    //   1   -> 257
    //   ...
    //   32  -> 288
    //   33..126 -> 33..126
    //   127 -> 289
    //   ...
    //   160 -> 322
    //   161..172 -> 161..172
    //   173 -> 323
    //   174..255 -> 174..255
    int32_t r = b;
    if ((b >= 33  && b <= 126)
     || (b >= 161 && b <= 172)
     || (b >= 174 && b <= 255)) {
        r = b;
    } else {
        // assign in order
        int32_t counter = 0;
        for (int32_t i = 0; i < 256; i++) {
            int32_t is_printable = (i >= 33  && i <= 126)
                                || (i >= 161 && i <= 172)
                                || (i >= 174 && i <= 255);
            if (!is_printable) {
                if (i == b) {
                    r = 256 + counter;
                    i = 256;  // exit-via-predicate
                }
                counter++;
            }
        }
    }
    return r;
}

// Encode codepoint cp as UTF-8 into out (up to 4 bytes). Returns
// number of bytes written.
static int32_t utf8_encode_cp(int32_t cp, char * out) {
    int32_t n = 0;
    if (cp < 0x80) {
        out[0] = (char)cp;
        n = 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6)  & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    return n;
}

static int32_t utf8_decode_one(const char * s, size_t n, int32_t * out_cp) {
    int32_t consumed = 0;
    if (n > 0) {
        unsigned char c0 = (unsigned char)s[0];
        if (c0 < 0x80) {
            *out_cp = c0;
            consumed = 1;
        } else if ((c0 & 0xE0) == 0xC0 && n >= 2) {
            *out_cp = ((c0 & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
            consumed = 2;
        } else if ((c0 & 0xF0) == 0xE0 && n >= 3) {
            *out_cp = ((c0 & 0x0F) << 12)
                    | (((unsigned char)s[1] & 0x3F) << 6)
                    | ((unsigned char)s[2] & 0x3F);
            consumed = 3;
        } else if ((c0 & 0xF8) == 0xF0 && n >= 4) {
            *out_cp = ((c0 & 0x07) << 18)
                    | (((unsigned char)s[1] & 0x3F) << 12)
                    | (((unsigned char)s[2] & 0x3F) << 6)
                    | ((unsigned char)s[3] & 0x3F);
            consumed = 4;
        } else {
            *out_cp = 0xFFFD;
            consumed = 1;
        }
    }
    return consumed;
}

static void tokenizer_build_byte_maps(struct tokenizer * t) {
    for (int32_t i = 0; i < 1024; i++) { t->uni_to_byte[i] = -1; }
    for (int32_t b = 0; b < 256; b++) {
        int32_t cp = bytes_to_unicode_cp(b);
        t->byte_to_uni[b] = cp;
        if (cp >= 0 && cp < 1024) { t->uni_to_byte[cp] = b; }
    }
}

static int32_t tokenizer_load(struct tokenizer * t, const struct gguf * g,
                        const struct slm_config * cfg) {
    memset(t, 0, sizeof(*t));
    t->vocab_size = cfg->vocab_size;
    t->bos_id     = cfg->bos_id;
    t->eos_id     = cfg->eos_id;
    tokenizer_build_byte_maps(t);
    const struct gguf_kv * tk = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!tk || tk->v.type != GGUF_VT_ARRAY
            || tk->v.arr_type != GGUF_VT_STR) {
        trace("missing tokenizer.ggml.tokens (str array)\n");
        return -1;
    }
    t->vocab_strs = (struct chars *)oom(
        calloc((size_t)t->vocab_size, sizeof(struct chars)));
    size_t c = 0;
    for (uint64_t i = 0; i < tk->v.arr_n; i++) {
        uint64_t n = gr_u64(tk->v.arr_data, &c);
        chars_put(&t->vocab_strs[i], (const char *)(tk->v.arr_data + c), n);
        c += n;
        s2i_put(&t->vocab_to_id,
                t->vocab_strs[i].data, t->vocab_strs[i].count, (int32_t)i);
    }

    const struct gguf_kv * mg = gguf_find_kv(g, "tokenizer.ggml.merges");
    if (mg && mg->v.type == GGUF_VT_ARRAY
           && mg->v.arr_type == GGUF_VT_STR) {
        size_t cc = 0;
        for (uint64_t i = 0; i < mg->v.arr_n; i++) {
            uint64_t n = gr_u64(mg->v.arr_data, &cc);
            const char * s = (const char *)(mg->v.arr_data + cc);
            s2i_put(&t->merge_rank, s, n, (int32_t)i);
            cc += n;
        }
    } else {
        trace("warning: no tokenizer.ggml.merges — BPE falls back to"
              " byte-token-only encoding\n");
    }
    // Known chat-framing specials (Qwen3 vocab). Looked up in the
    // vocab map; populated only when present, so non-chat GGUFs are
    // unaffected. Listed longest-first so the encoder's greedy match
    // picks the longest applicable token at each position.
    static const char * known[] = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        NULL,
    };
    t->n_specials = 0;
    for (int32_t i = 0; known[i] != NULL; i++) {
        int32_t slen = (int32_t)strlen(known[i]);
        int32_t id   = s2i_get(&t->vocab_to_id, known[i], slen, -1);
        if (id >= 0 && t->n_specials < TOK_MAX_SPECIALS) {
            t->specials[t->n_specials].str = known[i];
            t->specials[t->n_specials].len = slen;
            t->specials[t->n_specials].id  = id;
            t->n_specials++;
        }
    }
    return 0;
}

static void tokenizer_free(struct tokenizer * t) {
    if (t->vocab_strs) {
        for (int32_t i = 0; i < t->vocab_size; i++) {
            chars_free(&t->vocab_strs[i]);
        }
        free(t->vocab_strs);
    }
    s2i_free(&t->vocab_to_id);
    s2i_free(&t->merge_rank);
}

// Try to match any registered special token at `text[0..tlen)`.
// Returns the special's index in `t->specials` (>=0) and writes its
// byte length to `*sp_len`, or -1 if no special matches at position
// 0. Greedy longest-first match: specials are registered in
// longest-first order so this just returns the first hit.
static int32_t tokenizer_match_special(const struct tokenizer * t,
                                 const char * text, size_t tlen,
                                 int32_t * sp_len) {
    int32_t hit = -1;
    for (int32_t i = 0; i < t->n_specials && hit < 0; i++) {
        size_t slen = (size_t)t->specials[i].len;
        if (slen <= tlen && memcmp(text, t->specials[i].str, slen) == 0) {
            *sp_len = (int32_t)slen;
            hit     = i;
        }
    }
    return hit;
}

// BPE-encode a slice of raw text (no special-token recognition).
// `text[0..tlen)` is split on spaces with the GPT-2 convention that
// a leading space belongs to the next word, then each word is
// byte-level-remapped and greedy-BPE-merged. Returns the number of
// ids written to `out_ids` (bounded by `max_ids`).
static int32_t tokenizer_encode_bpe(const struct tokenizer * t,
                              const char * text, size_t tlen,
                              int32_t * out_ids, int32_t max_ids) {
    int32_t n_out = 0;
    size_t  pos   = 0;
    while (pos < tlen && n_out < max_ids) {
        // Find next "word" boundary: include leading space (if any).
        size_t word_start = pos;
        if (text[pos] == ' ') { pos++; }
        while (pos < tlen && text[pos] != ' ') { pos++; }
        size_t word_len = pos - word_start;

        // Build initial token list: one entry per UTF-8 byte, each
        // mapped through bytes-to-unicode.
        struct chars * toks = (struct chars *)oom(
            calloc(word_len, sizeof(struct chars)));
        int32_t n_toks = 0;
        for (size_t i = 0; i < word_len; i++) {
            int32_t cp = t->byte_to_uni[(unsigned char)text[word_start + i]];
            char    buf[4];
            int32_t nb = utf8_encode_cp(cp, buf);
            chars_put(&toks[n_toks++], buf, nb);
        }

        // Greedy BPE merge.
        int32_t changed = 1;
        while (changed && n_toks > 1) {
            changed = 0;
            int32_t best_idx  = -1;
            int32_t best_rank = INT32_MAX;
            // Scan adjacent pairs; find lowest rank.
            for (int32_t i = 0; i + 1 < n_toks; i++) {
                struct chars pair = {0};
                chars_put(&pair, toks[i].data,    toks[i].count);
                chars_put(&pair, " ",             1);
                chars_put(&pair, toks[i+1].data,  toks[i+1].count);
                int32_t rank = s2i_get(&t->merge_rank,
                                       pair.data, pair.count, INT32_MAX);
                chars_free(&pair);
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx  = i;
                }
            }
            if (best_idx >= 0) {
                // Merge toks[best_idx] + toks[best_idx+1].
                chars_put(&toks[best_idx],
                          toks[best_idx + 1].data, toks[best_idx + 1].count);
                chars_free(&toks[best_idx + 1]);
                for (int32_t j = best_idx + 1; j + 1 < n_toks; j++) {
                    toks[j] = toks[j + 1];
                }
                n_toks--;
                changed = 1;
            }
        }

        // Emit ids.
        for (int32_t i = 0; i < n_toks && n_out < max_ids; i++) {
            int32_t id = s2i_get(&t->vocab_to_id,
                                 toks[i].data, toks[i].count, -1);
            if (id < 0) {
                // Unknown sub-token. Should never happen for
                // byte-level BPE: every single byte is a known token.
                trace("unknown sub-token (len=%zu)\n", toks[i].count);
            } else {
                out_ids[n_out++] = id;
            }
        }
        for (int32_t i = 0; i < n_toks; i++) { chars_free(&toks[i]); }
        free(toks);
    }
    return n_out;
}

// Encode: input UTF-8 text -> token ids. Returns count.
//
// Two-level scan. The outer loop walks the text looking for
// registered special-token strings (`<|im_start|>` etc.) and emits
// their vocab IDs directly. The inner spans (text between specials)
// go through `tokenizer_encode_bpe` for the existing byte-level BPE.
// Without the outer scan a marker like `<|im_start|>` is split into
// six byte pieces, the model sees garbled framing instead of its
// trained chat envelope, and quality collapses.

__attribute__((unused))
static int32_t tokenizer_encode(const struct tokenizer * t,
                          const char * text,
                          int32_t * out_ids, int32_t max_ids) {
    int32_t n_out = 0;
    size_t  tlen  = strlen(text);
    size_t  pos   = 0;
    while (pos < tlen && n_out < max_ids) {
        // Scan ahead for the next special-token occurrence.
        size_t  sp_at  = tlen;
        int32_t sp_idx = -1;
        int32_t sp_len = 0;
        for (size_t i = pos; i < tlen && sp_idx < 0; i++) {
            int32_t mlen = 0;
            int32_t hit  = tokenizer_match_special(t, text + i, tlen - i, &mlen);
            if (hit >= 0) {
                sp_at  = i;
                sp_idx = hit;
                sp_len = mlen;
            }
        }
        // BPE-encode the text in front of the special (or all of it).
        if (sp_at > pos) {
            n_out += tokenizer_encode_bpe(t, text + pos, sp_at - pos,
                                    out_ids + n_out, max_ids - n_out);
        }
        if (sp_idx >= 0 && n_out < max_ids) {
            out_ids[n_out++] = t->specials[sp_idx].id;
            pos = sp_at + (size_t)sp_len;
        } else {
            pos = tlen;
        }
    }
    return n_out;
}

// Decode: token id -> raw UTF-8 bytes appended to `out`. Reverses the
// byte-level remap.

__attribute__((unused))
static void tokenizer_decode_one(const struct tokenizer * t,
                           int32_t id, struct chars * out) {
    if (id >= 0 && id < t->vocab_size) {
        const struct chars * s = &t->vocab_strs[id];
        size_t pos = 0;
        while (pos < s->count) {
            int32_t cp  = 0;
            int32_t adv = utf8_decode_one(s->data + pos,
                                          s->count - pos, &cp);
            if (adv == 0) { adv = 1; }
            if (cp >= 0 && cp < 1024 && t->uni_to_byte[cp] >= 0) {
                char b = (char)(unsigned char)t->uni_to_byte[cp];
                chars_put(out, &b, 1);
            } else {
                // Special token / non-byte: pass through as-is.
                chars_put(out, s->data + pos, adv);
            }
            pos += adv;
        }
    }
}

#endif  // TOKENIZER_C
