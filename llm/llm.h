// SPDX-License-Identifier: Apache-2.0
//
// llm.h - public C API for the Qwen3.5-0.8B Q4_K_M GGUF runner.
//
// This header is the boundary between the C implementation (llm.c +
// tensor.c, ~3200 LOC) and any caller: Swift via the Xcode bridging
// header (app/bridge.h), Objective-C, Python ctypes, another C
// program, etc. Everything in llm.c that isn't declared here is
// internal.
//
// Lifecycle:
//   1. llm_create(path)         - mmap the GGUF, allocate all KV/SSM
//                                 caches, parse weights. Returns NULL
//                                 only on out-of-memory; check
//                                 llm_loaded(ctx) for parse success.
//   2. llm_tokenize(...)        - byte-level BPE encode UTF-8 text
//                                 into token IDs.
//   3. llm_generate(...)        - streaming generation. The callback
//                                 fires once per generated token with
//                                 the decoded UTF-8 piece.
//   4. llm_destroy(ctx)         - free everything; safe on NULL.
//
// Thread-safety: a single llm_ctx is NOT thread-safe. It holds a
// streaming KV cache and per-layer SSM state that mutates on every
// forward pass. Use one ctx per concurrent inference stream.

#ifndef LLM_H
#define LLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle. Callers should never dereference; treat it as
// `void *` with type safety.
struct llm_ctx;

// Token streaming callback. `utf8` is a NUL-terminated UTF-8 piece
// (single token's decoded text, may be a partial Unicode codepoint
// for BBPE; concatenating successive pieces always yields valid
// UTF-8). Return non-zero to abort generation.
typedef int (*llm_token_cb)(const char * utf8, void * user);

// Open a GGUF and prepare the context. `path` is a filesystem path
// to a qwen35 architecture Q4_K_M GGUF (e.g. unsloth's release).
// On failure to parse, returns a non-NULL ctx with llm_loaded() == 0
// and an error string available via llm_get_error().
struct llm_ctx * llm_create(const char * path);

// Free ctx and all owned resources. Safe to call with NULL.
void llm_destroy(struct llm_ctx * ctx);

// 1 if the GGUF was successfully parsed and weights are usable.
// 0 if anything during load failed; call llm_get_error() to find
// out what.
int llm_loaded(const struct llm_ctx * ctx);

// Returns a NUL-terminated error message describing why load or a
// previous call failed. Empty string if no error. Lifetime is tied
// to the ctx; do not free.
const char * llm_get_error(const struct llm_ctx * ctx);

// Encode `text` into token IDs. Writes up to `max_ids` ids into
// `out_ids`. Returns the number of tokens written, or a negative
// value on overflow / error.
int llm_tokenize(struct llm_ctx * ctx, const char * text,
                 int32_t * out_ids, int max_ids);

// Run inference: prefill the prompt token ids, then sample up to
// `max_new` tokens, calling `cb(decoded_piece, user)` once per
// generated token. Stops early on EOS or when `cb` returns non-zero.
//
// `temperature` of 0 selects greedy/argmax (top-k is ignored).
// `top_k` of 0 or 1 also collapses to greedy; larger values do
// top-k sampling with the given temperature.
//
// Returns the number of tokens actually generated.
int llm_generate(struct llm_ctx * ctx,
                 const int32_t * prompt_ids, int prompt_n,
                 int max_new, float temperature, int top_k,
                 llm_token_cb cb, void * user);

// Model metadata accessors. Cheap (O(1)).
int llm_vocab_size(const struct llm_ctx * ctx);
int llm_eos_id    (const struct llm_ctx * ctx);
int llm_bos_id    (const struct llm_ctx * ctx);

// Timing stats from the most recent llm_generate() call.
// pp_per_sec: prompt-prefill throughput, tokens / second (wall time
//   of the prefill loop divided by prompt_n).
// tg_per_sec: token-generation throughput, tokens / second (wall
//   time of the decode loop divided by tokens generated).
// n_prefill, n_generated: token counts from the most recent call.
// All return 0 before any generate call has completed.
double  llm_pp_per_sec (const struct llm_ctx * ctx);
double  llm_tg_per_sec (const struct llm_ctx * ctx);
int32_t llm_n_prefill  (const struct llm_ctx * ctx);
int32_t llm_n_generated(const struct llm_ctx * ctx);

// The chat template Jinja string as stored in the GGUF under the
// `tokenizer.chat_template` KV. NULL if the file has no such KV
// (e.g. base completion models). Lifetime is tied to ctx; do not
// free, the string is mmap-backed.
//
// Mirrors llama.cpp's `llama_model_chat_template(model, NULL)`: the
// caller is expected to apply it, either by piping into a Jinja
// engine, or (recommended for this repo) by using the state machine
// in docs/DESIGN.md whose structure mirrors this template.
const char * llm_chat_template(const struct llm_ctx * ctx);

#ifdef __cplusplus
}
#endif

#endif // LLM_H
