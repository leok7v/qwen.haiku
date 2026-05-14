# qwen.haiku — Design

Durable design notes that don't churn week-to-week. The roadmap
(what's next, performance targets, app polish) lives in `PLAN.md`.

## Why a hand-written runner

llama.cpp's ggml is ~50 kLOC of pure-C that's portable, fast, and
well-tested — but huge for what we're doing. For a single model
family on a single class of hardware (Apple Silicon, CPU only),
~3 kLOC of focused code does the same job:

- **One model family**: Qwen3.5 (hybrid Gated DeltaNet + softmax
  attention). No need for the operator zoo that supports Llama,
  Mistral, Mixtral, Gemma, Phi, Whisper, CLIP, Stable Diffusion,
  Marian, Bloom, etc.
- **One quantization**: Q4_K_M (Q4_K + Q5_K + Q6_K + Q8_0 + a few
  fp32/fp16 tensors). No I-quants, no Q2/Q3, no GPTQ/AWQ.
- **One precision regime**: fp32 activations + Q8_K round-trip,
  bit-exact-mathematically to what llama.cpp's CPU backend does in
  `vec_dot_q[4-6]_K_q8_K_generic`. NEON int dot product is a
  planned perf optimization, not a correctness path.
- **One target**: Apple A-series / M-series CPUs, iOS + macOS. No
  CUDA, no Metal, no SYCL, no Vulkan. No Linux/Android port —
  callers can do that if they want, but we're not maintaining one.

This keeps the whole inference engine readable in an afternoon and
hackable in a day. The cost is generality.

## Architecture summary (Qwen3.5-0.8B)

24 transformer layers, hybrid pattern: every 4th layer is a
softmax-attention layer (3, 7, 11, 15, 19, 23), the rest are SSM
(Gated DeltaNet) layers. Shared by all 24:

- pre-norm `attn_norm.weight` (RMS, fp32)
- post-`linear_attn_out` residual add
- pre-FFN `post_attention_norm.weight` (RMS, fp32)
- SwiGLU FFN (`ffn_gate` Q4_K → SiLU × `ffn_up` Q4_K → `ffn_down` Q6_K)
- final residual add

**SSM layers** (Gated DeltaNet, autoregressive single-token path):

```
h_norm = rms_norm(h, attn_norm)
qkv_pre = attn_qkv · h_norm                  # Q5_K, conv input
z       = attn_gate · h_norm                 # Q4_K, FFN-style gate for y
b       = ssm_beta  · h_norm                 # Q8_0, → beta = sigmoid(b)
a       = ssm_alpha · h_norm                 # Q8_0, → g = -exp(A)·softplus(a+dt_bias)
conv_state.push(qkv_pre)                     # ring buffer, kernel=4
mixed   = silu(conv1d(conv_state))           # 6144 → Q/K/V split at 2048/2048
Q, K, V = mixed[0:2048], mixed[2048:4096], mixed[4096:6144]
Q, K    = l2_norm(Q, eps), l2_norm(K, eps)   # per-head, head_dim=128
Q       = Q / sqrt(128)
# Per-head recurrence (state[k, v]):
state  *= exp(g)
kv_mem  = state · K                          # [v] = sum_k state[k,v] · K[k]
delta   = (V - kv_mem) · beta
state  += outer(K, delta)
y_pre   = state · Q                          # [v] = sum_k state[k,v] · Q[k]
y       = silu(z) ⊙ rms_norm(y_pre, ssm_norm) # gated norm
ssm_out = y · ssm_out.weight                 # Q5_K, back to hidden_dim
```

State (`state[k_dim=128, v_dim=128]`) is per-head per-layer and
mutates on every forward pass. There are 16 SSM heads, so each
layer's state is 16 × 128 × 128 = 262144 floats.

**Attention layers** (full softmax, GQA 8 → 2, head_dim=256):

```
h_norm   = rms_norm(h, attn_norm)
Qcur_full = attn_q · h_norm                  # Q5_K, dim 2*(n_h*hd) = 4096
# Per-head head-interleaved split: [Q[256], gate[256]] × 8 heads.
Q, gate  = split_per_head(Qcur_full)
Q        = rms_norm(Q, attn_q_norm)          # per-head, weight dim 256
K        = attn_k · h_norm                   # Q4_K
V        = attn_v · h_norm                   # Q6_K
K        = rms_norm(K, attn_k_norm)
Q, K     = mrope(Q), mrope(K)                # interleaved, sections [11,11,10,0]
attn     = causal_attention(Q, K_cache[0..pos], V_cache[0..pos])
attn    *= sigmoid(gate)                     # output gate, qwen35-only
attn_out = attn · attn_output.weight         # Q4_K
```

Critical detail: the causal mask must allow attending to positions
0..pos (inclusive). The tensor_attention helper takes a `k_offset`
parameter such that `kv_max = k_offset + q + 1` for query index q.
With single-token decode (n_q=1, q=0), `k_offset = pos`.

## Tensor file format

GGUF v3 with `general.architecture = "qwen35"`. Tensor types in the
unsloth Q4_K_M release:

| tensor                          | type   | shape         | role                              |
|---------------------------------|--------|---------------|-----------------------------------|
| `token_embd.weight`             | Q6_K   | (1024, 248320)| tied with `output`                |
| `output_norm.weight`            | fp32   | (1024,)       | final RMS                          |
| `blk.N.attn_norm.weight`        | fp32   | (1024,)       | pre-attn/SSM RMS                  |
| `blk.N.post_attention_norm.weight` | fp32 | (1024,)      | pre-FFN RMS                       |
| `blk.N.ffn_gate.weight`         | Q4_K   | (1024, 3584)  | SwiGLU gate                       |
| `blk.N.ffn_up.weight`           | Q4_K   | (1024, 3584)  | SwiGLU up                         |
| `blk.N.ffn_down.weight`         | Q6_K   | (3584, 1024)  | SwiGLU down                       |
| `blk.N.attn_qkv.weight`         | Q5_K   | (1024, 6144)  | SSM conv input (qkv mixed)         |
| `blk.N.attn_gate.weight`        | Q4_K   | (1024, 2048)  | SSM z (silu gate for y_norm)      |
| `blk.N.attn_q.weight`           | Q5_K   | (1024, 4096)  | attention layers: Q+gate         |
| `blk.N.attn_k.weight`           | Q4_K   | (1024, 512)   | attention layers: K               |
| `blk.N.attn_v.weight`           | Q6_K   | (1024, 512)   | attention layers: V               |
| `blk.N.attn_q_norm.weight`      | fp32   | (256,)        | attention layers: Q RMS           |
| `blk.N.attn_k_norm.weight`      | fp32   | (256,)        | attention layers: K RMS           |
| `blk.N.attn_output.weight`      | Q4_K   | (2048, 1024)  | attention layers: output proj     |
| `blk.N.ssm_a`                   | fp32   | (16,)         | SSM: pre-computed `-exp(A_log)`    |
| `blk.N.ssm_alpha.weight`        | Q8_0   | (1024, 16)    | SSM: a → softplus → g_log         |
| `blk.N.ssm_beta.weight`         | Q8_0   | (1024, 16)    | SSM: b → sigmoid → beta           |
| `blk.N.ssm_conv1d.weight`       | fp32   | (4, 6144)     | SSM: conv1d kernel                |
| `blk.N.ssm_dt.bias`             | fp32   | (16,)         | SSM: dt bias for softplus(a + bias)|
| `blk.N.ssm_norm.weight`         | fp32   | (128,)        | SSM: pre-output RMS               |
| `blk.N.ssm_out.weight`          | Q5_K   | (2048, 1024)  | SSM: linear_attn_out proj         |

Layer types:
- SSM (no `ssm_a` tensor for the attention layers): 0, 1, 2, 4, 5, 6,
  8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22
- Attention: 3, 7, 11, 15, 19, 23

The runner probes `blk.N.ssm_a` to decide which path to take per
layer — no need to hardcode the attention positions.

Output is **tied to `token_embd.weight`**: there is no
`output.weight` tensor in the GGUF; the runner uses `token_embd`
itself as the LM head weight.

## Tokenizer

Byte-level BPE. Vocab (248320 entries) and merges (247587 entries)
come straight from the GGUF KVs `tokenizer.ggml.tokens` and
`tokenizer.ggml.merges`. BOS = 11 (`,`), EOS = 248046. The model
does NOT expect BOS to be prepended — verified against
llama-eval-callback dumping the same 5-token sequence for
"The capital of France is" as our tokenizer produces.

## Chat-template state machine (deferred; design)

The Qwen3.5 chat template is **stored in the GGUF** as a Jinja
string under the KV `tokenizer.chat_template` (string, ~5 KB).
llama.cpp exposes it via `llama_model_chat_template(model, NULL)`
where the NULL second argument selects the default template. We do
the same: the runner reads it at load time and exposes it through
`llm_chat_template(ctx)` — no need to hardcode the Jinja text or
ship a separate copy.

The Jinja string is too large to embed a full Jinja parser for, but
a **lightweight state machine** can cover the cases we care about
without pulling in `jinja2cpp`/MiniJinja. The GGUF-stored Jinja
serves as the runtime contract — when implementing the state
machine, match the template's behavior exactly; treat the stored
Jinja as ground truth and (optionally) sanity-check at load time
that key markers (`<|im_start|>`, `<|im_end|>`, `<think>`,
`</think>`) appear where expected.

Inputs to the state machine:

- `messages: [(role, content)]`  — roles ∈ {system, user, assistant, tool}
- `reasoning: on | off`           — toggles `<think>...</think>` framing
- `reasoning_effort: low|med|high`— hint to the model, prepended in system
- `add_generation_prompt: bool`   — append `<|im_start|>assistant\n`

Output: a single UTF-8 string ready to feed into `llm_tokenize`.

Algorithm (linear, no Jinja eval):

```
out = ""
if messages[0].role == "system":
    out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n"
    msgs = messages[1..]
else:
    msgs = messages
for m in msgs:
    if m.role == "user":
        out += "<|im_start|>user\n" + m.content + "<|im_end|>\n"
    elif m.role == "assistant":
        if reasoning_on and last_query_index_of(m):
            out += "<|im_start|>assistant\n<think>\n" + m.reasoning
            out +=     "\n</think>\n\n"   + m.content + "<|im_end|>\n"
        else:
            out += "<|im_start|>assistant\n" + m.content + "<|im_end|>\n"
    elif m.role == "tool":
        out += "<|im_start|>user\n<tool_response>\n" + m.content
        out +=    "\n</tool_response><|im_end|>\n"
if add_generation_prompt:
    out += "<|im_start|>assistant\n"
    if reasoning_on:
        out += "<think>\n"              // open the reasoning block
    else:
        out += "<think>\n\n</think>\n\n" // empty reasoning, skip
```

Reasoning-effort handling: prepend a marker to the SYSTEM message:

| effort | prepend |
|---|---|
| low    | `(brief: 1–2 sentences)\n\n`                          |
| medium | (no prepend; default)                                  |
| high   | `(think carefully step-by-step before answering)\n\n`  |

Streaming-decode behavior:
- When `reasoning_on`, the model emits `<think>...</think>` first.
  The Swift surface should detect the `</think>` token and split the
  stream into two channels: `reasoningChunk(s)` and `contentChunk(s)`.
- When `reasoning_off`, the `<think>\n\n</think>\n\n` is already in the
  prompt, so the model jumps straight to content.

Light prompt-massaging:
- Strip leading whitespace on user content; collapse internal
  consecutive newlines beyond 2.
- Detect tool-response wrapping (`<tool_response>…</tool_response>`)
  vs free-form text in messages with `role=tool`.
- For the first turn after a system, add a no-op user `"hi"` if
  none is provided (the template raises if no user query is
  present).

Not implemented yet — `app/Qwen.swift` exposes the raw `generate`
surface only; the C `--repl` mode uses the minimum framing
`<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`.

## Numerical fidelity

All four k-quant dequants (Q4_K, Q5_K, Q6_K) are bit-exact against
ggml's reference (verified by `tools/q[4-6]k_diff.cpp` linking
`libggml-base.a` and comparing first-N super-blocks). Q8_K
activation round-trip is byte-exact against
`quantize_row_q8_K_ref` (`q8k_diff.cpp`), using:

- `iscale = -127.f / xmax` where `xmax` is the SIGNED value of the
  element with largest absolute magnitude
- `nearest_int(iscale * x)` via the fp32 bit-trick
  (`(int)((x + 12582912.f) bits & 0x7fffff) - 0x400000`), matching
  llama.cpp's `nearest_int`. NOT `roundf`, which uses
  round-half-away-from-zero and differs at half-values.
- Clamp to `[-128, 127]`.

RMS norm and L2 norm accumulators are `double`, cast to `float` for
the final `1/sqrt(...)`. Matmul accumulators are `double`. RoPE
uses per-element `powf(base, -2i/rotary_dim)` rather than
incremental `scale *= step` — the latter accumulates rounding
across rotary pairs at moderate `rotary_dim`.

## Performance — current floor and where the headroom is

Today's runner is **single-threaded scalar fp32**. On an Apple M3
during generation, Activity Monitor reports ~100% CPU (one core
saturated) on a machine that has 6 performance + 4 efficiency
cores. That's ~1/6 of peak available compute, before any SIMD or
AMX. The performance ceiling we can reach without trading away the
"~3 kLOC, no deps, single file" rules is substantially higher.

Roadmap, in decreasing order of expected return-on-complexity:

### 1. Multi-thread matmul rows (likely 3-5×)
Every weight matmul (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `fp32`)
computes `out[j] = Σ w[j, :] · x[:]` for j in 0..out_f. The output
features are **completely independent** — perfect data parallelism.
`dispatch_apply` (GCD on macOS/iOS) or a tiny worker pool (pthread)
can saturate all perf cores with one extra include and ~30 LOC per
kernel. Activations and weights are read-only during the matmul; no
synchronization beyond join. Same applies to:
- per-head attention QK·V dot products
- per-head SSM recurrence (16 SSM heads ↔ 16 independent updates)
- the SwiGLU element-wise mul-add in FFN

### 2. NEON int dot product for k-quants (3-10× per matmul)
Apple Silicon has the `vdotq_s32` instruction (4× int8 × int8 → int32 SIMD
accumulate). Llama.cpp's CPU backend uses it directly in
`ggml_vec_dot_q[4-6]_K_q8_K`. Port verbatim (~80 LOC per quant
type) and keep activations as `block_q8_K { float d; int8_t qs[256]; }`
all the way through — no fp32 round-trip in the inner loop. This
is also the bit-exact path against llama.cpp's CPU output, which
is useful as a correctness oracle.

### 3. fp16 activations (~1.5-2× from halved bandwidth)
Currently all intermediates are `float`. Apple Silicon supports IEEE
fp16 arithmetic natively (`_Float16` / `__fp16`). Switching the
residual stream + KV cache + attention intermediates to fp16 halves
memory bandwidth — the binding constraint for matmul on this class
of CPU. KV cache already stores fp16; extending to activations is
the larger change but mechanical. RMS/L2 norms should keep their
`double` accumulators.

### 4. Apple AMX via Accelerate (~10-20× for matmul, but adds a dep)
Apple's undocumented matrix coprocessor is accessible through the
shipped-with-the-OS `Accelerate.framework` (`vDSP_mmul` /
`cblas_sgemm`, `BNNSMatMul`). This is the fastest CPU matmul on
M-series by a wide margin. The cost is dropping the "no deps" rule
for one library — Accelerate is Apple-provided and always present,
not a third party, so this trade-off is on the table. Worth
considering once #1 and #2 are landed and benchmarked.

### 5. Cache-blocked matmul (1.5-2× on top of #1)
Weight rows are ~1-4 KB each (one row of Q4_K is 144 bytes per
super-block, 1024-col row = 576 bytes; Q6_K row = 840 bytes).
After dequant the row fits in L1 (32 KB on Firestorm/Avalanche).
Tile output features in blocks of 8-16 so the activation row stays
hot across all those output features — pays for itself when N_out
is large (vocab head 248320 rows).

### 6. SSM kernel fusion (~10% on SSM-heavy layers)
The Gated DeltaNet inner loop has 5 sequential per-head passes:
`state *= g`, `kv_mem`, `delta`, `state +=`, `out`. Each touches
the same 16 KB per-head state buffer. Fuse the last three passes
into one loop reading `state[k, v]` once per (k, v) cell; that
eliminates two full L1 sweeps per token per head.

### 7. Weight prefetch overlap
During token N's forward pass, start prefetching weights for
token N+1's first matmul. Apple Silicon prefetcher is reasonable
but explicit `__builtin_prefetch` hints on the next layer's
`attn_norm.weight` can help when the weight working set exceeds
L2 (~32 MB shared). Try after #1 — the parallelization will move
the bottleneck.

### 8. Metal compute (out of scope)
Once on Apple Silicon, the GPU shares unified memory with the CPU.
A Metal compute pipeline for the dense matmul kernels would push
the runner to 50-100+ tok/s for the 0.8B model. This is a larger
project (separate runtime, MTLBuffer management, kernel sources)
and intentionally OUT of scope for qwen.haiku — keeps it CPU-only
and dependency-free. If someone wants a GPU runner they should
fork.

### What NOT to chase
- **Quantization of attention/SSM intermediates** — already fp32;
  precision was hard-won to make output coherent. Specifically:
  `RMSNorm` and `L2-norm` accumulate sum-of-squares in `double`,
  matmul kernels accumulate in `double`, Q8_K activation
  round-trip uses `nearest_int` (banker's rounding) not `roundf`,
  RoPE recomputes `powf(base, -2i/r)` per pair instead of
  multiplicative `scale *= step`. Reverting any of these
  individually makes the output look fine on short prompts and
  degenerate on long ones. The combined precision floor is the
  best we can do without int8 SIMD.
- **Multi-batch decode** — qwen.haiku is built for one user, one
  prompt at a time. Don't add batching complexity.
- **KV cache compression / quantization** — KV cache is already
  fp16; quantizing further would hurt long-context quality and
  this model's `max_position` is 262144 anyway — most users won't
  fill the cache.

## Coding discipline (apply to all C and Swift in this repo)

- Single entry, single exit. No early `return`. No `guard let ... else { return }`.
- No `break` / `continue` in loops (switch-case `break` is fine).
- No `bool ok` / `int rc` flags whose only purpose is to be
  inspected at the exit. Piggyback on existing state.
- Decompose long functions into helpers named by their sub-result.
- Match neighbouring code's style. Greenfield files default to the
  conventions in the existing `llm.c` / `tensor.c`.
- Default to no comments; add one only when the *why* is
  non-obvious (a hidden constraint, a workaround for a specific
  bug, behavior that would surprise a reader). Don't restate the
  *what*.
