# qwen.haiku — Plan and Open Questions

This is the forward-looking roadmap. Background context — design
rationale, every numerical bug fix during the rnd prototyping
phase, and the chat-template state-machine spec — lives in
`docs/DESIGN.md`. Stable here, churn there.

## Current state

The C runner produces coherent text from the unsloth
Qwen3.5-0.8B-Q4_K_M GGUF on Apple Silicon. Verified against
multiple prompts:

| prompt                                | continuation                                                                |
|---------------------------------------|------------------------------------------------------------------------------|
| The capital of France is              | a city in the Île-de-France region of France. It is the capital of …          |
| Hello, my name is                     | abella. I am a student at the University of the West Indies…                  |
| Once upon a time                      | , a young man named Alex, who was a student at a university…                 |
| Python is a programming language that | is used to write code for computers. It is a general-purpose…                |

End-to-end inference path is correct:
- GGUF v3 reader (KV map + tensor table + alignment)
- Byte-level BPE tokenizer (vocab + merges from GGUF, no external)
- All quant dequants bit-exact against ggml's reference (verified by
  `tools/q[4-6]k_diff.cpp`)
- Q8_K activation round-trip matches `quantize_row_q8_K_ref`
- Streaming KV cache with correct causal `k_offset = pos`
- Gated DeltaNet SSM recurrence (per HF transformers reference and
  llama.cpp's `build_delta_net_autoregressive`)
- MRoPE with `[T,H,W]` cyclic section assignment
- Q-norm / K-norm + attention output gate
- Greedy + top-k sampling

## Roadmap

### Phase A — App polish (next)
- Wire up `app/App.swift` SwiftUI scaffold so a first-launch run
  walks the user through the ~504 MB model download with progress.
- Verify the macOS sandboxed build actually writes to
  `~/Library/Containers/<bundle>/Data/Library/Caches/Qwen/` and the
  iOS build to its sandbox equivalent.
- Cancel button during streaming generation.
- Token/s counter in the status bar (cheap — `mach_absolute_time()`).
- Resume an interrupted download (URL.swift already supports it via
  the sidecar `.meta` file; needs UI affordance).

### Phase B — Chat template (done)
- Qwen3 chat-template state machine landed Swift-side in
  `app/Chat.swift`: `ChatMessage` + `ChatTemplate.apply` frame a
  conversation into `<|im_start|>role\n...<|im_end|>\n` envelopes
  per `docs/DESIGN.md`, with reasoning on/off and
  reasoning_effort low/med/high hooks. `ChatStreamFilter` mirrors
  the unframing side: detects `<|im_end|>` / `<|endoftext|>` in the
  streamed token output (with a 16-byte holdback for markers split
  across pieces) and signals end-of-turn so the view model cancels
  generation cleanly.
- `app/App.swift` is a chat surface now: editable system prompt
  header, scrolling message list (user in tinted bubble, assistant
  full-width), input bar with Send/Stop. Sampling defaults moved to
  T=1.0 / top_k=20 (Qwen3.5 non-thinking).
- C `--repl` keeps its minimal hard-coded framing (intentional;
  it's a single-turn command-line poke, not a chat surface).
- Follow-ups gated behind their own tasks:
  - top_p / min_p / presence_penalty / repetition_penalty (#21)
  - `<think>...</think>` split into a separate reasoning channel (#22)

### Phase C investigation: interleaved prefetch in matmul inner loops

Leo's `oblast` repo (https://github.com/leok7v/oblast) is believed
to have AVX2/AVX-512 matmul kernels that do **interleaved
prefetch** in the inner multiply loop (issue a prefetch for the
weight row N iterations ahead of the current consumer). A quick
glance at the ggml NEON Q4_K/Q5_K/Q6_K paths we ported here
(`q[456]k_dot_q8k_neon` in tensor.c) suggests they do NOT do this -
they stream the weight bytes via `vld1q_u8` and rely on the
hardware prefetcher.

Investigate:
1. Confirm whether oblast's AVX2/AVX-512 kernels actually do
   interleaved software prefetch (vs hardware-only).
2. Confirm whether ggml's NEON paths use `__builtin_prefetch` or
   `prfm pldl1keep` anywhere. Spot check `ggml-cpu/arch/arm/quants.c`
   for `prefetch` or `prfm` references.
3. If oblast does it and ggml does not, port the prefetch pattern
   into our Q[456]K dot helpers and bench.
4. Apple Silicon has a strong hardware prefetcher that often beats
   software prefetch on streaming loads, so the win is not
   guaranteed. Measurement, not theory.

### Phase C — Performance
Current cold-start throughput on Apple M3 is ~1.5 tok/s for the
0.8B model. The 24-layer forward pass spends nearly all time in the
Q4_K / Q5_K / Q6_K matmul kernels.

Targets, in priority order:
1. **NEON int8 dot product** (`vdotq_s32`). Bit-exact against
   llama.cpp's `vec_dot_q[4-6]_K_q8_K_generic`, eliminates the
   fp32 round-trip step. ~80 LOC per quant type. Expected 3-5x.
2. **Multi-thread matmul rows.** Each output feature is an
   independent dot product over the weight row. GCD or std::thread.
   Expected 3-4x on M3 (8 perf cores).
3. **fp16 activations.** Halves bandwidth and matches what
   llama.cpp does internally. Touches all kernels; do after #1 and
   #2 are landed and bench numbers are clean.

### Phase D — More models
The current runner is qwen35-specific (hybrid SSM+attention,
attention output gate, ssm_a layout). Generalizing to:
- Plain Qwen3 (pure transformer, no SSM layers, no output gate).
  Probably ~100 LOC of branching in the forward pass.
- Qwen2.5 (different RoPE convention, no MRoPE sections).
- Llama 3.x (different rope base, GQA, no Q/K norm).

Worth scoping carefully — could double the code size if done
poorly. Probably gate behind a `qwen35_only` build flag for now and
revisit once Phase A-C are landed.

## Open questions

- **iOS background download.** `URL.swift` supports background
  URLSession with a completion handler hook
  (`backgroundSessionCompletionHandler` global). Need to wire it
  into the iOS AppDelegate's
  `application(_:handleEventsForBackgroundURLSession:completionHandler:)`.
  For now the default no-op handler is fine — only matters when the
  user backgrounds the app mid-download.
- **App Store distribution.** The model file is ~504 MB; we
  intentionally download it post-install rather than bundling it
  into the IPA. App Review may flag this; an alternative is to
  bundle a smaller fallback model. Defer until we actually try to
  submit.
- **Model source pinning.** Currently pointed at
  `unsloth/Qwen3.5-0.8B-GGUF` on HuggingFace. If unsloth re-quantizes
  the file the SHA changes and the cache key becomes stale. Long
  term: ship a known-good SHA256 in code, verify after download,
  refuse to load mismatches.

## Investigation log: greedy parity vs llama.cpp (2026-05-14)

`tools/parity.sh` runs our `./Build/cli/llm --temperature 0` vs
`llama-completion --temp 0 -ngl 0 -st -no-cnv` on the same four
prompts. Output diverges on every prompt. Both runners are
self-deterministic. `tools/rolodex.sh` shows the same Qwen3.5-0.8B
GGUF produces clean haikus under im.ai's runner but wobbles under
ours, so the model is not the ceiling.

Layer-by-layer comparison on the single-token prompt "Hello" via
`llama-eval-callback -ngl 0`:

| tensor | ours | llama.cpp | diff |
|---|---|---|---|
| attn_norm-0 (head: 1.7735, -2.8266, -0.8377) | match | match | **bit-identical** |
| node_18 / attn_qkv-0 (head: -0.2255, -0.7863, 0.3339) | match | match | **bit-identical** |
| linear_attn_out-0 / ssm_out-0 (head: -0.6552 vs -0.6565) | close | close | ~0.2 % |
| post_ffn-23 (head: -0.2254 vs -0.1617) | diverged | -- | ~30 % |

So: embeddings, RMS norm, weight-load, and per-layer projections
match to FP precision. The ~0.2 % first-layer SSM drift compounds
to ~30 % by layer 23, which is enough to flip argmax on tokens
with close logits but does not represent a math bug. llama.cpp's
qwen35 SSM uses a **chunked parallel formulation with SOLVE_TRI**
(eval-callback shows `NEG -> MUL -> SOLVE_TRI -> ADD`,
`attn_inter_chunk + v_attn_chunk`); ours is sequential recurrent.
Mathematically equivalent, numerically different floating-point
summation order.

Findings:
- **EOG token set was incomplete.** llama.cpp stops on 5 tokens
  (`<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`,
  `<|file_sep|>`); we only stopped on `<|im_end|>` (= eos = eot in
  this GGUF's KV). Fixed: `struct llm_config` gains `stop_ids[8]`
  populated at tokenizer load by string lookup; `llm_generate`
  checks the array.
- **SSM math is correct but numerically distant from llama.cpp.**
  Sum-level drift after layer 0: conv_silu 0.01 %, y_norm 0.09 %,
  ssm_out 0.1 %. So drift enters at every intermediate, not just at
  the Q-quant matmul. Visible head/tail elements were
  `%9.4f`-printed and looked bit-identical; the actual 5th-6th
  decimal place differs everywhere. After 24 layers × ~80 ops, the
  ULP-level drifts compound to the 30-50 % element-wise drift at
  layer 23.
- **A "chunked SSM rewrite" or "Q-quant matmul reorder" alone will
  not close the gap.** Bit-matching llama.cpp would require
  mirroring their exact FP op order in conv1d, RMS norm, SiLU,
  l2_norm, the SSM update, the gated norm, AND every matmul -
  effectively re-using ggml's kernels directly. Not feasible
  inside the 3 kLOC budget.

Future investigation:
- Check whether MRoPE rotation order matches (rope type 40 ==
  "qwen35 mrope" in llama.cpp's enum; we ported a `tensor_rope_imrope`
  variant). Diff at the rope output for a known prompt.
- The `</thrank>` hallucination on "What is love?" persists even
  with the new sampler; suspect a numerical-edge argmax flip at
  some specific decode step. Catch with `LLM_TRACE_TOKENS=1` and
  see whether llama.cpp picks the same token at that position.

## Things that look like bugs but aren't

A fresh agent (or human) should know about these before reaching for
the precision/concurrency knobs.

- **Repetitive output on completion-style prompts** ("Hello" →
  "HelloHelloHello…", "One word answer: …" → loops on a phrase).
  The unsloth `Qwen3.5-0.8B-GGUF` we currently ship is the **base**
  foundation model, not Instruct. It has no instruction tuning so
  it just continues whatever pattern it sees until `max_new`. The
  user-facing **Stop** button covers this. Switching to the
  Instruct variant + applying the chat-template state machine
  (Phase B) is the real fix.

- **Small per-token output drift between runs** — currently zero
  on the same input (fp32 + double accumulators, no `-ffast-math`,
  no thread reordering). If you parallelize matmul rows (Phase C
  #1), output may diverge by a few ULP under different scheduler
  orderings. That's expected and harmless; do NOT add locks to
  serialize.

- **Single-core CPU usage in Activity Monitor (~100 %).** Yes, the
  runner is currently single-threaded scalar fp32. See the
  Performance roadmap in `docs/DESIGN.md` for the 6× headroom
  available on M-series CPUs.

- **The `--single` CLI hangs for ~1 s before first token.**
  That's `llm_create` mmapping and parsing ~500 MB of weights.
  The macOS app does this on a detached task so the UI stays
  responsive.

- **`Hello, my name is` → `abella. I am a student at the
  University of the West Indies…`** That's the model — not us.
  Base completion model imagines a plausible continuation.

## Smoke test

If you change anything in `llm/`, run these to catch obvious
regressions before chasing precision or concurrency issues:

```bash
cd llm && make
QWEN_GGUF=~/Downloads/.../Qwen3.5-0.8B-Q4_K_M.gguf ../Build/cli/llm --self-test
# Expected: 3 forward passes complete, argmax tokens 65/67/68 (or
# similar), no non-finite logits. Synthetic-weight only — no GGUF
# needed if QWEN_GGUF isn't set.

QWEN_GGUF=... ../Build/cli/llm --single "The capital of France is" --max-new 20
# Expected continuation contains "Paris" or "Île-de-France" or
# similar geographical context. NOT "...000000000000" or single
# repeated tokens (those indicate the k_offset bug or a precision
# regression).

QWEN_GGUF=... ../Build/cli/llm --single "Once upon a time" --max-new 20
# Expected: a fictional/narrative continuation. Not a dictionary
# definition of "time".
```

For numerical sanity on quant kernels:
```bash
clang++ -O2 tools/q4k_diff.cpp -L<llama.cpp build>/ggml/src \
  -lggml-base -lggml-cpu -o /tmp/q4k_diff
/tmp/q4k_diff <gguf> blk.0.ffn_gate.weight
# Expected: max_abs = 0 (bit-exact against ggml's dequant).
```

## Tools (`tools/`)

Diagnostic utilities, all standalone C/C++ — none link the runner.

| file | purpose |
|---|---|
| `dump_kvs.c` | print every KV in a GGUF (architecture, tokenizer config, chat_template, etc.) |
| `q4k_diff.cpp` | dequantize Q4_K block via runner's path and via ggml's reference, report max abs diff |
| `q5k_diff.cpp` | same for Q5_K |
| `q6k_diff.cpp` | same for Q6_K |
| `q8k_diff.cpp` | exercise the Q8_K activation round-trip against `quantize_row_q8_K_ref` |

The `q*_diff` tools link `libggml-base.a` and `libggml-cpu.a` from
a llama.cpp build — handy when porting a new quant kernel and
needing a bit-exact oracle. Build with `clang++ -O2 …` as in the
smoke test recipe above.

## Hard rules

- **No SwiftPM.** No `Package.swift`. The repo uses a plain
  `.xcodeproj` with a bridging header.
- **No third-party dependencies.** No llama.cpp, no ggml, no MLX,
  no swift-numerics, no Hugging Face SDK. Foundation + SwiftUI on
  the Swift side; libc + libm on the C side.
- **Single-file C engine.** `llm.c` includes `tensor.c` whole; both
  files are self-contained. New algorithm code goes into one of
  those two files unless it's truly orthogonal.
- **Coding style.** Single-entry / single-exit, no goto, no
  break/continue inside loops (switch-case break is fine), no
  `bool ok` flags. Decompose by responsibility. Match
  neighbouring code first.
