# qwen.haiku

A small, single-file CPU LLM runner for the Qwen3.5-0.8B Q4_K_M GGUF,
wrapped in a minimal SwiftUI app for iOS and macOS.

The whole inference engine is one C file plus its tensor library —
no dependencies on llama.cpp, ggml, Accelerate, BLAS, or any
third-party runtime. The Swift app calls into it via a single
bridging header.

## Layout

```
qwen.haiku/
├── llm/                  C inference engine (CPU, no deps)
│   ├── llm.c             entry point + model loader + forward pass
│   ├── tensor.c          tensor + quant kernels (#included by llm.c)
│   ├── llm.h             public C API (also the Swift bridge target)
│   └── Makefile          standalone CLI build (clang, C17, -O3)
├── app/                  SwiftUI app target (iOS + macOS)
│   ├── App.swift         @main, SwiftUI scene, view model
│   ├── Qwen.swift        Swift wrapper around the C API
│   ├── ModelDownloader.swift  fetch GGUF into Library/Caches/
│   ├── bridge.h          Swift ↔ C bridging header
│   ├── extensions/       URL.swift (resumable download) etc.
│   └── utils/            Trace, AnyCodable, background handler
├── tools/                Standalone C diagnostic utilities
│   ├── dump_kvs.c                 GGUF inspector
│   ├── q4k_diff.cpp               verify Q4_K dequant vs ggml
│   ├── q5k_diff.cpp               verify Q5_K dequant vs ggml
│   ├── q6k_diff.cpp               verify Q6_K dequant vs ggml
│   └── q8k_diff.cpp               verify Q8_K round-trip vs ggml
├── docs/                 Design docs + chat template spec
├── QwenHaiku.xcodeproj/  Xcode project (iOS + macOS app target)
├── PLAN.md               Roadmap + open questions
└── LICENSE               Apache 2.0
```

## Building

**Standalone CLI (no Xcode needed):**

```bash
cd llm && make             # writes Build/cli/llm at repo root
QWEN_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf ../Build/cli/llm --single "Hello, my name is"
```

The Makefile builds out-of-source: artifacts go into `Build/cli/`
at the repo root, so the `llm/` directory stays clean. The CLI
supports `--self-test` (synthetic-weight smoke test, no model file
needed), `--single "prompt"` (one-shot generation), and `--repl`
(interactive chat using the Qwen3 chat template). `make clean`
nukes `Build/cli/`.

**iOS / macOS app:**

```
open QwenHaiku.xcodeproj
```

The app downloads the model on first launch (~504 MB, cached in
`Library/Caches/Qwen/`) and offers a single prompt → completion
flow. Both platforms supported (iOS 18+, macOS 15+).

## What the runner supports

- Qwen3.5-0.8B hybrid architecture (Gated DeltaNet SSM + softmax
  attention every 4th layer)
- Q4_K_M GGUF with mixed Q4_K / Q5_K / Q6_K / Q8_0 / fp32 / fp16
  tensors, no offline conversion step
- Interleaved MRoPE with section partitioning
- Per-head Q-norm / K-norm
- Attention output gate (sigmoid)
- Byte-level BPE tokenizer parsed straight from the GGUF
- Streaming KV cache (single-token autoregressive decode)
- Greedy and top-k sampling with temperature

See `PLAN.md` for what's deferred, what's known to work, and the
chat template state machine spec.

## License

Apache 2.0. See `LICENSE`.
