#!/usr/bin/env bash
# perf-bench.sh - compare ./Build/cli/llm (CPU-NEON, threaded) against
# llama.cpp's llama-bench on both Metal and CPU for the same GGUF.
#
# llama-bench's standard report is pp512 (prompt-processing 512
# tokens) and tg128 (token-generation 128 tokens). We mirror that by
# feeding our runner a ~512-token prompt and asking for 128 new
# tokens. Token counts will not be exactly 512 (depends on how the
# real text tokenizes) but they will be in the same ballpark, and
# tok/s is the metric we care about.
set -u
OURS="${OURS:-./Build/cli/llm}"
LLAMA_BENCH="${LLAMA_BENCH:-/Users/leo/github.com/ggml-org/llama.cpp/build/bin/llama-bench}"
GGUF="${QWEN_GGUF:-$HOME/Library/Containers/io.github.leok7v.QwenHaiku/Data/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf}"
N_GEN="${N_GEN:-128}"
REPS="${REPS:-3}"

# Build a long-ish prompt: a ~2 KB paragraph that tokenizes to
# roughly 512 BBPE tokens with this vocab.
PROMPT_TEXT="$(printf 'The history of the silicon transistor begins in the late 1940s at Bell Laboratories, where William Shockley, John Bardeen and Walter Brattain demonstrated the first working point-contact transistor. The bipolar junction transistor that followed was the cornerstone of all integrated circuits until the metal-oxide-semiconductor field-effect transistor, the MOSFET, became practical in the early 1960s. Through successive nodes - microns, then submicron, then nanometers - the MOSFET shrank by orders of magnitude. Each generation roughly doubled the transistor count per chip every two years, a regularity that Gordon Moore observed and that came to be known as Moore'\''s Law. By the 2000s a single die held a billion transistors, by the 2020s well over fifty billion. The thermal and quantum-mechanical limits that once seemed comfortably distant began to bite: gate-oxide tunnelling, drain-induced barrier lowering, hot-carrier injection, and the simple wall of how much heat a square centimetre of die can shed under a heat spreader. The industry responded with strained silicon, copper interconnects, high-k dielectrics, FinFETs, gate-all-around transistors, and chiplet packaging that side-stepped the reticle limit. None of these were silver bullets; each bought a node or two of headroom before the next bottleneck appeared.')"
echo "=== ours (./Build/cli/llm, CPU NEON + threaded) ==="
OURS_LOG="$(mktemp)"
for i in $(seq 1 "$REPS"); do
    QWEN_GGUF="$GGUF" "$OURS" --single "$PROMPT_TEXT" --temperature 0 \
        --max-new "$N_GEN" >/dev/null 2>"$OURS_LOG"
    grep -E "^pp:" "$OURS_LOG" | tail -1
done
rm -f "$OURS_LOG"

echo
echo "=== llama-bench Metal (default -ngl 99) ==="
"$LLAMA_BENCH" -m "$GGUF" -p 512 -n "$N_GEN" -r "$REPS" 2>/dev/null \
    | grep -E "^\| qwen35"

echo
echo "=== llama-bench CPU (-ngl 0) ==="
"$LLAMA_BENCH" -m "$GGUF" -p 512 -n "$N_GEN" -r "$REPS" -ngl 0 2>/dev/null \
    | grep -E "^\| qwen35"
