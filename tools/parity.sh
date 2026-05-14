#!/usr/bin/env bash
# parity.sh - greedy-mode token-by-token comparison between our runner
# and llama.cpp's llama-cli on the same GGUF.
#
# At temperature=0 both samplers reduce to argmax, so identical
# weights + identical math should produce identical token sequences.
# Any divergence is a real numerical or tokenizer bug worth chasing.
#
# Run from repo root: tools/parity.sh
set -u
OURS="${OURS:-./Build/cli/llm}"
# llama.cpp renamed the single-shot CLI to `llama-completion`;
# `llama-cli` is now conversation/chat-only. Allow override via env.
LLAMA_CLI="${LLAMA_CLI:-/Users/leo/github.com/ggml-org/llama.cpp/build/bin/llama-completion}"
GGUF="${QWEN_GGUF:-$HOME/Library/Containers/io.github.leok7v.QwenHaiku/Data/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf}"
N="${N:-32}"
# Use "set" rather than "<<<" so each prompt is one $1.
prompts=(
    "The capital of France is"
    "Once upon a time"
    "Python is a programming language that"
    "Hello, my name is"
)
fail=0
for p in "${prompts[@]}"; do
    # Ours: --single echoes the prompt before the continuation; carve
    # off everything before/including the prompt.
    ours_raw="$("$OURS" --single "$p" --temperature 0 --max-new "$N" 2>/dev/null \
        | awk '/^---$/{flag=1; next} flag')"
    ours_tail="${ours_raw#*$p}"
    # Theirs: llama-completion in single-turn raw mode. Counter-
    # intuitive: --log-disable suppresses the generation output too
    # (not just banners), so we keep logging on, send the metal/load
    # noise to stderr, and strip the echoed prompt from stdout
    # ourselves.
    theirs_raw="$("$LLAMA_CLI" -m "$GGUF" -p "$p" -n "$N" --temp 0 \
        -st -no-cnv --no-warmup -ngl 0 2>/dev/null)"
    theirs="${theirs_raw#$p}"
    # Strip trailing whitespace; both runners may add a final \n.
    ours_clean="$(printf '%s' "$ours_tail" | sed -e 's/[[:space:]]*$//')"
    theirs_clean="$(printf '%s' "$theirs"  | sed -e 's/[[:space:]]*$//')"
    if [ "$ours_clean" = "$theirs_clean" ]; then
        echo "parity: OK    \"$p\""
    else
        echo "parity: DIFF  \"$p\""
        echo "  ours:   $ours_clean"
        echo "  theirs: $theirs_clean"
        fail=1
    fi
done
if [ "$fail" = 0 ]; then
    echo "parity: PASS (greedy token-for-token vs llama-cli)"
else
    echo "parity: FAIL"
fi
exit "$fail"
