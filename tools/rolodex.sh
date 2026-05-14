#!/usr/bin/env bash
# rolodex.sh - run each starter prompt through the CLI with the in-app
# chat preset and print the raw model output. Used to eyeball quality
# without going through the SwiftUI surface.
set -u
OURS="${OURS:-./Build/cli/llm}"
GGUF="${QWEN_GGUF:-$HOME/Library/Containers/io.github.leok7v.QwenHaiku/Data/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf}"
SEED="${SEED:-42}"
N="${N:-160}"
SYS="${SYS:-You are helpful assistant.}"
prompts=(
    "What is the meaning of life?"
    "What is consciousness, really?"
    "Why do we dream?"
    "What is love?"
    "How do we find happiness?"
    "What makes us human?"
    "What is time?"
    "What is the nature of reality?"
    "Why is there something rather than nothing?"
    "What is free will?"
    "What happens when we die?"
    "How do memories shape who we are?"
)
export QWEN_GGUF="$GGUF"
i=0
for q in "${prompts[@]}"; do
    i=$((i+1))
    framed="$(printf '<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$SYS" "$q")"
    echo "=========================="
    echo "[$i/${#prompts[@]}] $q"
    echo "--------------------------"
    "$OURS" --single "$framed" --temperature 0.7 --top-k 40 --top-p 0.9 \
        --min-p 0.05 --rep-penalty 1.25 --rep-window 64 \
        --min-new 8 --seed "$SEED" --max-new "$N" 2>/dev/null \
        | awk '/^---$/{flag=1; next} flag' \
        | awk 'BEGIN{p=0} {
            if (p) { print; next }
            i = index($0, "</think>")
            if (i > 0) {
                rest = substr($0, i + length("</think>"))
                print rest
                p = 1
            }
        }'
done
echo "=========================="
