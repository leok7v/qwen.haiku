#!/usr/bin/env bash
# Smoke-test parity gate for the C runner.
# Runs --self-test and four documented prompts at temperature 0, then
# md5-compares BOTH the continuation text AND the sampled token IDs
# against checked-in baselines.
#
# Two gates per prompt:
#   - stdout md5  vs tools/bench.baseline.md5         (TIGHT: any
#                                                      stdout byte
#                                                      shift fails)
#   - tokens md5  vs tools/bench.baseline.tokens.md5  (LOOSE: same
#                                                      argmax tokens
#                                                      survive ULP
#                                                      drift)
# Token mode is the parity gate that survives threading + NEON
# rewrites that shift logits by a few ULP but keep the argmax stable.
#
# Exit 0: every check matched. Exit non-zero: a mismatch or build error.
#
# Usage:   tools/bench.sh                  (uses default GGUF path)
#          QWEN_GGUF=/path/model.gguf tools/bench.sh
#          tools/bench.sh --update         (rewrite both baselines)

set -u
cd "$(dirname "$0")/.."

LLM="Build/cli/llm"
BASELINE_STDOUT="tools/bench.baseline.md5"
BASELINE_TOKENS="tools/bench.baseline.tokens.md5"
DEFAULT_GGUF="$HOME/Library/Containers/io.github.leok7v.QwenHaiku/Data/Library/Caches/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf"
: "${QWEN_GGUF:=$DEFAULT_GGUF}"
export QWEN_GGUF
export LLM_TRACE_TOKENS=1

PROMPTS=(
    "capital_of_france:The capital of France is"
    "once_upon_a_time:Once upon a time"
    "python_is:Python is a programming language that"
    "hello_my_name_is:Hello, my name is"
)
MAX_NEW=30
UPDATE=0
if [[ "${1:-}" == "--update" ]]; then UPDATE=1; fi

if [[ ! -x "$LLM" ]]; then
    echo "bench: $LLM missing - run (cd llm && make) first" >&2
    exit 2
fi
if [[ ! -f "$QWEN_GGUF" ]]; then
    echo "bench: GGUF not found at $QWEN_GGUF" >&2
    exit 2
fi

fail=0
new_baseline_stdout=""
new_baseline_tokens=""
tmp_err=$(mktemp)
trap 'rm -f "$tmp_err"' EXIT

md5_of() {
    if command -v md5 >/dev/null 2>&1; then
        printf '%s' "$1" | md5 -q
    else
        printf '%s' "$1" | md5sum | cut -c1-32
    fi
}

check_md5() {
    local kind="$1" name="$2" md5="$3" baseline="$4"
    if (( UPDATE == 1 )); then return 0; fi
    local expected
    expected=$(awk -v n="$name" '$2==n {print $1}' "$baseline" 2>/dev/null)
    if [[ -z "$expected" ]]; then
        echo "bench: $name $kind MISSING baseline" >&2
        fail=1
    elif [[ "$md5" != "$expected" ]]; then
        echo "bench: $name $kind FAIL  got=$md5  expected=$expected" >&2
        fail=1
    else
        echo "bench: $name $kind OK    $md5"
    fi
}

st=$("$LLM" --self-test 2>/dev/null)
if ! grep -q "argmax=65" <<<"$st" || \
   ! grep -q "argmax=67" <<<"$st" || \
   ! grep -q "argmax=68" <<<"$st"; then
    echo "bench: --self-test FAIL (expected argmax 65/67/68)" >&2
    fail=1
else
    echo "bench: --self-test OK"
fi

for entry in "${PROMPTS[@]}"; do
    name="${entry%%:*}"
    prompt="${entry#*:}"
    out=$("$LLM" --single "$prompt" --max-new $MAX_NEW --temperature 0 2>"$tmp_err")
    tokens=$(grep '^\[tok\] ' "$tmp_err")
    md5_stdout=$(md5_of "$out")
    md5_tokens=$(md5_of "$tokens")
    new_baseline_stdout+="$md5_stdout  $name"$'\n'
    new_baseline_tokens+="$md5_tokens  $name"$'\n'
    check_md5 stdout "$name" "$md5_stdout" "$BASELINE_STDOUT"
    check_md5 tokens "$name" "$md5_tokens" "$BASELINE_TOKENS"
done

if (( UPDATE == 1 )); then
    printf '%s' "$new_baseline_stdout" > "$BASELINE_STDOUT"
    printf '%s' "$new_baseline_tokens" > "$BASELINE_TOKENS"
    echo "bench: baselines rewritten ($BASELINE_STDOUT, $BASELINE_TOKENS)"
    exit 0
fi

if (( fail == 0 )); then
    echo "bench: PASS (self-test + ${#PROMPTS[@]} prompts, stdout + tokens both green)"
fi
exit $fail
