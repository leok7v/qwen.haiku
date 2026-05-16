#!/usr/bin/env bash
# rnd/sweep.sh — cross-host build + test sweep.
#
# Drives the full llm CLI battery (`--chat-test` is the load-bearing
# parity gate) on every host in `rnd/build+test.md`. Writes captured
# output to `rnd/sweep-results/<host>.out` so post-hoc diffing is easy.
#
# Skips unreachable hosts (5s connect timeout). Run from the repo root:
#
#     ./rnd/sweep.sh                 # all hosts
#     ./rnd/sweep.sh halo2 khadas    # subset
#
# Authoritative gate: every host must reproduce the Mac's --chat-test
# hashes (d2ba984b228fff7a / f12add080343c386 / a99b9d0705cc0220).

set -u
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"

mkdir -p rnd/sweep-results

# Authoritative hashes (Mac CLI --chat-test).
EXPECTED="d2ba984b228fff7a f12add080343c386 a99b9d0705cc0220"

ALL_HOSTS="apple halo2 x mbp15 agi mb-air-2012 pixel5 khadas"
HOSTS="${*:-$ALL_HOSTS}"

# GGUF path is host-specific; the Mac and most ssh boxes use:
GGUF_MAC="/Users/leo/Library/Containers/io.github.leok7v.QwenHaiku/Data/Library/Application Support/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf"

# Sync the source tree to a host. Excludes Build outputs + AI tooling
# dirs. Uses rsync for diff-only updates so the wire is short on
# subsequent runs.
sync_to() {
    local h=$1 path=$2
    rsync -az --delete \
        --exclude=Build/ \
        --exclude=tmp/ \
        --exclude=.agents \
        --exclude=.claude \
        --exclude=.gemini \
        --exclude='*.gguf' \
        --exclude=DerivedData/ \
        ./ "$h:$path/"
}

# Run the standard battery; emit a final "summary:" line for grepability.
# NOTE on quoting: $path and $gguf often contain '$HOME' as a literal,
# which the remote shell must expand. So we drop single-quotes inside
# the ssh command body — the outer double-quote of the ssh argument
# expands $path locally to the LITERAL `$HOME/qwen.haiku` string, then
# remote bash sees `cd $HOME/qwen.haiku` and expands properly.
remote_battery() {
    local h=$1 path=$2 gguf=$3
    ssh "$h" "set -e; cd $path && \
        echo '--- build'            && make -C slm 2>&1 | tail -10 && \
        echo '--- self-test'        && ./Build/cli/llm --self-test    2>&1 | tail -5  && \
        echo '--- think-test'       && ./Build/cli/llm --think-test   2>&1 | tail -3  && \
        echo '--- jinja-test'       && ./Build/cli/llm --jinja-test   2>&1 | tail -3  && \
        echo '--- chunked-test'     && ./Build/cli/llm --chunked-test 2>&1 | tail -5  && \
        echo '--- agent-test'       && ./Build/cli/llm --agent-test   2>&1 | tail -3  && \
        echo '--- chat-test (hash gate)' && QWEN_GGUF=$gguf ./Build/cli/llm --chat-test 2>&1 | tail -10 && \
        echo '--- qwen-test'        && ./Build/cli/qwen-test          2>&1 | tail -5  && \
        echo '--- bench'            && QWEN_GGUF=$gguf ./Build/cli/llm --bench         2>&1 | tail -5"
}

# Extract the three chat-test hashes (pass 0 only) and the bench line,
# then emit a single grep-friendly summary line per host.
summarize() {
    local out_file=$1 host_label=$2
    local got bench
    got=$(grep -oE 'hash=[0-9a-f]{16}' "$out_file" | head -3 | sed 's/hash=//' | tr '\n' ' ' | sed 's/ $//')
    bench=$(grep -E '^bench: ' "$out_file" | tail -1 | sed 's/^bench: //')
    local verdict
    if [ "$got" = "$EXPECTED" ]; then verdict="PASS"; else verdict="FAIL"; fi
    echo "summary: $host_label: $verdict | $bench | hashes=$got" | tee -a "$out_file"
}

skip_msg() {
    local h=$1 out_file=$2
    echo "summary: $h: SKIPPED (unreachable)" | tee "$out_file"
}

reachable() {
    local h=$1
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" true 2>/dev/null
}

run_apple() {
    local out=rnd/sweep-results/apple.out
    echo "host: Apple M-series (local), NEON+dotprod, macOS arm64" > "$out"
    {
        make -C slm 2>&1 | tail -10
        echo '--- self-test'    ; ./Build/cli/llm --self-test    2>&1 | tail -5
        echo '--- think-test'   ; ./Build/cli/llm --think-test   2>&1 | tail -3
        echo '--- jinja-test'   ; ./Build/cli/llm --jinja-test   2>&1 | tail -3
        echo '--- chunked-test' ; ./Build/cli/llm --chunked-test 2>&1 | tail -5
        echo '--- agent-test'   ; ./Build/cli/llm --agent-test   2>&1 | tail -3
        echo '--- chat-test'    ; ./Build/cli/llm --chat-test    2>&1 | tail -10
        echo '--- qwen-test'    ; ./Build/cli/qwen-test          2>&1 | tail -5
        echo '--- bench'        ; QWEN_GGUF="$GGUF_MAC" ./Build/cli/llm --bench 2>&1 | tail -5
        if [ -x ./Build/cli/qwen-swift-cli ]; then
            echo '--- swift-cli chat-test'
            QWEN_GGUF="$GGUF_MAC" ./Build/cli/qwen-swift-cli chat-test 2>&1 | tail -10
        fi
    } >> "$out"
    summarize "$out" "Apple M-series"
}

run_ssh_host() {
    local h=$1 label=$2 remote_path=$3 gguf=$4
    local out=rnd/sweep-results/$h.out
    echo "host: $label, $h" > "$out"
    if ! reachable "$h"; then
        skip_msg "$h" "$out"
        return
    fi
    sync_to "$h" "$remote_path" 2>&1 | tail -5 >> "$out"
    remote_battery "$h" "$remote_path" "$gguf" >> "$out" 2>&1
    summarize "$out" "$label"
}

run_pixel5() {
    local out=rnd/sweep-results/pixel5.out
    echo "host: Pixel 5 (Snapdragon 765G), NEON+dotprod, Android arm64" > "$out"
    if ! adb get-state 2>/dev/null | grep -q device; then
        skip_msg pixel5 "$out"
        return
    fi
    # Cross-compile + push happens here. NDK path is well-known.
    local NDK=/opt/homebrew/share/android-ndk/toolchains/llvm/prebuilt/darwin-x86_64
    if [ ! -x "$NDK/bin/clang" ]; then
        skip_msg pixel5 "$out"
        echo "summary: pixel5: SKIPPED (NDK not found at $NDK)" >> "$out"
        return
    fi
    # Build Android target (no libcurl; tools.c compiled out via
    # -DLLM_NO_TOOLS). Drop -static — bionic libc doesn't statically
    # link cleanly; the binary just uses the system libc on-device.
    "$NDK/bin/clang" --target=aarch64-linux-android24 \
        -O3 -std=c17 -Wall -Wextra \
        -Wno-unused-parameter -Wno-missing-braces \
        -DLLM_CLI -DLLM_NO_TOOLS \
        -fvectorize -fslp-vectorize \
        slm/slm.c -lm -o Build/cli/llm-android 2>&1 | tail -5 >> "$out"
    adb push Build/cli/llm-android /data/local/tmp/llm 2>&1 | tail -3 >> "$out"
    # If GGUF isn't already on-device, push it (~530MB; one-time).
    if ! adb shell '[ -f /data/local/tmp/qwen.gguf ]' 2>/dev/null; then
        echo 'pushing gguf to device (one-time)...' >> "$out"
        adb push "$GGUF_MAC" /data/local/tmp/qwen.gguf 2>&1 | tail -3 >> "$out"
    fi
    {
        echo '--- chat-test'
        adb shell 'taskset -a f0 sh -c "QWEN_GGUF=/data/local/tmp/qwen.gguf /data/local/tmp/llm --chat-test"' 2>&1 | tail -10
        echo '--- bench'
        adb shell 'taskset -a f0 sh -c "QWEN_GGUF=/data/local/tmp/qwen.gguf /data/local/tmp/llm --bench"' 2>&1 | tail -8
    } >> "$out"
    summarize "$out" "Pixel 5"
}

# ---------------------------------------------------------------------------
# Run requested hosts.
# ---------------------------------------------------------------------------

for h in $HOSTS; do
    case "$h" in
        apple)        run_apple ;;
        halo2)        run_ssh_host halo2        "AMD Zen 5 (Strix Halo)"  '$HOME/qwen.haiku' '$HOME/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        x)            run_ssh_host x            "Intel Haswell"           '$HOME/qwen.haiku' '$HOME/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        mbp15)        run_ssh_host mbp15        "Intel Ivy Bridge (mbp15)" '/Users/agi/qwen.haiku' '/Users/agi/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        agi)          run_ssh_host agi          "Intel Ivy Bridge (agi)"   '$HOME/qwen.haiku' '$HOME/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        mb-air-2012)  run_ssh_host mb-air-2012  "Intel Ivy Bridge (Win)"   '/c/Users/leo/qwen.haiku' '/c/Users/leo/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        pixel5)       run_pixel5 ;;
        khadas)       run_ssh_host khadas       "Amlogic A311D (Khadas)"   '$HOME/qwen.haiku' '$HOME/qwen.haiku/tmp/Qwen3.5-0.8B-Q4_K_M.gguf' ;;
        *) echo "unknown host: $h (known: $ALL_HOSTS)" >&2 ;;
    esac
done

echo
echo '==== sweep summary ===='
grep -h '^summary:' rnd/sweep-results/*.out 2>/dev/null
