#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# tests/parity_chat.py - multi-turn chat parity test: our llm vs llama.cpp.
#
# Drives both binaries through the SAME 3-turn conversation at
# temperature=0 (greedy) and compares per-turn assistant replies
# token-for-token. Greedy mode makes the comparison deterministic
# regardless of RNG/seed implementation differences — both samplers
# reduce to argmax-of-logits, so identical weights + identical math
# should produce identical token sequences. Any divergence is a real
# numerical or tokenizer bug worth chasing.
#
# Why Python (not bash):
#   - llama-server lifecycle (Popen + wait-for-/health + terminate)
#     is fiddly in bash, clean in Python with subprocess + urllib.
#   - The /v1/chat/completions JSON payload is naturally expressed
#     with stdlib `json`; bash would shell out to jq for every turn.
#   - Multi-turn requires accumulating a Python list of messages
#     across iterations; bash arrays of JSON strings are painful.
#
# Stdlib only — no `requests` or other deps. Run from repo root:
#
#   tests/parity_chat.py
#
# Env knobs:
#   QWEN_GGUF             path to the GGUF (default: ./tmp/...)
#   LLAMA_SERVER          path to llama-server binary
#   OURS                  path to our llm binary
#   PARITY_PORT           local port for llama-server (default 8123)
#   PARITY_MAX_NEW        tokens per turn (default 30)

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT     = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_GGUF  = os.path.join(REPO_ROOT, "tmp", "Qwen3.5-0.8B-Q4_K_M.gguf")
DEFAULT_SERVE = "/Users/leo/github.com/ggml-org/llama.cpp/build/bin/llama-server"
DEFAULT_OURS  = os.path.join(REPO_ROOT, "Build", "cli", "llm")

GGUF         = os.environ.get("QWEN_GGUF",   DEFAULT_GGUF)
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", DEFAULT_SERVE)
OURS         = os.environ.get("OURS",         DEFAULT_OURS)
PORT         = int(os.environ.get("PARITY_PORT",    "8123"))
MAX_NEW      = int(os.environ.get("PARITY_MAX_NEW", "30"))

# The three turns the C run_chat_test uses. Sharing them lets a
# future "compare against the C chat-test hashes" extension lay
# right on top.
TURNS = [
    "Hi! Just say hello back.",
    "What did I just ask?",
    "Thanks!",
]


def fail(msg):
    print("parity-chat: FAIL — " + msg, file=sys.stderr)
    sys.exit(1)


def boot_server():
    """Start llama-server, wait for /health, return Popen handle."""
    if not os.path.isfile(LLAMA_SERVER):
        fail("llama-server not found at " + LLAMA_SERVER)
    if not os.path.isfile(GGUF):
        fail("GGUF not found at " + GGUF)
    args = [
        LLAMA_SERVER, "-m", GGUF,
        "--port", str(PORT),
        "--ctx-size", "4096",
        "--no-warmup",
        # No GPU offload — match what our llm runs (pure CPU).
        "-ngl", "0",
        # Use the GGUF's actual tokenizer.chat_template (the same
        # Jinja our llm/jinja-template.c hand-translates). Without
        # --jinja, llama-server uses a built-in fallback template
        # that differs from the GGUF's, so the model sees different
        # framing bytes from each side and parity drifts.
        "--jinja",
    ]
    proc = subprocess.Popen(args,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    # /health flips to 200 as soon as the HTTP layer is up — which
    # is BEFORE the model is loaded into memory. A request at that
    # point returns 503 "Loading model". Poll with a tiny actual
    # completion until it succeeds, which means the model is loaded.
    probe_url  = "http://127.0.0.1:" + str(PORT) + "/v1/chat/completions"
    probe_body = json.dumps({
        "messages":   [{"role": "user", "content": "x"}],
        "max_tokens": 1,
        "stream":     False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    deadline = time.time() + 90.0
    ready    = False
    while time.time() < deadline and not ready:
        if proc.poll() is not None:
            fail("llama-server exited during startup")
        try:
            req = urllib.request.Request(
                probe_url, data=probe_body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as r:
                if r.status == 200:
                    ready = True
        except urllib.error.HTTPError:
            # 503 = still loading model. Keep polling.
            time.sleep(1.0)
        except urllib.error.URLError:
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    if not ready:
        proc.terminate()
        fail("llama-server did not become ready within 90s")
    print("parity-chat: llama-server up on port " + str(PORT))
    return proc


def ask_server(messages):
    """POST /v1/chat/completions with messages list, return the
    assistant reply text. Temperature 0 for greedy."""
    payload = {
        "messages":       messages,
        "temperature":    0.0,
        "top_p":          1.0,
        "max_tokens":     MAX_NEW,
        "stream":         False,
        "cache_prompt":   True,  # let server reuse prior KV
        # Match our llm's default framing: the gen prompt pre-fills
        # an empty `<think>\n\n</think>\n\n` block, model jumps
        # straight to content. Without this kwarg llama-server opens
        # a real `<think>` block and the reply lands in
        # `reasoning_content`, not `content`, making the diff a wash.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:" + str(PORT) + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120.0) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return resp["choices"][0]["message"]["content"]


def run_ours():
    """Run our llm --chat with greedy preset + all turns; parse the
    per-turn replies from stdout. Returns list[str]."""
    if not os.path.isfile(OURS):
        fail("our llm binary not found at " + OURS + " (run make in llm/)")
    args = [OURS, "--chat", "--preset", "greedy",
            "--max-new", str(MAX_NEW)]
    for t in TURNS:
        args += ["-p", t]
    env = os.environ.copy()
    env["QWEN_GGUF"] = GGUF
    res = subprocess.run(args, capture_output=True, env=env,
                         timeout=180.0)
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", errors="replace"))
        fail("our llm --chat exited " + str(res.returncode))
    out = res.stdout.decode("utf-8", errors="replace")
    # The run_chat output shape:
    #   --- turn 1/N ---
    #   [user] ...
    #   [assistant] <reply text>
    #   --- turn 2/N ---
    #   ...
    replies = []
    # Split on the assistant marker, then trim each block at the
    # next "--- turn" or end of string.
    parts = out.split("[assistant] ")
    for i in range(1, len(parts)):
        chunk = parts[i]
        m = re.search(r"\n---\s+turn ", chunk)
        if m:
            chunk = chunk[:m.start()]
        replies.append(chunk.rstrip("\n"))
    if len(replies) != len(TURNS):
        fail("parsed " + str(len(replies)) + " replies from our llm, "
             "expected " + str(len(TURNS)))
    return replies


def compare(ours, theirs):
    """Strip trailing whitespace + assistant-end markers from both
    sides before comparing. Returns list[(turn_idx, is_match)]."""
    results = []
    for i, (a, b) in enumerate(zip(ours, theirs)):
        a_clean = a.rstrip()
        b_clean = b.rstrip()
        results.append((i, a_clean == b_clean, a_clean, b_clean))
    return results


def main():
    print("parity-chat: GGUF=" + GGUF)
    print("parity-chat: " + str(len(TURNS)) + " turns,"
          + " max_new=" + str(MAX_NEW) + ", greedy (temp=0)")
    ours = run_ours()
    print("parity-chat: ours replies collected")
    server = boot_server()
    try:
        history = []
        theirs  = []
        for turn in TURNS:
            history.append({"role": "user", "content": turn})
            reply = ask_server(history)
            theirs.append(reply)
            history.append({"role": "assistant", "content": reply})
            print("parity-chat: llama-server turn " + str(len(theirs))
                  + "/" + str(len(TURNS)) + " replied "
                  + str(len(reply)) + " bytes")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
    results = compare(ours, theirs)
    fail_count = 0
    for (i, ok, a, b) in results:
        if ok:
            print("parity-chat: turn " + str(i + 1) + ": OK")
        else:
            fail_count += 1
            print("parity-chat: turn " + str(i + 1) + ": DIFF")
            print("  ours:   " + repr(a))
            print("  theirs: " + repr(b))
    if fail_count == 0:
        print("parity-chat: PASS (" + str(len(TURNS))
              + " turns identical greedy decode)")
        return 0
    print("parity-chat: FAIL (" + str(fail_count) + "/"
          + str(len(TURNS)) + " turns diverged)")
    # Some divergence is expected: even at temperature=0 (greedy),
    # accumulated fp32 ULP drift across a long multi-turn prompt
    # can flip an argmax pick, and that pick changes every later
    # token. Single-token / single-turn parity (tools/parity.sh)
    # is the gold standard; multi-turn parity is a stress test
    # that catches LARGER drift (whole sentences off), not 1-ULP
    # rounding.
    return 1


if __name__ == "__main__":
    sys.exit(main())
