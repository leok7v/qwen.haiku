# Snapshot/Restore + Atomic Web-Research Tool — Implementation Plan

**Author**: distilled from a design discussion between Leo and Claude on
2026-05-16. Picked up here so a fresh context can execute end-to-end
without re-deriving the design.

## TL;DR

Rebuild the `--tools` path so the model only ever does **one** tool
call per turn (no multi-step tool dance the 0.8B gets confused by),
backed by:

1. SSM+KV **snapshot/restore** primitives (cheap because the Mamba
   half is fixed-size memcpy-able state).
2. An **atomic `websearch` tool** that does DDG → pick URL → fetch →
   distill all in C, returning curated content instead of raw
   snippets.
3. **Snapshot before** the model emits `<tool_call>`, **restore after**
   the tool dispatches, **inject the distilled content** as if the
   assistant just wrote it as preamble, then let the model finish the
   answer. The model never sees a `<tool_response>` round-trip.

Plus: graduated **debug levels 0..9** with `trace()` calls throughout
so the CLI's `--tools --verbose --repl` and the macOS Debug tab
both surface every prompt, reasoning chunk, tool query, raw search
result, distilled text, snapshot/restore event, and final answer in
a controlled-verbosity stream.

Plus: replace the macOS UI's Debug chip toggle with a 0..9 stepper.

## HARD RULES — read before writing code

The following discipline is **not optional** for any code touched in
this plan. New code that violates these rules will be rejected on
review.

### SESE (single-entry / single-exit)

See `.agents/skills/sese/SKILL.md` and `.agents/memory/coding-philosophy.md`.

- Every function has **one** entry (the opening brace) and **one** exit
  (the closing brace `return r;` where `r` was constructed along the
  way).
- **No** mid-function `return`. No early-bailout `return`.
- **No** `goto`. **No** `break` (except inside `switch` grammar).
  **No** `continue`.
- **No** synthetic `bool ok = ...; if (!ok) { ... }` flag layered on
  top of the function's real state. The function's "did it work?"
  signal is the *return value or the output struct's field that
  already records success* (e.g. `tool_result.ok`).
- Use a **result variable** initialised to the default outcome at the
  top of the function and assigned in branches.

Example (good):
```c
static int do_thing(...) {
    int r = -1;                    // default outcome
    if (precondition_ok) {
        do_work();
        r = compute_result();
    }
    return r;
}
```

Example (bad):
```c
static int do_thing(...) {
    if (!precondition_ok) return -1;   // early return — REJECTED
    do_work();
    return compute_result();
}
```

### Other ground rules (from `.agents/memory/`)

- **Per-symbol `__attribute__((unused))`** for functions intentionally
  compiled but unused in some build configuration. **NO** target-wide
  `-Wno-unused-function`. See `feedback-per-symbol-unused.md`.
- **No Co-Authored-By trailer** on commits. See
  `feedback-no-coauthored-trailer.md`.
- **Single canonical API** — collapse parallel variants aggressively.
  Don't add new public functions when a flag/field would do. See
  `feedback-single-canonical-api.md`.
- **REPL, not server** — qwen.haiku is single-client, append-only KV.
  Don't add server-style multi-turn dispatch loops on top of the
  embedded agent loop. See `feedback-repl-not-server.md`.
- **Named enums over magic numbers** for CLI dispatch chains. See
  `feedback-enum-not-magic-numbers.md`.
- **`struct chars` over `char [N]`** for any string built up
  dynamically. The fixed-size sites that remain (`char nm[48]` in the
  forward path) are intentional — don't churn those.
- **Max line width < 80 cols.** Comma-align multi-call stanzas.
- **READ-DO-NOT-GREP-REPLACE.** Sed-based rewrites must be carefully
  bounded; a previous regex sweep across function boundaries trashed
  `slm_load_weights` because `snprintf(nm, ...)` lines exist in both
  GGUF tensor-name lookups and trace-call blocks. Read the function,
  identify each block, edit precisely.
- **Match neighbouring code first.** Don't introduce stylistic
  novelties that fight the rest of the file.
- **No comments that explain WHAT the code does** — well-named
  identifiers do that. Comments are for WHY: hidden constraints,
  workarounds for specific bugs, subtle invariants.
- **No dead code.** If a function is genuinely unreferenced after a
  refactor, delete it. `__attribute__((unused))` is for sometimes-
  unused-in-some-build, not always-unused-everywhere.

### Bench-parity gate

After every commit in this plan, **`Build/cli/slm --chat-test` must
print `08547e07b5084555` three times across both passes.** The hashes
are the load-bearing regression gate. The new tool-call architecture
changes only the `--tools` path; the plain growing-KV chat path is
unaffected and the gate exercises only that path with default
`ctrl.tools=true` (no actual tool dispatch happens because there's
no network in the test prompts). If the hashes move, something is
wrong with the changes — investigate before continuing.

## Codebase landmarks the implementation touches

### Files to modify

| File | What changes |
|---|---|
| `slm/model.c` | Add snapshot/restore. Add KV/SSM cache snapshot impl. |
| `slm/slm.h` | Public `slm_ctx_snapshot` / `_restore` / `_snapshot_free`. |
| `slm/tools.c` | Fix `tools_ddg_parse` to extract URLs. Rewrite `tools_websearch` to atomic chain (search → pick → fetch → distill). |
| `slm/agent.c` | Drop `agent_cascade_run` (and its support code: `PROMPT_ROUTER/EVALUATOR/SYNTHESIZER`, `agent_extract_url`, `agent_cascade_phase`, `agent_mux_cb`). |
| `slm/slm.c` | Modify `slm_generate`'s embedded agent loop to use snapshot/restore + inject-as-preamble pattern instead of the `<tool_response>` round-trip. Update `run_repl` / `run_single` / `run_chat` to drop the `agent_cascade_run` dispatch branch — they now use plain `slm_generate` with `ctrl.tools=true` and the new in-loop snapshot/restore. Add graduated `trace()` calls per debug level. |
| `app/App.swift` | Replace Debug ChipToggle with a 0..9 stepper. `vm.debug: Bool` → `vm.debugLevel: Int32`. Update `loadModel` / `send` / `clearChat` / `streamAssistant` references. |
| `slm/utils/trace.c` | (no change — already exists, ready for use) |

### Existing relevant code (file:line snapshots — may drift)

- `slm/model.c::slm_ctx_create` — currently `calloc` + `kv_init` +
  `ssm_cache_init` + `rng_seed`. Snapshot needs to deep-copy these
  caches.
- `slm/model.c::kv_init` / `kv_free` / `kv_row_k` / `kv_row_v` —
  KV cache layout: `_Float16 k[n_layers * max_position * n_kv_heads * head_dim]`.
- `slm/model.c::ssm_cache_init` / `ssm_cache_free` — SSM cache.
  `g_qh_trace_fp` is file-static; not part of ctx state.
- `slm/slm.c::slm_generate` (line ~850) — embedded agent loop. Where
  `<tool_call>` parses, dispatches, injects `<tool_response>`.
- `slm/tools.c::tools_ddg_parse` (line ~430) — two-pass HTML scanner.
  Currently extracts title + snippet, drops the URL.
- `slm/agent.c::agent_cascade_run` — to be deleted.
- `slm/slm.c::run_repl` / `run_single` / `run_chat` — three sites that
  currently branch on `with_tools` to dispatch into
  `agent_cascade_run`. To be flipped to plain `slm_generate`.
- `app/App.swift::SLMViewModel::debug` — Bool, exposed as a ChipToggle.

## Existing tests / gates

- `Build/cli/slm --chat-test` — parity gate, must remain
  `08547e07b5084555 × 3`.
- `Build/cli/slm --all-tests` — runs qwen_self_test, chunked_self_test,
  jinja_self_test, agent_parser_test, tools_self_test (live network),
  slm_think_test. Should still pass.
- `Build/cli/qwen-test` — must build with zero warnings.
- `xcodebuild -destination 'platform=macOS'` — must build.
- New smoke test: `Build/cli/slm --tools --verbose --repl` then type
  `What's the BTC price today?`. With debug 5+ we should see, in
  order: model emits tool_call, tool dispatches DDG (raw body
  trace), top URL picked, fetch HTTP status, distill output, restore
  event, model resumes with answer.

## Implementation plan

### Step 0 — Pre-flight

- Read `.agents/memory/coding-philosophy.md` and
  `.agents/skills/sese/SKILL.md`. Internalise SESE.
- Read `slm/slm.c::slm_generate` (the embedded agent loop) end-to-end
  before touching it. ~120 lines; understand the current flow:
  1. push user prompt into messages
  2. format turn delta, tokenize, append ids
  3. loop (up to SLM_GENERATE_TOOL_ITER_CAP):
     a. init think filter
     b. `slm_generate_raw(...)` with `slm_split_trampoline` cb
     c. if tool_call_ready: parse, dispatch, build tool_response
        injection, tokenize, append ids; LOOP back.
     d. else: done.
- Read `slm/tools.c::tools_ddg_parse`. Understand the two-pass
  structure. Plan the URL extraction.
- Read `slm/agent.c::agent_cascade_run` to confirm what gets deleted.
- Run `./Build/cli/slm --chat-test` and save the baseline hashes
  (should be `08547e07b5084555` × 3).

### Step 1 — Snapshot/restore primitives (model.c + slm.h)

Add to `slm/slm.h`:
```c
struct slm_snapshot;  // opaque

// Capture the full per-conversation mutable state at a point in time:
//   - KV cache rows [0..pos)
//   - SSM conv_state ring + ssm_state + conv_head
//   - pos, rng state
// Returns a heap-allocated snapshot; caller owns. NULL on OOM.
struct slm_snapshot * slm_ctx_snapshot(const struct slm_ctx * c);

// Restore the ctx's mutable state from the snapshot. The snapshot
// must have been taken from the same model+ctx layout (i.e. the same
// slm_model with the same hyperparams). Cheap memcpy.
void slm_ctx_restore(struct slm_ctx * c,
                     const struct slm_snapshot * s);

// Free a snapshot's allocations.
void slm_snapshot_free(struct slm_snapshot * s);
```

Add struct + impls in `slm/model.c`. Snapshot stores:
- `int32_t kv_used` (= `c->pos`)
- `_Float16 * k_copy` (size = `n_layers * pos * n_kv_heads * head_dim * sizeof(_Float16)`)
- `_Float16 * v_copy` (same)
- `float * conv_state_copy` (full ring size)
- `float * ssm_state_copy` (full)
- `int32_t * conv_head_copy` (n_layers)
- `int32_t pos`
- `struct rng rng`

**Don't snapshot `ids` / `messages`** — those are append-only history
caller-managed (the tool flow's caller decides whether to truncate
them). Document this.

SESE per function. Compile, run parity gate.

### Step 2 — Fix `tools_ddg_parse` URL extraction (tools.c)

DDG-lite result link shape (real markup, observed):
```html
<a class="result-link" rel="nofollow" href="https://example.com/page">Title</a>
```

Walk back from each `rel="nofollow"` match to find the preceding
`href="..."` within the same `<a>` opener. Emit:
```
1. <title>
   <url>
   <snippet>
   
2. <title>
   ...
```

Numbered so the picker can refer to indices later if useful. Output
goes via `chars_printf` into the result buffer that `tools_websearch`
returns.

SESE per function. Build, no parity change expected (this only
affects the `--tools` path which the parity gate doesn't exercise).

### Step 3 — Atomic `tools_websearch` chain (tools.c)

Rewrite `tools_websearch(query, max_results, out)`:
1. DDG scrape (as today) → get raw HTML body.
2. Parse to a small `struct tools_search_hit { char * title; char * url; char * snippet; } hits[MAX_HITS=8]` using the URL-aware pass above.
3. Pick the top URL: take `hits[0].url` (heuristic: DDG ranks; first non-aggregator). Stub for now — first hit period.
4. `tools_fetch(top_url, 15, &fetch_res)`.
5. If fetch ok: `tools_distill(fetch_res.body, …, &distill_res)`. Truncate distilled text to say 4000 chars.
6. Format result as:
   ```
   Web search result for "<query>":
   Source: <top_url>
   
   <distilled body>
   ```
7. Trace events (per debug level):
   - 1+: query
   - 3+: number of hits, chosen URL
   - 5+: hit list (first 3 titles)
   - 7+: raw HTML body size, fetch HTTP status
   - 9+: full raw HTML body (truncated to 2k), full distill output

SESE. Free all intermediates on every path (use result variable).
Build, parity unchanged.

### Step 4 — Wire snapshot/restore in `slm_generate` (slm.c)

In `slm_generate`'s agent loop, replace the `<tool_response>`
injection with a snapshot+restore+preamble injection:

Before the decode loop:
```c
struct slm_snapshot * snap = NULL;
if (c->ctrl.tools) { snap = slm_ctx_snapshot(c); }
```

On tool dispatch (current branch when tool_call_ready):
```c
if (snap != NULL) {
    slm_ctx_restore(c, snap);
    // Also rewind the ids array (delete tokens added during the
    // aborted generation).
}
struct chars preamble = {0};
chars_printf(&preamble,
    "Based on a web search, here is the relevant information:\n\n"
    "%s\n\nNow I will answer the user's question.\n\n",
    result_text);
// Tokenize preamble + assistant-frame-ending sequence; prefill it.
```

The cleanest preamble framing: NOT a synthetic user turn (that
re-asks the user). Inject as a continuation of the assistant turn:
"I searched and found: ...". The KV state at the snapshot point ends
right at the assistant-prefix tokens; appending plain content tokens
continues the assistant turn naturally. The model picks up from
there and writes the actual answer.

Then continue the decode loop ONCE (no second iteration) — the
model now writes the final answer with the search result baked into
its in-context buffer.

`SLM_GENERATE_TOOL_ITER_CAP` becomes 2 (one for the tool_call emit,
one for the answer). The cap is just a safety net.

SESE the loop body. Parity gate: still 08547e07 × 3 (this code path
doesn't fire in chat-test because the test prompts don't trigger
tool calls).

### Step 5 — Drop `agent_cascade_run` (agent.c + slm.c)

- Delete from agent.c: `PROMPT_ROUTER`, `PROMPT_EVALUATOR`,
  `PROMPT_SYNTHESIZER`, `agent_extract_url`, `agent_mux_cb` +
  `agent_mux_cb_fn`, `agent_cascade_phase`, `agent_cascade_run`.
  Keep: `AGENT_TOOL_WEBSEARCH/FETCH/DISTILL`, `agent_call_param`,
  `agent_free_calls`, `agent_carve`, `agent_json_string`/_value/
  _skip_ws, `agent_parse_json_call`, `agent_parse_tool_calls`,
  `agent_dispatch`, `agent_capture_box` (+ cb fn), `agent_run`
  (the --ask path), `agent_parser_test`.
- Update slm.c's `run_repl` / `run_single` / `run_chat`: drop the
  `if (with_tools) { ... cascade ... }` branch. With tools, just
  set `ctrl.tools = true` on the ctx and use plain `slm_generate`
  — the new snapshot/restore logic handles the tool dance.
- Trim CLI auto-clamp logic if no longer relevant (T=0.0 etc. is
  still wanted for tool-emit determinism).

Build, parity, `slm --tools --verbose --repl` smoke.

### Step 6 — Graduated debug levels via `trace()` calls

Define implicit levels (no enum needed; inline `if (level >= N)`
checks):
- **0**: silent except errors.
- **1**: high-level events (turn start, tool dispatch start/end,
  snapshot/restore, errors).
- **3**: + tool args, hit count, chosen URL.
- **5**: + reasoning text on (when ctrl.think), top hit list.
- **7**: + raw web body size, fetch HTTP status, full preamble text.
- **9**: + full raw HTML body (truncated), full distill output.

Sprinkle `if (c->ctrl.debug >= N) trace(...)` calls in tools.c,
slm.c's agent loop, agent.c. Use the per-level threshold each site
believes appropriate. Re-document the levels in slm.h's footnote (4)
for ctrl.debug.

### Step 7 — UI: Debug toggle → 0..9 stepper (App.swift)

In `SLMViewModel`:
- Change `var debug: Bool = true` to `var debugLevel: Int32 = 1`.
- Update `loadModel` / `clearChat` / `streamAssistant`:
  `let debugLv: Int32 = self.debugLevel` (drop the `? 1 : 0` ternary).

In `ChatView::toggleRow`:
- Replace `ChipToggle("Debug", isOn: $vm.debug)` with a stepper:
  ```swift
  Stepper("Debug \(vm.debugLevel)",
          value: $vm.debugLevel, in: 0...9)
      .font(.caption.monospaced())
      .controlSize(.small)
      .fixedSize()
  ```

Or use a compact `Picker` segmented control. Pick the one that fits
the toggle row's space budget on macOS at minWidth 600.

Build with xcodebuild, smoke-launch the app.

### Step 8 — Smoke test + parity + commit

- `make -C slm clean && make` (CLI + qwen-test, zero warnings)
- `./Build/cli/slm --chat-test` → 08547e07 × 3 PASS
- `./Build/cli/slm --tools --verbose --repl` then `What's the BTC price today?`
  - With debug at default (1): minimal events
  - Bump `--debug 5` or via env var: should see hit list, chosen URL
  - Bump to 9: see raw HTML and distill output
- `xcodebuild -destination 'platform=macOS'`: BUILD SUCCEEDED.
- Launch macOS app, type the same question with Tools on and Debug
  stepper at 5: Debug tab shows the full pipeline trace.

Commit per step (or per ~2 steps) so the diff is reviewable.
Suggested commit ordering:
- Commit A: snapshot/restore + DDG URL fix.
- Commit B: atomic websearch + slm_generate rewiring.
- Commit C: drop cascade + UI stepper + graduated trace levels.

Each commit must build clean and pass parity.

## Open design questions (resolve while implementing)

1. **`ids` array on snapshot/restore**: the snapshot captures `pos`
   (the KV position cursor) but the ctx's `ids` array (token history
   for repetition penalty + ctx_tokens accessor) keeps growing. On
   restore, should we truncate `ids` back to its snapshot length? The
   conservative answer is YES — they should stay in sync with `pos`.
   Add `size_t ids_count` to the snapshot, restore truncates
   `c->ids.count`.
2. **Heuristic for top URL pick**: first hit is OK for now. Future:
   skip aggregator domains (wikipedia, reddit) for time-sensitive
   queries like "price today", prefer financial-data sites. Out of
   scope for this commit.
3. **Distill truncation budget**: 4000 chars is a guess. Tune by
   eyeballing the BTC-price smoke test output.
4. **Preamble text**: the exact wording matters for the 0.8B's
   continuation quality. Worth trying 2-3 variants and picking the
   one that gives the cleanest BTC-price answer.

## What to NOT do in this commit

- Don't add multi-step / multi-tool support (one tool call per turn
  by design).
- Don't refactor the SIMD dispatcher / tensor kernels.
- Don't change the chat-template framing — bench-parity gate locks
  it down.
- Don't migrate more `fprintf(stderr, …)` to `trace()` beyond the
  sites this plan touches.
- Don't change the macOS minWidth/minHeight or window-tab settings.
- Don't touch the existing iOS build path.

## Memory snapshot to update after completion

Write a project memory entry summarising what landed: shape of
`slm_ctx_snapshot`, the snapshot+restore+preamble flow that replaces
the cascade, the new debug levels semantics. Drop the
`project-slm-repl-websearch-plan.md` memory entry (superseded).
