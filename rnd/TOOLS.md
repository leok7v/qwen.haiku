# TOOLS — handoff for the next R&D pass

**Status:** Multi-tool architecture landed 2026-05-17 / 18 (commits `1a2bdd2` → `e3f088a`). Replaced the previous DDG-HTML-scrape + URL-pick + fetch+distill pipeline. This doc is a fast on-ramp for the next agent picking up where we left off. Pair with [`rnd/WEBSEARCH.md`](WEBSEARCH.md) for the original R&D notes that motivated the rewrite.

---

## What's in the box

Six narrow per-intent tools. Each returns a small structured-data answer (typically 100-500 bytes). No HTML pipeline, no URL-pick second pass.

| Tool | Backend | Returns | Use for |
|------|---------|---------|---------|
| `wikipedia(query)` | en.wikipedia.org OpenSearch + REST summary | 200-500 char extract | Facts, definitions, biographies, concepts |
| `time_now(timezone)` | `timeapi.io` | ISO 8601 + day of week + tz | "What time is it in X" |
| `weather(latitude, longitude)` | `api.open-meteo.com` | Current temp + 2-day high/low + WMO description | Weather (caller supplies lat/lon) |
| `crypto_price(symbol, vs)` | `api.coingecko.com` | "Current price of X: N USD" | "Price of bitcoin / ETH / ..." |
| `ip_geo()` | `ip-api.com` (HTTP) | City, region, country, lat/lon, timezone, ISP | "Where am I", "my location" |
| `websearch(query)` | `api.mwmbl.org` (open-source search) | Top-3 hits w/ pre-segmented snippets | Long-tail fallback |

UA for all calls: `qwen.haiku/0.1 (https://github.com/leok7v/qwen.haiku; leo.kuznetsov@gmail.com)`. Wikipedia's WMF policy requires this shape (`appname/version (contact-url; contact-email)`); the other APIs accept anything but a clean identifying UA is good citizenship.

Tool JSON specs are in `slm/agent.c::AGENT_TOOL_*` (one per tool, each ~5-10 lines). Dispatch is `agent_dispatch` in the same file. Implementations are in `slm/tools.c` (~1100 LOC after the rewrite).

---

## How the flow works end-to-end

```
USER:    "What time is it in Tokyo right now?"

         ↓ slm_generate iter 0
         model emits <tool_call><function=time_now>
                       <parameter=timezone>Asia/Tokyo</parameter>
                     </function></tool_call>
         filter recognises markers, captures inner body
         ↓
         agent_parse_tool_calls → 1 call parsed
         agent_dispatch → tools_time_now("Asia/Tokyo", &r)
         r.body = "Current time in Asia/Tokyo: Monday, 2026-05-18T10:43:50…"
         ↓ slm_ctx_restore(snap)
         inject canonical Qwen tool-response framing:
             <tool_call>…</tool_call><|im_end|>
             <|im_start|>user
             <tool_response>
             Current time in Asia/Tokyo: Monday, …
             </tool_response><|im_end|>
             <|im_start|>assistant
             <think>

             </think>

         ↓ slm_generate iter 1 (tool_round=false, discard_tool_calls=true)
         model emits "Here is the current time in Tokyo:\n\n**Monday, 2026-05-18T10:43:50…**"
         filter passes content through; stray <tool_call> markup (if any)
         is silently absorbed via discard_tool_calls.

USER sees:    Here is the current time in Tokyo:
              **Monday, 2026-05-18T10:43:50.132276**
```

The single most important architectural detail: **the post-tool preamble uses canonical Qwen `<tool_response>` framing**, not raw prose. The earlier "Tool result:\n\n%s\n\nAnswer the user's question..." injection put the model off-distribution and the 0.8B routinely emitted a duplicate `<tool_call>` as its "answer" instead. Search `slm/slm.c` for `CANONICAL QWEN tool-response framing` to find this block.

---

## The think filter — the two flags that matter

`struct slm_think_filter` in `slm/slm.c` has two flags worth understanding before touching the dispatch path:

- **`recognize_tool_calls`** — when true, the filter recognises `<tool_call>...</tool_call>` markers, routes the inner body into `f->tool_call`, and signals `tool_call_ready` on close (which halts decoding so the caller can dispatch). When false, the markup flows through as plain content (leaks). We set this `= with_tools` for every iter.

- **`discard_tool_calls`** — new flag for the post-dispatch iter. When true, the filter still recognises markers (keeps them out of visible content) but on close it silently clears the buffer and keeps decoding — no `tool_call_ready`, no halt. We set this `= with_tools && !tool_round` so it activates on iter 1+.

Without this pair, iter 1 had no good options: `recognize=false` leaked raw markup; `recognize=true` without discard halted decoding mid-answer. With both, the answer iter can write a clean response even if the model emits a stray tool_call inside it.

---

## Known model-side failure modes (and what works against them)

The 0.8B is small and has predictable failure patterns. Document each as you encounter new ones; the existing list:

1. **"I cannot provide real-time information" refusal on iter 1.** Fix: explicit "USE that result as the source of truth … Do NOT refuse" wording in the HARD RULES nudge in `jinja_emit_tools_prelude`. Keep nudge tool-agnostic (don't name `websearch` etc. — it anchors the model to that one tool and fights the per-tool descriptions in the `<tools>` block).

2. **Boilerplate sys prompt biases toward refusal.** `--repl` passed `"You are a helpful assistant."`; on tool-response turns the 0.8B then said "I'm sorry, I can't…". Empty system text (matching `--single` and the e2e test) fixes this. If you ever add system text back, validate against the e2e probes.

3. **Re-opening `<think>` on iter 1.** Even with the gen prompt's closed `<think></think>`, the model sometimes re-opens think and routes its actual answer to `.reasoning`. The user sees `[think] …` lines but `cb->content` stays empty. The e2e test now captures both streams; if you write a new test, do the same.

4. **Broken markup: `<tool_call>` without `<function=…>` opener.** Probe 1 (Wikipedia) in the e2e test reliably triggers this — model emits `<tool_call>\n<parameter=query>\nproton\n</parameter>\n</function>\n</tool_call>` (no `<function=wikipedia>` opener). Parser returns 0 calls. Two known fix directions, both unimplemented:
   - **Lenient parser**: infer function name from `<parameter=…>` names + tool descriptions. Risky if multiple tools share a parameter name.
   - **System prompt tightening**: a follow-up example block right after the existing `<tool_call>` template showing `<function=NAME>` clearly bound. Community's enhanced jinja does something like this.

5. **Model overrides tool data with training data.** The bitcoin probe gets `crypto_price` data but the 0.8B sometimes still writes the wrong price from training (saw $43,500 and $64,500 across runs while the actual was $77,772). The HARD RULES nudge addresses this textually ("trust the numbers, dates, names…") but the 0.8B isn't always trustworthy enough. Bigger model or instruct-tuning would help; in the meantime accept that on rare queries the model embellishes the answer with stale memory.

6. **URL-pick was a no-op.** Worth knowing for context: the previous DDG flow used a "model picks URL" intermediate phase. Across every e2e probe in every run, the model emitted `1` (or nothing → fallback to 1). It never actually chose. That entire phase was deleted in the rewrite.

---

## What `tools-e2e-test` does (and its known limits)

`tools_e2e_test` in `slm/slm.c` runs 5 live probes against a fresh ctx each — one per tool. Grades PASS if (a) the model emitted something parseable and (b) the captured reply (content + reasoning) is > 8 chars after stripping.

**Current pass rate: 4/5.** The failing probe is wikipedia (broken markup, see (4) above).

Known false-positive class on the gate: a probe that "passes" doesn't mean the model used the tool data. Probe 3 (bitcoin) has been seen to pass with 339 chars of "$64,500 from late 2024" — training data, not tool data. Tightening the gate to *require* a substring from the tool response (e.g., the actual `$77,772` from coingecko) is a clean follow-up.

The test takes 2-4 min on M-series at T=0 because each probe is a fresh prefill of a ~3.9 KB system block.

---

## Architecture rationale (one-liners)

- **Six tools, not one.** The single generic `websearch` did two unrelated jobs poorly: intent classification (URL-pick) and content extraction (HTML distill). Both jobs are unnecessary for the common case — most user questions map cleanly to a structured-data API that returns the answer directly.
- **No URL-pick step.** The model never actually picked. The sort by content-length was doing all the work.
- **No HTML distill.** Pages where the distiller worked were the easy cases (mostly Wikipedia). Pages where it failed (SPAs / heavy nav chrome) produced 50 chars from 200 KB and the model couldn't answer.
- **Canonical Qwen `<tool_response>` framing.** Off-distribution framing made the 0.8B emit duplicate tool_calls as answers. In-distribution framing → real answers.
- **Tool-agnostic HARD RULES nudge.** Naming specific tools in the nudge anchors the 0.8B to those tools and fights the per-tool descriptions. Generic "use the most specific tool listed above" wording is more stable across changes to the tool set.

---

## Things to try next (sorted by leverage)

1. **Lenient `<tool_call>` parser.** Probe 1 of `--tools-e2e-test` fails because the model omits the `<function=…>` opener. If exactly one tool has a matching parameter set, infer the function name. Cleanly raises e2e to 5/5.

2. **Stricter test gate.** Make `probe_ok` also require that some keyword from the tool response appears in the model's reply. Filters out "passed with stale training data" false positives like probe 3.

3. **Geocoding for `weather(location)`.** Currently `weather` requires explicit `latitude` and `longitude`. The model has to either (a) know them from training (works for famous cities), or (b) call `wikipedia(city)` first to dig coordinates from the article. Option b is fragile. Open-Meteo has a geocoding endpoint (`https://geocoding-api.open-meteo.com/v1/search?name=Tokyo`). Adding a single `tools_open_meteo_geocode` helper and chaining it inside `tools_weather` when only the name is provided would make the tool more usable.

4. **`currency(from, to)`.** Listed in WEBSEARCH.md as a candidate but not implemented. Frankfurter redirected during the original probe; `https://api.frankfurter.app/latest?from=USD&to=EUR` with `-L` (curl follow-redirect) should work. Or `https://api.exchangerate.host/latest?base=USD&symbols=EUR`. Pick one and add it.

5. **News tool.** Lots of "current events" questions go to `websearch` and fail because mwmbl's index is sparse for breaking news. A free RSS-aggregator-style endpoint or a Hacker News / Reddit JSON feed would cover this.

6. **Local / "near me" tool.** None of the free APIs cover `nearest coffee shop` etc. WEBSEARCH.md punted this as out-of-scope (Maps APIs aren't free at scale). Skip unless you have a plan.

7. **Make the tool advert smarter.** Currently we emit all 6 tool specs into every system prompt — 3.9 KB. Most queries only need one tool. A heuristic that pre-classifies the query and emits only the 2-3 most-relevant tool specs would shrink the system block and arguably help routing. But you'd be reinventing the URL-pick layer in a different place. Probably not worth it yet.

---

## Files & where to look

- **`slm/tools.c`** — all six tool implementations, the tiny JSON value extractor (`tools_json_str/num/find_key/scope`), `tools_api_get` (clean API GET, no browser fingerprinting). LLM_NO_TOOLS branch at the bottom for libcurl-free builds.
- **`slm/agent.c`** — `AGENT_TOOL_*` JSON specs (~5-10 lines each), `agent_dispatch` that routes by name, `agent_parse_tool_calls` that pulls `<function=…>` calls out of model output.
- **`slm/jinja.c`** — `K_TOOL_INSTRUCTIONS` (the official Qwen3.5 template's tool format example) and the appended HARD RULES nudge. Both live inside `jinja_emit_tools_prelude`.
- **`slm/slm.c`** — `slm_generate` dispatch loop (the `if (snap != NULL && box.filter.tool_call_ready)` block), `slm_think_filter` definition, `tools_e2e_test` probes, `run_repl` system-prompt setup.
- **`rnd/WEBSEARCH.md`** — the R&D notes that motivated the rewrite. Six candidate APIs, ~20 query probes, design rationale.

Memory snapshots worth reading first: `project-tools-e2e-baseline`, `project-slm-atomic-tools`, `qwen35-tool-calling-community-findings`.

---

## One mental model that helps

Don't think of "tools" as a thing the model is great at and we just need to wire correctly. Think of it as: **the 0.8B is a calculator with bad handwriting**. It can read a prompt and write a structured tool call ~80% of the time. When it succeeds and the canonical `<tool_response>` comes back, it can usually quote the data correctly. Each step that fails (broken markup, off-distribution framing, contradictory system prompt, refusal-anchored phrasing) drops that ~80% by another 10-20 points. The win comes from removing failure surfaces, not from "training the model better" or "writing smarter prompts" beyond a point.

The DDG pipeline had ~5 failure surfaces stacked (DDG ad URLs, URL-pick that never picked, HTML strip that produced 55 chars from 200 KB, raw "Tool result:" framing, contradictory system prompt). The new pipeline has ~1 (model emitting broken markup). Keep removing surfaces. Don't add them back.
