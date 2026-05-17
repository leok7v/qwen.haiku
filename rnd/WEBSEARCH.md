# WEBSEARCH R&D — alternatives to DuckDuckGo HTML scraping

**Status:** R&D notes captured 2026-05-17. Empirical probe results across 6 candidate APIs and ~20 representative queries. Conclusions and a proposed architecture, but **no production code yet** — design pass required before implementation.

**Motivation:** Our current `websearch` tool does DDG HTML scrape → URL pick → fetch the chosen URL → HTML strip → inject. The pipeline fails repeatedly:
1. DDG returns ad-redirect URLs (`/y.js?ad_domain=...`) and help/disclaimer pages as top hits.
2. Many "winning" pages are heavy-JS / SPA where our HTML stripper extracts only the `<title>` (Merriam-Webster: 200 KB → 55 chars).
3. CamelCase-concatenated nav chrome drowns the actual data (CoinMarketCap: actual price `$78,382` buried at offset 1290 in 1450 chars of "CryptocurrenciesRankingCategoriesHistoricalSnapshots...").
4. Live-data queries (time, weather, current price) require a successful fetch + extraction every time, with no structured fallback.

The 0.8B model can't recover from these failure modes — it either refuses ("I cannot search the internet directly...") or invents data.

---

## Tested candidates

### 1. Wikipedia OpenSearch + REST summary (★★★★★)

Two-step chain, both endpoints free, **WMF User-Agent policy** is the only hard requirement (`User-Agent: appname/version (contact-url; contact-email)`).

```
# Step 1: resolve query → canonical article titles
curl -H 'User-Agent: ...' \
  'https://en.wikipedia.org/w/api.php?action=opensearch&search=proton&format=json&limit=5'
# → ["proton",["Proton","Proton Holdings","Proton AG","Proton therapy","Proton Mail"],...]

# Step 2: fetch clean structured summary for hit[0]
curl -H 'User-Agent: ...' \
  'https://en.wikipedia.org/api/rest_v1/page/summary/Proton'
# → JSON with .title, .description, .extract (200-500 char clean paragraph)
```

**Coverage probe** (10 queries, all stopword-stripped to noun phrase):

| Query                          | OpenSearch hit       | Summary extract (first ~80 chars)                                      |
|--------------------------------|----------------------|-------------------------------------------------------------------------|
| `proton`                       | Proton               | "A proton is a stable subatomic particle, symbol p, H+, or 1H+..."     |
| `Tom Cruise`                   | Tom_Cruise           | "Thomas Cruise Mapother IV is an American actor and film producer..."  |
| `4th president United States`  | (redirect) → James_Madison | "James Madison was an American statesman, diplomat, and Founding Father..." |
| `capital Australia`            | Capital_of_Australia | "Canberra is the capital city of Australia and the largest population..." |
| `Canberra`                     | Canberra             | "Canberra is the capital city of Australia..."                          |
| `ubiquitous`                   | Ubiquitous → Omnipresence | "Omnipresence or ubiquity is the attribute of being present anywhere..." |
| `acai`                         | Acai → Açaí_palm     | "The açaí palm (/əˈsaɪ.iː/ ə-SY-ee; Portuguese: [asaˈi]..."             |
| `vanilla`                      | Vanilla              | "Vanilla is a spice derived from orchids of the genus Vanilla..."       |
| `strep throat`                 | Strep_throat → Streptococcal_pharyngitis | "...streptococcal sore throat, is pharyngitis caused by..." |
| `running shoe`                 | Running_shoe → Sneakers | "Sneakers (US) or trainers (UK), also known by a wide variety..."      |
| `James Madison`                | James_Madison        | "James Madison was an American statesman, diplomat..."                  |
| `boil egg`                     | Boiled_egg           | "Boiled eggs are a food typically made using chicken eggs..."           |

**Misses** (OpenSearch returns `[]`):
- `vanilla bean origin` — extra words; works with just `vanilla`
- `strep throat symptoms` — extra word; works with just `strep throat`
- `PlayStation 5 price` — extra word; works with just `PlayStation 5`

**Implication:** OpenSearch wants **noun phrase only**. The SLM either needs to be prompted to issue clean queries, or we need a stopword-stripper between SLM and the API call. Trivial regex: strip `\b(price|cost|definition|meaning|origin|symptoms|cure|how to|where|when|why|what is)\b`.

**Strengths:**
- 200-500 char clean prose extracts, no nav chrome, no HTML.
- Redirects resolve automatically (e.g., `Capital_of_Australia` → article about Canberra).
- WMF infra is fast and rarely down.
- Free, generous rate limits (~200 req/s).
- Stable URL schemes since 2015.

**Weaknesses:**
- Encyclopedic only — no live data (current price, current time, current weather).
- OpenSearch needs noun-phrase queries.
- Some intents resolve to disambiguation pages (e.g., `Vanilla` is the spice, but `Ubiquitous` → `Omnipresence` which is theology-leaning).

---

### 2. mwmbl (api.mwmbl.org) (★★★☆☆)

Non-profit open-source search engine, ~500M URL index. Free, no key, no User-Agent requirement.

```
curl 'https://api.mwmbl.org/api/v1/search/?s=proton'
# → JSON list of {url, title (segments with is_bold), extract (segments), source}
```

**Coverage probe** (13 queries):

| Query                       | Top-3 quality                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------|
| `proton`                    | ★★★★★ Wikipedia hit[0], extract IS the definition                                     |
| `PlayStation 5 price`       | ★★★★☆ Wikipedia PlayStation_5 article, then real review sites                          |
| `boil an egg`               | ★★★★☆ ehow.com + Wikipedia Boiled_egg                                                  |
| `USD EUR exchange rate`     | ★★★★★ coincodex extract literally says "You can convert 100 USD to **91.71 EUR**"     |
| `best running shoes women`  | ★★★★☆ Tom's Guide, TechRadar, CNET, WIRED                                              |
| `bitcoin price today`       | ★★★☆☆ Wikipedia Bitcoin first (no live price), then bitcoin.org, etc.                  |
| `Tom Cruise age`            | ★☆☆☆☆ **Nile cruise tour packages** match "cruise"; Tom Cruise filmography at rank 6   |
| `4th president`             | ★★★☆☆ Daaji (Heartfulness) at rank 1, then James Madison at rank 2-3                   |
| `capital of Australia`      | ★★★☆☆ Australian_Capital_Territory at rank 1 (extract mentions Canberra), then noise   |
| `ubiquitous meaning`        | ★★☆☆☆ Punjabi/Urdu dictionaries, Wikipedia "Ubiquitous (album)"                        |
| `acai pronunciation`        | ★★★☆☆ MTL Blog at rank 1, Wikipedia Açaí_palm at rank 2 (extract has IPA)             |
| `vanilla origin`            | ★☆☆☆☆ Vanilla Ice rapper, gizmodo DNS error page, PlayStation repair                  |
| `current time Tokyo`        | ★★☆☆☆ Wikipedia Japan_Standard_Time, time.is (no live time visible), Tokyo Review     |
| `nearest coffee shop`       | ★☆☆☆☆ Kopi_luwak Wikipedia article, generic blog spam                                  |

**Strengths:**
- **Extracts are pre-segmented** — no HTML strip needed. The JSON already has `title.value` and `extract.value` strings.
- Wikipedia articles surface naturally for encyclopedic queries.
- For reviews / how-to queries, mwmbl's curated index gives real content sites (Tom's Guide, CNET, WIRED) without DDG's ad pollution.
- Extracts sometimes contain the literal answer (the USD/EUR result above).

**Weaknesses:**
- Tiny index → "Tom Cruise age" finds Nile cruise pages because Tom Cruise's main article is outranked.
- Disambiguation is weak — "vanilla origin" returns Vanilla Ice instead of the spice.
- No live data (time, weather, current price).
- "Nearest X" doesn't work — needs location-aware ranking.
- Quality varies wildly: top-tier for some queries, garbage for others. Not safe to use as a single source.

**Best use:** **Secondary fallback** when Wikipedia OpenSearch returns nothing and the query isn't time/weather/currency-specific. Skip the URL-pick + fetch + distill — just use the pre-segmented extracts directly.

---

### 3. Specialized live-data APIs (★★★★★ each, in their niche)

These return **structured JSON answers**, not pages to scrape. ~100-500 byte responses, no extraction layer.

#### a) `ip-api.com` — IP geolocation

```
curl 'http://ip-api.com/json/?fields=status,country,regionName,city,lat,lon,timezone,isp,query'
# → {"status":"success","country":"United States","regionName":"California",
#    "city":"Union City","lat":37.5958,"lon":-122.0191,
#    "timezone":"America/Los_Angeles","isp":"Comcast Cable Communications, LLC",
#    "query":"71.198.226.204"}
```
- Free, no key, no HTTPS for free tier (HTTPS on paid).
- Solves "Determine my location by IP" in one call.
- Lat/lon enables chaining to weather/time APIs.

#### b) `timeapi.io` — current time per timezone

```
curl 'https://timeapi.io/api/Time/current/zone?timeZone=Asia/Tokyo'
# → {"year":2026,"month":5,"day":18,"hour":8,"minute":11,"seconds":22,
#    "dateTime":"2026-05-18T08:11:22.5144158","timeZone":"Asia/Tokyo",
#    "dayOfWeek":"Monday","dstActive":false}
```
- Free, no key, HTTPS.
- IANA timezone names accepted.
- Solves "what time is it in X" definitively.

`worldtimeapi.org` was **down** during probe (connection reset by peer). `timeapi.io` is the working alternative.

#### c) `open-meteo.com` — weather forecast

```
curl 'https://api.open-meteo.com/v1/forecast?latitude=37.77&longitude=-122.42&current=temperature_2m,weather_code&daily=temperature_2m_max,temperature_2m_min,weather_code&timezone=America/Los_Angeles&forecast_days=2'
# → JSON with .current.temperature_2m, .daily.temperature_2m_max[0..1],
#   .daily.temperature_2m_min[0..1], .daily.weather_code[0..1]
```
- Free, no key, HTTPS.
- WMO weather codes (clear-sky=0, fog=45, etc.) need a small lookup table.
- Forecast up to 16 days, hourly granularity.
- Needs lat/lon — pair with Wikipedia OpenSearch on city name, or use Open-Meteo's geocoding endpoint.

#### d) `coingecko.com` — crypto prices

```
curl 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
# → {"bitcoin":{"usd":77772}}
```
- Free, no key.
- 33 byte response.
- Solves "Bitcoin price today" in one call. CoinMarketCap fiasco averted entirely.

#### e) Currency exchange — TBD

`frankfurter.app` returned a 301 redirect during probe (might need `-L`). `exchangerate.host` is another option. **Not yet confirmed working** — needs follow-up.

---

## What DDG was buying us — and where it fails

DDG's value proposition: arbitrary-string query in, ranked URL list out, covers the long tail.

**Where DDG is the only option:**
- Local queries (`nearest coffee shop`, `Is Target open right now?`) — none of the alternatives have location-aware ranking.
- Niche reviews / consumer comparison (`best running shoes`, `top rated air fryer`) — mwmbl handles some, but its index is thin.
- News / very recent events not yet indexed by Wikipedia or specialized APIs.
- Anything where the user's intent isn't pre-classifiable.

**Where DDG fails (and a specialized API would not):**
- Encyclopedic facts → Wikipedia REST is 10× cleaner.
- Current time → timeapi.io is exact.
- Weather → open-meteo is exact.
- Crypto / FX → coingecko / frankfurter are exact.
- IP geolocation → ip-api is exact.
- Wikipedia-able definitions → REST summary beats every commercial result.

---

## Proposed architecture: multi-tool, intent-routed

Replace single `websearch(query)` tool with **per-intent narrow tools**. The Qwen3.5 XML tool spec supports multiple tools natively — the model emits whichever `<function=...>` matches the user's intent.

```
wikipedia(query)        # encyclopedic facts, definitions, biographies
time_now(timezone)      # current time anywhere
weather(location)       # current + forecast
currency(from, to)      # exchange rates
crypto_price(symbol)    # BTC, ETH, ...
ip_geo()                # the caller's own IP location
websearch(query)        # fallback for the long tail (mwmbl → DDG)
```

Why narrow tools beat one generic `websearch`:

1. **No HTML pipeline.** Each tool returns ~100-500 byte structured answer. No 200 KB fetches. No `<script>`/`<style>` stripping. No CamelCase nav chrome. No URL-pick second pass.
2. **Intent is explicit.** The model picks the tool by name; failure modes are observable per tool. We can A/B test each one independently.
3. **Token budget collapses.** Current preamble: 1-5 KB of distilled HTML. New: 200 char structured JSON. The 0.8B's working memory fits one tool result trivially.
4. **No URL-pick step.** Currently the URL-pick model invocation always returns `1=fallback` and the heuristic sort wins. With narrow tools there's no URL to pick.
5. **Clean fallback.** `websearch(query)` stays as the catch-all but is rarely needed.

**Cascade for what's now `websearch`:**

```
websearch(query):
  result = mwmbl_search(query)               # pre-segmented extracts, no scrape
  if result is empty or all-low-quality:
    result = wikipedia_opensearch_chain(query)  # one more shot before DDG
  if still empty:
    result = ddg_scrape(query)               # last resort, existing pipeline
  return result.extracts[:3]                 # send 3 snippets, not URLs to fetch
```

**Risks:**

- The 0.8B may not reliably route to the right tool. Mitigation: explicit examples in the system prelude ("Use `wikipedia(...)` for facts, `time_now(...)` for time, ..."). Tested community fix in `qwen35-enhanced.jinja` is to include a follow-up example block; we already do similar.
- More tools = more JSON in the system prompt. ~300 chars per tool spec × 7 tools = ~2.1 KB system block. Tolerable; we currently spend that much on K_TOOL_INSTRUCTIONS alone.
- API churn: third parties can change schemas or rate-limit. Mitigation: each tool has its own failure surface; one going down doesn't break the others. Wikipedia is the most stable bet; the others are commodity.

---

## Implementation order (when we go)

1. **Wikipedia chain first.** Highest information density per byte. Replaces 60-70% of current `websearch` calls. Easiest to test (offline-cacheable fixtures).
2. **`time_now` + `weather` + `crypto_price` + `currency`.** Each is a 30-line tool. Together they kill the four worst e2e probe categories.
3. **`ip_geo`.** Specific to the "my location" intent. Easy.
4. **`mwmbl_search` replaces the DDG hit-list parser.** No HTML strip needed; result body is the formatted hit list. Keep DDG as next-level fallback.
5. **Drop the URL-pick phase entirely.** With narrow tools nothing needs URL picking. The fallback `websearch` cascade just returns the top-3 snippets.

Each step is independently shippable. Steps 1-3 deliver almost all the value without touching the DDG path.

---

## Open questions

- **WMF User-Agent compliance.** Wikipedia REST policy says the UA must include "appname/version (contact-url; contact-email)". We need a real contact URL. Easy: `qwen-haiku/0.1 (https://github.com/leok7v/qwen.haiku; <leo's email>)`.
- **Geocoding for `weather(location)`.** Open-Meteo has its own geocoding endpoint; or chain to Wikipedia OpenSearch and pull `lat/lon` from the article's coordinates field. Need to verify the latter works for cities.
- **Stopword stripping for Wikipedia OpenSearch.** Probably 10 regexes ("price", "cost", "definition", "meaning", "origin", "symptoms", "how to", "where is", "when did", "what is"). Should this run in C (deterministic) or be in the system prompt (model self-strips)?
- **Currency API.** Frankfurter redirected; need to confirm working alternative. `https://api.frankfurter.app/latest` (current spelling) may work with `-L`. Or `exchangerate.host`.
- **Local queries** (`nearest coffee shop`, `Is Target open`). These need a Maps API (Google Places, OpenStreetMap Nominatim, Apple MapKit). Not free at scale; punt for now and treat as out-of-scope.
- **Tool routing prompt.** Need example block in the system prelude showing 3-4 sample (query → tool) mappings so the 0.8B picks correctly. Validate against the e2e probe set.

---

## Probe artifacts

Raw command outputs captured in this session:
- `https://api.mwmbl.org/api/v1/search/?s=proton` → wiki hit[0] + 6 lower-quality hits
- `https://en.wikipedia.org/w/api.php?action=opensearch&search=...` → title list per query
- `https://en.wikipedia.org/api/rest_v1/page/summary/...` → clean extracts
- `http://ip-api.com/json/...` → 150 byte location JSON
- `https://timeapi.io/api/Time/current/zone?timeZone=Asia/Tokyo` → 180 byte time JSON
- `https://api.open-meteo.com/v1/forecast?...` → forecast JSON
- `https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd` → `{"bitcoin":{"usd":77772}}`

Re-run any of these to refresh; all should be idempotent.

---

## Bottom line

The current "single generic `websearch`" tool is doing **two unrelated jobs poorly**:
(1) intent classification (the URL-pick step) and
(2) content extraction (the HTML distill step).

Both jobs are unnecessary for the common case. Most user questions map cleanly to a structured-data API that returns the answer directly. Replace the generic tool with **6-7 narrow tools that each return ~200 byte JSON answers**, and keep DDG as the explicit fallback for the long tail. The 0.8B model's working memory and tool-routing accuracy both improve when the tools are narrower; failure modes become observable per tool rather than tangled in the URL-pick + HTML-distill chain.

Wikipedia OpenSearch + REST summary is the highest-leverage first move. Builds in ~100 LOC, no new dependencies (libcurl already there, JSON we can parse with a tiny scanner since the response shape is fixed). Validate against the existing e2e probe set: I expect 4/4 PASS once probes 1 (Tokyo time → `time_now`), 2 (Bitcoin → `crypto_price`), 3 (SF weather → `weather`), 4 (USD/EUR → `currency`) are all routed to their specialized tools.
