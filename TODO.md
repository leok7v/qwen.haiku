# TODO

Pending decisions Leo needs to make. Agents read this at session start and surface a short reminder per entry. Edit when an item is greenlit / dropped / postponed; don't auto-implement.

- **tests/test-nihs.c** — Needle-in-a-Haystack context-retention test. Drop a known token / fact at position N in a long context, ask the model to recall it after K turns. Goal: verify slm's growing-KV chat preserves long-range info across the 0.8B's small attention budget; track recall % as a regression metric. Discussed 2026-05-16; awaiting go.
