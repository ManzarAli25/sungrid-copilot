# Classifier x filter-mode experiment — 2026-10-31 12:45

22 single-turn RAG cases. hit@5 = an expected source chunk is in the top 5.

| Config | Category acc. | hit@5 (filter only) | hit@5 (with fallback) | Fallback used | Misses |
|---|---|---|---|---|---|
| jev/strict | 22/22 | 19/22 | 22/22 | 3/22 | — |
| jev/widened | 22/22 | 21/22 | 22/22 | 1/22 | — |
| llm/strict | 22/22 | 19/22 | 22/22 | 3/22 | — |
| llm/widened | 22/22 | 21/22 | 22/22 | 1/22 | — |
