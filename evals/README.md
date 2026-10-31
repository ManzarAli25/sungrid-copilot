# Evaluation

```bash
python -m evals.run_evals                  # full run: agent + deterministic checks + DeepEval
python -m evals.run_evals --skip-deepeval  # deterministic layer only (no judge cost)
python -m evals.run_evals --compare        # classifier backend x filter mode retrieval experiment
python -m evals.run_evals --eval-mode hybrid --out run_hybrid   # Jev decides judge verdicts
```

Results are written to `evals/results/` (`latest.md` / `latest.json`, `compare.md`).

## Golden set: `golden.yaml`

35 hand-written cases, each traceable to a document in `data/docs/`:

| Type | Cases | What they cover |
|---|---|---|
| RAG | 23 | One or more per document; doc 06 from both the rebate and billing side; the "rebate not on my bill" cross-category case; an unanswerable question (the income threshold number is not in the docs); a policy trap (the multi-family tier is announced but not finalised) |
| Eligibility | 10 | Eligible, the $4,000 cap, each hard-gate failure (income, ZIP, installer, size), missing info, vague installer praise that must not count as approval |
| Multi-turn | 3 (within the above) | "yes they're approved", "what if I shrink it to 9 kW?", "how long do I have to appeal *that*?" |
| Off-topic | 2 | Weather, coding help |

`tests/test_golden.py` checks that every referenced chunk and category exists, so the golden
set cannot silently drift from the corpus.

## Two layers

**1. Deterministic checks.** Exact, free, run on every case, no LLM judge:
category accuracy, route, retrieval hit@5, tool called / not called, exact tool arguments,
verdict, and a hard-gate string check (ineligible ⇒ "not eligible" and no `$`).

**2. DeepEval metrics.** LLM-as-a-judge, for qualities code can't check:

| Metric | Applied to | Question it answers |
|---|---|---|
| Faithfulness | RAG | Is every claim in the answer supported by the retrieved chunks? |
| Answer Relevancy | RAG | Does the answer address what was asked? |
| Contextual Precision | RAG | Are relevant chunks ranked above irrelevant ones? |
| Contextual Recall | RAG | Did retrieval return everything the reference answer needs? |
| Tool Correctness | Eligibility | Right tool, right input parameters? |
| Hard gate (G-Eval) | Eligibility | Ineligible ⇒ reason only, no amount or "maybe"; eligible ⇒ exactly the service's amount |

The judge is `JUDGE_MODEL` (default `openai/gpt-6-luna`), a different model family from the
agent model (`anthropic/claude-haiku-5.5`), to avoid self-preference bias.

## Latest results

| Layer 1 check | Result |
|---|---|
| category, route | 35/35 |
| retrieval hit@5 | 23/23 |
| tool not called when it shouldn't be | 27/27 |
| tool args, verdict | 8/8 |
| hard gate (string check) | 4/4 |

| DeepEval metric | Mean | Passed |
|---|---|---|
| Faithfulness | 0.99 | 22/23 |
| Answer Relevancy | 0.96 | 23/23 |
| Contextual Precision | 0.95 | 23/23 |
| Contextual Recall | 1.00 | 23/23 |
| Tool Correctness | 1.00 | 8/8 |
| Hard gate (G-Eval) | 1.00 | 8/8 |

The one Faithfulness failure (`rag-cert-lapse`, 0.75) is borderline: the answer states the rule
correctly, then adds an over-cautious hedge that partly contradicts it.

### Classifier x filter-mode experiment (`--compare`, 22 single-turn RAG cases)

| Config | Category acc. | hit@5 (filter only) | hit@5 (with fallback) |
|---|---|---|---|
| jev / strict | 22/22 | 19/22 | 22/22 |
| jev / widened | 22/22 | 21/22 | 22/22 |
| llm / strict | 22/22 | 19/22 | 22/22 |
| llm / widened | 22/22 | 21/22 | 22/22 |

Widening the filter recovers 2 of the 3 cases a strict filter loses, and the fallback recovers
the rest. Jev and the LLM classifier tie on this small, clean set; Jev's advantages here are
calibrated probabilities (which make the widening rule principled), latency and guaranteed
structured output, not measured accuracy.

## What the evals found and changed

| Finding | Change |
|---|---|
| Multi-family tier question confidently classified `incentive_rebate` (0.98); the answer lives in Company Updates (0.669 vs 0.530 filtered) | Relative fallback: override the filter when an out-of-filter hit beats the best filtered hit by ≥ 0.10 |
| "approved installer" extracted as unknown in 2/30 runs | Clearer field description with examples. 120/120 on positive and negative probes |
| Answers padded with unasked-for details; Rooftop eligibility check offered for a battery question | Answer prompt: direct answer first, only relevant details, offer the check only for the member's own Rooftop eligibility. Answer Relevancy 0.76 → 0.96 |
| G-Eval hard-gate rubric penalised the correct phrase "No rebate amount applies" | Rubric clarified: no dollar figure, estimate or range |
| Golden answers starting with a bare "Yes."/"No." could not be attributed to a chunk | Rewritten as full sentences |

Three of these were defects in the evaluation itself (rubric, golden answers, judge), not the
product. Reading failures before changing the system was what told them apart.

## Judge reliability: `llm` vs `hybrid`

DeepEval 4.3 can let Jev make the per-claim decisions while the LLM extracts claims and writes
reasons (`--eval-mode hybrid`). On the same pre-fix system the two modes disagreed:

| Metric | hybrid | llm |
|---|---|---|
| Faithfulness | 0.84 (17/23) | 0.99 (23/23) |
| Answer Relevancy | 0.98 (23/23) | 0.76 (15/23) |

Reading the failures: the hybrid Faithfulness reasons quoted `P(yes)=…` values that never
appear in the answers (the reason writer mistook Jev's decision probabilities for answer
content), while the `llm` judge's low relevancy scores were correct (answers really were
padded, and fixing that raised the score). So `llm` is the default. Hybrid stays available as
a second opinion. Caveat: each run regenerated answers, so this compares judges on similar,
not identical, outputs.

## Known limits of this evaluation

- **35 cases is a smoke test, not a benchmark.** It catches regressions and known failure
  modes. It can't give tight confidence intervals.
- **Single runs hide flakiness.** A 7% extraction failure passed one run and failed the next.
  Critical behaviours were probed 30× separately; a production suite should run key cases N
  times and track pass rates.
- **Faithfulness only sees retrieved chunks.** Correct behaviour that comes from the system
  prompt (e.g. offering the eligibility check) can read as "unsupported".
- **Golden answers were written by the same person who built the system.** Independent
  labelling, or real member questions, would be a stronger test.
