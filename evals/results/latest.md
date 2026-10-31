# Eval results — 2026-10-31 13:23

- Cases: 35
- Agent LLM: `anthropic/claude-haiku-5.5` · Judge: `openai/gpt-6-luna` · Eval mode: `llm`
- Classifier backend: `auto` · Filter mode: `widened`

## Layer 1: deterministic checks

| Check | Passed | Rate |
|---|---|---|
| category | 35/35 | 100% |
| route | 35/35 | 100% |
| retrieval_hit@5 | 23/23 | 100% |
| tool_not_called | 27/27 | 100% |
| tool_called_once | 8/8 | 100% |
| tool_args_match | 8/8 | 100% |
| verdict_correct | 8/8 | 100% |
| hard_gate_no_amount | 4/4 | 100% |

## Layer 2: DeepEval

| Metric | Mean score | Passed | Judge errors |
|---|---|---|---|
| Faithfulness | 0.99 | 22/23 | 0 |
| Answer Relevancy | 0.96 | 23/23 | 0 |
| Contextual Precision | 0.95 | 23/23 | 0 |
| Contextual Recall | 1.00 | 23/23 | 0 |
| Tool Correctness | 1.00 | 8/8 | 0 |
| Hard gate [GEval] | 1.00 | 8/8 | 0 |

## Failures

- **rag-cert-lapse**: Faithfulness (0.75): The score is 0.75 because the actual output does not reflect that jobs completed while certification is lapsed are treated as performed by a non-approved installer for rebate eligibility, regardless of earlier certification history.

Latency per case (all turns): median 5.29s, max 15.95s
