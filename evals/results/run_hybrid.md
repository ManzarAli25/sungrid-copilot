# Eval results — 2026-10-31 13:12

- Cases: 35
- Agent LLM: `anthropic/claude-haiku-5.5` · Judge: `openai/gpt-6-luna` · Eval mode: `hybrid`
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
| Faithfulness | 0.84 | 17/23 | 0 |
| Answer Relevancy | 0.98 | 23/23 | 0 |
| Contextual Precision | 0.97 | 23/23 | 0 |
| Contextual Recall | 0.96 | 21/23 | 0 |
| Tool Correctness | 1.00 | 8/8 | 0 |
| Hard gate [GEval] | 0.88 | 6/8 | 0 |

## Failures

- **rag-bill-dates**: Faithfulness (0.75): The score is 0.75 because the actual output states P(yes)=0.10, which contradicts the retrieval context.
- **rag-battery-without-rooftop**: Faithfulness (0.67): The score is 0.67 because the actual output gives conflicting values for P(yes), stating both 0.10 and 0.23.
- **rag-reversal-appeal-path**: Contextual Recall (0.67): The score is 0.67 because sentence 1’s appeal-process distinction and sentence 2’s 30-day notice are supported by the two high-confidence reasons, while another detail in the expected output has weaker support. The provided information does not specify node numbers in retrieval context, so they cannot be identified.
- **rag-cert-lapse**: Faithfulness (0.67): The score is 0.67 because the actual output gives conflicting values for P(yes), stating both 0.17 and 0.28.
- **rag-micro-inverters**: Faithfulness (0.75): The score is 0.75 because the actual output states P(yes)=0.32, which contradicts the retrieval context.
- **rag-battery-clearance**: Faithfulness (0.67): The score is 0.67 because the actual output gives P(yes) = 0.17, which contradicts the retrieval context.
- **rag-proxy-vote**: Contextual Recall (0.50): The score is 0.50 because sentence 1 is supported by node 1 in retrieval context (P(yes)=0.97), but node 2 (P(yes)=0.43) provides weaker support for the proxy-voting details and deadline.
- **rag-multifamily-tier**: Faithfulness (0.75): The score is 0.75 because the actual output gives conflicting values for P(yes): 0.20 and 0.26.
- **elig-zip-fail**: Hard gate [GEval] (0.70): The output correctly says the household with ZIP 10001 is not eligible and gives the tool’s exact reason that the ZIP is outside the SunGrid service region. However, “No rebate amount applies” mentions a rebate amount despite the instruction not to mention any amount when eligible=false.
- **elig-installer-fail**: Hard gate [GEval] (0.70): The response clearly says the household is not eligible and gives the tool’s exact reason: the installer is not on the approved list. However, “No rebate amount applies” mentions a rebate amount despite the instruction for an ineligible result to avoid discussing amounts; the appeals-process offer is also unsupported by the tool output.

Latency per case (all turns): median 7.34s, max 18.2s
