# Eval results — 2026-10-31 13:17

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
| Faithfulness | 0.99 | 23/23 | 0 |
| Answer Relevancy | 0.76 | 15/23 | 0 |
| Contextual Precision | 0.93 | 23/23 | 0 |
| Contextual Recall | 0.96 | 21/23 | 0 |
| Tool Correctness | 1.00 | 8/8 | 0 |
| Hard gate [GEval] | 0.89 | 6/8 | 0 |

## Failures

- **rag-bill-dates**: Answer Relevancy (0.50): The response earns 0.50 because it provides some information about bill and autopay timing, but discussion of failed charges, account status after retries, and general support guidance does not answer the timing questions.
- **rag-trueup**: Answer Relevancy (0.43): The score is 0.43 because the response appears to address the true-up somewhat, but much of it discusses who it applies to, billing, partial-enrollment calculations, and contact information rather than when it happens.
- **rag-rebate-not-on-bill**: Answer Relevancy (0.56): The score is 0.56 because the response is partly related to rebates and billing, but it focuses on reversal rules and timing rather than telling you whom to contact about a missing rebate.
- **rag-appeal-deadline**: Answer Relevancy (0.50): The score is 0.50 because the response is partly related to appealing a rebate decision, but much of it discusses billing notices, filing instructions, or eligibility checks rather than stating how long you have to appeal.
- **rag-battery-amount**: Answer Relevancy (0.38): The score is 0.38 because the response is broadly about the battery incentive, but it focuses on eligibility, application, and technical requirements rather than stating how much the add-on incentive pays.
- **rag-low-income-credit**: Answer Relevancy (0.27): The answer was only partly relevant: it addressed the credit amount and whether solar panels are required, but included extensive unrelated details about billing, net metering, eligibility verification, and applying, keeping the score low.
- **rag-income-threshold-unanswerable**: Answer Relevancy (0.36): The response is only partially relevant because it discusses general rebate eligibility requirements, but it never provides the specific income threshold number the user asked for.
- **rag-reversal-appeal-path**: Contextual Recall (0.67): The score is 0.67 because sentence 1’s appeals-process distinction and sentence 2’s notice timing are supported by node 2 in retrieval context, while the standalone “No” in sentence 1 is unsupported.
- **rag-cert-lapse**: Contextual Recall (0.50): The score is 0.50 because the substantive claim in sentence 1 is supported by node 1 in retrieval context, but the standalone “Yes” is not supported by any node in retrieval context.
- **rag-micro-inverters**: Answer Relevancy (0.57): The score is 0.57 because the response appears partly relevant, but it includes tangents about panel layouts, battery compatibility, and support instructions that do not explain when micro-inverters are required.
- **elig-income-fail**: Hard gate [GEval] (0.70): It clearly says the household is not eligible and gives the tool's exact reason, but “No rebate amount applies” still mentions a rebate amount, which the instructions prohibit for an ineligible result.
- **elig-installer-fail**: Hard gate [GEval] (0.70): The response correctly says the household is not eligible and gives the tool’s exact reason: the installer is not on the approved list. However, “No rebate amount applies” mentions a rebate amount despite the instruction not to mention any amount when eligible=false.

Latency per case (all turns): median 7.04s, max 18.45s
