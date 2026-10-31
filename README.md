# SunGrid Cooperative Copilot

A prototype support copilot for the (fictional) SunGrid solar cooperative. It answers member
questions with retrieval-augmented generation over an internal knowledge base, and it calls
a separate eligibility service for rebate-eligibility questions instead of answering them
from documents.

**Live demo:** https://sungrid-copilot.onrender.com (password-protected; available on request.
Free tier, so the first load after a period of inactivity can take up to a minute.)

## Stack

| Concern | Choice |
|---|---|
| LLM + embeddings | [OpenRouter](https://openrouter.ai) (OpenAI-compatible API) |
| Vector store | Qdrant: Cloud free tier, or local embedded mode with no account |
| Query classifier | TypeSafe **Jev** (typed decision model), with an LLM fallback |
| Agent framework | LangGraph |
| Evaluation | DeepEval |

## Project layout

```
config/categories.yaml        category definitions + document→category tags (source of truth)
data/docs/                    the 14 knowledge-base documents
sungrid_copilot/
  config.py                   settings from environment / .env
  ingestion/                  chunking + embedding + upsert into Qdrant
  retrieval/                  category-filtered vector search
  classifier/                 query → category (Jev or LLM backend)
  agent/                      LangGraph agent
  integrations/stub_tools.py  eligibility service stub (unmodified, owned by "another team")
scripts/                      CLI entry points (ingest, ask)
evals/                        DeepEval test suite
tests/                        unit + guard tests
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then fill in OPENROUTER_API_KEY (the only required key)
```

| Variable | Required | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Used for chat, embeddings and the eval judge |
| `QDRANT_URL`, `QDRANT_API_KEY` | no | Leave empty to use local embedded Qdrant |
| `TYPESAFE_API_KEY` | no | Leave empty to classify with the LLM fallback |

## Agent graph

Generated from the compiled LangGraph graph with `python -m scripts.draw_graph`. Solid arrows
are fixed edges and dotted arrows are conditional edges.

<!-- agent-graph:start -->
```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	contextualize(contextualize)
	classify(classify)
	detect_intent(detect_intent)
	extract_params(extract_params)
	ask_for_missing(ask_for_missing)
	check_eligibility(check_eligibility)
	compose_verdict(compose_verdict)
	retrieve(retrieve)
	generate_answer(generate_answer)
	decline(decline)
	__end__([<p>__end__</p>]):::last
	__start__ --> contextualize;
	check_eligibility --> compose_verdict;
	classify --> detect_intent;
	contextualize --> classify;
	detect_intent -. &nbsp;off_topic&nbsp; .-> decline;
	detect_intent -. &nbsp;eligibility&nbsp; .-> extract_params;
	detect_intent -. &nbsp;general&nbsp; .-> retrieve;
	extract_params -. &nbsp;missing&nbsp; .-> ask_for_missing;
	extract_params -. &nbsp;complete&nbsp; .-> check_eligibility;
	retrieve --> generate_answer;
	ask_for_missing --> __end__;
	compose_verdict --> __end__;
	decline --> __end__;
	generate_answer --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
<!-- agent-graph:end -->

| Node | Responsibility |
|---|---|
| `contextualize` | Rewrites a follow-up ("how long do I have to appeal *that*?") into a standalone question using the last 3 exchanges. Skipped on the first turn. The rewrite drives classification, intent and retrieval |
| `classify` | Standalone query → one primary category plus retrieval filter (Jev or LLM). **Never chooses the path.** |
| `detect_intent` | The routing decision: eligibility check, general question, or off-topic. Aware of the conversation, so "what if it were 9 kW?" re-runs the check |
| `extract_params` | LLM structured extraction of the 4 eligibility inputs. All are optional, so nothing is guessed. Reads the member's **raw** message plus the question just asked (so "yes they're approved" resolves), never earlier values |
| `ask_for_missing` | Asks only for the missing inputs. The values given so far persist in the conversation thread |
| `check_eligibility` | **The only node that calls the eligibility service** (`stub_tools.check_rebate_eligibility`, unmodified) |
| `compose_verdict` | Hard gate: the verdict text comes from a code template, and an ineligible result never includes an amount |
| `retrieve` → `generate_answer` | Category-filtered vector search, then a grounded answer that cites its sources |
| `decline` | Polite refusal for off-topic queries |

## Running

```bash
python -m scripts.ingest              # build/refresh the vector store (idempotent; --recreate to rebuild)
python -m scripts.ask "When are bills issued?"
python -m scripts.ask                 # interactive multi-turn chat
python -m scripts.ask --debug ...     # also show category, filter, route, sources and tool calls
python -m scripts.draw_graph          # print the agent graph as Mermaid
pytest                                # unit + guard tests (no API keys needed)
```

### Web UI (Streamlit)

```bash
streamlit run app.py      # http://localhost:8501
```

Four tabs: **Copilot** (chat with an "under the hood" panel per answer showing the rewritten
query, category probabilities, retrieval filter, retrieved chunks with scores, fallback, and the
eligibility tool's input and output), **How it works** (live agent graph), **Knowledge base**
(documents, category tags and the reasons for them, chunks) and **Evaluation** (latest results).

**Deploying to Render:** push the repo, then in Render choose *New → Blueprint* and select it.
`render.yaml` defines the service. Enter `OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
`TYPESAFE_API_KEY` and **`APP_PASSWORD`** when prompted. Use Qdrant Cloud (already ingested),
because Render's disk is ephemeral. `APP_PASSWORD` gates the app so a public URL can't spend
your API credits, and `MAX_TURNS_PER_SESSION` (default 40) caps each session. Vercel isn't
suitable: Streamlit needs a long-running server with WebSockets, not serverless functions.

### Evaluation

```bash
python -m evals.run_evals                  # 35 golden cases: deterministic checks + DeepEval metrics
python -m evals.run_evals --skip-deepeval  # deterministic checks only (no judge cost)
python -m evals.run_evals --compare        # classifier backend x filter mode experiment
```

Latest run: **100% on every deterministic check** (category, route, retrieval hit@5, tool
arguments, verdict, hard gate). DeepEval: Faithfulness 0.99, Answer Relevancy 0.96, Contextual
Precision 0.95, Contextual Recall 1.00, Tool Correctness 1.00, hard-gate G-Eval 1.00.
Methodology, the judge-mode comparison and what the evals changed are in
[`evals/README.md`](evals/README.md).

## Assumptions

### 1. Single primary category, but the retrieval filter can be widened
The brief asks for a classifier that assigns **a single primary category** and uses
**that category** as a retrieval filter. The classifier always outputs exactly one primary
category (`Classification.primary`).

By default (`FILTER_MODE=widened`), the retrieval filter **also includes any runner-up
category with probability ≥ 0.10** (at most 3 categories). This departs from the most literal
reading of the brief on purpose:

- **Evidence:** for *"My rebate hasn't shown up on my bill, who do I contact?"* Jev returns
  `incentive_rebate 0.84 / billing_account 0.16`. The correct answer is in the **Billing & Account
  FAQs** doc. Filtering on the primary category alone hides it, and the top hit becomes a
  related but wrong section (score 0.588). The widened filter returns the exact answer
  first (score 0.748).
- **Why it's still a filter, not routing:** every relevant query runs the same retrieval step.
  The category only narrows which chunks are searched.
- **Literal behaviour is one setting away:** `FILTER_MODE=strict` filters on the primary category
  only. Measured on 22 RAG cases, the filter alone retrieves the right chunk in **19/22 (strict)
  vs 21/22 (widened)**. With the fallback (assumption 4), both reach 22/22.

### 2. `non_relevant` queries skip retrieval
There is no category to filter to, so the copilot declines politely instead of searching.

### 3. Ambiguous documents are tagged with several categories
Doc 06 (rebate reversal / billing adjustment), doc 07 (installer certification) and doc 10
(low-income bill credit) are tagged with more than one category. The reasoning for each is
recorded next to the tag in `config/categories.yaml`. Doc 06's **primary** category is
`incentive_rebate`, because both the reversal decision and the appeal path belong to the
Incentive Program committee; Billing only executes the repayment schedule.

### 4. The filter can be overridden by clearly better evidence
Retrieval always runs a filtered and an unfiltered search (one embedding, two vector queries).
The unfiltered results are used when:
- **absolute:** the best filtered hit scores below 0.45 (cosine). From a small probe, relevant
  queries scored 0.49–0.75 and an off-topic query scored 0.15; or
- **relative:** the best unfiltered hit beats the best filtered hit by ≥ 0.10. Added after the
  golden set showed *"What rebate will the new multi-family tier get?"* confidently classified
  `incentive_rebate` (0.98), while the answer is in Company Updates (0.669 vs 0.530).

So the category acts as a strong preference, not a wall. Both thresholds were tuned on small
probes and are validated by the eval suite, not treated as final.

### 5. Eligibility inputs are never guessed
All four eligibility inputs are optional during extraction. Anything the member didn't state
is asked for, not inferred. "My installer is great" is not approval. The LLM never calls the
tool directly, because tool-calling models tend to fill required arguments with plausible guesses.

## Known limitations

- **The eligibility service reports only the first failed check.** A household failing both
  income and system size hears about one reason. That's the stub's behaviour, deliberately
  unchanged.
- **The tool covers the Rooftop Rebate only.** Battery add-on and low-income credit questions
  are answered from documents. Their eligibility can't be checked by a tool.
- **The income threshold value isn't in the documents** (only that it's "published annually").
  The copilot says so rather than quoting the stub's internal `$120,000`.
- **Conversation state is in memory** (`InMemorySaver`). It's lost on restart. Production would
  use a persistent checkpointer (Postgres/Redis).
- **Each follow-up turn costs one extra LLM call** for query rewriting. First turns cost nothing extra.
- **Thresholds (0.10 widening, 0.45 / 0.10 fallback) are tuned on a small set.** They're
  sensible defaults with eval evidence, not optimised values.
- **The eval set is 35 hand-written cases.** It's a regression and failure-mode suite, not a
  statistically strong benchmark. See [`evals/README.md`](evals/README.md#known-limits-of-this-evaluation).

## Design doc

*TODO*: see `DESIGN.md`.
