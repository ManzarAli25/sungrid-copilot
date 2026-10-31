# SunGrid Copilot

A prototype support copilot for SunGrid, a fictional solar cooperative. It answers member
questions from the 14 knowledge-base docs (RAG), and for rooftop rebate eligibility it calls
the eligibility service instead of answering from the docs.

Built with LangGraph, Qdrant, OpenRouter (LLM and embeddings), TypeSafe Jev (query classifier,
with an LLM fallback) and DeepEval (evals).

- **Live demo:** https://sungrid-copilot.onrender.com (free tier, so the first load can take
  up to a minute)
- **Design write-up (Part C):** [`DESIGN.md`](DESIGN.md), also as [`DESIGN.pdf`](DESIGN.pdf)
- **Evaluation details:** [`evals/README.md`](evals/README.md)

## 1. Install

You need Python 3.10 or newer.

```bash
git clone https://github.com/ManzarAli25/sungrid-copilot.git
cd sungrid-copilot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure

Copy the example config and add your keys. `.env` is git-ignored, so keys never get committed.

```bash
cp .env.example .env
```

Only the OpenRouter key is required. Everything else has a working default.

| Variable | Required | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Chat model, embeddings and the eval judge |
| `QDRANT_URL`, `QDRANT_API_KEY` | no | Qdrant Cloud. Leave empty to run Qdrant locally (no account needed) |
| `TYPESAFE_API_KEY` | no | Jev classifier. Leave empty to classify with the LLM instead |
| `CLASSIFIER_BACKEND` | no | `auto` (default: Jev if there's a key, else LLM), `jev` or `llm` |
| `FILTER_MODE` | no | `widened` (default) or `strict`, see assumption 1 |

Models are set in `.env.example` too (`LLM_MODEL`, `JUDGE_MODEL`, `EMBEDDING_MODEL`).

## 3. Run it end to end

**1. Build the vector store.** Chunks the docs, embeds them and stores them in Qdrant. It's safe
to re-run: it updates chunks instead of duplicating them.

```bash
python -m scripts.ingest              # add --recreate to rebuild from scratch
```

**2. Ask questions from the terminal.**

```bash
python -m scripts.ask "When are bills issued?"
python -m scripts.ask                 # interactive chat, remembers the conversation
python -m scripts.ask --debug "..."   # also shows category, filter, route, chunks and tool calls
```

Try an eligibility question like *"Am I eligible for the rooftop rebate? ZIP 94103, income
90k, 6 kW, approved installer"*, then follow up with *"what if it were 12 kW?"*.

**3. Or use the web UI.**

```bash
streamlit run app.py                  # http://localhost:8501
```

It has the chat with an "under the hood" panel for every answer (category probabilities,
filter, retrieved chunks, tool input/output, timings per step), the agent graph, the knowledge
base with its category tags, and the latest eval results. To deploy it on Render, create a
Blueprint from this repo (`render.yaml`) and use Qdrant Cloud, since Render's disk is wiped on
restart.

**4. Run the tests.** No API keys needed.

```bash
pytest
```

**5. Run the evals.**

```bash
python -m evals.run_evals                  # 35 golden questions: exact checks + DeepEval metrics
python -m evals.run_evals --skip-deepeval  # exact checks only, no judge cost
python -m evals.run_evals --compare        # Jev vs LLM, strict vs widened filter
```

Latest run: 100% on every exact check (category, route, retrieval, tool inputs, verdict, hard
gate). DeepEval: Faithfulness 0.99, Answer Relevancy 0.96, Contextual Precision 0.95,
Contextual Recall 1.00, Tool Correctness 1.00, hard-gate G-Eval 1.00.

## 4. Assumptions

**1. One primary category, but the filter can include a second one.** The brief asks for one
primary category used as the retrieval filter. The classifier always gives exactly one primary
category. But by default we also add any other category with probability ≥ 0.10 to the filter.
The reason: for *"My rebate hasn't shown up on my bill, who do I contact?"* Jev says
`incentive_rebate 0.84 / billing_account 0.16`, and the right answer is in the Billing FAQs.
Filtering on the primary category alone hides it (top hit 0.588, the wrong section). With the
second category it comes first (0.748). Across 22 questions, the filter alone finds the right
chunk 19/22 times in strict mode vs 21/22 widened. It's still a filter, not routing: every
question goes through the same retrieval step. `FILTER_MODE=strict` gives the literal behaviour.

**2. Off-topic questions skip retrieval.** If the category is `non_relevant` there's nothing to
filter on, so the copilot politely declines instead of searching.

**3. Ambiguous docs get more than one category.** Doc 06 (rebate reversal / billing adjustment),
doc 07 (installer certification) and doc 10 (low-income bill credit) each have two categories.
The reason for each is written next to it in `config/categories.yaml`. Doc 06's primary category
is `incentive_rebate`, because the Incentive Program committee makes the reversal decision and
handles appeals; Billing only runs the repayment schedule.

**4. Better results outside the filter win.** Retrieval runs a filtered and an unfiltered search
together. We use the unfiltered results if the filtered ones are weak (best score below 0.45) or
if the best result outside the filter beats the best filtered one by 0.10 or more. We added the
second rule after the evals showed *"What rebate will the new multi-family tier get?"*
classified `incentive_rebate` at 0.98, while the answer is in Company Updates (0.669 vs 0.530).

**5. Eligibility inputs are never guessed.** Anything the member didn't say is asked for, not
assumed. "My installer is great" doesn't count as approved. The LLM doesn't call the tool
itself, because tool-calling models tend to fill missing arguments with guesses.

**6. The eligibility service only reports the first failed check.** If a household fails income
and system size, they hear one reason. That's how the stub works, and we didn't change it.

**7. The tool only covers the rooftop rebate.** Questions about the battery add-on or the
low-income credit are answered from the docs.

**8. The income threshold isn't in the docs.** The docs only say it's published every year, so
the copilot says that instead of quoting the stub's internal `$120,000`.

## Project structure

```
config/categories.yaml        categories and which docs belong to them (with reasons)
data/docs/                    the 14 knowledge-base docs
sungrid_copilot/
  config.py                   settings from .env
  ingestion/                  chunking, embedding, storing in Qdrant
  retrieval/                  filtered vector search with fallback
  classifier/                 question → category (Jev or LLM)
  agent/                      the LangGraph agent
  integrations/stub_tools.py  the eligibility service stub, unmodified
scripts/                      ingest, ask, draw_graph
evals/                        golden set, eval runner, results
tests/                        unit and guard tests
app.py                        Streamlit web UI
```

The agent graph, generated from the code with `python -m scripts.draw_graph`. Solid arrows are
fixed edges, dotted arrows are conditional.

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
