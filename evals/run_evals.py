"""Evaluate the SunGrid copilot against the golden set.

Two layers:
  1. Deterministic checks (no LLM judge): category, route, retrieval hit@5, tool args,
     verdict, hard gate. Cheap, exact, and run on every case.
  2. DeepEval metrics (LLM-as-a-judge, optionally with Jev deciding): faithfulness, answer
     relevancy, contextual precision/recall for RAG cases; tool correctness and a G-Eval
     hard-gate rubric for eligibility cases.

Usage:
    python -m evals.run_evals                      # full run (agent + both layers)
    python -m evals.run_evals --skip-deepeval      # deterministic layer only (fast, cheap)
    python -m evals.run_evals --only rag-trueup elig-cap
    python -m evals.run_evals --eval-mode llm      # judge decisions by the LLM only
    python -m evals.run_evals --compare            # classifier x filter-mode retrieval experiment
"""

import argparse
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from sungrid_copilot.config import get_settings

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVALS_DIR / "golden.yaml"
RESULTS_DIR = EVALS_DIR / "results"


# --- golden set --------------------------------------------------------------------------


def load_cases(only: list[str] | None = None) -> list[dict]:
    cases = yaml.safe_load(GOLDEN_FILE.read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if not only or c["id"] in only]


def short_id(chunk_id: str) -> str:
    """"06_ambiguous_rebate_billing_adjustments::overview" -> "06::overview"."""
    doc, section = chunk_id.split("::", 1)
    return f"{doc[:2]}::{section}"


def accepted_categories(case: dict) -> list[str]:
    cat = case["category"]
    return (cat if isinstance(cat, list) else [cat]) + case.get("also_ok", [])


# --- running the agent -------------------------------------------------------------------


def run_case(graph, case: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    start = time.perf_counter()
    for turn in case["turns"]:
        state = graph.invoke({"query": turn}, config)
    state["latency_s"] = round(time.perf_counter() - start, 2)
    return state


# --- layer 1: deterministic checks -------------------------------------------------------


def deterministic_checks(case: dict, state: dict) -> dict[str, bool]:
    checks = {
        "category": state.get("category") in accepted_categories(case),
        "route": state.get("route") == case["route"],
    }
    tool_calls = state.get("tool_calls") or []

    if case.get("sources"):
        retrieved = {short_id(c["chunk_id"]) for c in state.get("retrieved") or []}
        checks["retrieval_hit@5"] = any(s in retrieved for s in case["sources"])

    if "tool_args" in case:
        checks["tool_called_once"] = len(tool_calls) == 1
        if tool_calls:
            checks["tool_args_match"] = tool_calls[0]["args"] == case["tool_args"]
            checks["verdict_correct"] = tool_calls[0]["result"]["eligible"] == case["eligible"]
    else:
        checks["tool_not_called"] = not tool_calls  # RAG / off-topic / needs-info must never call it

    if case.get("eligible") is False:
        answer = state.get("answer", "")
        checks["hard_gate_no_amount"] = "not eligible" in answer and "$" not in answer
    return checks


# --- layer 2: DeepEval -------------------------------------------------------------------


def build_judges(eval_mode: str):
    from deepeval.models import OpenRouterModel, TypeSafeModel

    class JudgeModel(OpenRouterModel):
        # G-Eval asks the judge for token log-probs, and DeepEval only skips that for its own
        # OpenAI/Azure classes. The default judge (gpt-6-luna, a reasoning model) rejects
        # logprobs with a 400. G-Eval catches AttributeError from this call and falls back to
        # a plain JSON score, so raise it here.
        def generate_raw_response(self, *args, **kwargs):
            raise AttributeError("log-prob scoring not supported by this judge")

        async def a_generate_raw_response(self, *args, **kwargs):
            raise AttributeError("log-prob scoring not supported by this judge")

    s = get_settings()
    judge = JudgeModel(
        model=s.judge_model, api_key=s.openrouter_api_key, base_url=s.openrouter_base_url, temperature=0
    )
    jev = TypeSafeModel(model=s.jev_model, api_key=s.typesafe_api_key) if eval_mode != "llm" else None
    return judge, jev


def to_test_case(case: dict, state: dict):
    from deepeval.test_case import LLMTestCase, ToolCall

    common = dict(
        name=case["id"],
        input=state.get("standalone_query") or case["turns"][-1],
        actual_output=state.get("answer", ""),
        tags=case.get("tags"),
    )
    if case["type"] == "rag":
        return LLMTestCase(
            **common,
            expected_output=case["expected_output"],
            retrieval_context=[f"{c['header']}\n{c['text']}" for c in state.get("retrieved") or []],
        )
    if "tool_args" in case:
        called = [
            ToolCall(name=t["name"], input_parameters=t["args"], output=t["result"])
            for t in state.get("tool_calls") or []
        ]
        return LLMTestCase(
            **common,
            tools_called=called,
            expected_tools=[ToolCall(name="check_rebate_eligibility", input_parameters=case["tool_args"])],
        )
    return None  # off-topic / needs-info cases are fully covered by layer 1


def rag_metrics(judge, jev, eval_mode):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    kw = dict(model=judge, system_one_model=jev, eval_mode=eval_mode)
    return [
        FaithfulnessMetric(threshold=0.8, **kw),
        AnswerRelevancyMetric(threshold=0.7, **kw),
        ContextualPrecisionMetric(threshold=0.5, **kw),
        ContextualRecallMetric(threshold=0.7, **kw),
    ]


def eligibility_metrics(judge, jev, eval_mode):
    from deepeval.metrics import GEval, ToolCorrectnessMetric
    from deepeval.test_case import SingleTurnParams, ToolCallParams

    hard_gate = GEval(
        name="Hard gate",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
        ],
        evaluation_steps=[
            "Read the output of the check_rebate_eligibility tool in the tools called.",
            "If the tool returned eligible=false: the actual output must clearly say the household "
            "is not eligible and give the tool's reason. It must NOT state any dollar figure, "
            "estimate or range, and must not suggest a partial or possible rebate. Stating that "
            "no rebate applies is correct and must not be penalised.",
            "If the tool returned eligible=true: the actual output must report the tool's "
            "estimated_rebate_usd exactly and must not add eligibility conditions of its own.",
            "Heavily penalise any eligibility claim or amount not supported by the tool output.",
        ],
        threshold=0.8,
        model=judge,  # G-Eval scoring is LLM-only in DeepEval
    )
    tool_correctness = ToolCorrectnessMetric(
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        threshold=1.0,
        model=judge,
        system_one_model=jev,
        eval_mode=eval_mode,
    )
    return [tool_correctness, hard_gate]


def run_deepeval(test_cases, metrics) -> dict[str, list[dict]]:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig

    if not test_cases:
        return {}
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=8),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        error_config=ErrorConfig(ignore_errors=True),  # a failed judge call is reported, not fatal
        # Off: every run should judge fresh answers, and concurrent cache-file writes crash on Windows.
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    return {
        tr.name: [
            {"metric": m.name, "score": m.score, "success": m.success, "reason": m.reason,
             "error": m.error}
            for m in tr.metrics_data or []
        ]
        for tr in result.test_results
    }


# --- report ------------------------------------------------------------------------------


def summarise(rows: list[dict], eval_mode: str) -> str:
    s = get_settings()
    lines = [
        f"# Eval results — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Cases: {len(rows)}",
        f"- Agent LLM: `{s.llm_model}` · Judge: `{s.judge_model}` · Eval mode: `{eval_mode}`",
        f"- Classifier backend: `{s.classifier_backend}` · Filter mode: `{s.filter_mode}`",
        "",
        "## Layer 1: deterministic checks",
        "",
        "| Check | Passed | Rate |",
        "|---|---|---|",
    ]
    tally = defaultdict(lambda: [0, 0])
    for r in rows:
        for name, ok in r["checks"].items():
            tally[name][0] += ok
            tally[name][1] += 1
    for name, (ok, n) in tally.items():
        lines.append(f"| {name} | {ok}/{n} | {ok / n:.0%} |")

    scores = defaultdict(list)
    for r in rows:
        for m in r.get("deepeval") or []:
            scores[m["metric"]].append(m)
    if scores:
        lines += ["", "## Layer 2: DeepEval", "",
                  "| Metric | Mean score | Passed | Judge errors |", "|---|---|---|---|"]
        for name, ms in scores.items():
            valid = [m["score"] for m in ms if m["score"] is not None]
            mean = sum(valid) / len(valid) if valid else float("nan")
            errors = sum(1 for m in ms if m.get("error"))
            lines.append(f"| {name} | {mean:.2f} | {sum(bool(m['success']) for m in ms)}/{len(ms)} | {errors} |")

    failures = []
    for r in rows:
        bad = [k for k, ok in r["checks"].items() if not ok]
        bad += [f"{m['metric']} ({m['score']:.2f}): {m['reason']}" for m in r.get("deepeval") or []
                if not m["success"] and m["score"] is not None]
        if bad:
            failures.append(f"- **{r['id']}**: " + "; ".join(bad))
    lines += ["", "## Failures", ""] + (failures or ["None."])

    latencies = sorted(r["latency_s"] for r in rows)
    lines += ["", f"Latency per case (all turns): median {latencies[len(latencies) // 2]}s, "
              f"max {latencies[-1]}s"]
    return "\n".join(lines) + "\n"


def run(args) -> None:
    from sungrid_copilot.agent.graph import build_graph

    cases = load_cases(args.only)
    graph = build_graph()
    rows, states = [], {}
    for case in cases:
        state = run_case(graph, case)
        states[case["id"]] = state
        checks = deterministic_checks(case, state)
        rows.append({"id": case["id"], "checks": checks, "latency_s": state["latency_s"],
                     "category": state.get("category"), "route": state.get("route"),
                     "answer": state.get("answer")})
        print(f"{'PASS' if all(checks.values()) else 'FAIL'}  {case['id']:<34} {state['latency_s']:>5}s")

    if not args.skip_deepeval:
        judge, jev = build_judges(args.eval_mode)
        by_id = {c["id"]: c for c in cases}
        rag_tcs, elig_tcs = [], []
        for cid, state in states.items():
            tc = to_test_case(by_id[cid], state)
            if tc is not None:
                (rag_tcs if by_id[cid]["type"] == "rag" else elig_tcs).append(tc)
        print(f"\nDeepEval: {len(rag_tcs)} RAG cases, {len(elig_tcs)} tool cases (eval mode: {args.eval_mode})")
        results = run_deepeval(rag_tcs, rag_metrics(judge, jev, args.eval_mode))
        results |= run_deepeval(elig_tcs, eligibility_metrics(judge, jev, args.eval_mode))
        for r in rows:
            r["deepeval"] = results.get(r["id"], [])

    RESULTS_DIR.mkdir(exist_ok=True)
    report = summarise(rows, args.eval_mode if not args.skip_deepeval else "skipped")
    (RESULTS_DIR / f"{args.out}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"{args.out}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("\n" + report)
    print(f"Saved {RESULTS_DIR / (args.out + '.md')}")


# --- experiment: classifier backend x filter mode ----------------------------------------


def compare(args) -> None:
    """Retrieval hit@5 and classifier accuracy for each backend x filter mode, on single-turn
    RAG cases. Only classification + retrieval run (no generation, no judge)."""
    from sungrid_copilot.classifier.classifier import classify_with_jev, classify_with_llm
    from sungrid_copilot.embeddings import embed_query
    from sungrid_copilot.retrieval.retriever import _search, should_override_filter

    cases = [c for c in load_cases(args.only) if c["type"] == "rag" and len(c["turns"]) == 1]
    backends = {"jev": classify_with_jev, "llm": classify_with_llm}
    stats = defaultdict(lambda: defaultdict(int))
    misses = defaultdict(list)

    for case in cases:
        query = case["turns"][0]
        vector = embed_query(query)
        unfiltered = _search(vector, None, 5)
        for backend, fn in backends.items():
            for mode in ("strict", "widened"):
                c = fn(query, mode=mode)
                filtered = _search(vector, c.filter_categories, 5) if c.filter_categories else []
                fell_back = should_override_filter(filtered, unfiltered)
                final = unfiltered if fell_back else filtered
                hit_filtered = any(short_id(h.chunk_id) in case["sources"] for h in filtered)
                hit_final = any(short_id(h.chunk_id) in case["sources"] for h in final)
                key = f"{backend}/{mode}"
                stats[key]["category"] += c.primary in accepted_categories(case)
                stats[key]["hit_filtered"] += hit_filtered
                stats[key]["hit_final"] += hit_final
                stats[key]["fallback"] += fell_back
                if not hit_final:
                    misses[key].append(case["id"])

    n = len(cases)
    lines = [
        f"# Classifier x filter-mode experiment — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{n} single-turn RAG cases. hit@5 = an expected source chunk is in the top 5.",
        "",
        "| Config | Category acc. | hit@5 (filter only) | hit@5 (with fallback) | Fallback used | Misses |",
        "|---|---|---|---|---|---|",
    ]
    for key, st in stats.items():
        lines.append(
            f"| {key} | {st['category']}/{n} | {st['hit_filtered']}/{n} | {st['hit_final']}/{n} "
            f"| {st['fallback']}/{n} | {', '.join(misses[key]) or '—'} |"
        )
    report = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "compare.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {RESULTS_DIR / 'compare.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the SunGrid copilot")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--skip-deepeval", action="store_true", help="deterministic layer only")
    parser.add_argument(
        "--eval-mode",
        choices=["llm", "hybrid", "system_one"],
        default="llm",
        help="who makes the judge's decisions: llm (default) or hybrid (LLM extracts, Jev decides; see evals/README)",
    )
    parser.add_argument("--compare", action="store_true", help="classifier x filter-mode experiment")
    parser.add_argument("--out", default="latest", help="results file name (without extension)")
    args = parser.parse_args()
    compare(args) if args.compare else run(args)


if __name__ == "__main__":
    main()
