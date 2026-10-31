"""The SunGrid copilot as a LangGraph state graph.

    START → contextualize → classify → detect_intent ─┬─ eligibility → extract_params ─┬─ complete → check_eligibility → compose_verdict → END
                                                      │                                └─ missing  → ask_for_missing → END
                                                      ├─ general     → retrieve → generate_answer → END
                                                      └─ off-topic   → decline → END

- contextualize: rewrites follow-ups into a standalone question using conversation history.
- classify: category → retrieval filter only (Part A). It never decides the path.
- detect_intent: the routing decision (Part B).
- check_eligibility: the only node that calls the eligibility service, kept apart from retrieval.
"""

import operator
import time
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sungrid_copilot.agent import eligibility as elig
from sungrid_copilot.agent.contextualize import contextualize
from sungrid_copilot.agent.intent import ELIGIBILITY, detect_intent
from sungrid_copilot.classifier.classifier import classify
from sungrid_copilot.integrations.eligibility_tool import eligibility_tool
from sungrid_copilot.retrieval.retriever import retrieve


class AgentState(TypedDict, total=False):
    # input for this turn
    query: str
    # persisted across turns in one conversation (thread)
    history: Annotated[list[dict], operator.add]  # reducer: node updates are appended, not replaced
    eligibility_params: dict
    awaiting_fields: list[str]
    # per-turn outputs
    standalone_query: str
    category: str
    category_probabilities: dict
    filter_categories: list[str]
    intent: str
    route: str
    retrieved: list[dict]
    used_fallback: bool
    tool_calls: list[dict]
    answer: str
    timings: dict  # seconds per node for this turn (shown in the UI)


def _timed(name: str, node: Callable[[AgentState], dict], first: bool = False):
    """Wrap a node to record its duration in state["timings"]. The first node of a turn
    starts a fresh dict; the graph is sequential, so later nodes extend it."""

    def wrapper(state: AgentState) -> dict:
        start = time.perf_counter()
        update = node(state)
        previous = {} if first else (state.get("timings") or {})
        return {**update, "timings": {**previous, name: round(time.perf_counter() - start, 2)}}

    return wrapper


def _finish(state: AgentState, route: str, answer: str, **extra) -> dict:
    """Final update of a turn: the answer, plus this exchange appended to the history."""
    turn = [{"role": "member", "content": state["query"]}, {"role": "assistant", "content": answer}]
    return {"route": route, "answer": answer, "history": turn, **extra}


def contextualize_node(state: AgentState) -> dict:
    return {
        "standalone_query": contextualize(state["query"], state.get("history") or []),
        # reset per-turn outputs from any previous turn in this thread
        "retrieved": [],
        "tool_calls": [],
        "used_fallback": False,
    }


def classify_node(state: AgentState) -> dict:
    c = classify(state["standalone_query"])
    return {
        "category": c.primary,
        "category_probabilities": c.probabilities,
        "filter_categories": c.filter_categories,
    }


def intent_node(state: AgentState) -> dict:
    return {
        "intent": detect_intent(
            state["standalone_query"], state.get("awaiting_fields"), state.get("eligibility_params")
        )
    }


def route_by_intent(state: AgentState) -> str:
    if state["intent"] == ELIGIBILITY:
        return "eligibility"
    if state["category"] == "non_relevant":
        return "off_topic"
    return "general"


def extract_params_node(state: AgentState) -> dict:
    # Raw message, not the rewrite: only values the member actually stated in this turn.
    new = elig.extract_params(state["query"], state.get("awaiting_fields"))
    return {"eligibility_params": elig.merge(state.get("eligibility_params") or {}, new)}


def route_after_extraction(state: AgentState) -> str:
    return "missing" if elig.missing_fields(state["eligibility_params"]) else "complete"


def ask_for_missing_node(state: AgentState) -> dict:
    missing = elig.missing_fields(state["eligibility_params"])
    return _finish(
        state, "eligibility_needs_info", elig.ask_for_missing_message(missing), awaiting_fields=missing
    )


def check_eligibility_node(state: AgentState) -> dict:
    args = {f: state["eligibility_params"][f] for f in elig.REQUIRED_FIELDS}
    result = eligibility_tool.invoke(args)
    return {"tool_calls": [{"name": eligibility_tool.name, "args": args, "result": result}]}


def compose_verdict_node(state: AgentState) -> dict:
    result = state["tool_calls"][-1]["result"]
    return _finish(state, "eligibility_check", elig.verdict_message(result), awaiting_fields=[])


def retrieve_node(state: AgentState) -> dict:
    r = retrieve(state["standalone_query"], state["filter_categories"])
    return {"retrieved": [vars(c) for c in r.chunks], "used_fallback": r.used_fallback}


def generate_answer_node(state: AgentState) -> dict:
    from sungrid_copilot.agent.answer import generate_answer

    answer = generate_answer(state["standalone_query"], state["retrieved"])
    return _finish(state, "rag", answer, awaiting_fields=[])


def decline_node(state: AgentState) -> dict:
    answer = (
        "I can only help with SunGrid Cooperative topics: membership and policies, rebates and "
        "incentives, billing, installation, and company news."
    )
    return _finish(state, "off_topic", answer, awaiting_fields=[])


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("contextualize", _timed("contextualize", contextualize_node, first=True))
    g.add_node("classify", _timed("classify", classify_node))
    g.add_node("detect_intent", _timed("detect_intent", intent_node))
    g.add_node("extract_params", _timed("extract_params", extract_params_node))
    g.add_node("ask_for_missing", _timed("ask_for_missing", ask_for_missing_node))
    g.add_node("check_eligibility", _timed("check_eligibility", check_eligibility_node))
    g.add_node("compose_verdict", _timed("compose_verdict", compose_verdict_node))
    g.add_node("retrieve", _timed("retrieve", retrieve_node))
    g.add_node("generate_answer", _timed("generate_answer", generate_answer_node))
    g.add_node("decline", _timed("decline", decline_node))

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "classify")
    g.add_edge("classify", "detect_intent")
    g.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {"eligibility": "extract_params", "general": "retrieve", "off_topic": "decline"},
    )
    g.add_conditional_edges(
        "extract_params",
        route_after_extraction,
        {"missing": "ask_for_missing", "complete": "check_eligibility"},
    )
    g.add_edge("check_eligibility", "compose_verdict")
    g.add_edge("retrieve", "generate_answer")
    for terminal in ("ask_for_missing", "compose_verdict", "generate_answer", "decline"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())
