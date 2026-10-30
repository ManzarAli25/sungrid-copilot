"""Ask the SunGrid copilot.

Usage:
    python -m scripts.ask "When are bills issued?"   # one question
    python -m scripts.ask                             # interactive chat (multi-turn)
    python -m scripts.ask --debug ...                 # also show category, route, sources, tool calls
"""

import argparse
import uuid

from sungrid_copilot.agent.graph import build_graph


def show(state: dict, debug: bool) -> None:
    print(f"\n{state['answer']}\n")
    if not debug:
        return
    if state.get("standalone_query") != state.get("query"):
        print(f"  rewritten: {state.get('standalone_query')}")
    probs = {k: round(v, 2) for k, v in (state.get("category_probabilities") or {}).items() if v >= 0.01}
    print(f"  category={state.get('category')} {probs}  filter={state.get('filter_categories')}")
    print(f"  intent={state.get('intent')}  route={state.get('route')}")
    for c in state.get("retrieved") or []:
        print(f"  [{c['score']:.3f}] {c['header']}")
    if state.get("used_fallback"):
        print("  (filtered results were weak — used unfiltered fallback)")
    for t in state.get("tool_calls") or []:
        print(f"  tool {t['name']}({t['args']}) -> {t['result']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the SunGrid copilot")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if args.question:
        show(graph.invoke({"query": args.question}, config), args.debug)
        return

    print("SunGrid copilot. Type 'exit' to quit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("exit", "quit"):
            break
        if query:
            show(graph.invoke({"query": query}, config), args.debug)


if __name__ == "__main__":
    main()
