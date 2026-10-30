"""Print the agent graph as Mermaid (rendered natively by GitHub), or save it as a PNG.

Usage:
    python -m scripts.draw_graph                 # print Mermaid source
    python -m scripts.draw_graph --png graph.png # render PNG (uses the mermaid.ink web API)
"""

import argparse

from sungrid_copilot.agent.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--png", metavar="PATH", help="write a PNG instead of printing Mermaid")
    args = parser.parse_args()

    graph = build_graph().get_graph()
    if args.png:
        with open(args.png, "wb") as f:
            f.write(graph.draw_mermaid_png())
        print(f"Wrote {args.png}")
    else:
        print(graph.draw_mermaid())


if __name__ == "__main__":
    main()
