"""Streamlit front end for the SunGrid copilot.

    streamlit run app.py

Tabs: chat with an "under the hood" view of every answer, the agent graph, the knowledge base
and its category tags, and the latest evaluation results.
"""

import hmac
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from sungrid_copilot.config import PROJECT_ROOT, get_settings, load_category_config

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
MAX_TURNS_PER_SESSION = int(os.getenv("MAX_TURNS_PER_SESSION", "40"))

EXAMPLES = {
    "General questions": [
        "When are bills issued, and when does autopay charge me?",
        "How much is the battery storage add-on incentive?",
        "When are micro-inverters required?",
    ],
    "Ambiguous / cross-category": [
        "My rebate hasn't shown up on my bill yet. Who do I contact?",
        "My rebate got reversed and I'm on autopay. What happens?",
        "What rebate will the new multi-family housing tier get?",
    ],
    "Eligibility tool + hard gate": [
        "Am I eligible for the rooftop rebate? ZIP 94103, income $90,000, 6 kW, approved installer.",
        "Can I get the rooftop rebate? 94105, we make 150k, 8 kW, SunGrid-approved installer.",
        "Am I eligible for the rooftop rebate? I live in 94103.",
    ],
    "Off-topic": ["What's the weather in Paris today?"],
}

st.set_page_config(page_title="SunGrid Copilot", page_icon="☀️", layout="wide")


# --- access control ------------------------------------------------------------------------


def require_password() -> None:
    """If APP_PASSWORD is set, gate the app so a public URL can't spend the API keys."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return
    st.title("☀️ SunGrid Copilot")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, expected):
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()


# --- cached resources ----------------------------------------------------------------------


@st.cache_resource
def get_graph():
    from sungrid_copilot.agent.graph import build_graph

    return build_graph()


@st.cache_resource
def get_chunks():
    from sungrid_copilot.ingestion.pipeline import load_chunks

    return load_chunks()


# --- chat ----------------------------------------------------------------------------------


def init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("turns", 0)


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


def ask(query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    if st.session_state.turns >= MAX_TURNS_PER_SESSION:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Demo limit reached for this session. Refresh to start again."}
        )
        return
    st.session_state.turns += 1
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    graph = get_graph()  # outside the timer: loading the agent is start-up cost, not answer latency
    start = time.perf_counter()
    try:
        state = graph.invoke({"query": query}, config)
    except Exception as exc:  # surface API/network errors in the UI instead of a stack trace
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {exc}"})
        return
    details = {k: v for k, v in state.items() if k not in ("history",)}
    details["latency_s"] = round(time.perf_counter() - start, 2)
    st.session_state.messages.append({"role": "assistant", "content": state["answer"], "details": details})


ROUTE_LABELS = {
    "rag": "📄 Retrieval + answer",
    "eligibility_check": "🛠️ Eligibility tool",
    "eligibility_needs_info": "❓ Asked for missing info",
    "off_topic": "🚫 Declined (off-topic)",
}


def render_details(d: dict) -> None:
    with st.expander("🔍 Under the hood", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", ROUTE_LABELS.get(d.get("route"), d.get("route")))
        c2.metric("Primary category", d.get("category", "—"))
        c3.metric("Intent", (d.get("intent") or "—").replace("_", " "))
        c4.metric("Latency", f"{d.get('latency_s', 0)} s")

        if d.get("timings"):
            steps = " → ".join(f"{name} {secs}s" for name, secs in d["timings"].items())
            st.caption(f"⏱️ Per step: {steps}")

        if d.get("standalone_query") and d.get("standalone_query") != d.get("query"):
            st.markdown(f"**Rewritten follow-up:** _{d['standalone_query']}_")

        left, right = st.columns(2)
        with left:
            st.markdown("**Category probabilities** (Jev)")
            probs = d.get("category_probabilities") or {}
            if probs:
                df = pd.DataFrame(
                    sorted(probs.items(), key=lambda kv: -kv[1]), columns=["category", "probability"]
                )
                st.dataframe(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")
                    },
                )
            else:
                st.caption("LLM classifier backend: no calibrated probabilities.")
        with right:
            st.markdown("**Retrieval filter**")
            filt = d.get("filter_categories") or []
            st.write(", ".join(f"`{c}`" for c in filt) if filt else "_none (retrieval skipped or not used)_")
            if len(filt) > 1:
                st.caption("Widened: runner-up category ≥ 0.10 probability was added to the filter.")
            if d.get("used_fallback"):
                st.warning("Filter overridden: unfiltered results were clearly better (or filtered ones weak).")

        if d.get("retrieved"):
            st.markdown("**Retrieved chunks** (top 5, cosine similarity)")
            for c in d["retrieved"]:
                with st.container(border=True):
                    st.markdown(f"`{c['score']:.3f}` **{c['header']}**  · tags: {', '.join(c['categories'])}")
                    st.caption(c["text"])

        for t in d.get("tool_calls") or []:
            st.markdown("**Eligibility service call** (`stub_tools.check_rebate_eligibility`, unmodified)")
            a, b = st.columns(2)
            a.markdown("Input")
            a.json(t["args"])
            b.markdown("Output")
            b.json(t["result"])
            if not t["result"]["eligible"]:
                st.info("Hard gate: ineligible → verdict built from a code template; no amount is shown.")

        if d.get("route") == "eligibility_needs_info":
            st.markdown("**Known so far:**")
            st.json(d.get("eligibility_params") or {})


def chat_tab() -> None:
    st.caption(
        "Ask about SunGrid policies, rebates, billing, installation or company news. "
        "Eligibility questions call the eligibility service instead of answering from documents. "
        "Follow-ups use the conversation (try *“what if it were 9 kW?”* after an eligibility check)."
    )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("details"):
                render_details(msg["details"])

    if query := st.chat_input("Ask the SunGrid copilot…"):
        with st.spinner("Thinking…"):
            ask(query)
        st.rerun()


# --- how it works --------------------------------------------------------------------------


def mermaid(code: str, height: int = 820) -> None:
    # The code is passed as a JSON string, not inlined as HTML: LangGraph's Mermaid contains
    # `<p>` tags and `&nbsp;`, which the browser would otherwise parse before Mermaid reads it.
    # Trusted input only: it comes from our own compiled graph, never from users or the LLM.
    st.iframe(
        f"""
        <div id="graph"></div>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: false, theme: "neutral" }});
          const {{ svg }} = await mermaid.render("agent-graph", {json.dumps(code)});
          document.getElementById("graph").innerHTML = svg;
        </script>
        """,
        height=height,
    )


def how_it_works_tab() -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Agent graph (LangGraph)")
        st.caption("Drawn from the compiled graph. Dotted arrows are conditional edges.")
        mermaid(get_graph().get_graph().draw_mermaid())
    with right:
        st.subheader("Nodes")
        st.markdown(
            """
| Node | Job |
|---|---|
| `contextualize` | Rewrites follow-ups into standalone questions |
| `classify` | One primary category + retrieval filter. **Never picks the path** |
| `detect_intent` | Routing: eligibility / general / off-topic |
| `extract_params` | Pulls the 4 eligibility inputs; missing = asked, never guessed |
| `ask_for_missing` | Asks only for what's missing |
| `check_eligibility` | **Only** node that calls the eligibility service |
| `compose_verdict` | Hard gate: code template, no amount when ineligible |
| `retrieve` | Filtered vector search with fallback |
| `generate_answer` | Grounded, cited answer |
| `decline` | Off-topic refusal |
"""
        )
        s = get_settings()
        st.subheader("Configuration")
        st.markdown(
            f"""
- **Agent LLM:** `{s.llm_model}`
- **Embeddings:** `{s.embedding_model}` ({s.embedding_dim}-d)
- **Vector store:** Qdrant {'Cloud' if s.qdrant_url else 'local'} · `{s.qdrant_collection}`
- **Classifier:** {'Jev (TypeSafe)' if s.typesafe_api_key and s.classifier_backend != 'llm' else 'LLM fallback'}
- **Filter mode:** `{s.filter_mode}`
"""
        )


# --- knowledge base ------------------------------------------------------------------------


def knowledge_base_tab() -> None:
    cfg = load_category_config()
    chunks = get_chunks()
    counts = pd.Series([c.doc_id for c in chunks]).value_counts()

    st.subheader("Documents and category tags")
    st.caption("Tags come from `config/categories.yaml`. The first tag is primary; filters match any tag.")
    rows = [
        {
            "document": doc_id,
            "categories": ", ".join(entry["categories"]),
            "chunks": int(counts.get(doc_id, 0)),
            "why": (entry.get("note") or "").strip(),
        }
        for doc_id, entry in cfg["documents"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Category definitions (read by the classifier)"):
        for name, c in cfg["categories"].items():
            st.markdown(f"**`{name}`**: {c['description'].strip()}")

    st.subheader("Chunks")
    doc = st.selectbox("Document", list(cfg["documents"]), index=5)
    for c in [c for c in chunks if c.doc_id == doc]:
        with st.container(border=True):
            st.markdown(f"**{c.header}**")
            st.caption("Embedded text = header + body. The header carries the chunk's origin into the vector.")
            st.write(c.text)


# --- evaluation ----------------------------------------------------------------------------


def evaluation_tab() -> None:
    latest = RESULTS_DIR / "latest.md"
    if not latest.exists():
        st.info("No results yet. Run `python -m evals.run_evals`.")
        return
    left, right = st.columns(2)
    with left:
        st.markdown(latest.read_text(encoding="utf-8"))
    with right:
        compare = RESULTS_DIR / "compare.md"
        if compare.exists():
            st.markdown(compare.read_text(encoding="utf-8"))

    data = RESULTS_DIR / "latest.json"
    if data.exists():
        st.subheader("Per-case results")
        rows = []
        for r in json.loads(data.read_text(encoding="utf-8")):
            row = {
                "case": r["id"],
                "route": r.get("route"),
                "category": r.get("category"),
                "checks": f"{sum(r['checks'].values())}/{len(r['checks'])}",
            }
            for m in r.get("deepeval") or []:
                row[m["metric"].replace(" [GEval]", "")] = None if m["score"] is None else round(m["score"], 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Methodology and what the evals changed: `evals/README.md` in the repository.")


# --- layout --------------------------------------------------------------------------------


def sidebar() -> None:
    with st.sidebar:
        st.header("☀️ SunGrid Copilot")
        st.caption("RAG + agent prototype · LangGraph · Qdrant · Jev · DeepEval")
        if st.button("🔄 New conversation", width="stretch"):
            reset_conversation()
            st.rerun()
        st.divider()
        st.subheader("Try an example")
        for group, questions in EXAMPLES.items():
            st.markdown(f"**{group}**")
            for q in questions:
                if st.button(q, key=f"ex-{q}", width="stretch"):
                    with st.spinner("Thinking…"):
                        ask(q)
                    st.rerun()


def main() -> None:
    require_password()
    init_session()
    if not get_settings().openrouter_api_key:
        st.error("OPENROUTER_API_KEY is not set. See `.env.example`.")
        st.stop()
    # Warm-up: import and build the agent once per server process, before the first question.
    # On a small instance this takes tens of seconds; cached afterwards for every session.
    with st.spinner("Loading the copilot (first visit after a restart takes a little longer)…"):
        get_graph()
        get_chunks()
    sidebar()
    tabs = st.tabs(["💬 Copilot", "🧭 How it works", "📚 Knowledge base", "📊 Evaluation"])
    with tabs[0]:
        chat_tab()
    with tabs[1]:
        how_it_works_tab()
    with tabs[2]:
        knowledge_base_tab()
    with tabs[3]:
        evaluation_tab()


main()
