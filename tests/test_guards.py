"""Guard tests: cheap checks that catch config drift and accidental edits."""

import hashlib

from sungrid_copilot.config import DOCS_DIR, PROJECT_ROOT, load_category_config

# SHA-256 of the starter-kit stub_tools.py (LF line endings). The stub represents a
# service another team owns — if this fails, someone edited it.
STUB_SHA256 = "94625eef007754b061bb167b6ba60c3e7f08e3c0529b29b7763eb9954fec2389"


def test_stub_tools_is_unmodified():
    raw = (PROJECT_ROOT / "sungrid_copilot" / "integrations" / "stub_tools.py").read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == STUB_SHA256


def test_readme_graph_matches_compiled_graph():
    from sungrid_copilot.agent.graph import build_graph

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- agent-graph:start -->")[1].split("<!-- agent-graph:end -->")[0]
    mermaid = build_graph().get_graph().draw_mermaid().strip()
    assert mermaid in block, "README graph is stale — run `python -m scripts.draw_graph` and paste"


def test_every_doc_is_tagged_with_known_categories():
    cfg = load_category_config()
    known = set(cfg["categories"]) - {"non_relevant"}
    doc_ids = {p.stem for p in DOCS_DIR.glob("*.md")}

    assert doc_ids == set(cfg["documents"]), "docs/ and categories.yaml are out of sync"
    for doc_id, entry in cfg["documents"].items():
        assert entry["categories"], f"{doc_id} has no categories"
        assert set(entry["categories"]) <= known, f"{doc_id} uses an undefined category"
