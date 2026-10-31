"""Guard the golden set itself: every referenced chunk and category must exist."""

from evals.run_evals import accepted_categories, load_cases, short_id
from sungrid_copilot.config import load_category_config
from sungrid_copilot.ingestion.pipeline import load_chunks


def test_golden_sources_exist():
    known = {short_id(c.chunk_id) for c in load_chunks()}
    for case in load_cases():
        for source in case.get("sources", []):
            assert source in known, f"{case['id']}: unknown source {source}"


def test_golden_categories_exist():
    known = set(load_category_config()["categories"])
    for case in load_cases():
        assert set(accepted_categories(case)) <= known, case["id"]


def test_golden_case_shape():
    ids = [c["id"] for c in load_cases()]
    assert len(ids) == len(set(ids))
    for case in load_cases():
        if case["type"] == "rag":
            assert case.get("sources") and case.get("expected_output"), case["id"]
        if "tool_args" in case:
            assert "eligible" in case, case["id"]
