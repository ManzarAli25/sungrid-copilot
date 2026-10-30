from sungrid_copilot.agent.contextualize import HISTORY_TURNS, contextualize, format_history
from sungrid_copilot.agent.graph import _finish


def test_first_turn_is_not_rewritten():
    # No history → returned as-is without calling the LLM (no API key needed for this test).
    assert contextualize("When are bills issued?", []) == "When are bills issued?"


def test_history_window_is_limited():
    history = [{"role": "member", "content": f"m{i}"} for i in range(20)]
    lines = format_history(history).splitlines()
    assert len(lines) == 2 * HISTORY_TURNS
    assert lines[-1] == "member: m19"


def test_finish_appends_exchange_to_history():
    update = _finish({"query": "hi"}, "rag", "hello")
    assert update["history"] == [
        {"role": "member", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
