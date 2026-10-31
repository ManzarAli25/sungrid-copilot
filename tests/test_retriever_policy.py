from sungrid_copilot.retrieval.retriever import RetrievedChunk, should_override_filter


def _hits(*scores):
    return [RetrievedChunk(f"c{i}", "h", "t", [], s) for i, s in enumerate(scores)]


def test_keeps_filter_when_filtered_results_are_competitive():
    assert not should_override_filter(_hits(0.70, 0.60), _hits(0.72, 0.70))


def test_overrides_when_filtered_results_are_weak():
    assert should_override_filter(_hits(0.30), _hits(0.32))


def test_overrides_when_outside_hit_is_clearly_better():
    # Real case: multi-family tier question, filtered 0.530 vs Company Updates 0.669
    assert should_override_filter(_hits(0.530), _hits(0.669))


def test_overrides_when_filter_returns_nothing():
    assert should_override_filter([], _hits(0.5))
