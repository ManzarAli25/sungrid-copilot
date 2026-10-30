from sungrid_copilot.classifier.classifier import select_filter_categories


def _probs(**kw):
    base = dict.fromkeys(
        ["program_policies", "incentive_rebate", "billing_account",
         "technical_installation", "company_updates", "non_relevant"], 0.0)
    return {**base, **kw}


def test_confident_single_category():
    assert select_filter_categories(_probs(incentive_rebate=1.0)) == ["incentive_rebate"]


def test_plausible_runner_up_widens_filter():
    # Real Jev output for "My rebate hasn't shown up on my bill, who do I contact?"
    probs = _probs(incentive_rebate=0.84, billing_account=0.16)
    assert select_filter_categories(probs) == ["incentive_rebate", "billing_account"]


def test_strict_mode_filters_on_primary_only():
    probs = _probs(incentive_rebate=0.84, billing_account=0.16)
    assert select_filter_categories(probs, mode="strict") == ["incentive_rebate"]


def test_noise_below_threshold_is_ignored():
    probs = _probs(program_policies=0.95, company_updates=0.05)
    assert select_filter_categories(probs) == ["program_policies"]


def test_non_relevant_primary_skips_retrieval():
    assert select_filter_categories(_probs(non_relevant=0.9, company_updates=0.1)) == []


def test_non_relevant_is_never_a_filter_value():
    probs = _probs(company_updates=0.6, non_relevant=0.4)
    assert select_filter_categories(probs) == ["company_updates"]
