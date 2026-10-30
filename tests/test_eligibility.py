from sungrid_copilot.agent import eligibility as elig
from sungrid_copilot.agent.eligibility import EligibilityParams
from sungrid_copilot.integrations.eligibility_tool import eligibility_tool


def test_tool_wraps_stub_with_unchanged_signature():
    assert set(eligibility_tool.args) == set(elig.REQUIRED_FIELDS)


def test_missing_fields_are_asked_for_not_guessed():
    params = {"household_zip": "94103", "annual_income_usd": 90000}
    assert elig.missing_fields(params) == ["system_size_kw", "installer_approved"]
    msg = elig.ask_for_missing_message(elig.missing_fields(params))
    assert "size of your solar system" in msg and "approved installer list" in msg


def test_installer_false_counts_as_provided():
    params = dict(household_zip="94103", annual_income_usd=1, system_size_kw=5, installer_approved=False)
    assert elig.missing_fields(params) == []


def test_merge_keeps_earlier_values_and_overrides_with_new():
    known = {"household_zip": "94103", "system_size_kw": 5.0}
    new = EligibilityParams(system_size_kw=8.0, annual_income_usd=90000)
    merged = elig.merge(known, new)
    assert merged == {"household_zip": "94103", "system_size_kw": 8.0, "annual_income_usd": 90000}


def test_sanitize_drops_invalid_values():
    p = elig.sanitize(EligibilityParams(household_zip="941", annual_income_usd=-5, system_size_kw=0))
    assert p.household_zip is None and p.annual_income_usd is None and p.system_size_kw is None


def test_hard_gate_ineligible_verdict_has_no_amount():
    result = eligibility_tool.invoke(
        dict(household_zip="94103", annual_income_usd=200_000, system_size_kw=6.0, installer_approved=True)
    )
    assert result["eligible"] is False
    msg = elig.verdict_message(result)
    assert "not eligible" in msg and "$" not in msg
    assert "income exceeds" in msg


def test_eligible_verdict_reports_service_amount():
    result = eligibility_tool.invoke(
        dict(household_zip="94103", annual_income_usd=90_000, system_size_kw=6.0, installer_approved=True)
    )
    assert result == {"eligible": True, "reason": "All eligibility checks passed.", "estimated_rebate_usd": 2400.0}
    assert "$2,400.00" in elig.verdict_message(result)
