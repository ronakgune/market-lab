"""Baseline tests for market_lab/fundamentals.py (Stream 7 quality gate).

Fixtures are tiny hand-built companyfacts payloads in the real EDGAR
structure: ACME passes every leg; FAILCO fails every leg. The
point-in-time tests are the money tests — a fact is visible only
from its `filed` date, and a restatement supersedes the original
only from the restatement's filed date.
"""
import copy
import json
from datetime import date
from pathlib import Path

from market_lab.fundamentals import evaluate_quality, quality_as_of

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


ACME = _load("acme_companyfacts.json")
FAILCO = _load("failco_companyfacts.json")


def test_acme_passes_all_legs():
    report = evaluate_quality("ACME", ACME, date(2025, 3, 1))
    assert report.passes
    assert report.reasons == ()
    assert report.roe == 1_200_000_000 / 10_000_000_000
    assert report.latest_fcf == 1_200_000_000
    assert report.fcf_positive_in_window == 5
    assert len(report.fcf_by_year) == 5
    assert report.leverage == 11_000_000_000 / 20_000_000_000


def test_acme_passes_at_an_earlier_date_with_earlier_facts():
    report = evaluate_quality("ACME", ACME, date(2023, 6, 1))
    assert report.passes
    assert report.net_income == 1_000_000_000  # FY2022, the latest filed by then
    assert report.equity == 8_000_000_000
    assert len(report.fcf_by_year) == 3  # 2020-2022 only
    assert report.leverage == 9_000_000_000 / 16_000_000_000


def test_fact_filed_in_2024_is_invisible_to_a_2023_query():
    report = evaluate_quality("ACME", ACME, date(2023, 12, 31))
    # FY2023 net income was filed 2024-02-15: not yet knowable, so the
    # trailing figure is FY2022's — at its restated 950M value, since
    # the 10-K/A (filed 2023-06-15) WAS knowable by this date.
    assert report.net_income == 950_000_000
    assert report.equity == 8_000_000_000
    later = evaluate_quality("ACME", ACME, date(2024, 3, 1))
    assert later.net_income == 1_100_000_000
    assert later.equity == 9_000_000_000


def test_restatement_supersedes_only_from_its_filed_date():
    before = evaluate_quality("ACME", ACME, date(2023, 6, 1))
    assert before.net_income == 1_000_000_000  # original FY2022 10-K
    after = evaluate_quality("ACME", ACME, date(2023, 7, 1))
    assert after.net_income == 950_000_000  # 10-K/A filed 2023-06-15


def test_failco_fails_every_leg_with_reasons():
    report = evaluate_quality("FAILCO", FAILCO, date(2025, 6, 1))
    assert not report.passes
    assert report.roe is not None and report.roe < 0
    assert report.leverage == 7_600_000_000 / 8_000_000_000
    joined = " ".join(report.reasons)
    assert "ROE" in joined
    assert "fiscal years" in joined  # only 2 matched years < 3 required
    assert "leverage" in joined


def test_fcf_majority_rule_failure():
    facts = copy.deepcopy(ACME)
    capex = facts["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"]
    for entry in capex:
        if entry["end"] >= "2022-12-31":
            entry["val"] = 5_000_000_000  # FCF deeply negative 2022-2024
    report = evaluate_quality("ACME", facts, date(2025, 3, 1))
    assert not report.passes
    assert report.fcf_positive_in_window == 2
    assert any("positive in 2 of the last 5" in r for r in report.reasons)


def test_fcf_latest_year_negative_fails_even_with_majority():
    facts = copy.deepcopy(ACME)
    capex = facts["facts"]["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"]
    for entry in capex:
        if entry["end"] == "2024-12-31":
            entry["val"] = 2_500_000_000  # latest-year FCF = -600M
    report = evaluate_quality("ACME", facts, date(2025, 3, 1))
    assert not report.passes
    assert report.fcf_positive_in_window == 4
    assert any("latest fiscal-year FCF is not positive" in r for r in report.reasons)
    assert not any("positive in" in r for r in report.reasons)


def test_missing_data_fails_closed():
    report = evaluate_quality("ZZZZ", {"facts": {"us-gaap": {}}}, date(2025, 1, 1))
    assert not report.passes
    assert report.roe is None and report.leverage is None and report.latest_fcf is None
    assert len(report.reasons) >= 3


def test_non_company_symbols_are_excluded_with_reason():
    for symbol, needle in (("SPY", "index ETF"), ("VIXY", "volatility"), ("IBIT", "crypto")):
        report = quality_as_of(symbol, date(2025, 1, 1), facts={})
        assert not report.passes
        assert needle in report.reasons[0]
