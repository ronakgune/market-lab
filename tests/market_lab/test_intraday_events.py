from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from market_lab.intraday.events import (
    EarningsEvent,
    classify_session,
    edgar_user_agent,
    events_from_submissions,
    fetch_earnings_events,
    fetch_submissions,
    next_trading_day,
)

FIXTURE = Path(__file__).parent / "fixtures" / "intraday" / "edgar_submissions_acme.json"
ET = ZoneInfo("America/New_York")
FEB = (date(2026, 2, 1), date(2026, 2, 28))


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _events() -> list[EarningsEvent]:
    return events_from_submissions("ACME", 1234567, _fixture(), *FEB)


def test_only_item_202_8k_in_range():
    events = _events()
    assert [e.accession_number for e in events] == [
        "0001234567-26-000001",  # Feb 3 premarket
        "0001234567-26-000002",  # Feb 5 afterhours
        "0001234567-26-000004",  # Feb 6 afterhours (Friday)
        "0001234567-26-000003",  # Feb 9 intraday
    ]


def test_session_classification_and_actionable_days():
    by_acc = {e.accession_number: e for e in _events()}

    pre = by_acc["0001234567-26-000001"]
    assert pre.session == "premarket"
    assert pre.accepted_et == datetime(2026, 2, 3, 7, 15, tzinfo=ET)
    assert pre.actionable_day == date(2026, 2, 3)

    ah = by_acc["0001234567-26-000002"]
    assert ah.session == "afterhours"
    assert ah.accepted_et.time() == time(16, 20)
    assert ah.actionable_day == date(2026, 2, 6)  # next trading day

    fri = by_acc["0001234567-26-000004"]
    assert fri.session == "afterhours"
    assert fri.actionable_day == date(2026, 2, 9)  # Friday evening rolls to Monday

    intra = by_acc["0001234567-26-000003"]
    assert intra.session == "intraday"
    assert intra.actionable_day == date(2026, 2, 9)
    assert intra.cik == 1234567 and intra.ticker == "ACME"


def test_classify_session_boundaries():
    def at(h: int, m: int) -> datetime:
        return datetime(2026, 2, 3, h, m, tzinfo=ET)

    assert classify_session(at(9, 29)) == "premarket"
    assert classify_session(at(9, 30)) == "intraday"
    assert classify_session(at(15, 59)) == "intraday"
    assert classify_session(at(16, 0)) == "afterhours"


def test_next_trading_day_skips_weekend():
    assert next_trading_day(date(2026, 2, 6)) == date(2026, 2, 9)  # Fri -> Mon
    assert next_trading_day(date(2026, 2, 4)) == date(2026, 2, 5)  # Wed -> Thu


def test_fetch_earnings_events_with_injected_seams():
    calls = []

    def fake_fetch(ticker: str, cik: int) -> dict:
        calls.append((ticker, cik))
        return _fixture()

    events = fetch_earnings_events(
        ["ACME"], *FEB,
        submissions_fetcher=fake_fetch,
        cik_resolver=lambda ticker: 1234567,
    )
    assert calls == [("ACME", 1234567)]
    assert len(events) == 4


def test_user_agent_placeholder_and_env(monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    ua = edgar_user_agent()
    assert "REPLACE-ME" in ua and "example.com" in ua
    monkeypatch.setenv("EDGAR_USER_AGENT", "test-agent/1.0 (dev@example.com)")
    assert edgar_user_agent() == "test-agent/1.0 (dev@example.com)"


def test_fetch_submissions_caches_raw_json(tmp_path):
    calls = []

    def http_get(url: str, headers: dict) -> dict:
        calls.append((url, headers))
        return _fixture()

    payload = fetch_submissions(1234567, http_get=http_get, cache_dir=tmp_path)
    assert payload["name"] == "ACME CORP"
    assert calls[0][0] == "https://data.sec.gov/submissions/CIK0001234567.json"
    assert (tmp_path / "CIK0001234567.json").exists()

    def boom(url: str, headers: dict) -> dict:
        raise AssertionError("second call must come from cache")

    again = fetch_submissions(1234567, http_get=boom, cache_dir=tmp_path)
    assert again == payload
