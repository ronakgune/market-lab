from __future__ import annotations

from datetime import datetime, timezone

import pytest
from alpaca.data.enums import Adjustment, DataFeed

from market_lab.intraday.alpaca_fetcher import MissingCredentialsError, fetch_minute_bars

UTC = timezone.utc
START = datetime(2026, 2, 3, 8, 0, tzinfo=UTC)  # 03:00 ET: premarket included by design
END = datetime(2026, 2, 3, 16, 0, tzinfo=UTC)


def _raw(ts: str, close: float = 100.0) -> dict:
    return {"t": ts, "o": close, "h": close + 0.3, "l": close - 0.3, "c": close, "v": 250}


class FakePager:
    """Recorded two-page fetcher: page 1 points at page 2 via next_page_token."""

    def __init__(self):
        self.calls: list[dict] = []
        self.pages = [
            {"bars": {"ACME": [_raw("2026-02-03T08:00:00Z"),  # 03:00 ET premarket bar
                               _raw("2026-02-03T14:30:00Z")]},
             "next_page_token": "tok-2"},
            {"bars": {"ACME": [_raw("2026-02-03T14:31:00Z"),
                               {"t": "2026-02-03T14:32:00Z", "o": 0, "h": 0, "l": 0, "c": 0, "v": 0}]},
             "next_page_token": None},
        ]

    def __call__(self, params: dict) -> dict:
        self.calls.append(dict(params))
        return self.pages[len(self.calls) - 1]


def test_pagination_followed_until_exhausted():
    pager = FakePager()
    bars, source = fetch_minute_bars("ACME", START, END, page_fetcher=pager)
    assert len(pager.calls) == 2
    assert pager.calls[1]["page_token"] == "tok-2"
    # 3 valid bars (the zero bar is invalid and skipped); premarket bar kept.
    assert len(bars) == 3
    assert bars[0].timestamp == datetime(2026, 2, 3, 8, 0, tzinfo=UTC)
    assert source == "alpaca:sip"


def test_feed_and_adjustment_passed_through():
    pager = FakePager()
    bars, source = fetch_minute_bars("ACME", START, END, feed=DataFeed.IEX,
                                     adjustment=Adjustment.SPLIT, page_fetcher=pager)
    assert pager.calls[0]["feed"] == "iex"
    assert pager.calls[0]["adjustment"] == "split"
    assert pager.calls[0]["timeframe"] == "1Min"
    assert source == "alpaca:iex"


def test_defaults_are_sip_and_all():
    pager = FakePager()
    fetch_minute_bars("ACME", START, END, page_fetcher=pager)
    assert pager.calls[0]["feed"] == "sip"
    assert pager.calls[0]["adjustment"] == "all"


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(MissingCredentialsError):
        fetch_minute_bars("ACME", START, END)


def test_naive_bounds_rejected():
    with pytest.raises(ValueError):
        fetch_minute_bars("ACME", datetime(2026, 2, 3), END, page_fetcher=FakePager())


def test_client_injection_builds_page_fetcher():
    class StubClient:
        def __init__(self):
            self.paths = []

        def get(self, path, params):
            self.paths.append(path)
            return {"bars": {"ACME": [_raw("2026-02-03T14:30:00Z")]}, "next_page_token": None}

    stub = StubClient()
    bars, _ = fetch_minute_bars("ACME", START, END, client=stub)
    assert stub.paths == ["/stocks/bars"]
    assert len(bars) == 1
