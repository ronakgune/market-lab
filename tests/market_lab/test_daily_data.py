"""Baseline tests for the daily-bar data spine (market_lab.daily_data).

The spine's contract: real bars or a loud failure — never invented data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from market_lab import daily_data
from market_lab.data import fetch_prices


def raw_bar(day: str, close: float, volume: int = 1_000_000) -> dict:
    return {
        "t": f"{day}T05:00:00Z",  # midnight ET in EST
        "o": close * 0.99, "h": close * 1.01, "l": close * 0.98,
        "c": close, "v": volume,
    }


class FakePager:
    """Serves canned pages keyed by page_token (None = first page)."""

    def __init__(self, pages: dict):
        self.pages = pages
        self.calls: list[dict] = []

    def __call__(self, params: dict) -> dict:
        self.calls.append(dict(params))
        return self.pages[params.get("page_token")]


def test_fetch_daily_bars_paginates_dedupes_and_validates():
    pager = FakePager({
        None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0), raw_bar("2024-01-03", 471.0)]},
               "next_page_token": "p2"},
        "p2": {"bars": {"SPY": [raw_bar("2024-01-03", 471.5),  # duplicate date: last write wins
                                raw_bar("2024-01-04", 0.0),     # invalid: non-positive close, dropped
                                raw_bar("2024-01-05", 473.0)]},
               "next_page_token": None},
    })
    bars, source = daily_data.fetch_daily_bars(
        "SPY", date(2024, 1, 1), date(2024, 1, 31), page_fetcher=pager)
    assert source == "alpaca:sip"
    assert [b.date for b in bars] == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
    assert bars[1].close == 471.5
    assert len(pager.calls) == 2
    assert pager.calls[0]["timeframe"] == "1Day"
    assert pager.calls[0]["feed"] == "sip"
    assert pager.calls[0]["adjustment"] == "all"


def test_fetch_daily_bars_rejects_inverted_range():
    with pytest.raises(ValueError):
        daily_data.fetch_daily_bars("SPY", date(2024, 2, 1), date(2024, 1, 1),
                                    page_fetcher=FakePager({}))


def test_fetch_daily_bars_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(daily_data.MissingCredentialsError):
        daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31))


def test_backfill_writes_cache_and_provenance(tmp_path, monkeypatch):
    price_dir = tmp_path / "prices"
    monkeypatch.setattr("market_lab.data.PRICE_DIR", price_dir)
    monkeypatch.setattr("market_lab.daily_data.price_path",
                        lambda s: Path(price_dir) / f"{s}.csv")
    monkeypatch.setattr("market_lab.data.ensure_dirs", lambda: None)
    pager = FakePager({None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]},
                               "next_page_token": None}})
    results = daily_data.backfill_daily(["SPY"], date(2024, 1, 1), date(2024, 1, 31),
                                        page_fetcher=pager)
    assert results[0].bars == 1 and not results[0].error
    bars, meta = daily_data.load_daily_bars("SPY")
    assert len(bars) == 1 and bars[0].close == 470.0
    assert meta["source"] == "alpaca:sip" and meta["feed"] == "sip"
    assert meta["adjustment"] == "all" and meta["bars"] == 1


def test_load_daily_bars_refuses_cache_without_provenance(tmp_path, monkeypatch):
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    (price_dir / "SPY.csv").write_text(
        "date,open,high,low,close,volume\n2024-01-02,1,1,1,1,100\n")
    monkeypatch.setattr("market_lab.daily_data.price_path",
                        lambda s: Path(price_dir) / f"{s}.csv")
    monkeypatch.setattr("market_lab.data.PRICE_DIR", price_dir)
    with pytest.raises(FileNotFoundError):
        daily_data.load_daily_bars("SPY")


def test_backfill_failure_is_recorded_not_faked(tmp_path, monkeypatch):
    monkeypatch.setattr("market_lab.data.PRICE_DIR", tmp_path / "prices")
    monkeypatch.setattr("market_lab.data.ensure_dirs", lambda: None)

    def boom(params):
        raise ConnectionError("network down")

    results = daily_data.backfill_daily(["SPY"], date(2024, 1, 1), date(2024, 1, 31),
                                        page_fetcher=boom)
    assert results[0].bars == 0 and "ConnectionError" in results[0].error


def test_fetch_prices_require_live_refuses_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr("market_lab.data.PRICE_DIR", tmp_path / "prices")
    monkeypatch.setattr("market_lab.data.SYNTHETIC_PRICE_DIR", tmp_path / "synthetic")
    monkeypatch.setattr("market_lab.data.ensure_dirs", lambda: None)
    bars, source = fetch_prices("FAKE", days=60, prefer_network=False)
    assert source == "synthetic" and bars  # legacy default still works offline
    with pytest.raises(RuntimeError):
        fetch_prices("FAKE", days=60, prefer_network=False, require_live=True)
