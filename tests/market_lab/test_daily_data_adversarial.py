"""Adversarial tests for the daily-bar data spine (market_lab.daily_data)
and the require_live guard in market_lab.data.fetch_prices.

The spine's contract: real bars or a loud failure — never invented data,
never mislabeled data. These tests attack that contract: forged or
mismatched provenance sidecars, hostile bar payloads, timezone edges,
pagination traps, and every return point of fetch_prices under
require_live=True. No network, no credentials: fake pagers + tmp dirs.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from market_lab import daily_data
from market_lab.data import Bar, fetch_prices, save_prices


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def raw_bar(day: str, close: float, volume: int = 1_000_000) -> dict:
    return {
        "t": f"{day}T05:00:00Z",  # midnight ET in EST
        "o": close * 0.99, "h": close * 1.01, "l": close * 0.98,
        "c": close, "v": volume,
    }


def raw_ts(ts: str, close: float, volume: int = 1_000_000) -> dict:
    return {"t": ts, "o": close * 0.99, "h": close * 1.01,
            "l": close * 0.98, "c": close, "v": volume}


class FakePager:
    """Serves canned pages keyed by page_token (None = first page)."""

    def __init__(self, pages: dict):
        self.pages = pages
        self.calls: list[dict] = []

    def __call__(self, params: dict) -> dict:
        self.calls.append(dict(params))
        return self.pages[params.get("page_token")]


def per_symbol_pager(pages_by_symbol: dict):
    def fetch(params: dict) -> dict:
        return pages_by_symbol[params["symbols"]]
    return fetch


def make_bar(d: date, close: float = 100.0) -> Bar:
    return Bar(d, close * 0.99, close * 1.01, close * 0.98, close, 1_000)


def make_bars(end: date, n: int = 40, close: float = 100.0) -> list[Bar]:
    bars: list[Bar] = []
    d = end
    while len(bars) < n:
        if d.weekday() < 5:
            bars.append(make_bar(d, close))
        d -= timedelta(days=1)
    return sorted(bars, key=lambda b: b.date)


@pytest.fixture()
def cache_env(tmp_path, monkeypatch):
    """Point the real CSV cache + synthetic cache at tmp dirs."""
    price_dir = tmp_path / "prices"
    monkeypatch.setattr("market_lab.data.PRICE_DIR", price_dir)
    monkeypatch.setattr("market_lab.data.SYNTHETIC_PRICE_DIR", tmp_path / "synthetic")
    monkeypatch.setattr("market_lab.data.ensure_dirs", lambda: None)
    return price_dir


def write_cache_with_meta(sym: str, bar_list: list[Bar], **meta_overrides) -> dict:
    """Write a CSV cache plus a sidecar that is consistent unless overridden."""
    save_prices(sym, bar_list)
    meta = {
        "symbol": sym.upper(),
        "source": "alpaca:sip",
        "feed": "sip",
        "adjustment": "all",
        "fetched_at": "2026-11-06T00:00:00+00:00",
        "requested_start": "2024-01-01",
        "requested_end": "2024-12-31",
        "bars": len(bar_list),
        "first": bar_list[0].date.isoformat(),
        "last": bar_list[-1].date.isoformat(),
    }
    meta.update(meta_overrides)
    daily_data.sidecar_path(sym).write_text(json.dumps(meta), encoding="utf-8")
    return meta


THREE_BARS = [make_bar(date(2024, 1, 2), 470.0),
              make_bar(date(2024, 1, 3), 471.0),
              make_bar(date(2024, 1, 4), 472.0)]


# --------------------------------------------------------------------------
# 1. provenance forgery / confusion in load_daily_bars
# --------------------------------------------------------------------------

def test_load_rejects_sidecar_claiming_more_bars_than_csv_holds(cache_env):
    # Sidecar says 2500 bars (a real backfill); the CSV holds only 3.
    write_cache_with_meta("SPY", THREE_BARS, bars=2500)
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_sidecar_when_csv_has_extra_bars(cache_env):
    # Bars appended to the CSV after the backfill: count drifts the other way.
    write_cache_with_meta("SPY", THREE_BARS, bars=2,
                          last=THREE_BARS[1].date.isoformat())
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_sidecar_with_wrong_first_date(cache_env):
    write_cache_with_meta("SPY", THREE_BARS, first="2023-12-29")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_sidecar_with_wrong_last_date(cache_env):
    write_cache_with_meta("SPY", THREE_BARS, last="2024-01-31")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_sidecar_of_another_symbol(cache_env):
    # SPY's genuine sidecar placed next to QQQ's CSV, counts/ranges aligned
    # so ONLY the symbol identity can catch the swap.
    save_prices("QQQ", THREE_BARS)
    spy_meta = write_cache_with_meta("SPY", THREE_BARS)
    daily_data.sidecar_path("QQQ").write_text(json.dumps(spy_meta), encoding="utf-8")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("QQQ")


def test_load_rejects_symlinked_sidecar_of_another_symbol(cache_env):
    write_cache_with_meta("SPY", THREE_BARS)
    save_prices("QQQ", THREE_BARS)
    daily_data.sidecar_path("QQQ").symlink_to(daily_data.sidecar_path("SPY"))
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("QQQ")


def test_load_accepts_symlinked_csv_when_provenance_is_consistent(cache_env):
    # A symlink is not inherently an attack: same symbol, consistent meta.
    write_cache_with_meta("SPY", THREE_BARS)
    elsewhere = cache_env / "elsewhere.csv"
    elsewhere.write_text(daily_data.price_path("SPY").read_text(encoding="utf-8"),
                         encoding="utf-8")
    daily_data.price_path("SPY").unlink()
    daily_data.price_path("SPY").symlink_to(elsewhere)
    bars, meta = daily_data.load_daily_bars("SPY")
    assert len(bars) == 3 and meta["symbol"] == "SPY"


def test_load_rejects_empty_sidecar_object(cache_env):
    save_prices("SPY", THREE_BARS)
    daily_data.sidecar_path("SPY").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_non_object_sidecar(cache_env):
    save_prices("SPY", THREE_BARS)
    daily_data.sidecar_path("SPY").write_text('["alpaca:sip"]', encoding="utf-8")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_rejects_non_alpaca_source_even_when_consistent(cache_env):
    # A perfectly self-consistent sidecar for some other provenance is not
    # admissible on the Alpaca spine.
    write_cache_with_meta("SPY", THREE_BARS, source="yfinance")
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


def test_load_accepts_lowercase_symbol_in_sidecar(cache_env):
    write_cache_with_meta("SPY", THREE_BARS, symbol="spy")
    bars, meta = daily_data.load_daily_bars("SPY")
    assert len(bars) == 3


def test_load_sidecar_without_csv_is_file_not_found(cache_env):
    meta = {
        "symbol": "SPY", "source": "alpaca:sip", "feed": "sip",
        "adjustment": "all", "fetched_at": "2026-11-06T00:00:00+00:00",
        "requested_start": "2024-01-01", "requested_end": "2024-12-31",
        "bars": 3, "first": "2024-01-02", "last": "2024-01-04",
    }
    daily_data.sidecar_path("SPY").parent.mkdir(parents=True, exist_ok=True)
    daily_data.sidecar_path("SPY").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        daily_data.load_daily_bars("SPY")


def test_load_malformed_json_sidecar_fails_loudly(cache_env):
    save_prices("SPY", THREE_BARS)
    daily_data.sidecar_path("SPY").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):  # JSONDecodeError is a ValueError
        daily_data.load_daily_bars("SPY")


def test_load_rejects_stale_sidecar_after_truncated_rewrite(cache_env):
    # Simulates a re-backfill that died mid-save: the CSV was truncated to
    # fewer rows but the previous, fuller sidecar survived.
    write_cache_with_meta("SPY", THREE_BARS)
    save_prices("SPY", THREE_BARS[:1])
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("SPY")


# --------------------------------------------------------------------------
# 2. bar parsing
# --------------------------------------------------------------------------

def test_fetch_accepts_string_numbers():
    bar = raw_bar("2024-01-02", 470.0)
    bar.update({"o": "465.30", "h": "474.70", "l": "460.60",
                "c": "470.00", "v": "1000000"})
    pager = FakePager({None: {"bars": {"SPY": [bar]}, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert len(bars) == 1 and bars[0].close == 470.0 and bars[0].volume == 1_000_000


def test_fetch_drops_bar_missing_volume_keeps_siblings():
    no_v = raw_bar("2024-01-02", 470.0)
    del no_v["v"]
    pager = FakePager({None: {"bars": {"SPY": [no_v, raw_bar("2024-01-03", 471.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 3)]


def test_fetch_drops_junk_entries_keeps_valid():
    pager = FakePager({None: {"bars": {"SPY": [None, "junk", 42,
                                                raw_bar("2024-01-02", 470.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert len(bars) == 1 and bars[0].close == 470.0


def test_fetch_float_volume_is_truncated_not_invented():
    # Policy pin: venue sends integer volume; a float is floored by int().
    pager = FakePager({None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0, volume=100)]},
                               "next_page_token": None}})
    pager.pages[None]["bars"]["SPY"][0]["v"] = 100.9
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert bars[0].volume == 100


def test_fetch_drops_naive_timestamp():
    # No offset: the ET date of a naive timestamp depends on the host's
    # local timezone, so the bar cannot be dated honestly. Drop it.
    pager = FakePager({None: {"bars": {"SPY": [raw_ts("2024-01-02T05:00:00", 470.0),
                                                raw_bar("2024-01-03", 471.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 3)]


def test_fetch_dst_spring_forward_week():
    pager = FakePager({None: {"bars": {"SPY": [
        raw_ts("2024-03-08T05:00:00Z", 510.0),   # midnight EST
        raw_ts("2024-03-10T05:00:00Z", 511.0),   # midnight EST, spring-forward day
        raw_ts("2024-03-11T04:00:00Z", 512.0),   # midnight EDT
    ]}, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 3, 1), date(2024, 3, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 3, 8), date(2024, 3, 10), date(2024, 3, 11)]


def test_fetch_dst_fall_back_week():
    pager = FakePager({None: {"bars": {"SPY": [
        raw_ts("2024-11-01T04:00:00Z", 570.0),   # midnight EDT
        raw_ts("2024-11-03T04:00:00Z", 571.0),   # midnight EDT, fall-back day
        raw_ts("2024-11-04T05:00:00Z", 572.0),   # midnight EST
    ]}, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 11, 1), date(2024, 11, 30),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 11, 1), date(2024, 11, 3), date(2024, 11, 4)]


def test_fetch_midnight_utc_belongs_to_prior_et_date():
    # Policy pin: a bar's identity is its ET date. 2024-01-02T00:00Z is
    # 2024-01-01 19:00 in New York.
    pager = FakePager({None: {"bars": {"SPY": [raw_ts("2024-01-02T00:00:00Z", 470.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 1)]


def test_fetch_duplicate_timestamps_last_write_wins():
    pager = FakePager({
        None: {"bars": {"SPY": [raw_bar("2024-01-02", 100.0),
                                raw_bar("2024-01-02", 101.0),  # same page: later wins
                                raw_bar("2024-01-03", 200.0)]},
               "next_page_token": "p2"},
        "p2": {"bars": {"SPY": [raw_bar("2024-01-03", 201.0)]},  # later page wins
               "next_page_token": None},
    })
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [(b.date, b.close) for b in bars] == [(date(2024, 1, 2), 101.0),
                                                  (date(2024, 1, 3), 201.0)]


def test_fetch_sorts_bars_returned_out_of_order_across_pages():
    pager = FakePager({
        None: {"bars": {"SPY": [raw_bar("2024-01-05", 473.0)]}, "next_page_token": "p2"},
        "p2": {"bars": {"SPY": [raw_bar("2024-01-03", 471.0)]}, "next_page_token": None},
    })
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 3), date(2024, 1, 5)]


def test_fetch_ignores_other_symbols_mixed_into_page():
    pager = FakePager({None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)],
                                       "QQQ": [raw_bar("2024-01-02", 400.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert len(bars) == 1 and bars[0].close == 470.0


def test_fetch_matches_symbol_key_case_insensitively():
    lower_key = FakePager({None: {"bars": {"spy": [raw_bar("2024-01-02", 470.0)]},
                                   "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=lower_key)
    assert len(bars) == 1
    lower_request = FakePager({None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]},
                                       "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("spy", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=lower_request)
    assert len(bars) == 1


# --------------------------------------------------------------------------
# 3. range discipline (boundaries are ET dates)
# --------------------------------------------------------------------------

def test_fetch_range_boundaries_are_inclusive():
    pager = FakePager({None: {"bars": {"SPY": [
        raw_bar("2023-12-29", 460.0),   # before start
        raw_bar("2024-01-02", 470.0),   # exactly start
        raw_bar("2024-01-31", 480.0),   # exactly end
        raw_bar("2024-02-01", 490.0),   # after end (05:00Z stamp)
    ]}, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 2), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 2), date(2024, 1, 31)]


def test_fetch_timezone_can_pull_a_bar_into_range():
    # Stamped Feb 1 in UTC, but Jan 31 19:30 ET: in range for end=Jan 31.
    pager = FakePager({None: {"bars": {"SPY": [raw_ts("2024-02-01T00:30:00Z", 480.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert [b.date for b in bars] == [date(2024, 1, 31)]


def test_fetch_timezone_can_push_a_bar_out_of_range():
    # Stamped Jan 1 in UTC, but Dec 31 2023 21:00 ET: before start=Jan 1.
    pager = FakePager({None: {"bars": {"SPY": [raw_ts("2024-01-01T02:00:00Z", 460.0)]},
                               "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert bars == []


# --------------------------------------------------------------------------
# 4. pagination attacks
# --------------------------------------------------------------------------

def test_fetch_same_token_forever_raises_instead_of_looping():
    calls = []

    def stuck(params):
        calls.append(params)
        return {"bars": {"SPY": []}, "next_page_token": "loop"}

    with pytest.raises(RuntimeError):
        daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                    page_fetcher=stuck)
    assert len(calls) <= 3


def test_fetch_token_cycle_raises():
    pager = FakePager({
        None: {"bars": {"SPY": []}, "next_page_token": "A"},
        "A": {"bars": {"SPY": []}, "next_page_token": "B"},
        "B": {"bars": {"SPY": []}, "next_page_token": "A"},
    })
    with pytest.raises(RuntimeError):
        daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                    page_fetcher=pager)


def test_fetch_empty_pages_with_fresh_tokens_still_terminate():
    pager = FakePager({
        None: {"bars": {"SPY": []}, "next_page_token": "p2"},
        "p2": {"bars": {}, "next_page_token": "p3"},
        "p3": {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]}, "next_page_token": None},
    })
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert len(bars) == 1 and len(pager.calls) == 3


def test_fetch_page_missing_bars_key_is_an_empty_page():
    pager = FakePager({
        None: {"next_page_token": "p2"},
        "p2": {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]}, "next_page_token": None},
    })
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert len(bars) == 1


def test_fetch_bars_null_is_an_empty_page():
    pager = FakePager({None: {"bars": None, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert bars == []


def test_fetch_symbol_bar_list_null_is_an_empty_page():
    pager = FakePager({None: {"bars": {"SPY": None}, "next_page_token": None}})
    bars, _ = daily_data.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 31),
                                          page_fetcher=pager)
    assert bars == []


# --------------------------------------------------------------------------
# 5. require_live: trace every return point of fetch_prices
# --------------------------------------------------------------------------

def _stub_yfinance(monkeypatch, download):
    stub = types.SimpleNamespace(download=download)
    monkeypatch.setitem(sys.modules, "yfinance", stub)


class _FakeFrame:
    def __init__(self, rows):
        self._rows = rows
        self.empty = not rows

    def tail(self, n):
        return self

    def iterrows(self):
        return iter(self._rows)


def _frame_rows(n: int):
    return [
        (datetime(2026, 11, 2) + timedelta(days=i),
         {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 12_345})
        for i in range(n)
    ]


def test_require_live_fresh_real_cache_is_served(cache_env):
    save_prices("SPY", make_bars(date.today()))
    bars, source = fetch_prices("SPY", days=30, prefer_network=False, require_live=True)
    assert source == "cache" and len(bars) == 30


def test_require_live_stale_real_cache_raises_not_served(cache_env):
    # A real but stale cache must NOT be served under require_live, and must
    # NOT quietly fall through to synthetic either: it raises.
    save_prices("SPY", make_bars(date.today() - timedelta(days=30)))
    with pytest.raises(RuntimeError):
        fetch_prices("SPY", days=30, prefer_network=False, require_live=True)


def test_stale_real_cache_without_require_live_falls_to_synthetic(cache_env):
    # Pins the legacy boundary: only require_live closes the synthetic door.
    save_prices("SPY", make_bars(date.today() - timedelta(days=30)))
    bars, source = fetch_prices("SPY", days=30, prefer_network=False)
    assert source == "synthetic" and bars


def test_require_live_network_failure_with_fresh_cache_serves_cache(cache_env, monkeypatch):
    save_prices("SPY", make_bars(date.today()))

    def boom(*a, **k):
        raise ConnectionError("network down")

    _stub_yfinance(monkeypatch, boom)
    bars, source = fetch_prices("SPY", days=30, prefer_network=True, require_live=True)
    assert source == "cache" and len(bars) == 30


def test_require_live_empty_network_result_and_no_cache_raises(cache_env, monkeypatch):
    _stub_yfinance(monkeypatch, lambda *a, **k: _FakeFrame([]))
    with pytest.raises(RuntimeError):
        fetch_prices("SPY", days=30, prefer_network=True, require_live=True)


def test_require_live_short_network_result_is_not_served_or_cached(cache_env, monkeypatch):
    # yfinance returns a few valid bars (below the min-days floor): they are
    # neither returned nor persisted as a "real" cache under require_live.
    _stub_yfinance(monkeypatch, lambda *a, **k: _FakeFrame(_frame_rows(5)))
    with pytest.raises(RuntimeError):
        fetch_prices("SPY", days=60, prefer_network=True, require_live=True)
    assert not daily_data.price_path("SPY").exists()


# --------------------------------------------------------------------------
# 6. backfill integrity
# --------------------------------------------------------------------------

def test_backfill_dashed_symbol_round_trip(cache_env):
    pager = per_symbol_pager({
        "BRK-B": {"bars": {"BRK-B": [raw_bar("2024-01-02", 400.0)]},
                  "next_page_token": None},
    })
    results = daily_data.backfill_daily(["BRK-B"], date(2024, 1, 1), date(2024, 1, 31),
                                        page_fetcher=pager)
    assert results[0].bars == 1 and not results[0].error
    assert (cache_env / "BRK_B.csv").exists()  # sanitized on disk
    bars, meta = daily_data.load_daily_bars("BRK-B")
    assert bars[0].close == 400.0 and meta["symbol"] == "BRK-B"


def test_backfill_sanitized_path_collision_cannot_mislabel(cache_env):
    # price_path maps both "BRK-B" and "BRK/B" to BRK_B.csv: the second
    # backfill overwrites the first's files. Loading the FIRST symbol must
    # then fail loudly (its provenance now names the other symbol), never
    # return the second symbol's bars under the first symbol's name.
    pager = per_symbol_pager({
        "BRK-B": {"bars": {"BRK-B": [raw_bar("2024-01-02", 111.0)]},
                  "next_page_token": None},
        "BRK/B": {"bars": {"BRK/B": [raw_bar("2024-01-02", 222.0)]},
                  "next_page_token": None},
    })
    results = daily_data.backfill_daily(["BRK-B", "BRK/B"], date(2024, 1, 1),
                                        date(2024, 1, 31), page_fetcher=pager)
    assert all(not r.error for r in results)
    bars, meta = daily_data.load_daily_bars("BRK/B")
    assert bars[0].close == 222.0
    with pytest.raises(ValueError):
        daily_data.load_daily_bars("BRK-B")


def test_backfill_sidecar_write_failure_is_recorded_and_load_refuses(cache_env, monkeypatch):
    pager = FakePager({None: {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]},
                               "next_page_token": None}})
    real_sidecar_path = daily_data.sidecar_path
    with monkeypatch.context() as m:
        m.setattr(daily_data, "sidecar_path",
                  lambda s: cache_env / "missing-dir" / "SPY.csv.meta.json")
        results = daily_data.backfill_daily(["SPY"], date(2024, 1, 1), date(2024, 1, 31),
                                            page_fetcher=pager)
    assert results[0].error  # failure recorded, not hidden
    assert not real_sidecar_path("SPY").exists()
    with pytest.raises(FileNotFoundError):
        daily_data.load_daily_bars("SPY")  # CSV exists but has no provenance


def test_backfill_one_symbol_failure_does_not_poison_others(cache_env):
    def fetch(params):
        if params["symbols"] == "BAD":
            raise ConnectionError("venue down for BAD")
        return {"bars": {"SPY": [raw_bar("2024-01-02", 470.0)]},
                "next_page_token": None}

    results = daily_data.backfill_daily(["BAD", "SPY"], date(2024, 1, 1),
                                        date(2024, 1, 31), page_fetcher=fetch)
    by_symbol = {r.symbol: r for r in results}
    assert "ConnectionError" in by_symbol["BAD"].error
    assert by_symbol["SPY"].bars == 1 and not by_symbol["SPY"].error
    with pytest.raises(FileNotFoundError):
        daily_data.load_daily_bars("BAD")
    bars, _ = daily_data.load_daily_bars("SPY")
    assert len(bars) == 1
