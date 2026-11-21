"""Tests for scripts/laggard_watchlist_check.py (Laggard Top Five watchlist).

Covers: state load/save round-trip, EDGAR watermark advancement, flag
logic on fixture filing lists (a new 8-K flags; a price-only change does
not), and the missing-credentials failure naming MissingCredentialsError.
All network seams are injected fakes; nothing here touches Alpaca or SEC.
"""
import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from market_lab.data import Bar
from market_lab.intraday.alpaca_fetcher import MissingCredentialsError

_spec = importlib.util.spec_from_file_location(
    "laggard_watchlist_check",
    Path(__file__).resolve().parents[2] / "scripts" / "laggard_watchlist_check.py",
)
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)

CIK_MAP = {"INTU": 896878, "APP": 1751008, "NOW": 1373715, "ADBE": 796343, "IT": 1057786}


def _filing(form, filed, accession):
    return {"form": form, "filing_date": filed, "accession": accession}


def _payload(filings):
    return {
        "filings": {
            "recent": {
                "form": [f["form"] for f in filings],
                "filingDate": [f["filing_date"] for f in filings],
                "accessionNumber": [f["accession"] for f in filings],
            }
        }
    }


def _bars(symbol, last_close):
    return (
        [
            Bar(date(2026, 11, 20), 1.0, 1.0, 1.0, wl.THESIS[symbol]["reference_price"], 100),
            Bar(date(2026, 11, 21), 1.0, 1.0, 1.0, last_close, 100),
        ],
        {"symbol": symbol, "source": "fixture"},
    )


def _ok_backfill(symbols, start, end):
    return [SimpleNamespace(symbol=s, error="") for s in symbols]


def _run(tmp_path, submissions, last_close=100.0):
    state_path = tmp_path / "state.json"
    report_dir = tmp_path / "checks"
    code, lines = wl.run_check(
        state_path,
        report_dir,
        today=date(2026, 11, 21),
        backfill=_ok_backfill,
        load_bars=lambda s: _bars(s, last_close),
        cik_map=CIK_MAP,
        submissions_fetcher=lambda cik: submissions[cik],
    )
    return code, lines, state_path, report_dir


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")


# ------------------------------------------------------------------ state

def test_default_state_carries_thesis_and_reference_prices():
    state = wl.default_state()
    assert state["reference_date"] == "2026-11-20"
    expected = {"INTU": 270.44, "APP": 348.29, "NOW": 87.61, "ADBE": 217.71, "IT": 174.49}
    for ticker, price in expected.items():
        entry = state["tickers"][ticker]
        assert entry["reference_price"] == price
        assert entry["status"] == "watching"
        assert entry["invalidation_triggers"]
        assert entry["confirming_events"]
        assert entry["edgar_watermark"] == {"filing_date": None, "accessions": []}
    # The APP insider pattern and the ADBE dated catalyst ride along as data.
    assert any("insider selling" in n for n in state["tickers"]["APP"]["notes"])
    assert any("Dec 9, 2026" in n for n in state["tickers"]["ADBE"]["notes"])


def test_state_save_load_round_trip(tmp_path):
    state = wl.default_state()
    state["tickers"]["NOW"]["last_close"] = 90.12
    state["tickers"]["NOW"]["edgar_watermark"] = {
        "filing_date": "2026-11-19",
        "accessions": ["0001373715-26-000001"],
    }
    path = tmp_path / "nested" / "state.json"
    wl.save_state(path, state)
    assert wl.load_state(path) == state
    assert wl.load_state(tmp_path / "missing.json") is None


# -------------------------------------------------------------- watermark

def test_parse_recent_filings_filters_to_watched_forms():
    payload = _payload(
        [
            _filing("8-K", "2026-11-10", "a1"),
            _filing("S-8", "2026-11-11", "a2"),
            _filing("4", "2026-11-12", "a3"),
            _filing("10-Q", "2026-11-09", "a4"),
            _filing("DEFA14A", "2026-11-13", "a5"),
        ]
    )
    filings = wl.parse_recent_filings(payload)
    assert [f["accession"] for f in filings] == ["a4", "a1", "a3"]  # sorted by date


def test_filings_since_and_watermark_advancement():
    filings = [
        _filing("8-K", "2026-11-18", "a1"),
        _filing("4", "2026-11-20", "a2"),
        _filing("4", "2026-11-20", "a3"),
        _filing("10-Q", "2026-11-21", "a4"),
    ]
    assert wl.filings_since(filings, None) == filings
    wm = wl.advance_watermark(filings[:3], None)
    assert wm == {"filing_date": "2026-11-20", "accessions": ["a2", "a3"]}
    # Same-date accession already seen is not new; the later 10-Q is.
    assert wl.filings_since(filings, wm) == [filings[3]]
    # Advancing over the same frontier merges accessions, never regresses.
    wm2 = wl.advance_watermark([_filing("4", "2026-11-20", "a9")], wm)
    assert wm2 == {"filing_date": "2026-11-20", "accessions": ["a2", "a3", "a9"]}
    assert wl.advance_watermark([], wm) == wm


# ------------------------------------------------------------------ flags

def test_flag_lines_one_per_new_filing():
    new = [_filing("8-K", "2026-11-21", "acc-1"), _filing("4", "2026-11-21", "acc-2")]
    lines = wl.flag_lines("INTU", new, Path("/tmp/check.json"))
    assert len(lines) == 2
    assert lines[0].startswith("WATCHLIST_FLAG INTU new 8-K filing acc-1 filed 2026-11-21")
    assert lines[1].startswith("WATCHLIST_FLAG INTU new 4 filing acc-2")
    assert wl.flag_lines("INTU", [], Path("/tmp/check.json")) == []


def test_baseline_run_is_quiet_and_sets_watermarks(tmp_path):
    submissions = {cik: _payload([_filing("8-K", "2026-11-10", f"acc-{cik}")]) for cik in CIK_MAP.values()}
    code, lines, state_path, report_dir = _run(tmp_path, submissions)
    assert code == 0
    assert lines == ["WATCHLIST_QUIET"]  # historical filings are baseline, NOT flagged
    state = wl.load_state(state_path)
    for ticker in wl.TICKERS:
        assert state["tickers"][ticker]["edgar_watermark"]["filing_date"] == "2026-11-10"
        assert state["tickers"][ticker]["last_check"] == "2026-11-21"
    reports = list(report_dir.glob("check-*.json"))
    assert len(reports) == 1


def test_price_only_change_never_flags(tmp_path):
    submissions = {cik: _payload([]) for cik in CIK_MAP.values()}
    # A violent move vs the reference price, with zero new filings: quiet.
    code, lines, state_path, _ = _run(tmp_path, submissions, last_close=1.0)
    assert lines == ["WATCHLIST_QUIET"]
    state = wl.load_state(state_path)
    assert state["tickers"]["INTU"]["pct_vs_reference"] < -99  # recorded as context only


def test_new_8k_on_second_run_flags(tmp_path):
    submissions = {cik: _payload([]) for cik in CIK_MAP.values()}
    _run(tmp_path, submissions)  # baseline
    intu_cik = CIK_MAP["INTU"]
    submissions[intu_cik] = _payload([_filing("8-K", "2026-11-21", "acc-new")])
    code, lines, _, report_dir = _run(tmp_path, submissions)
    assert code == 0
    assert len(lines) == 1
    assert lines[0].startswith("WATCHLIST_FLAG INTU new 8-K filing acc-new filed 2026-11-21")
    assert "check-" in lines[0]  # carries the check-report path


def test_failed_scan_does_not_advance_watermark(tmp_path):
    submissions = {cik: _payload([_filing("10-K", "2026-11-01", f"acc-{cik}")]) for cik in CIK_MAP.values()}
    _run(tmp_path, submissions)  # baseline

    def flaky(cik):
        if cik == CIK_MAP["ADBE"]:
            raise RuntimeError("sec down")
        return submissions[cik]

    state_path = tmp_path / "state.json"
    code, lines = wl.run_check(
        state_path,
        tmp_path / "checks",
        today=date(2026, 11, 21),
        backfill=_ok_backfill,
        load_bars=lambda s: _bars(s, 100.0),
        cik_map=CIK_MAP,
        submissions_fetcher=flaky,
    )
    assert code == 1
    assert any("WATCHLIST_ERROR" in line and "ADBE" in line for line in lines)
    state = wl.load_state(state_path)
    assert state["tickers"]["ADBE"]["edgar_watermark"]["filing_date"] == "2026-11-01"


# -------------------------------------------------------------- credentials

def test_missing_credentials_fail_by_name(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(MissingCredentialsError):
        wl.ensure_credentials()
    with pytest.raises(MissingCredentialsError):
        wl.run_check()
    # main() converts it to the named stdout error, exit code 2.
    assert wl.main([]) == 2
