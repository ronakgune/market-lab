"""Baseline tests for scripts/weekly_rollup.py: book summary math
(equity, week change, recent fills), missing-state handling, and the
carry section over a real funding store.
"""
import importlib.util
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from market_lab.broker import Portfolio, Position, save_portfolio
from market_lab.crypto_carry import FundingPoint, append_points

_spec = importlib.util.spec_from_file_location(
    "weekly_rollup",
    Path(__file__).resolve().parents[2] / "scripts" / "weekly_rollup.py",
)
rollup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rollup)

AS_OF = date(2026, 1, 16)  # a Friday


def _write_book(tmp_path: Path):
    state = tmp_path / "state.json"
    equity = tmp_path / "equity.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    save_portfolio(Portfolio(cash=5_000.0, positions={"SPY": Position("SPY", 100, 200.0)}), state)
    marks = [
        (AS_OF - timedelta(days=10), 25_000.0),
        (AS_OF - timedelta(days=3), 25_500.0),
        (AS_OF, 25_750.0),
    ]
    equity.write_text("".join(
        json.dumps({"date": d.isoformat(), "equity": e}) + "\n" for d, e in marks
    ))
    rows = [
        {"accepted": True, "side": "BUY", "symbol": "SPY", "quantity": 10,
         "fill_price": 200.0, "execution_date": (AS_OF - timedelta(days=2)).isoformat()},
        {"accepted": True, "side": "BUY", "symbol": "SPY", "quantity": 5,
         "fill_price": 190.0, "execution_date": (AS_OF - timedelta(days=10)).isoformat()},
        {"accepted": False, "side": "SELL", "symbol": "SPY", "quantity": 3,
         "fill_price": None, "execution_date": (AS_OF - timedelta(days=1)).isoformat()},
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return state, equity, ledger


def test_summarize_book_math(tmp_path):
    state, equity, ledger = _write_book(tmp_path)
    text = "\n".join(rollup.summarize_book("Test book", state, equity, ledger, 25_000.0, AS_OF))
    assert "Equity: $25,750.00" in text
    assert "cash $5,000.00" in text
    assert "SPY 100 @ $200.00" in text
    # Baseline is the last mark at least 7 days old (the -10d mark).
    assert "Week change: +750.00 (+3.00%)" in text
    # Only the in-week accepted fill is listed.
    assert "Fills this week: 1" in text
    assert "BUY 10 SPY @ $200.00" in text


def test_summarize_book_missing_state(tmp_path):
    text = "\n".join(rollup.summarize_book(
        "Empty book", tmp_path / "nope.json", tmp_path / "nope.jsonl",
        tmp_path / "nope2.jsonl", 25_000.0, AS_OF,
    ))
    assert "No state yet" in text


def test_summarize_carry_no_data_and_with_data(tmp_path):
    missing = "\n".join(rollup.summarize_carry(tmp_path / "funding.jsonl"))
    assert "No funding data stored yet" in missing

    path = tmp_path / "funding.jsonl"
    now = datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc)
    points = [
        FundingPoint("kraken", "BTC", now - timedelta(days=d), 0.0001, 8.0)
        for d in range(25)
    ]
    append_points(path, points)
    text = "\n".join(rollup.summarize_carry(path))
    assert "kraken BTC" in text
    assert "25 days covered" in text
    # 0.01% per 8h annualizes to ~10.95% > 8% trigger, sustained 25 days.
    assert "trigger MET" in text


def test_build_rollup_combines_sections(tmp_path):
    state, equity, ledger = _write_book(tmp_path)
    report = rollup.build_rollup(
        as_of=AS_OF,
        xs_paths=(state, equity, ledger),
        vt_paths=(tmp_path / "none.json", tmp_path / "none.jsonl", tmp_path / "none2.jsonl"),
        carry_path=tmp_path / "funding.jsonl",
    )
    assert "XS momentum (Stream 2)" in report
    assert "vt_trend forward (Stream 1)" in report
    assert "No state yet" in report  # vt book has not run
    assert "Crypto carry (Stream 6" in report
    assert "No funding data stored yet" in report
