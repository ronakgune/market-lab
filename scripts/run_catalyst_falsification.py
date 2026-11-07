#!/usr/bin/env python3
"""Morning Catalyst falsification sprint (Stream 3).

Pre-registered design (market_lab/catalyst_study.py): |gap| >= 5%
events on 12 liquid names, 2019-2026, continuation entry at the open,
+0.5% target vs -0.5% stop race over minute bars to 12:00 ET, realistic
costs (~0.28R/trade). Minute bars come from the intraday store; missing
event days are fetched from Alpaca SIP and written to the store, so
re-runs are free and the study is reproducible.

Full-sample and post-2023-holdout results are recorded in the program
trial ledger BEFORE the verdict prints. The stream survives only if
the holdout passes the kill rule (net Sharpe >= 1.0, PF >= 1.3, no
single loss worse than -5%).

Usage:
    python scripts/run_catalyst_falsification.py [--no-fetch]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.catalyst_study import (
    COST_R,
    StudyStats,
    aggregate,
    find_gap_events,
    kill_rule_verdict,
    simulate_trade,
)
from market_lab.config import PRICE_DIR
from market_lab.daily_data import load_daily_bars
from market_lab.intraday.alpaca_fetcher import fetch_minute_bars
from market_lab.intraday.store import bars_for_day, write_bars
from market_lab.validation import TrialLedger

SYMBOLS = ["AAPL", "AMD", "AMZN", "AVGO", "COIN", "GOOGL",
           "META", "MSFT", "MSTR", "NVDA", "SMCI", "TSLA"]
HOLDOUT_START_YEAR = 2023


def morning_bars(symbol: str, day, allow_fetch: bool):
    bars = bars_for_day(symbol, day)
    if bars or not allow_fetch:
        return bars
    start = datetime.combine(day, time(12, 0), tzinfo=timezone.utc)
    end = datetime.combine(day, time(18, 0), tzinfo=timezone.utc)
    try:
        fetched, _source = fetch_minute_bars(symbol, start, end)
    except Exception:
        return []  # one event's fetch failure must not kill a 741-event study
    if fetched:
        write_bars(fetched)
    return bars_for_day(symbol, day)


def _row(label: str, stats: StudyStats) -> str:
    return (f"{label:<22} {stats.trades:>5} {stats.win_rate * 100:>6.1f}% "
            f"{stats.profit_factor:>7.2f} {stats.mean_net_r:>+8.3f} "
            f"{stats.sharpe_daily_r:>8.2f} {stats.worst_trade_frac * 100:>+7.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use only bars already in the intraday store")
    args = parser.parse_args()

    bars_by_symbol = {}
    for symbol in SYMBOLS:
        bars, _meta = load_daily_bars(symbol)
        bars_by_symbol[symbol] = bars
    events = find_gap_events(bars_by_symbol)
    print(f"{len(events)} gap events (|gap| >= 5%, {SYMBOLS[0]}..{SYMBOLS[-1]}, 2019+)")

    trades, skipped = [], 0
    for i, event in enumerate(events, 1):
        minute_bars = morning_bars(event.symbol, event.day, allow_fetch=not args.no_fetch)
        result = simulate_trade(minute_bars, event.direction) if minute_bars else None
        if result is None:
            skipped += 1
            continue
        trades.append(result)
        if i % 100 == 0:
            print(f"  ...{i}/{len(events)} events, {len(trades)} trades, {skipped} skipped")

    full = aggregate(trades)
    holdout = aggregate([t for t in trades if t.day.year >= HOLDOUT_START_YEAR])
    survives, verdict = kill_rule_verdict(holdout)

    params = {
        "min_gap": 0.05, "target_frac": 0.005, "stop_frac": 0.005,
        "deadline_minutes": 150, "cost_r": round(COST_R, 4),
        "symbols": SYMBOLS, "start_year": 2019, "holdout_start_year": HOLDOUT_START_YEAR,
        "skipped_no_data": skipped,
    }

    def metrics(s: StudyStats) -> dict:
        pf = s.profit_factor
        return {
            "trades": float(s.trades), "win_rate": round(s.win_rate, 4),
            "profit_factor": round(pf, 3) if pf != float("inf") else 1e9,
            "mean_net_r": round(s.mean_net_r, 4), "sharpe_daily_r": round(s.sharpe_daily_r, 3),
            "worst_trade_frac": round(s.worst_trade_frac, 4), "total_net_r": round(s.total_net_r, 2),
        }

    dataset = {
        "symbols": ",".join(SYMBOLS), "events": len(events),
        "source": "alpaca:sip minute bars via intraday store; events from provenanced daily cache",
        "first": trades[0].day.isoformat() if trades else "",
        "last": trades[-1].day.isoformat() if trades else "",
    }
    ledger = TrialLedger(str(PRICE_DIR.parent / "trials.jsonl"))
    full_rec = ledger.record(
        stream="morning_catalyst", strategy="gap_continuation_0p5_race",
        params=params, dataset=dataset, metrics=metrics(full),
        notes="FULL sample 2019-2026. Falsification sprint; simplifications favor "
              "the strategy (see catalyst_study docstring).")
    holdout_rec = ledger.record(
        stream="morning_catalyst", strategy="gap_continuation_0p5_race",
        params=params, dataset=dataset, metrics=metrics(holdout),
        notes=f"HOLDOUT 2023+. Kill rule verdict: {verdict}")
    full_id, holdout_id = full_rec.trial, holdout_rec.trial

    print(f"\nRecorded trial #{full_id} (full) and #{holdout_id} (holdout) in the ledger.")
    print(f"\n{'sample':<22} {'n':>5} {'win':>7} {'PF':>7} {'meanR':>8} {'Sharpe':>8} {'worst':>8}")
    print(_row("full 2019-2026", full))
    print(_row("holdout 2023+", holdout))
    print(f"\nCosts: {COST_R:.2f}R per trade (5bps/fill + $1/fill on $5,000). Skipped (no minute data): {skipped}")
    print(f"\nVERDICT: {verdict}")
    if not survives:
        print("Stream 3 closes permanently per the pre-registered rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
