#!/usr/bin/env python3
"""Phase-2 evaluation for the defined-risk premium-selling stream (Stream 4).

Runs the evaluation frozen in docs/premium-selling-preregistration.md
(registered 2026-11-17, approved before any results existed):

- Full window: every chain day 2020-01-10 -> 2026-11-13.
- Four stress reads, reported separately and recorded as their own
  trials: Feb-Apr 2020, Jan-Nov 2022, Jul-Aug 2024, Mar-Apr 2025.
  A stress read takes entries only inside its window (the IV gate's
  trailing history is seeded with the chain days before the window —
  prior data, causal) and manages those positions to completion on
  later chain days.

Data discipline:
- Every chain day is fetched through market_lab.options_history
  (per-date queries ONLY — range queries against this API silently
  return empty). The per-date cache with provenance sidecars is the
  checkpoint: a rerun resumes where the last one stopped.
- The evaluation refuses to simulate on unresolved dates. A date is
  resolved when it returns a ChainDay or a provenanced NoChainData.
  A transport failure is neither: failed dates are retried once,
  and if any remain, the run stops and reports the block WITHOUT
  recording trials — a chain day silently treated as absent would
  corrupt the gate history, the marks, and the management grid.
- Every run actually performed is recorded in the program-wide
  TrialLedger before results are printed.

Equity marks are end-of-chain-day: settle expired positions, apply
the registered management rules, attempt the registered entry, then
snapshot (a position entered today is marked at today's closing
cost — the spread is paid immediately, as it would be live).

Usage:
    python scripts/run_premium_selling_backtest.py
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import PRICE_DIR
from market_lab.daily_data import load_daily_bars
from market_lab.data import Bar
from market_lab.options_history import ChainDay, NoChainData, fetch_chain, load_chain
from market_lab.premium_selling import (
    BOOK_EQUITY, PaperBook, Position, build_iron_condor,
    build_put_credit_spread, evaluate_management, excursion_breach,
    iv_gate, Skip, Structure,
)
from market_lab.validation import TrialLedger

SYMBOL = "SPY"
WINDOW_START = date(2020, 1, 10)
WINDOW_END = date(2026, 11, 13)
STRESS_WINDOWS = [
    ("stress_2020_covid", date(2020, 2, 1), date(2020, 4, 30)),
    ("stress_2022_bear", date(2022, 1, 1), date(2022, 11, 30)),
    ("stress_2024_volspike", date(2024, 7, 1), date(2024, 8, 31)),
    ("stress_2025_tariff", date(2025, 3, 1), date(2025, 4, 30)),
]
LEDGER_PATH = PRICE_DIR.parent / "trials.jsonl"
CHAIN_WEEKDAYS = (0, 2, 4)  # Mon / Wed / Fri — the dataset's cadence


# ---------------------------------------------------------------------------
# Fetch phase (checkpointed by the options_history cache itself)
# ---------------------------------------------------------------------------

def candidate_dates(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() in CHAIN_WEEKDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


def resolve_dates(days: list[date], workers: int = 3) -> tuple[dict[date, ChainDay], list[date]]:
    """Resolve every candidate date to a ChainDay or provenanced absence.

    Returns (chains_by_day, unresolved). Cached dates cost nothing;
    uncached dates are fetched with a small thread pool, and failures
    are retried once sequentially before being declared unresolved.
    """
    chains: dict[date, ChainDay] = {}
    todo: list[date] = []
    for d in days:
        try:
            r = load_chain(SYMBOL, d)
            if isinstance(r, ChainDay):
                chains[d] = r
        except FileNotFoundError:
            todo.append(d)
        except ValueError:
            todo.append(d)  # refused cache entry: re-fetch to re-prove it
    print(f"fetch: {len(chains)} chain days cached, {len(todo)} dates to fetch")

    def _fetch(d: date):
        try:
            return d, fetch_chain(SYMBOL, d), None
        except Exception as exc:  # transport/server failure — never data
            return d, None, exc

    failed: list[date] = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (d, result, err) in enumerate(pool.map(_fetch, todo), 1):
                if err is not None:
                    failed.append(d)
                elif isinstance(result, ChainDay):
                    chains[d] = result
                if i % 100 == 0:
                    print(f"fetch: {i}/{len(todo)} resolved "
                          f"({len(chains)} chain days so far, {len(failed)} failed)")
    if failed:
        print(f"fetch: retrying {len(failed)} failed dates once, sequentially")
        still: list[date] = []
        for d in failed:
            _, result, err = _fetch(d)
            if err is not None:
                still.append(d)
            elif isinstance(result, ChainDay):
                chains[d] = result
        failed = still
    return chains, failed


# ---------------------------------------------------------------------------
# Simulation (the registered rules; machinery does the per-day work)
# ---------------------------------------------------------------------------

def _skip_bucket(reason: str) -> str:
    for prefix in (
        "no expiry with DTE in",
        "no PUT strike with |delta|",
        "no CALL strike with |delta|",
        "no available PUT strike below",
        "no available CALL strike above",
        "non-positive credit",
        "risk cap allows 0 contracts",
        "insufficient IV history",
        "IV percentile",
        "no IV index computable",
        "aggregate open-risk cap",
    ):
        if reason.startswith(prefix):
            return prefix
    return reason


class RunOutcome:
    def __init__(self) -> None:
        self.book = PaperBook()
        self.snapshots = []            # BookSnapshot per chain day simulated
        self.trade_log: list[dict] = []  # per closed trade, with max_loss + breach
        self.census: dict[str, int] = {}
        self.entries_by_kind: dict[str, int] = {}
        self.leg_missing_events = 0


def simulate(chains: dict[date, ChainDay], spy_bars: list[Bar],
             entry_start: date, entry_end: date, sim_end: date,
             seed_prior: list[ChainDay]) -> RunOutcome:
    out = RunOutcome()
    book = out.book
    spy_close = {b.date: b.close for b in spy_bars}
    spy_dates = [b.date for b in spy_bars]

    def close_on(d: date) -> float:
        if d in spy_close:
            return spy_close[d]
        idx = bisect.bisect_right(spy_dates, d) - 1
        return spy_close[spy_dates[idx]]

    def record_close(pos: Position, exit_day: date, trade) -> None:
        bars_slice = [b for b in spy_bars if b.date <= exit_day]
        breach = excursion_breach(pos, bars_slice)
        out.trade_log.append({
            "kind": trade.kind, "entry_date": pos.entry_date,
            "exit_date": trade.exit_date, "exit_reason": trade.exit_reason,
            "contracts": trade.contracts, "credit_total": trade.credit_total,
            "max_loss_total": pos.max_loss_total, "pnl": trade.pnl,
            "excursion_breach": breach is not None,
        })

    prior: list[ChainDay] = list(seed_prior)
    sim_days = [d for d in sorted(chains) if entry_start <= d <= sim_end]
    for day in sim_days:
        chain = chains[day]
        # 1) settle positions whose expiration has passed
        for pos in list(book.positions):
            if pos.expiration < day:
                trade = book.settle(pos, close_on(pos.expiration))
                record_close(pos, pos.expiration, trade)
        # 2) registered management on today's chain
        for pos in list(book.positions):
            dec = evaluate_management(pos, chain)
            if dec.reason == "leg_missing_from_chain_no_decision":
                out.leg_missing_events += 1
            if dec.action == "EXIT" and dec.close_cost_total is not None:
                trade = book.close(pos, day, dec.reason, dec.close_cost_total)
                record_close(pos, day, trade)
        # 3) registered entry attempt (one per chain day, condor first)
        if entry_start <= day <= entry_end:
            gate_ok, gate_reason = iv_gate(chain, prior)
            if not gate_ok:
                key = _skip_bucket(gate_reason)
                out.census[key] = out.census.get(key, 0) + 1
            else:
                structure = build_iron_condor(chain, day)
                if isinstance(structure, Skip):
                    key = _skip_bucket(structure.reason)
                    out.census[key] = out.census.get(key, 0) + 1
                    structure = build_put_credit_spread(chain, day)
                    if isinstance(structure, Skip):
                        key = _skip_bucket(structure.reason)
                        out.census[key] = out.census.get(key, 0) + 1
                        structure = None
                if isinstance(structure, Structure):
                    ok, why = book.can_open(structure)
                    if ok:
                        book.open(structure, day)
                        out.entries_by_kind[structure.kind] = \
                            out.entries_by_kind.get(structure.kind, 0) + 1
                    else:
                        key = _skip_bucket(why)
                        out.census[key] = out.census.get(key, 0) + 1
        # 4) end-of-day snapshot, then today becomes gate history
        out.snapshots.append(book.snapshot(chain))
        prior.append(chain)
    return out


# ---------------------------------------------------------------------------
# Metrics (survival-bar definitions are the pre-registration's)
# ---------------------------------------------------------------------------

def run_metrics(out: RunOutcome, spy_bars: list[Bar]) -> dict:
    snaps = out.snapshots
    equities = [s.equity for s in snaps]
    rets = [equities[i] / equities[i - 1] - 1.0
            for i in range(1, len(equities)) if equities[i - 1] > 0]
    spy_index = {b.date: i for i, b in enumerate(spy_bars)}
    gaps = [spy_index[snaps[i].day] - spy_index[snaps[i - 1].day]
            for i in range(1, len(snaps))
            if snaps[i].day in spy_index and snaps[i - 1].day in spy_index]
    median_gap = median(gaps) if gaps else 2
    if len(rets) > 2 and pstdev(rets) > 0:
        sharpe = mean(rets) / pstdev(rets) * math.sqrt(252.0 / median_gap)
    else:
        sharpe = 0.0
    peak = -math.inf
    max_dd_dollars = 0.0
    for e in equities:
        peak = max(peak, e)
        max_dd_dollars = max(max_dd_dollars, peak - e)
    by_month: dict[tuple[int, int], list[float]] = {}
    for s in snaps:
        by_month.setdefault((s.day.year, s.day.month), []).append(s.equity)
    monthly = []
    prev_last: float | None = None
    for key in sorted(by_month):
        vals = by_month[key]
        base = vals[0] if len(vals) > 1 or prev_last is None else prev_last
        monthly.append(vals[-1] / base - 1.0)
        prev_last = vals[-1]
    closed = out.trade_log
    stop_failures = sum(1 for t in closed
                        if t["max_loss_total"] > 0
                        and t["pnl"] <= -0.90 * t["max_loss_total"])
    breaches = sum(1 for t in closed if t["excursion_breach"])
    return {
        "snapshots": len(snaps),
        "final_equity": equities[-1] if equities else BOOK_EQUITY,
        "total_return": (equities[-1] / BOOK_EQUITY - 1.0) if equities else 0.0,
        "realized_pnl": out.book.realized_pnl,
        "sharpe": sharpe,
        "median_gap_trading_days": float(median_gap),
        "max_drawdown_dollars": max_dd_dollars,
        "max_drawdown_book_frac": max_dd_dollars / BOOK_EQUITY,
        "worst_month": min(monthly) if monthly else 0.0,
        "closed_trades": len(closed),
        "wins": sum(1 for t in closed if t["pnl"] > 0),
        "stop_failures": stop_failures,
        "stop_failure_rate": (stop_failures / len(closed)) if closed else 0.0,
        "excursion_breaches": breaches,
        "excursion_rate": (breaches / len(closed)) if closed else 0.0,
        "worst_trade_pnl": min((t["pnl"] for t in closed), default=0.0),
        "open_at_end": len(out.book.positions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=str, default=str(LEDGER_PATH))
    parser.add_argument("--fetch-only", action="store_true",
                        help="resolve all chain days, then stop before simulating")
    args = parser.parse_args()

    days = candidate_dates(WINDOW_START, WINDOW_END)
    chains, unresolved = resolve_dates(days)
    print(f"fetch: {len(chains)} chain days secured of {len(days)} candidate dates; "
          f"{len(days) - len(chains) - len(unresolved)} provenanced absences; "
          f"{len(unresolved)} unresolved")
    if unresolved:
        print("BLOCKED: unresolved dates after one retry "
              "(transport/server failures — not evidence of absence):")
        for d in unresolved[:20]:
            print(f"  {d}")
        if len(unresolved) > 20:
            print(f"  ... and {len(unresolved) - 20} more")
        print("No simulation run and no trials recorded. The cache keeps "
              "every resolved date; rerun to resume.")
        return 2
    if args.fetch_only:
        print("fetch-only: all dates resolved; not simulating.")
        return 0

    spy_bars, _ = load_daily_bars(SYMBOL)
    ledger = TrialLedger(args.ledger)
    print(f"ledger: {ledger.count()} trials present before this run")

    ordered_days = sorted(chains)
    runs: list[tuple[str, date, date, date]] = [
        ("full", WINDOW_START, WINDOW_END, WINDOW_END),
    ] + [(name, s, e, ordered_days[-1]) for name, s, e in STRESS_WINDOWS]

    results: dict[str, tuple[RunOutcome, dict]] = {}
    for name, entry_start, entry_end, sim_end in runs:
        seed = [chains[d] for d in ordered_days if d < entry_start]
        out = simulate(chains, spy_bars, entry_start, entry_end, sim_end, seed)
        m = run_metrics(out, spy_bars)
        results[name] = (out, m)
        ledger.record(
            stream="premium_selling",
            strategy="premium_selling",
            params={
                "window": name,
                "entry_start": entry_start.isoformat(),
                "entry_end": entry_end.isoformat(),
                "book_equity": BOOK_EQUITY,
                "fill_model": "shorts at bid, longs at ask; $0.65/contract/leg/side",
                "structures": "iron condor first, then put credit spread; expiry nearest "
                              "37 DTE in [30,45]; short |delta| nearest 0.13 in [0.10,0.16]; "
                              "wings = next available strike",
                "iv_gate": "chain IV index >= 60th pctile of trailing 60 chain days (>=20 prior)",
                "management": "hard exit DTE<=7; stop at 2x credit; take profit at 50% credit; "
                              "expiry settlement at SPY close",
                "sizing": "max loss <= $250/position; aggregate open risk <= $1,250",
                "preregistration": "docs/premium-selling-preregistration.md (2026-11-17)",
            },
            dataset={
                "symbol": SYMBOL,
                "chain_days": len(chains),
                "first": ordered_days[0].isoformat(),
                "last": ordered_days[-1].isoformat(),
                "source": "dolthub:post-no-preference/options@master via "
                          "market_lab.options_history (per-date queries, provenanced cache)",
            },
            metrics={
                "total_return": m["total_return"],
                "net_pnl": m["final_equity"] - BOOK_EQUITY,
                "realized_pnl": m["realized_pnl"],
                "sharpe": m["sharpe"],
                "max_drawdown_book_frac": m["max_drawdown_book_frac"],
                "worst_month": m["worst_month"],
                "closed_trades": float(m["closed_trades"]),
                "stop_failure_rate": m["stop_failure_rate"],
                "excursion_rate": m["excursion_rate"],
                "worst_trade_pnl": m["worst_trade_pnl"],
                "open_at_end": float(m["open_at_end"]),
            },
            notes=f"Stream 4 phase 2 (pre-registered); window={name}",
        )
        print(f"recorded trial for window={name}: closed={m['closed_trades']} "
              f"net=${m['final_equity'] - BOOK_EQUITY:,.0f} sharpe={m['sharpe']:.2f}")

    full_out, fm = results["full"]
    print(f"\n== FULL WINDOW 2020-01-10 -> 2026-11-13 ({len(chains)} chain days) ==")
    print(f"final equity ${fm['final_equity']:,.2f} (net ${fm['final_equity'] - BOOK_EQUITY:,.2f}, "
          f"realized ${fm['realized_pnl']:,.2f}); snapshots {fm['snapshots']}; "
          f"open at end {fm['open_at_end']}")
    print(f"Sharpe {fm['sharpe']:.2f} (median chain-day gap {fm['median_gap_trading_days']:.0f} "
          f"trading days) -> survival bar 1 (>0): {'PASS' if fm['sharpe'] > 0 else 'FAIL'}"
          f" | kill 1: {'TRIGGERED' if fm['sharpe'] <= 0 else 'no'}")
    print(f"max drawdown ${fm['max_drawdown_dollars']:,.0f} = {fm['max_drawdown_book_frac'] * 100:.1f}% "
          f"of book -> survival bar 2 (<=12%): {'PASS' if fm['max_drawdown_book_frac'] <= 0.12 else 'FAIL'}"
          f" | kill 1: {'TRIGGERED' if fm['max_drawdown_book_frac'] > 0.12 else 'no'}")
    print(f"worst month {fm['worst_month'] * 100:.1f}% -> survival bar 3 (>=-6%): "
          f"{'PASS' if fm['worst_month'] >= -0.06 else 'FAIL'}"
          f" | kill 1: {'TRIGGERED' if fm['worst_month'] < -0.06 else 'no'}")
    print(f"closed trades {fm['closed_trades']} (wins {fm['wins']}) -> survival bar 4 (>=40): "
          f"{'PASS' if fm['closed_trades'] >= 40 else 'FAIL'}"
          f" | kill 4: {'TRIGGERED' if fm['closed_trades'] < 40 else 'no'}")
    print(f"stop-execution failures {fm['stop_failures']}/{fm['closed_trades']} "
          f"= {fm['stop_failure_rate'] * 100:.1f}% -> kill 2 (>=25%): "
          f"{'TRIGGERED' if fm['stop_failure_rate'] >= 0.25 else 'no'}")
    print(f"excursion breaches {fm['excursion_breaches']}/{fm['closed_trades']} "
          f"= {fm['excursion_rate'] * 100:.1f}% -> kill 3 (>=25%): "
          f"{'TRIGGERED' if fm['excursion_rate'] >= 0.25 else 'no'}")
    print(f"entries by kind: {full_out.entries_by_kind}; "
          f"leg-missing no-decision events: {full_out.leg_missing_events}")
    print("skip census (full window, per chain-day entry attempt):")
    for reason, count in sorted(full_out.census.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {reason}")

    print("\n== STRESS READS (entries inside window; managed to completion) ==")
    for name, _, _ in STRESS_WINDOWS:
        _, m = results[name]
        print(f"{name:20s} closed={m['closed_trades']:3d} wins={m['wins']:3d} "
              f"realized=${m['realized_pnl']:>10,.0f} worst trade=${m['worst_trade_pnl']:>9,.0f} "
              f"maxDD={m['max_drawdown_book_frac'] * 100:5.1f}% of book "
              f"stop-fail={m['stop_failure_rate'] * 100:4.1f}% "
              f"excursions={m['excursion_rate'] * 100:4.1f}%")

    survives = (
        fm["sharpe"] > 0 and fm["max_drawdown_book_frac"] <= 0.12
        and fm["worst_month"] >= -0.06 and fm["closed_trades"] >= 40
        and fm["stop_failure_rate"] < 0.25 and fm["excursion_rate"] < 0.25
    )
    print(f"\nPRE-REGISTERED VERDICT (full window): {'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
    print(f"5 trials recorded in {args.ledger} (total ledger trials: {ledger.count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
