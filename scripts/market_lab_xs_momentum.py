#!/usr/bin/env python3
"""Forward paper tracker for the cross-sectional momentum stream (Stream 2).

The backtest (trials 9-10) is an upper bound: its universe is survivor-
selected. This tracker is the survivorship-free evidence — it trades the
pre-registered configuration forward, on provenanced Alpaca SIP daily
bars only, in an isolated $25K paper portfolio:

- Universe: the 11 full-history single names from the backtest runner.
  Forward, the selection can only pick from what exists today; nothing
  about tomorrow's winners is baked in.
- Monthly rebalance (every 21 SPY bars) to dual_momentum_targets
  (12-1 formation, top-3, SPY absolute gate), vol-scaled to a 15%
  target by the tracker's own marked equity history.
- Decisions are made on the latest close and queued as candidates;
  they fill at the NEXT bar's open on a later run — never same-bar.

Execution note: fills are executed locally under the repo's standard
ExecutionModel (5bps slippage + $1 commission), NOT through the shared
mock broker. The broker's risk gates (5% max order, 10% max position)
are sized for the ensemble portfolio and would reject this sleeve's
designed ~33% orders; loosening shared gates for one stream is the
wrong move, so the tracker keeps its own accounting and writes the
same Portfolio state and OrderDecision ledger formats.

Usage:
    python scripts/market_lab_xs_momentum.py [--refresh]

--refresh first re-runs the full 10-year backfill for the universe +
SPY (the cache writer replaces whole files, so a partial refresh
would truncate history). Without it, the run refuses a cache whose
latest bar is more than 7 days old.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.backtest import ExecutionModel
from market_lab.broker import (
    OrderCandidate,
    OrderDecision,
    Portfolio,
    Position,
    append_ledger,
    load_order_candidates,
    load_portfolio,
    save_order_candidates,
    save_portfolio,
)
from market_lab.config import (
    XS_MOMENTUM_CANDIDATES,
    XS_MOMENTUM_EQUITY_HISTORY,
    XS_MOMENTUM_LEDGER,
    XS_MOMENTUM_REBALANCE_LOG,
    XS_MOMENTUM_REPORT_DIR,
    XS_MOMENTUM_STARTING_CASH,
    XS_MOMENTUM_STATE,
    ensure_dirs,
)
from market_lab.daily_data import backfill_daily, load_daily_bars
from market_lab.portfolio_construction import _vol_scale, dual_momentum_targets

# Pre-registered configuration — identical to the backtest (trials 9-10).
UNIVERSE = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "MSFT", "MSTR", "NVDA", "SMCI", "TSLA"]
FORMATION_DAYS = 252
SKIP_DAYS = 21
TOP_N = 3
REBALANCE_EVERY = 21
TARGET_VOL = 0.15
STRATEGY = "xs_momentum"
EXECUTION = ExecutionModel()  # 5bps slippage + $1 commission, the house model
MIN_TRADE_NOTIONAL = 100.0  # below this, whole-share rounding makes the order dust
MAX_CACHE_AGE_DAYS = 7


def _latest_bar_date(bars_by_symbol, spy_bars) -> date:
    return min([bars[-1].date for bars in bars_by_symbol.values()] + [spy_bars[-1].date])


def _equity_history(path: Path) -> list[tuple[str, float]]:
    """Read marked equity history, skipping corrupt rows.

    This file is operational state, not an audit ledger: a torn final
    write (crash mid-append) or a hand-edited line must not kill every
    later run. A skipped row only shortens the vol-scaling history.
    Non-finite values are corruption by definition here — the writer
    serializes with allow_nan=False — and a NaN would crash pstdev
    downstream, so those rows are skipped too.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            day, equity = row["date"], float(row["equity"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if not isinstance(day, str) or not math.isfinite(equity):
            continue
        rows.append((day, equity))
    return rows


def _append_equity_mark(path: Path, day: str, equity: float, history: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not math.isfinite(equity):
        raise ValueError(f"refusing to write a non-finite equity mark: {equity}")
    if history and history[-1][0] >= day:
        # Already marked for this bar — and never append a mark older
        # than the latest one (e.g. the cache was replaced by an older
        # snapshot): marks must stay strictly date-increasing or the
        # vol-scaling returns computed from them are meaningless.
        return history
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"date": day, "equity": equity}, sort_keys=True, allow_nan=False) + "\n")
    return history + [(day, equity)]


def _last_rebalance_date(rebalance_log_path: Path) -> str | None:
    """Rebalances are logged as events in their own right: a rebalance that
    produces zero orders (book already on target, or an empty selection
    against an empty book) still counts, or every later run would keep
    re-deciding it."""
    if not rebalance_log_path.exists():
        return None
    dates = []
    for line in rebalance_log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            day = json.loads(line)["date"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # torn/corrupt row: a lost rebalance record only
            # causes an early re-decision, which diffs against the
            # current book and is therefore safe.
        if isinstance(day, str):
            dates.append(day)
    return max(dates) if dates else None


def _append_rebalance_log(path: Path, day: str, targets, scale: float, queued: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": day,
        "targets": {t.symbol: round(t.target_weight * scale, 6) for t in targets},
        "vol_scale": scale,
        "candidates_queued": queued,
    }
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def rebalance_due(spy_bars, last_rebalance: str | None) -> bool:
    if last_rebalance is None:
        return True
    bars_since = sum(1 for bar in spy_bars if bar.date.isoformat() > last_rebalance)
    return bars_since >= REBALANCE_EVERY


def targets_to_candidates(targets, scale: float, portfolio: Portfolio,
                          prices: dict[str, float], signal_date: str) -> list[OrderCandidate]:
    """Diff scaled target weights against current holdings -> whole-share orders.

    Every held name absent from the targets is sold in full; every target
    is moved to its scaled weight. Dust deltas (< MIN_TRADE_NOTIONAL) are
    skipped so whole-share rounding cannot churn the book.
    """
    equity = portfolio.equity(prices)
    if equity <= 0:
        return []
    wanted = {t.symbol: t.target_weight * scale for t in targets}
    candidates = []
    for symbol in sorted(set(wanted) | set(portfolio.positions)):
        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue
        current_qty = portfolio.positions.get(symbol, Position(symbol)).quantity
        target_qty = int(equity * wanted.get(symbol, 0.0) / price)
        delta = target_qty - current_qty
        if delta == 0 or abs(delta) * price < MIN_TRADE_NOTIONAL:
            continue
        side = "BUY" if delta > 0 else "SELL"
        candidates.append(OrderCandidate(
            side=side,
            symbol=symbol,
            quantity=abs(delta),
            strategy=STRATEGY,
            confidence=1.0,
            reason=f"xs momentum rebalance {signal_date}: target weight {wanted.get(symbol, 0.0):.1%} (vol scale {scale:.2f})",
            signal_date=signal_date,
            reference_close=price,
        ))
    return candidates


def execute_candidate(candidate: OrderCandidate, next_open: float, prices: dict[str, float],
                      portfolio_path: Path, ledger_path: Path, execution_date: str) -> OrderDecision:
    """Fill one candidate at the given open under EXECUTION; same state and
    ledger formats as the mock broker, with this sleeve's own gates:
    no shorting, whole shares, cash-checked, finite-only."""
    now = datetime.now(timezone.utc).isoformat()
    portfolio = load_portfolio(portfolio_path)
    symbol = candidate.symbol.upper()
    # Finiteness gate FIRST (mirrors broker.evaluate_order): every
    # comparison below is False for NaN, so a NaN quantity/open would
    # sail through the positivity and cash gates, be "accepted", write
    # a NaN fill to the ledger, and only then crash the strict state
    # writer — leaving ledger and portfolio permanently inconsistent.
    try:
        finite = math.isfinite(candidate.quantity) and math.isfinite(next_open)
    except TypeError:
        finite = False
    fill = EXECUTION.fill_price(candidate.side, next_open) if finite else None
    decision = None
    if not finite:
        decision = OrderDecision(False, candidate.side, symbol, candidate.quantity, next_open, None,
                                 "quantity and open must be finite", now, STRATEGY, candidate.signal_date, execution_date)
    elif candidate.quantity <= 0 or int(candidate.quantity) != candidate.quantity or next_open <= 0:
        decision = OrderDecision(False, candidate.side, symbol, candidate.quantity, next_open, None,
                                 "quantity must be a positive whole number of shares and open must be positive", now, STRATEGY, candidate.signal_date, execution_date)
    elif candidate.side == "SELL":
        held = portfolio.positions.get(symbol, Position(symbol)).quantity
        if candidate.quantity > held:
            decision = OrderDecision(False, candidate.side, symbol, candidate.quantity, next_open, None,
                                     "sell rejected: insufficient shares (no shorting)", now, STRATEGY, candidate.signal_date, execution_date)
        else:
            portfolio.cash += candidate.quantity * fill - EXECUTION.commission_per_trade
            remaining = held - candidate.quantity
            if remaining:
                portfolio.positions[symbol].quantity = remaining
            else:
                portfolio.positions.pop(symbol, None)
            decision = OrderDecision(True, candidate.side, symbol, candidate.quantity, next_open, fill,
                                     "accepted by xs tracker execution", now, STRATEGY, candidate.signal_date, execution_date)
    else:  # BUY
        cost = candidate.quantity * fill + EXECUTION.commission_per_trade
        if cost > portfolio.cash:
            decision = OrderDecision(False, candidate.side, symbol, candidate.quantity, next_open, None,
                                     "insufficient cash", now, STRATEGY, candidate.signal_date, execution_date)
        else:
            pos = portfolio.positions.get(symbol, Position(symbol))
            new_qty = pos.quantity + candidate.quantity
            pos.avg_price = (pos.quantity * pos.avg_price + candidate.quantity * fill) / new_qty
            pos.quantity = new_qty
            portfolio.positions[symbol] = pos
            portfolio.cash -= cost
            decision = OrderDecision(True, candidate.side, symbol, candidate.quantity, next_open, fill,
                                     "accepted by xs tracker execution", now, STRATEGY, candidate.signal_date, execution_date)
    append_ledger(decision, ledger_path)
    if decision.accepted:
        save_portfolio(portfolio, portfolio_path)
    return decision


def run(state_path: Path = XS_MOMENTUM_STATE, ledger_path: Path = XS_MOMENTUM_LEDGER,
        candidates_path: Path = XS_MOMENTUM_CANDIDATES, equity_path: Path = XS_MOMENTUM_EQUITY_HISTORY,
        rebalance_log_path: Path = XS_MOMENTUM_REBALANCE_LOG,
        report_dir: Path = XS_MOMENTUM_REPORT_DIR, refresh: bool = False,
        today: date | None = None) -> str:
    ensure_dirs()
    today = today or date.today()
    if refresh:
        results = backfill_daily(UNIVERSE + ["SPY"], today - timedelta(days=365 * 10 + 30), today - timedelta(days=1))
        failed = [r.symbol for r in results if r.error or r.bars == 0]
        if failed:
            raise RuntimeError(f"backfill failed for {failed}; refusing to track on stale data")

    bars_by_symbol, spy_bars = {}, None
    for symbol in UNIVERSE:
        bars, _meta = load_daily_bars(symbol)
        bars_by_symbol[symbol] = bars
    spy_bars, _spy_meta = load_daily_bars("SPY")
    latest = _latest_bar_date(bars_by_symbol, spy_bars)
    if (today - latest).days > MAX_CACHE_AGE_DAYS:
        raise RuntimeError(
            f"daily cache is stale (latest bar {latest}, {((today - latest).days)} days old); "
            "rerun with --refresh before tracking on it"
        )

    prices = {symbol: bars[-1].close for symbol, bars in bars_by_symbol.items()}
    prices["SPY"] = spy_bars[-1].close
    signal_date = latest.isoformat()

    if not state_path.exists():
        save_portfolio(Portfolio(cash=XS_MOMENTUM_STARTING_CASH, positions={}), state_path)
    portfolio = load_portfolio(state_path)

    # 1) Fill candidates queued by earlier runs, at the latest bar's open.
    decisions = []
    remaining = []
    pending = load_order_candidates(path=candidates_path)
    filled_symbols: set[str] = set()
    seen_candidates: set[tuple] = set()
    for candidate in pending:
        # A candidate's identity is (symbol, side, signal_date, quantity):
        # one rebalance emits at most one candidate per symbol, so an
        # exact duplicate line is the SAME order (file corruption, a
        # double write). It must be dropped outright — deferring it
        # would fill the same order again on the next run.
        identity = (candidate.symbol, candidate.side, candidate.signal_date, candidate.quantity)
        if identity in seen_candidates:
            continue
        seen_candidates.add(identity)
        if candidate.signal_date < signal_date and candidate.symbol in bars_by_symbol \
                and candidate.symbol not in filled_symbols:
            decisions.append(execute_candidate(
                candidate, bars_by_symbol[candidate.symbol][-1].open, prices,
                state_path, ledger_path, signal_date,
            ))
            filled_symbols.add(candidate.symbol)
        else:
            remaining.append(candidate)
    save_order_candidates(remaining, path=candidates_path)
    portfolio = load_portfolio(state_path)

    # 2) Mark today's equity (once). A non-finite mark means poisoned
    # price data (e.g. a NaN close in the cache): refuse the run here,
    # before the mark or any rebalance decision consumes it. The fills
    # above are already ledgered and saved consistently.
    equity = portfolio.equity(prices)
    if not math.isfinite(equity):
        raise RuntimeError(
            f"computed equity is not finite ({equity}); refusing to mark or rebalance on poisoned prices"
        )
    history = _append_equity_mark(equity_path, signal_date, equity, _equity_history(equity_path))

    # 3) Rebalance when due: queue candidates to fill at the next bar.
    queued: list[OrderCandidate] = []
    scale = 1.0
    targets = []
    due = rebalance_due(spy_bars, _last_rebalance_date(rebalance_log_path))
    if due:
        targets = dual_momentum_targets(
            bars_by_symbol, formation_days=FORMATION_DAYS, skip_days=SKIP_DAYS,
            top_n=TOP_N, max_weight=1.0, spy_bars=spy_bars,
        )
        # Vol scaling consumes returns between marks; only positive
        # marks can form a return base (a zero/negative mark is a
        # corrupt or wiped-out state, and _vol_scale's pstdev also
        # raises on any non-finite value that slipped through).
        scale = _vol_scale([value for _day, value in history if value > 0], TARGET_VOL, 60)
        queued = targets_to_candidates(targets, scale, portfolio, prices, signal_date)
        _append_rebalance_log(rebalance_log_path, signal_date, targets, scale, len(queued))
        if queued:
            existing = load_order_candidates(path=candidates_path)
            merged = {(c.symbol, c.signal_date, c.side): c for c in existing + queued}
            save_order_candidates(list(merged.values()), path=candidates_path)

    lines = [
        f"# XS Momentum Paper Tracker — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Status: forward paper tracking (Stream 2) — pre-registered config, isolated $25K ledger.",
        f"Data through: {signal_date} (Alpaca SIP daily, provenanced cache)",
        "",
        f"- Equity: ${equity:,.2f} (cash ${portfolio.cash:,.2f})",
        f"- Positions: " + (", ".join(
            f"{sym} {pos.quantity} @ ${pos.avg_price:.2f}" for sym, pos in sorted(portfolio.positions.items())
        ) or "none"),
        f"- Fills this run: {sum(1 for d in decisions if d.accepted)} accepted, {sum(1 for d in decisions if not d.accepted)} rejected",
        f"- Rebalance due: {'yes' if due else 'no'}; vol scale {scale:.2f}; targets: "
        + (", ".join(f"{t.symbol} {t.target_weight * scale:.1%}" for t in targets) or "—"),
        f"- Candidates queued: {len(queued)} (fill at next bar's open)",
        "",
        "*Research/paper-only. No live broker orders. No investment advice.*",
        "",
    ]
    report = "\n".join(lines)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (report_dir / f"xs_momentum-{stamp}.md").write_text(report)
    (report_dir / "latest.md").write_text(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XS momentum forward paper tracking")
    parser.add_argument("--refresh", action="store_true", help="Re-run the full daily backfill first")
    args = parser.parse_args()
    try:
        report = run(refresh=args.refresh)
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
