#!/usr/bin/env python3
"""Forward paper tracker for the vol-targeted trend stream (Stream 1, vt_trend).

The legacy scripts/market_lab_vt_trend.py is a 30-day mock-tracking
runner on the legacy data path (yfinance/cache/synthetic) behind the
mock broker's ensemble risk gates. This tracker is the Stream 1 sibling
of the XS momentum forward tracker (scripts/market_lab_xs_momentum.py):
the same evidence discipline, applied to vt_trend —

- Data: provenanced Alpaca SIP daily bars only, via
  market_lab.daily_data. No synthetic fallback, ever. Without
  --refresh, a cache whose latest bar is more than 7 days old is
  refused.
- Signals: generate_vt_trend_signal per symbol on the latest close,
  exactly as backtested (trials 1-4). Decisions are queued as
  candidates and fill at the NEXT bar's open on a later run — never
  same-bar.
- Execution: local fills under the repo's standard ExecutionModel
  (5bps slippage + $1 commission), in an isolated $25K paper
  portfolio with its own state/ledger under data/market-lab/
  vt_trend_forward/. The mock broker's gates are sized for the
  ensemble portfolio and would reject this sleeve's designed orders;
  like the XS tracker, this tracker keeps its own accounting in the
  same Portfolio / OrderDecision formats.

Book construction (interpretation, recorded here): the backtest trials
ran SPY and QQQ as separate $25K books. A single forward book cannot
give both symbols 100% of the same $25K, so the book is split into
two equal sleeves: each symbol's effective target weight is its
signal target_weight x 1/2. The vol-targeting shape within each
sleeve is unchanged; total gross exposure caps at 100% of the book.

A signal is evaluated once per bar date: the signal log is the
cadence record (the role the rebalance log plays for XS), so a
same-day re-run neither re-decides nor double-queues.

Usage:
    python scripts/market_lab_vt_trend_forward.py [--refresh]
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
    VT_TREND_FORWARD_CANDIDATES,
    VT_TREND_FORWARD_EQUITY_HISTORY,
    VT_TREND_FORWARD_LEDGER,
    VT_TREND_FORWARD_REPORT_DIR,
    VT_TREND_FORWARD_SIGNAL_LOG,
    VT_TREND_FORWARD_STARTING_CASH,
    VT_TREND_FORWARD_STATE,
    ensure_dirs,
)
from market_lab.daily_data import backfill_daily, load_daily_bars
from market_lab.signals import generate_vt_trend_signal

SYMBOLS = ["SPY", "QQQ"]
SLEEVE_WEIGHT = 1.0 / len(SYMBOLS)  # equal capital split across sleeves
STRATEGY = "vt_trend"
EXECUTION = ExecutionModel()  # 5bps slippage + $1 commission, the house model
MIN_TRADE_NOTIONAL = 100.0  # below this, whole-share rounding makes the order dust
MAX_CACHE_AGE_DAYS = 7


def _equity_history(path: Path) -> list[tuple[str, float]]:
    """Read marked equity history, skipping corrupt rows.

    Operational state, not an audit ledger: a torn final write or a
    hand-edited line must not kill every later run. Non-finite values
    are corruption by definition here — the writer serializes with
    allow_nan=False — so those rows are skipped too.
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
        # than the latest one: marks must stay strictly date-increasing.
        return history
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"date": day, "equity": equity}, sort_keys=True, allow_nan=False) + "\n")
    return history + [(day, equity)]


def _last_signal_date(signal_log_path: Path) -> str | None:
    """Signal evaluations are logged as events in their own right: an
    evaluation that queues zero orders (book already on target) still
    counts, or every later run on the same bar would keep re-deciding
    it. Corrupt rows are skipped — a lost record only causes a
    re-decision, which diffs against the current book and is safe."""
    if not signal_log_path.exists():
        return None
    dates = []
    for line in signal_log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            day = json.loads(line)["date"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(day, str):
            dates.append(day)
    return max(dates) if dates else None


def _append_signal_log(path: Path, day: str, signals: dict, targets: dict[str, float], queued: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": day,
        "targets": {sym: round(targets[sym], 6) for sym in sorted(targets)},
        "signal_target_weights": {
            sym: round(signals[sym].target_weight, 6) for sym in sorted(signals)
        },
        "actions": {sym: signals[sym].action for sym in sorted(signals)},
        "candidates_queued": queued,
    }
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def signal_targets(bars_by_symbol: dict) -> tuple[dict, dict[str, float]]:
    """Evaluate vt_trend per symbol; effective target = signal weight
    x sleeve share of the single $25K book."""
    signals, targets = {}, {}
    for symbol in SYMBOLS:
        sig = generate_vt_trend_signal(symbol, bars_by_symbol[symbol])
        signals[symbol] = sig
        targets[symbol] = sig.target_weight * SLEEVE_WEIGHT
    return signals, targets


def targets_to_candidates(targets: dict[str, float], portfolio: Portfolio,
                          prices: dict[str, float], signal_date: str,
                          signals: dict | None = None) -> list[OrderCandidate]:
    """Diff effective target weights against current holdings -> whole-share
    orders. Every held name absent from the targets is sold in full.
    Dust deltas (< MIN_TRADE_NOTIONAL) are skipped so whole-share
    rounding cannot churn the book."""
    equity = portfolio.equity(prices)
    if equity <= 0:
        return []
    candidates = []
    for symbol in sorted(set(targets) | set(portfolio.positions)):
        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue
        current_qty = portfolio.positions.get(symbol, Position(symbol)).quantity
        target_qty = int(equity * targets.get(symbol, 0.0) / price)
        delta = target_qty - current_qty
        if delta == 0 or abs(delta) * price < MIN_TRADE_NOTIONAL:
            continue
        side = "BUY" if delta > 0 else "SELL"
        sig = (signals or {}).get(symbol)
        reason = (
            f"vt_trend forward {signal_date}: target weight {targets.get(symbol, 0.0):.1%} "
            f"of book (signal {sig.action} {sig.target_weight:.2f} x {SLEEVE_WEIGHT:.2f} sleeve)"
            if sig is not None else
            f"vt_trend forward {signal_date}: target weight {targets.get(symbol, 0.0):.1%}"
        )
        candidates.append(OrderCandidate(
            side=side,
            symbol=symbol,
            quantity=abs(delta),
            strategy=STRATEGY,
            confidence=float(sig.confidence) if sig is not None else 1.0,
            reason=reason,
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
    # Finiteness gate FIRST (mirrors the XS tracker): every comparison
    # below is False for NaN, so a NaN quantity/open would sail through
    # the positivity and cash gates, be "accepted", write a NaN fill to
    # the ledger, and only then crash the strict state writer.
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
                                     "accepted by vt_trend forward tracker execution", now, STRATEGY, candidate.signal_date, execution_date)
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
                                     "accepted by vt_trend forward tracker execution", now, STRATEGY, candidate.signal_date, execution_date)
    append_ledger(decision, ledger_path)
    if decision.accepted:
        save_portfolio(portfolio, portfolio_path)
    return decision


def run(state_path: Path = VT_TREND_FORWARD_STATE, ledger_path: Path = VT_TREND_FORWARD_LEDGER,
        candidates_path: Path = VT_TREND_FORWARD_CANDIDATES,
        equity_path: Path = VT_TREND_FORWARD_EQUITY_HISTORY,
        signal_log_path: Path = VT_TREND_FORWARD_SIGNAL_LOG,
        report_dir: Path = VT_TREND_FORWARD_REPORT_DIR, refresh: bool = False,
        today: date | None = None) -> str:
    ensure_dirs()
    today = today or date.today()
    if refresh:
        results = backfill_daily(SYMBOLS, today - timedelta(days=365 * 10 + 30), today - timedelta(days=1))
        failed = [r.symbol for r in results if r.error or r.bars == 0]
        if failed:
            raise RuntimeError(f"backfill failed for {failed}; refusing to track on stale data")

    bars_by_symbol = {}
    for symbol in SYMBOLS:
        bars, _meta = load_daily_bars(symbol)
        bars_by_symbol[symbol] = bars
    latest = min(bars[-1].date for bars in bars_by_symbol.values())
    if (today - latest).days > MAX_CACHE_AGE_DAYS:
        raise RuntimeError(
            f"daily cache is stale (latest bar {latest}, {(today - latest).days} days old); "
            "rerun with --refresh before tracking on it"
        )

    prices = {symbol: bars[-1].close for symbol, bars in bars_by_symbol.items()}
    signal_date = latest.isoformat()

    if not state_path.exists():
        save_portfolio(Portfolio(cash=VT_TREND_FORWARD_STARTING_CASH, positions={}), state_path)
    portfolio = load_portfolio(state_path)

    # 1) Fill candidates queued by earlier runs, at the latest bar's open.
    decisions = []
    remaining = []
    pending = load_order_candidates(path=candidates_path)
    filled_symbols: set[str] = set()
    seen_candidates: set[tuple] = set()
    for candidate in pending:
        # A candidate's identity is (symbol, side, signal_date, quantity):
        # one evaluation emits at most one candidate per symbol, so an
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
    # price data: refuse the run here, before the mark or any signal
    # decision consumes it. The fills above are already ledgered and
    # saved consistently.
    equity = portfolio.equity(prices)
    if not math.isfinite(equity):
        raise RuntimeError(
            f"computed equity is not finite ({equity}); refusing to mark or decide on poisoned prices"
        )
    _append_equity_mark(equity_path, signal_date, equity, _equity_history(equity_path))

    # 3) Evaluate the signal once per bar date: queue candidates to
    # fill at the next bar. The signal log is the cadence record — a
    # bar date already logged was already decided, so a same-day
    # re-run stops here (idempotent).
    queued: list[OrderCandidate] = []
    signals: dict = {}
    targets: dict[str, float] = {}
    already_decided = _last_signal_date(signal_log_path) == signal_date
    if not already_decided:
        signals, targets = signal_targets(bars_by_symbol)
        queued = targets_to_candidates(targets, portfolio, prices, signal_date, signals)
        _append_signal_log(signal_log_path, signal_date, signals, targets, len(queued))
        if queued:
            existing = load_order_candidates(path=candidates_path)
            merged = {(c.symbol, c.signal_date, c.side): c for c in existing + queued}
            save_order_candidates(list(merged.values()), path=candidates_path)

    outcome = {
        "signal_date": signal_date,
        "equity": equity,
        "cash": portfolio.cash,
        "positions": {
            sym: pos.quantity for sym, pos in sorted(portfolio.positions.items())
        },
        "fills_accepted": sum(1 for d in decisions if d.accepted),
        "fills_rejected": sum(1 for d in decisions if not d.accepted),
        "fills": [
            {
                "symbol": d.symbol, "side": d.side, "quantity": d.quantity,
                "fill_price": d.fill_price, "accepted": d.accepted, "reason": d.reason,
            }
            for d in decisions
        ],
        "candidates_queued": len(queued),
        "queued": [
            {"symbol": c.symbol, "side": c.side, "quantity": c.quantity} for c in queued
        ],
        "targets": {sym: round(targets[sym], 6) for sym in sorted(targets)} if targets else {},
    }

    lines = [
        f"# vt_trend Forward Paper Tracker — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Status: forward paper tracking (Stream 1) — backtested vt_trend config, isolated $25K ledger, SPY/QQQ equal sleeves.",
        f"Data through: {signal_date} (Alpaca SIP daily, provenanced cache)",
        "",
        f"- Equity: ${equity:,.2f} (cash ${portfolio.cash:,.2f})",
        "- Positions: " + (", ".join(
            f"{sym} {pos.quantity} @ ${pos.avg_price:.2f}" for sym, pos in sorted(portfolio.positions.items())
        ) or "none"),
        f"- Fills this run: {outcome['fills_accepted']} accepted, {outcome['fills_rejected']} rejected",
        "- Targets: " + (
            ", ".join(
                f"{sym} {targets[sym]:.1%} (signal {signals[sym].action} {signals[sym].target_weight:.2f})"
                for sym in SYMBOLS
            ) if targets else f"already decided for {signal_date}"
        ),
        f"- Candidates queued: {len(queued)} (fill at next bar's open)",
        "",
        "*Research/paper-only. No live broker orders. No investment advice.*",
        "",
    ]
    report = "\n".join(lines)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (report_dir / f"vt_trend_forward-{stamp}.md").write_text(report)
    (report_dir / "latest.md").write_text(report)
    (report_dir / "last_run.json").write_text(
        json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vt_trend forward paper tracking")
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
