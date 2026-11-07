from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, pstdev

from .backtest import ExecutionModel
from .data import Bar
from .indicators import max_drawdown


@dataclass(frozen=True)
class MomentumTarget:
    symbol: str
    rank: int
    relative_score: float
    absolute_momentum: float
    target_weight: float
    reason: str


@dataclass(frozen=True)
class RebalanceSnapshot:
    decision_index: int
    fill_index: int
    targets: list[MomentumTarget]
    fill_prices: dict[str, float]
    equity_before: float
    equity_after: float
    vol_scale: float = 1.0


@dataclass(frozen=True)
class DualMomentumBacktestResult:
    total_return: float
    benchmark_return: float
    max_drawdown: float
    sharpe: float
    final_equity: float
    rebalances: int
    snapshots: list[RebalanceSnapshot]
    strategy: str = "dual_momentum"
    benchmark_symbol: str = ""
    costs_paid: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _price_at_or_before(bars: list[Bar], idx: int) -> float:
    if not bars:
        return 0.0
    return bars[max(0, min(idx, len(bars) - 1))].close


def _momentum(bars: list[Bar], decision_index: int, formation_days: int, skip_days: int) -> float | None:
    end_idx = decision_index - skip_days
    return _return_between(bars, end_idx - formation_days, end_idx)


def _return_between(bars: list[Bar], start_idx: int, end_idx: int) -> float | None:
    if start_idx < 0 or end_idx >= len(bars):
        return None
    start = bars[start_idx].close
    end = bars[end_idx].close
    if start <= 0:
        return None
    return end / start - 1.0


def dual_momentum_targets(
    bars_by_symbol: dict[str, list[Bar]],
    formation_days: int = 252,
    skip_days: int = 21,
    top_n: int = 3,
    absolute_threshold: float = 0.0,
    max_weight: float = 0.20,
    decision_index: int | None = None,
    spy_bars: list[Bar] | None = None,
) -> list[MomentumTarget]:
    """Select relative winners that also pass an absolute momentum filter.

    This is an auditable dual-momentum primitive: rank by formation return, reject
    symbols below the absolute threshold, then assign equal target weights capped per symbol.
    Execution is handled separately by the backtest/mock broker at the next open.
    """
    if formation_days <= 0:
        raise ValueError("formation_days must be positive")
    if skip_days < 0:
        raise ValueError("skip_days must be non-negative")
    if top_n <= 0:
        return []

    spy_threshold = absolute_threshold
    if spy_bars:
        if decision_index is None:
            spy_idx = len(spy_bars) - 1
        elif decision_index < len(spy_bars):
            spy_idx = decision_index
        else:
            spy_idx = -1
        if spy_idx >= 0:
            spy_momentum = _return_between(spy_bars, spy_idx - formation_days, spy_idx)
            if spy_momentum is not None:
                spy_threshold = max(absolute_threshold, spy_momentum)

    scores: list[tuple[str, float, float]] = []
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        if decision_index is None:
            idx = len(bars) - 1
        elif decision_index >= len(bars):
            continue
        else:
            idx = decision_index
        relative_score = _momentum(bars, idx, formation_days, skip_days)
        absolute_momentum = _return_between(bars, idx - formation_days, idx)
        if relative_score is None or absolute_momentum is None:
            continue
        scores.append((symbol.upper(), relative_score, absolute_momentum))

    scores.sort(key=lambda item: item[1], reverse=True)
    selected = [(symbol, relative_score, absolute_momentum) for symbol, relative_score, absolute_momentum in scores if absolute_momentum > spy_threshold][:top_n]
    if not selected:
        return []

    equal_weight = min(max_weight, 1.0 / len(selected))
    return [
        MomentumTarget(
            symbol=symbol,
            rank=rank,
            relative_score=relative_score,
            absolute_momentum=absolute_momentum,
            target_weight=equal_weight,
            reason=f"dual momentum rank {rank}: relative {relative_score:.1%}; absolute {absolute_momentum:.1%} > {spy_threshold:.1%} benchmark/absolute filter",
        )
        for rank, (symbol, relative_score, absolute_momentum) in enumerate(selected, start=1)
    ]


def _portfolio_value(cash: float, positions: dict[str, float], bars_by_symbol: dict[str, list[Bar]], idx: int, use_open: bool = False) -> float:
    total = cash
    for symbol, qty in positions.items():
        bars = bars_by_symbol.get(symbol, [])
        if not bars:
            continue
        bar = bars[max(0, min(idx, len(bars) - 1))]
        total += qty * (bar.open if use_open else bar.close)
    return total


def _align_on_common_dates(bars_by_symbol: dict[str, list[Bar]]) -> dict[str, list[Bar]]:
    dated = {symbol.upper(): {bar.date: bar for bar in bars} for symbol, bars in bars_by_symbol.items() if bars}
    if not dated:
        return {}
    common_dates = set.intersection(*(set(by_date) for by_date in dated.values()))
    if not common_dates:
        return {}
    ordered_dates = sorted(common_dates)
    return {symbol: [by_date[d] for d in ordered_dates] for symbol, by_date in dated.items()}


def _stats(equity_curve: list[float]) -> tuple[float, float, float]:
    rets = [float(equity_curve[i]) / float(equity_curve[i - 1]) - 1 for i in range(1, len(equity_curve)) if float(equity_curve[i - 1]) > 0 and isfinite(float(equity_curve[i])) and isfinite(float(equity_curve[i - 1]))]
    ret_std = pstdev(rets) if len(rets) > 2 else 0.0
    sharpe = (mean(rets) / ret_std * sqrt(252)) if len(rets) > 2 and ret_std > 0 else 0.0
    mdd = max_drawdown(equity_curve)
    total_return = equity_curve[-1] / equity_curve[0] - 1 if equity_curve and equity_curve[0] > 0 else 0.0
    return total_return, mdd, sharpe


def _vol_scale(equity_curve: list[float], target_vol: float | None, vol_window: int) -> float:
    """Daniel-Moskowitz style vol scaling: shrink exposure toward a constant
    volatility target using the strategy's own trailing realized vol.

    Returns 1.0 (no scaling) until at least 20 daily returns exist — early
    rebalances have no measurable vol, and inventing one would be worse.
    Long-only: the scale never exceeds 1.0 (no leverage).
    """
    if target_vol is None:
        return 1.0
    rets = [
        equity_curve[i] / equity_curve[i - 1] - 1.0
        for i in range(max(1, len(equity_curve) - vol_window), len(equity_curve))
        if equity_curve[i - 1] > 0
    ]
    if len(rets) < 20:
        return 1.0
    sd = pstdev(rets)
    if sd <= 0:
        return 1.0
    realized = sd * sqrt(252)
    return _clamp(target_vol / realized, 0.0, 1.0)


def run_dual_momentum_backtest(
    bars_by_symbol: dict[str, list[Bar]],
    formation_days: int = 252,
    skip_days: int = 21,
    top_n: int = 3,
    rebalance_every: int = 21,
    start_index: int | None = None,
    initial_cash: float = 10_000.0,
    max_weight: float = 0.20,
    absolute_threshold: float = 0.0,
    spy_bars: list[Bar] | None = None,
    execution: ExecutionModel | None = None,
    target_vol: float | None = None,
    vol_window: int = 60,
    benchmark_symbol: str = "SPY",
) -> DualMomentumBacktestResult:
    """Dual-momentum backtest with honest measurement defaults.

    - Costs are ON unless explicitly disabled: `execution=None` selects the
      repo's standard ExecutionModel (5bps slippage + $1/trade); pass
      ExecutionModel(0, 0) for a frictionless run, as a deliberate choice.
    - The benchmark is SPY: pass `spy_bars` (also used for the absolute
      momentum gate), or include the benchmark symbol in `bars_by_symbol`.
      It is never silently "the alphabetically first symbol" anymore.
    - `target_vol` enables vol scaling of target weights (see _vol_scale);
      the cross-sectional momentum literature treats this as mandatory
      risk management, and stream runners always set it.
    """
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if target_vol is not None and target_vol <= 0:
        raise ValueError("target_vol must be positive when set")
    if vol_window < 20:
        raise ValueError("vol_window must be at least 20 (the minimum returns for a vol estimate)")
    execution = execution or ExecutionModel()
    if not bars_by_symbol:
        return DualMomentumBacktestResult(0.0, 0.0, 0.0, 0.0, initial_cash, 0, [])

    normalized_bars_by_symbol = _align_on_common_dates(bars_by_symbol)
    if not normalized_bars_by_symbol:
        return DualMomentumBacktestResult(0.0, 0.0, 0.0, 0.0, initial_cash, 0, [])

    min_len = min(len(bars) for bars in normalized_bars_by_symbol.values())
    if min_len < formation_days + skip_days + 2:
        return DualMomentumBacktestResult(0.0, 0.0, 0.0, 0.0, initial_cash, 0, [])
    start = start_index if start_index is not None else formation_days + skip_days
    start = max(start, formation_days + skip_days)
    end = min_len - 1
    if start >= end:
        return DualMomentumBacktestResult(0.0, 0.0, 0.0, 0.0, initial_cash, 0, [])

    # Align SPY onto the universe's common calendar, index-for-index, so the
    # momentum gate and the benchmark read the same dates as the universe.
    aligned_spy: list[Bar] | None = None
    if spy_bars:
        reference = next(iter(normalized_bars_by_symbol.values()))
        spy_by_date = {bar.date: bar for bar in spy_bars}
        if any(bar.date not in spy_by_date for bar in reference):
            raise ValueError("spy_bars do not cover the universe calendar; refusing to misalign the benchmark/gate")
        aligned_spy = [spy_by_date[bar.date] for bar in reference]

    cash = initial_cash
    positions: dict[str, float] = {}
    equity_curve: list[float] = [initial_cash]
    snapshots: list[RebalanceSnapshot] = []
    costs_paid = 0.0

    for idx in range(start, end):
        if (idx - start) % rebalance_every == 0 and idx + 1 < min_len:
            equity_before = _portfolio_value(cash, positions, normalized_bars_by_symbol, idx)
            targets = dual_momentum_targets(
                normalized_bars_by_symbol,
                formation_days=formation_days,
                skip_days=skip_days,
                top_n=top_n,
                absolute_threshold=absolute_threshold,
                max_weight=max_weight,
                decision_index=idx,
                spy_bars=aligned_spy,
            )
            fill_idx = idx + 1
            scale = _vol_scale(equity_curve, target_vol, vol_window)
            equity_at_fill = _portfolio_value(cash, positions, normalized_bars_by_symbol, fill_idx, use_open=True)
            desired: dict[str, float] = {}
            fill_prices: dict[str, float] = {}
            for target in targets:
                open_px = normalized_bars_by_symbol[target.symbol][fill_idx].open
                if open_px <= 0:
                    continue
                buy_px = execution.fill_price("BUY", open_px)
                weight = _clamp(target.target_weight, 0.0, max_weight) * scale
                if weight <= 0:
                    continue
                # Clamp at zero: a negative equity mark (garbage opens) must
                # never turn a target weight into a short sale.
                desired[target.symbol] = max(0.0, equity_at_fill * weight / buy_px)
                fill_prices[target.symbol] = buy_px
            sells: dict[str, float] = {}
            buys: dict[str, float] = {}
            for symbol in set(positions) | set(desired):
                current = positions.get(symbol, 0.0)
                want = desired.get(symbol, 0.0)
                if want < current:
                    sells[symbol] = current - want
                elif want > current:
                    buys[symbol] = want - current
            new_positions = dict(positions)
            proceeds = 0.0
            for symbol, qty in sells.items():
                open_px = normalized_bars_by_symbol[symbol][fill_idx].open
                if open_px <= 0:
                    # Cannot execute at a non-positive price; carry the
                    # position (mirrors the buy-side guard on desired qtys).
                    continue
                sell_px = execution.fill_price("SELL", open_px)
                proceeds += qty * sell_px - execution.commission_per_trade
                costs_paid += qty * (open_px - sell_px) + execution.commission_per_trade
                new_positions[symbol] = positions[symbol] - qty
            budget = cash + proceeds
            # Scale quantities against the notional alone, reserving the
            # per-trade commissions from the budget first. Scaling against
            # notional+commissions but then charging commissions in full
            # overspends the budget by (1 - scale) x commissions on every
            # scaled rebalance — cash silently goes negative.
            total_notional = sum(qty * fill_prices[symbol] for symbol, qty in buys.items())
            commissions = execution.commission_per_trade * len(buys)
            buy_scale = 1.0
            if total_notional + commissions > budget:
                room = budget - commissions
                buy_scale = max(0.0, room) / total_notional if total_notional > 0 and room > 0 else 0.0
            spent = 0.0
            if buy_scale > 0:
                for symbol, qty in buys.items():
                    scaled_qty = qty * buy_scale
                    open_px = normalized_bars_by_symbol[symbol][fill_idx].open
                    spent += scaled_qty * fill_prices[symbol] + execution.commission_per_trade
                    costs_paid += scaled_qty * (fill_prices[symbol] - open_px) + execution.commission_per_trade
                    new_positions[symbol] = new_positions.get(symbol, 0.0) + scaled_qty
            cash = budget - spent
            positions = {symbol: qty for symbol, qty in new_positions.items() if qty > 1e-12}
            snapshots.append(RebalanceSnapshot(idx, fill_idx, targets, fill_prices, equity_before, _portfolio_value(cash, positions, normalized_bars_by_symbol, fill_idx), scale))
        equity_curve.append(_portfolio_value(cash, positions, normalized_bars_by_symbol, idx + 1))

    final_equity = equity_curve[-1]
    total_return, mdd, sharpe = _stats(equity_curve)
    bench_name = ""
    benchmark_return = 0.0
    if aligned_spy is not None:
        bench_name = benchmark_symbol
        b_start = aligned_spy[start].close
        b_end = aligned_spy[end].close
        benchmark_return = b_end / b_start - 1 if b_start > 0 else 0.0
    elif benchmark_symbol in normalized_bars_by_symbol:
        bench_name = benchmark_symbol
        bench_bars = normalized_bars_by_symbol[benchmark_symbol]
        benchmark_start = _price_at_or_before(bench_bars, start)
        benchmark_end = _price_at_or_before(bench_bars, end)
        benchmark_return = benchmark_end / benchmark_start - 1 if benchmark_start > 0 else 0.0
    return DualMomentumBacktestResult(total_return, benchmark_return, mdd, sharpe, final_equity, len(snapshots), snapshots, "dual_momentum", bench_name, costs_paid)
