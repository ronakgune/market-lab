"""Time Machine forecast: time-of-day bootstrap Monte Carlo for a plan.

Adapted from trader-timemachine in day-trader-skill (MIT,
Jakeschincariol/day-trader-skill) and re-engineered for Market Lab:

  * No network, no Yahoo. Input is ``SessionBar`` sequences built from our
    own Alpaca SIP minute bars in the Parquet store (see session.py).
  * Pure functions with a seeded RNG: same plan + same data + same seed
    gives the same result on any machine.
  * Improvements over the source: an explicit ``cost_bps`` round-trip cost
    deducted from R (the source counted a gap through the stop as -1R and
    charged nothing), strict plan validation, and the point-in-time pool
    guard lives in session.full_sessions(before=trade_date).

What it is: a reality check, not a prediction. Each replay rebuilds the
rest of the session bar-by-bar, drawing a real historical bar from the
same time-of-day slot (+/- ``window`` slots) of a past full session,
scaled to the current price. It also re-runs the opening-range rule on
every past session (history replay) so simulated odds sit next to what
the rule actually did.

Known limits (inherited, stated plainly): bars are drawn independently
per slot, so volatility clustering across bars is weakened; the pool is
only as representative as the lookback window; news/earnings regimes are
not in the bars. Treat the output as odds under "days like the pool",
never as a promise.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from .session import SessionBar, full_sessions, group_sessions, slots_per_session

MIN_SESSIONS_DEFAULT = 20


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    trade_date: date
    entry: float
    stop: float
    target: float
    entry_slot: int  # slot of the entry bar; replays start at entry_slot+1
    side: str = "long"  # "long" | "short"
    bar_minutes: int = 5

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        for name in ("entry", "stop", "target"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a positive finite price")
        if self.side == "long" and not (self.stop < self.entry < self.target):
            raise ValueError("long plan requires stop < entry < target")
        if self.side == "short" and not (self.target < self.entry < self.stop):
            raise ValueError("short plan requires target < entry < stop")
        n = slots_per_session(self.bar_minutes)
        if not (0 <= self.entry_slot < n):
            raise ValueError(f"entry_slot must be in [0, {n})")

    @property
    def risk(self) -> float:
        return (self.entry - self.stop) if self.side == "long" else (self.stop - self.entry)

    @property
    def target_r(self) -> float:
        gain = (self.target - self.entry) if self.side == "long" else (self.entry - self.target)
        return gain / self.risk


@dataclass(frozen=True)
class ForecastResult:
    replays: int
    seed: int
    window: int
    pool_sessions: int
    bars_left: int
    p_target: float
    p_stop: float
    p_neither: float
    p_green: float
    median_close: float
    p5_close: float
    p95_close: float
    avg_r: float  # net of cost_bps
    cost_bps: float
    verdict: str


@dataclass(frozen=True)
class HistoryTrade:
    day: date
    entry_slot: int
    entry: float
    stop: float
    target: float
    outcome: str  # "target" | "stop" | "close"
    r: float


@dataclass(frozen=True)
class HistoryResult:
    sessions: int
    triggered: int
    target: int
    stop: int
    close: int
    avg_r: float | None
    trades: tuple[HistoryTrade, ...]


def build_pool(
    sessions: Mapping[date, Sequence[SessionBar]], bar_minutes: int = 5
) -> list[list[tuple[float, float, float, float]]]:
    """pool[slot] = every session's bar at that slot as ratios to the
    previous bar's close: (open, high, low, close) / prev_close."""
    n = slots_per_session(bar_minutes)
    pool: list[list[tuple[float, float, float, float]]] = [[] for _ in range(n)]
    for day in sorted(sessions):
        bars = sorted(sessions[day], key=lambda b: b.slot)
        for i in range(1, n):
            prev_close = bars[i - 1].close
            b = bars[i]
            pool[i].append((b.open / prev_close, b.high / prev_close,
                            b.low / prev_close, b.close / prev_close))
    return pool


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def verdict_for(p_target: float, p_stop: float, target_r: float) -> str:
    """One honest line, weighing target share against break-even share.

    Break-even share of decisive replays is 1/(1+R): 50% at 1R, 33% at 2R.
    """
    decisive = p_target + p_stop
    if decisive <= 0:
        return "No replay hit the target or the stop; the exit-at-close price decides this plan."
    share = p_target / decisive
    breakeven = 1.0 / (1.0 + target_r)
    gap = share - breakeven
    if abs(gap) < 0.05:
        return "Close to a coin flip" if abs(target_r - 1.0) < 1e-9 else "Close to break-even"
    direction = "target" if gap > 0 else "stop"
    strength = "Strongly toward" if abs(gap) >= 0.15 else "Leans toward"
    return f"{strength} the {direction}"


def simulate_forecast(
    plan: TradePlan,
    sessions: Mapping[date, Sequence[SessionBar]],
    *,
    replays: int = 10_000,
    seed: int = 19,
    window: int = 2,
    min_sessions: int = MIN_SESSIONS_DEFAULT,
    cost_bps: float = 0.0,
    until_slot: int | None = None,
) -> ForecastResult:
    """Run the bootstrap Monte Carlo for ``plan`` over past full sessions.

    ``sessions`` must already be point-in-time safe (use
    session.full_sessions(before=plan.trade_date)); this function refuses
    any session on/after the trade date as a second line of defence.
    """
    if replays <= 0:
        raise ValueError("replays must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")
    pool_src = full_sessions(sessions, before=plan.trade_date,
                             bar_minutes=plan.bar_minutes)
    if len(pool_src) < min_sessions:
        raise ValueError(
            f"forecast needs at least {min_sessions} full sessions before "
            f"{plan.trade_date}, got {len(pool_src)}")
    n = slots_per_session(plan.bar_minutes)
    last = (n - 1) if until_slot is None else until_slot
    if not (plan.entry_slot < last < n):
        raise ValueError("until_slot must be after entry_slot and inside the session")
    pool = build_pool(pool_src, bar_minutes=plan.bar_minutes)
    for i in range(plan.entry_slot + 1, last + 1):
        if not pool[i]:
            raise ValueError(f"empty pool at slot {i}")

    long = plan.side == "long"
    risk = plan.risk
    cost_r = (plan.entry * (cost_bps / 10_000.0)) / risk if cost_bps else 0.0
    rng = random.Random(seed)

    def slot_pick(i: int) -> int:
        if window == 0:
            return i
        return min(n - 1, max(1, i + rng.randint(-window, window)))

    counts = {"target": 0, "stop": 0, "close": 0}
    closes: list[float] = []
    rs: list[float] = []
    green = 0
    for _ in range(replays):
        price = plan.entry
        outcome = "close"
        for i in range(plan.entry_slot + 1, last + 1):
            _o, h, l, c = rng.choice(pool[slot_pick(i)])
            hi, lo = price * h, price * l
            if outcome == "close":
                # Stop checked first: a bar touching both counts as the
                # stop (conservative, same rule as the source).
                if (lo <= plan.stop) if long else (hi >= plan.stop):
                    outcome = "stop"
                elif (hi >= plan.target) if long else (lo <= plan.target):
                    outcome = "target"
            price = price * c
        counts[outcome] += 1
        closes.append(price)
        if outcome == "target":
            r = plan.target_r
            green += 1
        elif outcome == "stop":
            r = -1.0
        else:
            r = ((price - plan.entry) if long else (plan.entry - price)) / risk
            if (price > plan.entry) if long else (price < plan.entry):
                green += 1
        rs.append(r - cost_r)
    srt = sorted(closes)
    return ForecastResult(
        replays=replays, seed=seed, window=window,
        pool_sessions=len(pool_src), bars_left=last - plan.entry_slot,
        p_target=counts["target"] / replays,
        p_stop=counts["stop"] / replays,
        p_neither=counts["close"] / replays,
        p_green=green / replays,
        median_close=_pct(srt, 0.5), p5_close=_pct(srt, 0.05),
        p95_close=_pct(srt, 0.95),
        avg_r=round(statistics.mean(rs), 4),
        cost_bps=cost_bps,
        verdict=verdict_for(counts["target"] / replays,
                            counts["stop"] / replays, plan.target_r),
    )


def _cents_away(x: float, long: bool) -> float:
    c = round(x * 100, 4)
    return (math.floor(c) if long else math.ceil(c)) / 100.0


def stop_for_rule(rule: str, or_high: float, or_low: float,
                  entry: float, side: str, stop_pct: float = 1.0) -> float:
    long = side == "long"
    if rule == "or_mid":
        return _cents_away((or_high + or_low) / 2, long)
    if rule == "or_extreme":
        return _cents_away(or_low if long else or_high, long)
    if rule == "pct":
        frac = stop_pct / 100.0
        return _cents_away(entry * (1 - frac) if long else entry * (1 + frac), long)
    raise ValueError(f"unknown stop rule: {rule!r}")


def history_replay(
    sessions: Mapping[date, Sequence[SessionBar]],
    *,
    or_bars: int = 3,
    stop_rule: str = "or_mid",
    stop_pct: float = 1.0,
    target_r: float = 1.0,
    side: str = "long",
    bar_minutes: int = 5,
) -> HistoryResult:
    """Re-run the opening-range breakout rule on every full session.

    Entry = first bar after the opening range to CLOSE beyond it; then
    stop-first bar-by-bar to the close. This is what the rule actually
    did historically — the non-simulated half of the Time Machine.
    """
    if or_bars <= 0:
        raise ValueError("or_bars must be positive")
    long = side == "long"
    n = slots_per_session(bar_minutes)
    pool = full_sessions(sessions, bar_minutes=bar_minutes)
    trades: list[HistoryTrade] = []
    for day in sorted(pool):
        bars = pool[day]
        or_high = max(b.high for b in bars[:or_bars])
        or_low = min(b.low for b in bars[:or_bars])
        entry_slot = next(
            (b.slot for b in bars[or_bars:]
             if (b.close > or_high if long else b.close < or_low)), None)
        if entry_slot is None:
            continue
        entry = round(bars[entry_slot].close, 2)
        stop = stop_for_rule(stop_rule, or_high, or_low, entry, side, stop_pct)
        risk = (entry - stop) if long else (stop - entry)
        if risk <= 0:
            continue
        target = round(entry + target_r * risk, 2) if long else round(entry - target_r * risk, 2)
        outcome = "close"
        for b in bars[entry_slot + 1:]:
            if (b.low <= stop) if long else (b.high >= stop):
                outcome = "stop"
                break
            if (b.high >= target) if long else (b.low <= target):
                outcome = "target"
                break
        close = bars[-1].close
        if outcome == "target":
            r = target_r
        elif outcome == "stop":
            r = -1.0
        else:
            r = ((close - entry) if long else (entry - close)) / risk
        trades.append(HistoryTrade(day, entry_slot, entry, stop, target,
                                   outcome, round(r, 4)))
    return HistoryResult(
        sessions=len(pool), triggered=len(trades),
        target=sum(1 for t in trades if t.outcome == "target"),
        stop=sum(1 for t in trades if t.outcome == "stop"),
        close=sum(1 for t in trades if t.outcome == "close"),
        avg_r=round(statistics.mean(t.r for t in trades), 4) if trades else None,
        trades=tuple(trades),
    )


def forecast_from_session_bars(
    plan: TradePlan,
    session_bars: Sequence[SessionBar],
    **kwargs,
) -> ForecastResult:
    """Convenience: group flat SessionBar list, then simulate."""
    return simulate_forecast(plan, group_sessions(session_bars), **kwargs)
