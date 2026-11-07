"""Morning Catalyst falsification study (Stream 3).

The promoted claim (Half Percent Club / ORB-style morning trading):
after a large overnight gap, trade the continuation in the gap
direction, scalp +0.5% against a -0.5% stop, and compound. The deep
research (Nov 6, 2026) found the claim does not replicate as a capital
strategy; this study is the pre-registered falsification sprint on OUR
data, with every simplification chosen to FAVOR the strategy, so a
kill is decisive:

- Events: |gap| >= 5% at the open (prior close -> today open) on liquid
  single names — known at the open, point-in-time clean.
- Entry at the first minute bar's open; target +0.5%, stop -0.5%.
- When one minute bar spans BOTH target and stop, the stop wins
  (the single conservative choice; everything else favors the claim).
- A bar opening beyond the stop exits at that open (real gap-through);
  symmetrically, a bar opening beyond the target fills at that open
  (a resting limit order gets the open print, before any intrabar dip).
- No exit by the deadline (12:00 ET) exits at that bar's close.
- Shorts pay the same costs as longs (borrow ignored — favors shorts).
- Costs: 5bps slippage per fill + $1 per fill on a $5,000 position,
  converted to R (1R = the 0.5% stop distance): ~0.28R per trade.

Kill rule (pre-registered): the strategy survives only if the
post-2023 holdout shows net Sharpe >= 1.0 AND profit factor >= 1.3
AND no single loss worse than -5% of notional. Otherwise the stream
closes permanently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

from .data import Bar
from .intraday.bars import MinuteBar

ET = ZoneInfo("America/New_York")
TARGET_FRAC = 0.005
STOP_FRAC = 0.005
DEADLINE_MINUTES = 150  # 9:30 -> 12:00 ET
POSITION_NOTIONAL = 5_000.0
SLIPPAGE_BPS = 5.0
COMMISSION_PER_FILL = 1.0

COST_FRAC = 2 * (SLIPPAGE_BPS / 10_000) + 2 * COMMISSION_PER_FILL / POSITION_NOTIONAL
COST_R = COST_FRAC / STOP_FRAC


@dataclass(frozen=True)
class GapEvent:
    symbol: str
    day: date
    direction: int  # +1 gap up (long), -1 gap down (short)
    gap: float
    daily_open: float


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    day: date
    direction: int
    gross_r: float
    net_r: float
    exit_reason: str  # "target" | "stop" | "gap_through" | "deadline"
    worst_loss_frac: float  # signed return on notional at exit, gross


def find_gap_events(bars_by_symbol: dict[str, list[Bar]], min_gap: float = 0.05,
                    start_year: int = 2019) -> list[GapEvent]:
    events = []
    for symbol, bars in bars_by_symbol.items():
        # Pair each day with the chronologically prior close. Sorting first
        # keeps an unsorted caller from silently creating lookahead events
        # (a day paired against a FUTURE close). Identity on sorted input.
        bars = sorted(bars, key=lambda b: b.date)
        for i in range(1, len(bars)):
            if bars[i].date.year < start_year or bars[i - 1].close <= 0:
                continue
            gap = bars[i].open / bars[i - 1].close - 1.0
            if abs(gap) >= min_gap:
                events.append(GapEvent(symbol, bars[i].date, 1 if gap > 0 else -1, gap, bars[i].open))
    events.sort(key=lambda e: (e.day, e.symbol))
    return events


def _session_bars(minute_bars: list[MinuteBar]) -> list[MinuteBar]:
    """Regular-session bars from the open through the deadline, ET."""
    session = []
    for bar in minute_bars:
        local = bar.timestamp.astimezone(ET)
        minutes = (local.hour - 9) * 60 + local.minute - 30
        if 0 <= minutes <= DEADLINE_MINUTES:
            session.append(bar)
    return session


def simulate_trade(minute_bars: list[MinuteBar], direction: int) -> TradeResult | None:
    """Run the +0.5%/-0.5% race on one event's morning bars.

    Returns None when the morning has too little data to trade (< 5 bars).
    """
    bars = _session_bars(minute_bars)
    if len(bars) < 5:
        return None
    # One trade = one morning. _session_bars filters by clock time only, so
    # bars from a LATER date at the same clock time would otherwise bleed
    # into this trade (the race would run into the next morning). Restrict
    # to the first session bar's ET date; identity on single-day input.
    day0 = bars[0].timestamp.astimezone(ET).date()
    bars = [b for b in bars if b.timestamp.astimezone(ET).date() == day0]
    if len(bars) < 5:
        return None
    entry = bars[0].open
    if entry <= 0:
        return None
    target = entry * (1 + TARGET_FRAC * direction)
    stop = entry * (1 - STOP_FRAC * direction)
    exit_price = bars[-1].close
    reason = "deadline"
    for bar in bars:  # the entry bar's own range counts: it opens at entry
        if direction > 0:
            if bar.open >= target:  # resting limit fills at the open print,
                exit_price, reason = bar.open, "target"  # before any intrabar dip
                break
            if bar.open <= stop:
                exit_price, reason = bar.open, "gap_through"
                break
            if bar.low <= stop:  # stop wins ties within a bar (conservative)
                exit_price, reason = stop, "stop"
                break
            if bar.high >= target:
                exit_price, reason = target, "target"
                break
        else:
            if bar.open <= target:  # resting limit fills at the open print
                exit_price, reason = bar.open, "target"
                break
            if bar.open >= stop:
                exit_price, reason = bar.open, "gap_through"
                break
            if bar.high >= stop:
                exit_price, reason = stop, "stop"
                break
            if bar.low <= target:
                exit_price, reason = target, "target"
                break
    gross_frac = (exit_price / entry - 1.0) * direction
    gross_r = gross_frac / STOP_FRAC
    first = bars[0]
    return TradeResult(
        symbol=first.symbol,
        day=first.timestamp.astimezone(ET).date(),
        direction=direction,
        gross_r=gross_r,
        net_r=gross_r - COST_R,
        exit_reason=reason,
        worst_loss_frac=gross_frac,
    )


@dataclass(frozen=True)
class StudyStats:
    trades: int
    win_rate: float
    profit_factor: float
    mean_net_r: float
    sharpe_daily_r: float
    worst_trade_frac: float
    total_net_r: float


def aggregate(trades: list[TradeResult]) -> StudyStats:
    if not trades:
        return StudyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    net = [t.net_r for t in trades]
    wins = sum(r for r in net if r > 0)
    losses = -sum(r for r in net if r < 0)
    profit_factor = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)
    daily: dict[date, float] = {}
    for trade in trades:
        daily[trade.day] = daily.get(trade.day, 0.0) + trade.net_r
    daily_series = [daily[d] for d in sorted(daily)]
    if len(daily_series) > 2 and pstdev(daily_series) > 0:
        sharpe = mean(daily_series) / pstdev(daily_series) * sqrt(252)
    else:
        sharpe = 0.0
    return StudyStats(
        trades=len(trades),
        win_rate=sum(1 for r in net if r > 0) / len(net),
        profit_factor=profit_factor,
        mean_net_r=mean(net),
        sharpe_daily_r=sharpe,
        worst_trade_frac=min(t.worst_loss_frac for t in trades),
        total_net_r=sum(net),
    )


def kill_rule_verdict(holdout: StudyStats) -> tuple[bool, str]:
    """Pre-registered: survival needs ALL of Sharpe >= 1.0, PF >= 1.3,
    and no single loss worse than -5% of notional in the holdout."""
    failures = []
    if holdout.sharpe_daily_r < 1.0:
        failures.append(f"holdout net Sharpe {holdout.sharpe_daily_r:.2f} < 1.0")
    if holdout.profit_factor < 1.3:
        failures.append(f"holdout profit factor {holdout.profit_factor:.2f} < 1.3")
    if holdout.worst_trade_frac <= -0.05:
        failures.append(f"worst holdout trade {holdout.worst_trade_frac:.1%} <= -5%")
    if failures:
        return False, "FALSIFIED — " + "; ".join(failures)
    return True, "SURVIVES the pre-registered kill rule"
