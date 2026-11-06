"""Session bars: aggregate 1-minute bars into regular-session bars.

Adapted in concept (not copied in data plumbing) from the day-trader-skill
scanner/timemachine (MIT, Jakeschincariol/day-trader-skill), re-built on
Market Lab's own types: input is ``MinuteBar`` (tz-aware UTC, mandatory
provenance) as stored in the Parquet/DuckDB store, never Yahoo payloads.

A regular session is 09:30-16:00 America/New_York. A 5-minute session has
78 bars (slots 0..77, slot 0 starts 09:30 ET). Only bars whose ET start
falls inside the session are kept; pre/post-market bars are dropped here
by design, because opening-range and time-of-day statistics are defined
on the regular session.

A session is *full* only when every slot is present exactly once. Half
days, halted names and gappy data are therefore excluded from forecast
pools instead of being silently interpolated — no bar is ever invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .bars import MinuteBar

ET = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
SESSION_MINUTES = 390  # 09:30 -> 16:00


@dataclass(frozen=True)
class SessionBar:
    """One aggregated bar inside the regular session (ET-aligned)."""

    symbol: str
    day: date  # ET trading day
    slot: int  # 0-based index from the session open, in units of bar_minutes
    start_et: datetime  # tz-aware, America/New_York
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_minutes: int = 5

    @property
    def end_et(self) -> datetime:
        return self.start_et + timedelta(minutes=self.bar_minutes)


def slots_per_session(bar_minutes: int = 5) -> int:
    if bar_minutes <= 0 or SESSION_MINUTES % bar_minutes != 0:
        raise ValueError("bar_minutes must be a positive divisor of 390")
    return SESSION_MINUTES // bar_minutes


def _slot_for(ts_et: datetime, bar_minutes: int) -> int | None:
    if ts_et.second or ts_et.microsecond:
        return None
    minutes = (ts_et.hour * 60 + ts_et.minute) - (9 * 60 + 30)
    if minutes < 0 or minutes >= SESSION_MINUTES:
        return None
    if minutes % bar_minutes != 0:
        return None
    return minutes // bar_minutes


def resample_to_session_bars(
    bars: Iterable[MinuteBar], bar_minutes: int = 5
) -> list[SessionBar]:
    """Aggregate 1-minute bars into ``bar_minutes`` session bars.

    Input may be unordered and span many days/symbols are NOT mixed: all
    bars must share one symbol. A session bar is emitted only when all of
    its constituent minutes are present (for 1-minute input); partial
    buckets are dropped rather than emitted as understated bars.
    """
    rows = sorted(bars, key=lambda b: b.timestamp)
    if not rows:
        return []
    symbol = rows[0].symbol.upper()
    if any(b.symbol.upper() != symbol for b in rows):
        raise ValueError("resample requires a single symbol")
    if bar_minutes == 1:
        grouped: dict[tuple[date, int], list[MinuteBar]] = {}
        for b in rows:
            ts_et = b.timestamp.astimezone(ET)
            slot = _slot_for(ts_et, 1)
            if slot is None:
                continue
            grouped.setdefault((ts_et.date(), slot), []).append(b)
        out = []
        for (day, slot), members in sorted(grouped.items()):
            if len(members) != 1:
                continue
            m = members[0]
            out.append(SessionBar(symbol, day, slot, m.timestamp.astimezone(ET),
                                  m.open, m.high, m.low, m.close, int(m.volume), 1))
        return out

    n_slots = slots_per_session(bar_minutes)
    grouped = {}
    for b in rows:
        ts_et = b.timestamp.astimezone(ET)
        minutes = (ts_et.hour * 60 + ts_et.minute) - (9 * 60 + 30)
        if minutes < 0 or minutes >= SESSION_MINUTES or ts_et.second or ts_et.microsecond:
            continue
        slot = minutes // bar_minutes
        grouped.setdefault((ts_et.date(), slot), []).append(b)
    out = []
    for (day, slot), members in sorted(grouped.items()):
        if len(members) != bar_minutes:
            continue  # incomplete bucket: drop, never fabricate
        # A duplicated minute must not mask a missing one: the bucket is
        # complete only if its minutes are exactly the slot's minutes.
        offsets = sorted(
            ((m.timestamp.astimezone(ET).hour * 60
              + m.timestamp.astimezone(ET).minute) - (9 * 60 + 30))
            - slot * bar_minutes
            for m in members)
        if offsets != list(range(bar_minutes)):
            continue
        members.sort(key=lambda b: b.timestamp)
        start_et = datetime.combine(
            day, SESSION_OPEN, tzinfo=ET) + timedelta(minutes=slot * bar_minutes)
        out.append(SessionBar(
            symbol=symbol, day=day, slot=slot, start_et=start_et,
            open=members[0].open,
            high=max(m.high for m in members),
            low=min(m.low for m in members),
            close=members[-1].close,
            volume=sum(int(m.volume) for m in members),
            bar_minutes=bar_minutes,
        ))
    _ = n_slots
    return out


def group_sessions(
    session_bars: Sequence[SessionBar],
) -> dict[date, list[SessionBar]]:
    """Group session bars by ET day, each day sorted by slot."""
    out: dict[date, list[SessionBar]] = {}
    for b in session_bars:
        out.setdefault(b.day, []).append(b)
    for day in out:
        out[day] = sorted(out[day], key=lambda b: b.slot)
    return out


def is_full_session(bars: Sequence[SessionBar], bar_minutes: int = 5) -> bool:
    n = slots_per_session(bar_minutes)
    if len(bars) != n:
        return False
    return [b.slot for b in sorted(bars, key=lambda b: b.slot)] == list(range(n))


def full_sessions(
    sessions: Mapping[date, Sequence[SessionBar]],
    *,
    before: date | None = None,
    bar_minutes: int = 5,
) -> dict[date, list[SessionBar]]:
    """Only complete sessions, optionally strictly before ``before``.

    The ``before`` filter is the point-in-time guard: the forecast pool for
    a trade on day D must never contain day D itself or anything after it.
    """
    out: dict[date, list[SessionBar]] = {}
    for day in sorted(sessions):
        if before is not None and day >= before:
            continue
        bars = sorted(sessions[day], key=lambda b: b.slot)
        if is_full_session(bars, bar_minutes=bar_minutes):
            out[day] = bars
    return out
