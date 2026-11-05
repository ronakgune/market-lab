"""MinuteBar: one minute of OHLCV for one symbol, always tz-aware UTC.

Mirrors the discipline of market_lab.data.Bar / _valid_bar, extended with a
timestamp and a mandatory provenance `source` string (e.g. "alpaca:sip",
"synthetic") so intraday data carries the same source tracking as the daily
data layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    timestamp: datetime  # tz-aware, UTC
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("MinuteBar timestamp must be tz-aware")
        if self.timestamp.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("MinuteBar timestamp must be in UTC")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("MinuteBar symbol must be non-empty")
        if not self.source or not self.source.strip():
            raise ValueError("MinuteBar source must be non-empty (provenance is mandatory)")


def _finite_price(value: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def valid_minute_bar(bar: MinuteBar) -> bool:
    """Same price sanity as data._valid_bar, plus OHLC containment."""
    if not isinstance(bar, MinuteBar):
        return False
    if not all(_finite_price(v) for v in (bar.open, bar.high, bar.low, bar.close)):
        return False
    if bar.high < bar.low or bar.volume < 0:
        return False
    return bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high
