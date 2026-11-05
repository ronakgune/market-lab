"""Intraday foundation: 1-minute bars, a Parquet/DuckDB bar store, an
Alpaca historical-bars fetcher (data API only), and point-in-time EDGAR
earnings events.

Research-only, like the rest of Market Lab: this package reads historical
market data and SEC filings. It contains no order placement of any kind and
never imports a trading/broker client.
"""
from __future__ import annotations

from .bars import MinuteBar, valid_minute_bar
from .events import EarningsEvent
from .store import bars_for_day, bars_for_window, write_bars

__all__ = [
    "EarningsEvent",
    "MinuteBar",
    "bars_for_day",
    "bars_for_window",
    "valid_minute_bar",
    "write_bars",
]
