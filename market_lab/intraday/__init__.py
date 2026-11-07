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
from .forecast import ForecastResult, HistoryResult, TradePlan, history_replay, simulate_forecast
from .scanner import DailySetup, OrbHit, build_universe, detect_orb, find_daily_setups, scan_daily
from .session import SessionBar, full_sessions, group_sessions, resample_to_session_bars
from .store import bars_for_day, bars_for_window, write_bars

__all__ = [
    "DailySetup",
    "EarningsEvent",
    "ForecastResult",
    "HistoryResult",
    "MinuteBar",
    "OrbHit",
    "SessionBar",
    "TradePlan",
    "bars_for_day",
    "bars_for_window",
    "build_universe",
    "detect_orb",
    "find_daily_setups",
    "full_sessions",
    "group_sessions",
    "history_replay",
    "resample_to_session_bars",
    "scan_daily",
    "simulate_forecast",
    "valid_minute_bar",
    "write_bars",
]
