from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from market_lab.intraday.bars import MinuteBar, valid_minute_bar

TS = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)


def _bar(**overrides) -> MinuteBar:
    fields = dict(symbol="ACME", timestamp=TS, open=100.0, high=100.5,
                  low=99.8, close=100.2, volume=1_000, source="alpaca:sip")
    fields.update(overrides)
    return MinuteBar(**fields)


def test_valid_bar():
    assert valid_minute_bar(_bar())


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        _bar(timestamp=datetime(2026, 2, 3, 14, 30))


def test_non_utc_timestamp_rejected():
    with pytest.raises(ValueError):
        _bar(timestamp=datetime(2026, 2, 3, 9, 30, tzinfo=ZoneInfo("America/New_York")))


def test_blank_symbol_and_source_rejected():
    with pytest.raises(ValueError):
        _bar(symbol=" ")
    with pytest.raises(ValueError):
        _bar(source="")


def test_invalid_prices():
    assert not valid_minute_bar(_bar(open=0.0))
    assert not valid_minute_bar(_bar(close=-1.0))
    assert not valid_minute_bar(_bar(high=99.0, low=100.0, open=99.5, close=99.5))


def test_ohlc_containment_and_volume():
    assert not valid_minute_bar(_bar(close=101.0))  # close above high
    assert not valid_minute_bar(_bar(open=99.0))  # open below low
    assert not valid_minute_bar(_bar(volume=-5))


def test_valid_minute_bar_rejects_other_types():
    assert not valid_minute_bar(object())
    assert not valid_minute_bar(None)
