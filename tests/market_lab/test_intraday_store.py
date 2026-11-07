from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from market_lab.intraday.bars import MinuteBar
from market_lab.intraday.store import (
    bars_for_day,
    bars_for_window,
    partition_provenance,
    write_bars,
)

UTC = timezone.utc
DAY1 = date(2026, 2, 3)  # Tuesday
DAY2 = date(2026, 2, 4)


def _bar(day: date, minute: int, close: float = 100.0, source: str = "alpaca:sip") -> MinuteBar:
    # 14:30 UTC = 09:30 ET in February (EST); minutes offset from the open.
    ts = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    return MinuteBar(symbol="ACME", timestamp=ts, open=close, high=close + 0.4,
                     low=close - 0.4, close=close, volume=500, source=source)


def test_round_trip_and_day_query(tmp_path):
    bars = [_bar(DAY1, m) for m in (2, 0, 1)]  # written out of order
    assert write_bars(bars, root=tmp_path) == 3
    got = bars_for_day("ACME", DAY1, root=tmp_path)
    assert [b.timestamp for b in got] == sorted(b.timestamp for b in bars)
    assert all(b.source == "alpaca:sip" for b in got)


def test_partition_layout_and_provenance(tmp_path):
    write_bars([_bar(DAY1, 0, source="synthetic")], root=tmp_path)
    pdir = tmp_path / "symbol=ACME" / "date=2026-02-03"
    assert (pdir / "bars.parquet").exists()
    prov = partition_provenance("ACME", DAY1, root=tmp_path)
    assert prov["sources"] == ["synthetic"]
    assert prov["rows"] == 1


def test_window_query_bounds(tmp_path):
    write_bars([_bar(DAY1, m) for m in range(10)], root=tmp_path)
    start = datetime(2026, 2, 3, 14, 33, tzinfo=UTC)
    end = datetime(2026, 2, 3, 14, 36, tzinfo=UTC)
    got = bars_for_window("ACME", start, end, root=tmp_path)
    assert len(got) == 3  # [start, end): minutes 3, 4, 5
    assert got[0].timestamp == start
    assert got[-1].timestamp < end


def test_window_spans_partitions(tmp_path):
    write_bars([_bar(DAY1, 0), _bar(DAY2, 0)], root=tmp_path)
    got = bars_for_window("ACME", datetime(2026, 2, 3, tzinfo=UTC),
                          datetime(2026, 2, 5, tzinfo=UTC), root=tmp_path)
    assert len(got) == 2
    assert got[0].timestamp < got[1].timestamp


def test_dedupe_keeps_first_write(tmp_path):
    assert write_bars([_bar(DAY1, 0, close=100.0)], root=tmp_path) == 1
    # Same timestamp, different close: the first write wins, 0 new rows.
    assert write_bars([_bar(DAY1, 0, close=999.0)], root=tmp_path) == 0
    got = bars_for_day("ACME", DAY1, root=tmp_path)
    assert len(got) == 1
    assert got[0].close == 100.0
    # A genuinely new timestamp is stored.
    assert write_bars([_bar(DAY1, 1)], root=tmp_path) == 1
    assert len(bars_for_day("ACME", DAY1, root=tmp_path)) == 2


def test_invalid_bar_rejected(tmp_path):
    bad = MinuteBar(symbol="ACME", timestamp=datetime(2026, 2, 3, 14, 30, tzinfo=UTC),
                    open=100.0, high=100.5, low=99.5, close=0.0, volume=10, source="alpaca:sip")
    with pytest.raises(ValueError):
        write_bars([bad], root=tmp_path)
    assert bars_for_day("ACME", DAY1, root=tmp_path) == []


def test_unknown_symbol_and_naive_window(tmp_path):
    assert bars_for_day("NOPE", DAY1, root=tmp_path) == []
    assert bars_for_window("NOPE", datetime(2026, 2, 3, tzinfo=UTC),
                           datetime(2026, 2, 4, tzinfo=UTC), root=tmp_path) == []
    with pytest.raises(ValueError):
        bars_for_window("ACME", datetime(2026, 2, 3), datetime(2026, 2, 4, tzinfo=UTC), root=tmp_path)
