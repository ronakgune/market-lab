"""Offline smoke check for the intraday foundation.

Generates deterministic synthetic minute bars for one symbol/day, stores
them in the SYNTHETIC quarantine store (INTRADAY_SYNTHETIC_BARS_DIR —
never the real bar store, mirroring the daily layer's synthetic
discipline), queries an event window back, and asserts the round-trip.

No network by construction: this script imports only config, bars, and
store from market_lab (no fetcher, no EDGAR, no SDKs).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from market_lab.config import INTRADAY_BARS_DIR, INTRADAY_SYNTHETIC_BARS_DIR
from market_lab.intraday.bars import MinuteBar
from market_lab.intraday.store import bars_for_day, bars_for_window, write_bars

SYMBOL = "SMOKE"
DAY = date(2026, 2, 3)  # a Tuesday
OPEN_UTC = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
MINUTES = 390  # one full regular session


def synthetic_session() -> list[MinuteBar]:
    bars: list[MinuteBar] = []
    price = 100.0
    for i in range(MINUTES):
        delta = ((i * 7) % 13 - 6) * 0.03  # deterministic walk, no RNG
        open_ = price
        close = round(price + delta, 4)
        bars.append(MinuteBar(
            symbol=SYMBOL, timestamp=OPEN_UTC + timedelta(minutes=i),
            open=open_, high=max(open_, close) + 0.02, low=min(open_, close) - 0.02,
            close=close, volume=1_000 + (i * 37) % 500, source="synthetic",
        ))
        price = close
    return bars


def main() -> None:
    root = INTRADAY_SYNTHETIC_BARS_DIR
    assert root != INTRADAY_BARS_DIR, "smoke must never write to the real bar store"
    assert "synthetic" in root.name, "smoke root must be the synthetic quarantine area"

    bars = synthetic_session()
    written = write_bars(bars, root=root)
    assert written == MINUTES, f"expected {MINUTES} new rows, wrote {written}"
    # Re-writing the same session is a no-op (first write wins).
    assert write_bars(bars, root=root) == 0

    day_bars = bars_for_day(SYMBOL, DAY, root=root)
    assert len(day_bars) == MINUTES
    assert all(b.source == "synthetic" for b in day_bars)

    window = bars_for_window(SYMBOL, OPEN_UTC, OPEN_UTC + timedelta(minutes=60), root=root)
    assert len(window) == 60, f"event window returned {len(window)} bars, expected 60"
    assert window == sorted(window, key=lambda b: b.timestamp)
    assert window[0].timestamp == OPEN_UTC

    print(f"SMOKE OK: {written} synthetic bars stored under {root}")
    print(f"SMOKE OK: day query returned {len(day_bars)} bars; "
          f"09:30-10:30 ET window returned {len(window)} bars, all source='synthetic'")


if __name__ == "__main__":
    main()
