#!/usr/bin/env python3
"""Backfill daily bars for strategy streams from Alpaca (SIP, adjusted).

Credentials come from the environment only (ALPACA_API_KEY /
ALPACA_API_SECRET), matching the intraday fetcher contract. Writes the
repo CSV cache plus a provenance sidecar per symbol via
``market_lab.daily_data.backfill_daily`` — no synthetic data, ever.

Usage:
    python scripts/backfill_daily.py [--symbols SPY QQQ ...] [--years 10]
                                     [--start YYYY-MM-DD] [--end YYYY-MM-DD]
Defaults: DEFAULT_UNIVERSE, 10 years ending today.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import DEFAULT_UNIVERSE
from market_lab.daily_data import backfill_daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    # Default end is yesterday: the free/paper SIP subscription rejects
    # queries touching recent (in-progress) data, and daily strategies
    # should never consume an incomplete bar anyway.
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=int(args.years * 365.25))
    symbols = args.symbols or DEFAULT_UNIVERSE

    results = backfill_daily(symbols, start, end)
    failures = 0
    for r in results:
        if r.error:
            failures += 1
            print(f"{r.symbol:6s} FAILED  {r.error}")
        else:
            print(f"{r.symbol:6s} {r.bars:5d} bars  {r.first} .. {r.last}  ({r.source})")
    print(f"\n{len(results) - failures}/{len(results)} symbols backfilled, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
