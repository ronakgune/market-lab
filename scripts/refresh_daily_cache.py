#!/usr/bin/env python3
"""Provenance-checked refresh of the daily price cache (pipeline stage 1).

Re-runs the full Alpaca SIP daily backfill (adjustment ALL) for the
pipeline universe via market_lab.daily_data.backfill_daily, then
verifies every symbol by loading it back through load_daily_bars —
which refuses any cache whose provenance sidecar is missing or
disagrees with the CSV. The cache writer replaces whole files, so the
refresh always fetches the full window; a partial refresh would
truncate history.

Universe: config.DEFAULT_UNIVERSE plus VIXY (the vol-regime overlay's
series, kept in the provenanced cache alongside the 21 defaults).

Credentials come from the environment only (ALPACA_API_KEY /
ALPACA_API_SECRET) — source ~/.config/market-lab/alpaca.env first.
Missing credentials surface as per-symbol MissingCredentialsError
results and a nonzero exit, never as substituted data.

Usage:
    python scripts/refresh_daily_cache.py [--symbols SPY QQQ ...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import DEFAULT_UNIVERSE
from market_lab.daily_data import BackfillResult, backfill_daily, load_daily_bars

EXTRA_SYMBOLS = ["VIXY"]  # vol-regime overlay series, cached with the universe
DEFAULT_SYMBOLS = list(DEFAULT_UNIVERSE) + [s for s in EXTRA_SYMBOLS if s not in DEFAULT_UNIVERSE]
WINDOW_DAYS = 365 * 10 + 30  # match the trackers' --refresh window


def refresh_cache(symbols: list[str] | None = None, today: date | None = None) -> list[BackfillResult]:
    """Backfill the full window for each symbol, then provenance-verify
    each one by loading it back. A symbol that backfills but fails
    verification is returned with its error set — a cache that cannot
    vouch for itself is a failed refresh, not a usable one."""
    today = today or date.today()
    symbols = symbols or DEFAULT_SYMBOLS
    results = backfill_daily(symbols, today - timedelta(days=WINDOW_DAYS), today - timedelta(days=1))
    verified: list[BackfillResult] = []
    for result in results:
        if result.error or result.bars == 0:
            verified.append(result)
            continue
        try:
            bars, _meta = load_daily_bars(result.symbol)
        except Exception as exc:  # provenance refusal is a refresh failure
            verified.append(BackfillResult(
                result.symbol, result.bars, result.first, result.last,
                result.source, f"provenance check failed: {type(exc).__name__}: {exc}",
            ))
            continue
        if not bars or bars[-1].date != result.last:
            verified.append(BackfillResult(
                result.symbol, result.bars, result.first, result.last,
                result.source, "provenance check failed: reloaded cache disagrees with backfill",
            ))
            continue
        verified.append(result)
    return verified


def failed_symbols(results: list[BackfillResult]) -> list[str]:
    return [r.symbol for r in results if r.error or r.bars == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the provenanced daily price cache")
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()
    results = refresh_cache(args.symbols)
    for r in results:
        if r.error:
            print(f"{r.symbol:6s} FAILED  {r.error}")
        else:
            print(f"{r.symbol:6s} {r.bars:5d} bars  {r.first} .. {r.last}  ({r.source}, provenance verified)")
    failed = failed_symbols(results)
    print(f"\n{len(results) - len(failed)}/{len(results)} symbols refreshed and verified, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
