#!/usr/bin/env python3
"""Capture a live prediction market into the frozen-descriptor store.

    python scripts/capture_prediction_market.py --venue kalshi --id <TICKER>
    python scripts/capture_prediction_market.py --venue polymarket --id <MARKET_ID>

The capture is preserved byte-exact in the store's raw archive. By
design it is NOT engine-admissible: the chassis normalizer admits only
synthetic fixtures, so a live capture imports as quarantined with the
source-class error recorded. That is the gate working, not a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urllib.request

from market_lab.prediction_markets.capture import (
    capture_kalshi_market,
    capture_polymarket_market,
)


def http_get(url: str, timeout: float = 25.0) -> tuple[bytes, str]:
    """The network lives here, outside the network-free chassis package."""
    request = urllib.request.Request(url, headers={"User-Agent": "market-lab-pm-capture/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get_content_type()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", choices=["kalshi", "polymarket"], required=True)
    parser.add_argument("--id", required=True, help="Kalshi ticker or Polymarket market id")
    args = parser.parse_args()

    capture = capture_kalshi_market if args.venue == "kalshi" else capture_polymarket_market
    result = capture(args.id, http_get)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
