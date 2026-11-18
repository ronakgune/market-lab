"""Coverage probe for the DoltHub community options dataset (Stream 4).

Determines which (symbol, date) chains exist in post-no-preference/options
by issuing one scoped COUNT query per date. The dataset's coverage is
spotty — chains exist only on some days — so phase 1 needs a map before
any evaluation window can be pre-registered.

Design:
- Resumable: results are checkpointed to
  data/market-lab/options/coverage_checkpoint.json after every probe;
  a restart skips dates already recorded. Errors are recorded as
  negative sentinels (-1 timeout/transport, -2 server error) and are
  NOT treated as "no chain" — a later run retries them.
- Polite: sequential requests, small sleep between probes, NO auth
  header (DoltHub returns 400 when one is sent).
- Probe sets: every Friday in the range, plus every weekday inside the
  named stress windows. --weekday N adds every weekday-N in the range
  (used when a weekday pattern emerges from the Friday/stress pass).

Usage:
    .venv/bin/python scripts/probe_options_coverage.py \
        [--symbol SPY] [--start 2019-01-01] [--end 2026-11-13] \
        [--weekday 2] [--sleep 0.25]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONS_DIR = ROOT / "data" / "market-lab" / "options"
CHECKPOINT = OPTIONS_DIR / "coverage_checkpoint.json"
API_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

STRESS_WINDOWS = [
    (date(2020, 2, 1), date(2020, 4, 30)),   # COVID crash
    (date(2022, 1, 1), date(2022, 11, 30)),  # 2022 bear / vol year
    (date(2024, 7, 1), date(2024, 8, 31)),   # Aug 2024 vol spike
    (date(2025, 3, 1), date(2025, 4, 30)),   # Apr 2025 tariff shock
]


def probe_dates(start: date, end: date, extra_weekday: int | None,
                skip_weekdays: frozenset[int] = frozenset()) -> list[date]:
    days: set[date] = set()
    d = start
    while d <= end:
        if d.weekday() == 4:  # every Friday
            days.add(d)
        if extra_weekday is not None and d.weekday() == extra_weekday:
            days.add(d)
        d += timedelta(days=1)
    for lo, hi in STRESS_WINDOWS:
        d = max(lo, start)
        while d <= min(hi, end):
            if d.weekday() < 5:  # weekdays only inside stress windows
                days.add(d)
            d += timedelta(days=1)
    return sorted(d for d in days if d.weekday() not in skip_weekdays)


def count_chain(symbol: str, day: date, timeout: float = 150.0) -> int:
    sql = (
        "SELECT COUNT(*) AS n FROM option_chain "
        f"WHERE act_symbol='{symbol}' AND date='{day.isoformat()}'"
    )
    url = API_BASE + "?q=" + urllib.parse.quote(sql)
    req = urllib.request.Request(url)  # deliberately no auth header
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    if payload.get("query_execution_status") != "Success":
        raise RuntimeError(f"query failed: {payload.get('query_execution_message')}")
    rows = payload.get("rows") or []
    if not rows:
        return 0
    return int(rows[0]["n"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--weekday", type=int, default=None,
                    help="also probe every weekday with this number (Mon=0)")
    ap.add_argument("--skip-weekdays", default="",
                    help="comma-separated weekday numbers to exclude "
                         "(e.g. '1,3' for Tue/Thu once the Mon/Wed/Fri "
                         "snapshot pattern is established)")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, int] = {}
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))

    skip = frozenset(int(x) for x in args.skip_weekdays.split(",") if x.strip())
    # Probe dates never recorded, plus retry dates recorded as errors.
    todo = [d for d in probe_dates(start, end, args.weekday, skip)
            if d.isoformat() not in checkpoint or checkpoint[d.isoformat()] < 0]
    print(f"{len(todo)} dates to probe ({len(checkpoint)} already recorded)", flush=True)

    for i, day in enumerate(todo, 1):
        key = day.isoformat()
        try:
            n = count_chain(args.symbol, day)
            checkpoint[key] = n
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            checkpoint[key] = -1
            print(f"{key}: transport error {exc}", flush=True)
        except Exception as exc:  # server-side error / deadline
            checkpoint[key] = -2
            print(f"{key}: error {type(exc).__name__}: {exc}", flush=True)
        CHECKPOINT.write_text(json.dumps(checkpoint, indent=1, sort_keys=True), encoding="utf-8")
        if i % 25 == 0:
            present = sum(1 for v in checkpoint.values() if v > 0)
            print(f"[{i}/{len(todo)}] recorded={len(checkpoint)} present={present}", flush=True)
        time.sleep(args.sleep)

    present = sorted(k for k, v in checkpoint.items() if v > 0)
    print(f"DONE recorded={len(checkpoint)} present={len(present)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
