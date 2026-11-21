#!/usr/bin/env python3
"""Daily paper pipeline: refresh -> vt_trend forward tracker -> report.

One run does, in order:

1. Dirty-tree guard: refuse to run if any of the pipeline's own files
   has uncommitted changes — an in-progress edit must never execute
   against live paper state (the same discipline the XS tracker job
   applies to its files).
2. Data refresh: provenance-checked full-window refresh of the daily
   price cache (scripts/refresh_daily_cache.py). Credentials come from
   the environment only — source ~/.config/market-lab/alpaca.env
   first; MissingCredentialsError is reported by name.
3. vt_trend forward tracker (scripts/market_lab_vt_trend_forward.py)
   on the just-refreshed cache.

Output contract for the scheduling layer: the first stdout line is
exactly one of
    PIPELINE_QUIET   — nothing reportable happened (mark-only day)
    PIPELINE_REPORT  — fills and/or newly queued candidates follow
and failures exit nonzero with the error on stderr (a PIPELINE_FAILURE
line is also printed to stdout so a captured-stdout reader sees it).
The XS momentum and crypto carry trackers keep their own jobs; this
pipeline never runs or edits them.

Usage:
    python scripts/daily_pipeline.py [--skip-refresh] [--symbols SPY QQQ ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

# The pipeline's own files: the guard refuses to run while any of
# these carries uncommitted changes in the tree being executed.
GUARDED_FILES = [
    "scripts/daily_pipeline.py",
    "scripts/refresh_daily_cache.py",
    "scripts/market_lab_vt_trend_forward.py",
    "market_lab/config.py",
    "market_lab/daily_data.py",
    "market_lab/signals.py",
]


class PipelineSkip(RuntimeError):
    """The run was refused before doing any work (e.g. dirty tree)."""


def uncommitted_changes(repo_root: Path, paths: list[str]) -> list[str]:
    """Porcelain-status the given paths; return the dirty file list.

    Raises PipelineSkip if git itself fails — an unverifiable tree is
    treated as a dirty one (fail closed)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", *paths],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise PipelineSkip(f"git status failed in {repo_root}: {proc.stderr.strip()}")
    dirty = []
    for line in proc.stdout.splitlines():
        if line.strip():
            dirty.append(line[3:].strip())
    return dirty


def _load_tracker_module(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "market_lab_vt_trend_forward", repo_root / "scripts" / "market_lab_vt_trend_forward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_refresh_module(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "refresh_daily_cache", repo_root / "scripts" / "refresh_daily_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PipelineOutcome:
    reportable: bool
    report: str
    fills_accepted: int = 0
    fills_rejected: int = 0
    candidates_queued: int = 0
    refreshed_symbols: int = 0
    signal_date: str = ""


def run_pipeline(repo_root: Path = REPO_ROOT, today: date | None = None,
                 symbols: list[str] | None = None, skip_refresh: bool = False,
                 tracker_module=None, refresh_module=None) -> PipelineOutcome:
    dirty = uncommitted_changes(repo_root, GUARDED_FILES)
    if dirty:
        raise PipelineSkip(
            "refusing to run: uncommitted changes in pipeline files: " + ", ".join(dirty)
        )

    refreshed = 0
    if not skip_refresh:
        refresh_module = refresh_module or _load_refresh_module(repo_root)
        results = refresh_module.refresh_cache(symbols, today)
        failed = refresh_module.failed_symbols(results)
        if failed:
            errors = {r.symbol: r.error for r in results if r.symbol in failed}
            raise RuntimeError(f"cache refresh failed for {failed}: {errors}")
        refreshed = len(results)

    tracker_module = tracker_module or _load_tracker_module(repo_root)
    report = tracker_module.run(refresh=False, today=today)
    outcome_path = tracker_module.VT_TREND_FORWARD_REPORT_DIR / "last_run.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    fills_accepted = int(outcome.get("fills_accepted", 0))
    fills_rejected = int(outcome.get("fills_rejected", 0))
    queued = int(outcome.get("candidates_queued", 0))
    reportable = (fills_accepted + fills_rejected) > 0 or queued > 0
    return PipelineOutcome(
        reportable=reportable,
        report=report,
        fills_accepted=fills_accepted,
        fills_rejected=fills_rejected,
        candidates_queued=queued,
        refreshed_symbols=refreshed,
        signal_date=str(outcome.get("signal_date", "")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily paper pipeline (refresh + vt_trend forward tracker)")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip the cache refresh stage (tracker still refuses a stale cache)")
    parser.add_argument("--symbols", nargs="*", default=None, help="Refresh universe override (default: pipeline universe)")
    args = parser.parse_args()
    try:
        outcome = run_pipeline(symbols=args.symbols, skip_refresh=args.skip_refresh)
    except PipelineSkip as exc:
        print(f"PIPELINE_FAILURE: skipped — {exc}", file=sys.stderr)
        print("PIPELINE_FAILURE")
        return 1
    except Exception as exc:
        print(f"PIPELINE_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("PIPELINE_FAILURE")
        return 1
    if outcome.reportable:
        print("PIPELINE_REPORT")
        print(
            f"Daily pipeline: refreshed {outcome.refreshed_symbols} symbols; "
            f"vt_trend fills {outcome.fills_accepted} accepted / {outcome.fills_rejected} rejected; "
            f"{outcome.candidates_queued} candidates queued."
        )
        print()
        print(outcome.report)
    else:
        print("PIPELINE_QUIET")
        print(
            f"Daily pipeline: refreshed {outcome.refreshed_symbols} symbols; "
            f"vt_trend mark-only day (data through {outcome.signal_date}), no fills, no new candidates."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
