"""Baseline tests for scripts/daily_pipeline.py: the dirty-tree guard
and the quiet/reportable outcome contract. Git fixtures are real
(temp repos); the refresh and tracker stages are fakes — the guard
must refuse before either of them is touched.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "daily_pipeline",
    Path(__file__).resolve().parents[2] / "scripts" / "daily_pipeline.py",
)
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline  # @dataclass resolution needs the module registered
_spec.loader.exec_module(pipeline)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    for rel in pipeline.GUARDED_FILES:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# guarded\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


class FakeRefresh:
    def __init__(self, failed=()):
        self.called = False
        self._failed = list(failed)

    def refresh_cache(self, symbols, today):
        self.called = True
        from types import SimpleNamespace
        return [SimpleNamespace(symbol="SPY", error="boom" if self._failed else "", bars=0)]

    def failed_symbols(self, results):
        return self._failed


class FakeTracker:
    VT_TREND_FORWARD_REPORT_DIR = None  # set per-test

    def __init__(self, outcome):
        self.called = False
        self._outcome = outcome

    def run(self, refresh=False, today=None):
        self.called = True
        report_dir = Path(self.VT_TREND_FORWARD_REPORT_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "last_run.json").write_text(json.dumps(self._outcome))
        return "# fake tracker report\n"


def test_uncommitted_changes_clean_then_dirty(tmp_path):
    repo = _make_repo(tmp_path)
    assert pipeline.uncommitted_changes(repo, pipeline.GUARDED_FILES) == []
    (repo / "market_lab" / "signals.py").write_text("# edited\n")
    dirty = pipeline.uncommitted_changes(repo, pipeline.GUARDED_FILES)
    assert dirty == ["market_lab/signals.py"]
    # A deleted guarded file is also a change that must refuse.
    (repo / "market_lab" / "signals.py").write_text("# edited again\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    (repo / "scripts" / "daily_pipeline.py").unlink()
    assert pipeline.uncommitted_changes(repo, pipeline.GUARDED_FILES) == ["scripts/daily_pipeline.py"]


def test_uncommitted_changes_fails_closed_outside_a_repo(tmp_path):
    with pytest.raises(pipeline.PipelineSkip, match="git status failed"):
        pipeline.uncommitted_changes(tmp_path, pipeline.GUARDED_FILES)


def test_pipeline_refuses_dirty_tree_before_any_stage(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "market_lab" / "config.py").write_text("# in-progress edit\n")
    refresh = FakeRefresh()
    tracker = FakeTracker({"fills_accepted": 0, "fills_rejected": 0, "candidates_queued": 0})
    tracker.VT_TREND_FORWARD_REPORT_DIR = tmp_path / "reports"
    with pytest.raises(pipeline.PipelineSkip, match="uncommitted changes"):
        pipeline.run_pipeline(repo_root=repo, tracker_module=tracker, refresh_module=refresh)
    assert not refresh.called
    assert not tracker.called


def test_pipeline_quiet_and_reportable_outcomes(tmp_path):
    repo = _make_repo(tmp_path)
    refresh = FakeRefresh()

    quiet = FakeTracker({"signal_date": "2026-01-02", "fills_accepted": 0,
                         "fills_rejected": 0, "candidates_queued": 0})
    quiet.VT_TREND_FORWARD_REPORT_DIR = tmp_path / "reports_quiet"
    outcome = pipeline.run_pipeline(repo_root=repo, tracker_module=quiet, refresh_module=refresh)
    assert outcome.reportable is False
    assert outcome.refreshed_symbols == 1
    assert outcome.signal_date == "2026-01-02"

    busy = FakeTracker({"signal_date": "2026-01-03", "fills_accepted": 1,
                        "fills_rejected": 0, "candidates_queued": 2})
    busy.VT_TREND_FORWARD_REPORT_DIR = tmp_path / "reports_busy"
    outcome = pipeline.run_pipeline(repo_root=repo, tracker_module=busy, refresh_module=refresh)
    assert outcome.reportable is True
    assert outcome.fills_accepted == 1
    assert outcome.candidates_queued == 2


def test_pipeline_refresh_failure_stops_before_tracker(tmp_path):
    repo = _make_repo(tmp_path)
    refresh = FakeRefresh(failed=["SPY"])
    tracker = FakeTracker({"fills_accepted": 0, "fills_rejected": 0, "candidates_queued": 0})
    tracker.VT_TREND_FORWARD_REPORT_DIR = tmp_path / "reports"
    with pytest.raises(RuntimeError, match="cache refresh failed"):
        pipeline.run_pipeline(repo_root=repo, tracker_module=tracker, refresh_module=refresh)
    assert not tracker.called
