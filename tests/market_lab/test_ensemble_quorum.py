"""Baseline tests for the quorum fail-safe ensemble guard."""

from __future__ import annotations

import pytest

from market_lab.ensemble import default_min_valid, guarded_ensemble
from market_lab.signals import Signal


def sig(action: str, confidence: float, weight: float = 0.0, strategy: str = "m") -> Signal:
    return Signal(
        symbol="SPY", action=action, confidence=confidence, reason="r",
        close=100.0, rsi14=None, sma20=None, sma50=None, volatility20=None,
        strategy=strategy, target_weight=weight, evidence={},
    )


FALLBACK = sig("HOLD", 0.0)


def test_default_quorum_rule():
    assert default_min_valid(1) == 1
    assert default_min_valid(3) == 3
    assert default_min_valid(7) == 4
    assert default_min_valid(12) == 4
    with pytest.raises(ValueError):
        default_min_valid(0)


def test_quorum_failure_goes_flat_and_names_failures():
    def boom():
        raise RuntimeError("member exploded")

    members = {
        "a": lambda: sig("BUY", 0.9, 0.5),
        "b": boom,
        "c": lambda: sig("BUY", 0.8, 0.5),
        "d": lambda: (_ for _ in ()).throw(ValueError("nope")),
    }
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=3)
    assert out.action == "SELL" and out.target_weight == 0.0
    assert out.strategy == "guarded_ensemble"
    assert "b" in out.reason and "d" in out.reason
    assert out.evidence["quorum_valid"] == 2.0


def test_member_with_nan_output_counts_as_failed():
    members = {
        "a": lambda: sig("BUY", float("nan"), 0.5),
        "b": lambda: sig("BUY", 0.9, 0.5),
    }
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL"  # only 1 valid of 2 needed -> fail-safe
    assert out.evidence["quorum_failed"] == "a"


def test_quorum_met_combines_with_thresholds():
    members = {
        "a": lambda: sig("BUY", 0.5, 0.4),
        "b": lambda: sig("BUY", 0.4, 0.7),
        "c": lambda: sig("HOLD", 0.1),
        "d": lambda: sig("SELL", 0.2),
    }
    out = guarded_ensemble(members, fallback=FALLBACK)  # default quorum = 4
    assert out.action == "BUY"
    assert out.target_weight == pytest.approx(0.7)  # max BUY member weight
    assert out.evidence["quorum_valid"] == 4.0


def test_quorum_met_but_no_consensus_holds():
    members = {
        "a": lambda: sig("BUY", 0.3, 0.5),
        "b": lambda: sig("SELL", 0.3),
        "c": lambda: sig("HOLD", 0.2),
        "d": lambda: sig("HOLD", 0.1),
    }
    out = guarded_ensemble(members, fallback=FALLBACK)
    assert out.action == "HOLD" and out.target_weight == 0.0


def test_empty_members_and_bad_min_valid_refused():
    with pytest.raises(ValueError):
        guarded_ensemble({}, fallback=FALLBACK)
    with pytest.raises(ValueError):
        guarded_ensemble({"a": lambda: sig("BUY", 0.5)}, fallback=FALLBACK, min_valid=2)
    with pytest.raises(ValueError):
        guarded_ensemble({"a": lambda: sig("BUY", 0.5)}, fallback=FALLBACK, min_valid=0)
