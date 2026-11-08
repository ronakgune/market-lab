"""Quorum fail-safe for signal ensembles.

Pattern dissected 2026-11-07 from BreakFree (ronakgune/BreakFree
@ 2f36051, ``VotingEngine.run``): each strategy votes in isolation —
one member raising marks its vote ERROR instead of crashing the
engine — and if fewer than a quorum of members produce valid votes,
the system goes to 100% cash rather than trading on a partial
ensemble. ``MIN_QUORUM`` there is ``max(1, min(n_strategies, 4))``:
at most 4, i.e. a majority-ish floor for its 7-member ensemble.

This module is that pattern rebuilt as a reusable guard for this
repo's ensembles (``signals.generate_ensemble_signal`` is the obvious
future adopter; adoption is deliberately NOT done here, so no live
signal behavior changes in the harvest commit). A member counts as
valid only if it returns a Signal with finite confidence and target
weight — a member that "succeeds" with NaN outputs is a failed
member, not a vote.

Combination, when quorum is met, mirrors the scoring thresholds of
``generate_ensemble_signal`` (buy >= 0.70, sell >= 0.45, confidence-
weighted) minus its SPY-relative guard, which belongs to the caller.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Mapping

from .signals import Signal

BUY_THRESHOLD = 0.70
SELL_THRESHOLD = 0.45


def default_min_valid(n_members: int) -> int:
    """BreakFree's quorum rule: at most 4 valid votes required."""
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")
    return max(1, min(n_members, 4))


def guarded_ensemble(
    members: Mapping[str, Callable[[], Signal]],
    *,
    fallback: Signal,
    min_valid: int | None = None,
) -> Signal:
    """Combine member signals, failing safe to flat below quorum.

    ``members`` maps a display name to a zero-argument callable that
    produces that member's Signal for the current bar (callers close
    over symbol/bars). ``fallback`` supplies the identity fields
    (symbol, close, indicator snapshot) for the fail-safe signal.

    Below quorum the result is a SELL to target weight 0.0 — this
    repo's go-flat signal — with strategy ``guarded_ensemble`` and a
    reason naming the failed members. At or above quorum the valid
    members are combined with generate_ensemble_signal's thresholds.
    """
    if not members:
        raise ValueError("members must not be empty")
    needed = default_min_valid(len(members)) if min_valid is None else min_valid
    if needed < 1 or needed > len(members):
        raise ValueError(f"min_valid must be in [1, {len(members)}], got {needed}")

    valid: dict[str, Signal] = {}
    failed: list[str] = []
    for name, member in members.items():
        try:
            sig = member()
            ok = (
                sig is not None
                and math.isfinite(sig.confidence)
                and math.isfinite(sig.target_weight)
            )
        except Exception:
            failed.append(name)
            continue
        if not ok:
            failed.append(name)
            continue
        valid[name] = sig

    quorum_evidence = {
        "quorum_valid": float(len(valid)),
        "quorum_needed": float(needed),
        "quorum_failed": ",".join(failed) if failed else None,
    }
    if len(valid) < needed:
        return replace(
            fallback,
            action="SELL",
            target_weight=0.0,
            strategy="guarded_ensemble",
            reason=(
                f"quorum failure: {len(valid)}/{len(members)} members valid "
                f"(need {needed}); fail-safe flat"
                + (f"; failed: {', '.join(failed)}" if failed else "")
            ),
            evidence={**fallback.evidence, **quorum_evidence},
        )

    family = list(valid.values())
    buy_score = sum(s.confidence for s in family if s.action == "BUY")
    sell_score = sum(s.confidence for s in family if s.action == "SELL")
    primary = max(family, key=lambda s: s.confidence)
    if buy_score >= BUY_THRESHOLD and buy_score > sell_score:
        action = "BUY"
        confidence = min(1.0, max(0.0, buy_score / len(family) + 0.35))
        target_weight = max((s.target_weight for s in family if s.action == "BUY"), default=0.0)
    elif sell_score >= SELL_THRESHOLD and sell_score >= buy_score:
        action = "SELL"
        confidence = min(1.0, max(0.0, sell_score / len(family) + 0.25))
        target_weight = 0.0
    else:
        action = "HOLD"
        confidence = min(1.0, max(0.0, max(buy_score, sell_score) / len(family)))
        target_weight = 0.0
    return Signal(
        symbol=fallback.symbol,
        action=action,
        confidence=confidence,
        reason="guarded ensemble: "
        + " | ".join(f"{name}={s.action}/{s.confidence:.2f}" for name, s in valid.items())
        + (f" | failed: {', '.join(failed)}" if failed else ""),
        close=fallback.close,
        rsi14=primary.rsi14,
        sma20=primary.sma20,
        sma50=primary.sma50,
        volatility20=primary.volatility20,
        strategy="guarded_ensemble",
        target_weight=target_weight,
        evidence={**quorum_evidence, "buy_score": buy_score, "sell_score": sell_score},
    )


__all__ = ["BUY_THRESHOLD", "SELL_THRESHOLD", "default_min_valid", "guarded_ensemble"]
