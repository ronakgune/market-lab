"""Volatility-regime overlay: VIX-spike veto + regime sizing.

Dissected 2026-11-07 from BreakFree (ronakgune/BreakFree @ 2f36051,
``breakfree_cli/strategies/voting.py`` — VolatilityGuard + VotingEngine)
and rebuilt to market-lab standards. Full dissection notes and the
lessons registry live in ``docs/negative-results.md``.

The harvested design, faithfully:

- The regime signal is a RATIO — ``vix_close / SMA(vix_close, 20)`` —
  never an absolute VIX level. BreakFree learned this the hard way: a
  hardcoded ``VIX > 30`` veto fed with VIXY prices (mean ~$47 in its
  dataset) pinned the system in cash through a bull run (the "VIXY
  trap"). A ratio is level-free, so it survives VIXY's decay and
  reverse splits; that is also why this module consumes VIXY bars and
  why split-adjusted (``adjustment=all``) data is safe here.
- VETO: ratio > 1.4 forces full flat (target weight 0) and latches for
  a 3-bar cooldown before the overlay re-evaluates.
- REGIME SIZING (long side only): ratio <= 1.0 -> scale 1.0;
  1.0 < ratio < 1.4 -> scale falls linearly from 0.8 to a 0.2 floor;
  ratio >= 1.4 -> 0.0 (the veto zone). Note the dissected mapping is
  discontinuous at ratio == 1.0 (1.0 from below, ~0.8 just above); the
  discontinuity is preserved deliberately — this module records the
  harvested design as-is so trials measure BreakFree's actual rule,
  and the quirk is logged in the negative-results registry.

Deliberate deviations from the source, both forced by this repo's
validation discipline:

- BreakFree's veto machine is stateful and keyed on wall-clock
  timestamps (``timedelta(days=3)`` — a Friday veto is eaten by the
  weekend). Here the whole overlay is PRECOMPUTED as a pure series:
  each date's state depends only on vol bars up to that date, so a
  signal wrapper can look states up by date, stays a pure function of
  ``(symbol, bars)``, and passes ``validation.check_lookahead``
  truncation unchanged. Cooldown is counted in bars (3 bars flat),
  not calendar days.
- Missing or warmup data means NEUTRAL (scale 1.0, no veto), matching
  BreakFree's insufficient-data behavior — but non-finite or
  non-positive vol closes are refused with ValueError rather than
  silently neutralized. No fabricated values, ever.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import replace
from datetime import date
from typing import Callable, Mapping, NamedTuple, Sequence

from .data import Bar
from .signals import Signal

REGIME_LOOKBACK = 20
VETO_RATIO = 1.4
COOLDOWN_BARS = 3


class OverlayState(NamedTuple):
    """The overlay's verdict for one trading day."""

    ratio: float | None  # vix_close / SMA(vix_close, lookback); None in warmup
    scale: float  # long-exposure multiplier in [0.0, 1.0]
    vetoed: bool  # True while the spike veto (incl. cooldown) forces flat


def vix_ratios(closes: Sequence[float], lookback: int = REGIME_LOOKBACK) -> list[float | None]:
    """Per-bar vol ratio series, aligned with ``closes``.

    Entry ``i`` is ``closes[i] / mean(closes[i - lookback + 1 : i + 1])``,
    or None until ``lookback`` bars exist. Refuses non-finite or
    non-positive closes: a vol proxy that reports those is broken data,
    and broken data must stop the run, not quietly scale a portfolio.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    out: list[float | None] = []
    window_sum = 0.0
    for i, close in enumerate(closes):
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"vol close at index {i} must be positive and finite, got {close}")
        window_sum += close
        if i >= lookback:
            window_sum -= closes[i - lookback]
        if i < lookback - 1:
            out.append(None)
        else:
            mean = window_sum / lookback
            out.append(close / mean)
    return out


def regime_scale(ratio: float | None, veto_ratio: float = VETO_RATIO) -> float:
    """Long-exposure multiplier for one day's ratio (veto handled separately).

    Faithful to the dissected mapping, discontinuity at 1.0 included.
    """
    if ratio is None:
        return 1.0
    if not math.isfinite(ratio):
        raise ValueError(f"ratio must be finite or None, got {ratio}")
    if ratio <= 1.0:
        return 1.0
    if ratio < veto_ratio:
        # Linear 0.8 -> 0.2 across (1.0, veto_ratio), floored at 0.2.
        return max(0.2, 0.8 - ((ratio - 1.0) / (veto_ratio - 1.0)) * 0.6)
    return 0.0


def overlay_states(
    ratios: Sequence[float | None],
    *,
    veto_ratio: float = VETO_RATIO,
    cooldown_bars: int = COOLDOWN_BARS,
) -> list[OverlayState]:
    """Run the veto state machine over a ratio series — a pure function.

    A ratio above ``veto_ratio`` on a live bar latches the veto for
    that bar and the next ``cooldown_bars - 1`` bars; while latched,
    ratios are not re-evaluated (BreakFree returns from cooldown before
    running its strategies, so a continuing spike does NOT extend the
    cooldown — it re-triggers only once the latch expires).
    """
    if cooldown_bars < 1:
        raise ValueError(f"cooldown_bars must be >= 1, got {cooldown_bars}")
    states: list[OverlayState] = []
    latch = 0  # bars of forced flat remaining, including the current bar
    for ratio in ratios:
        if ratio is not None and not math.isfinite(ratio):
            raise ValueError(f"ratio must be finite or None, got {ratio}")
        if latch > 0:
            states.append(OverlayState(ratio=ratio, scale=0.0, vetoed=True))
            latch -= 1
            continue
        if ratio is not None and ratio > veto_ratio:
            states.append(OverlayState(ratio=ratio, scale=0.0, vetoed=True))
            latch = cooldown_bars - 1
            continue
        states.append(OverlayState(ratio=ratio, scale=regime_scale(ratio, veto_ratio), vetoed=False))
    return states


def overlay_by_date(
    vol_bars: Sequence[Bar],
    *,
    lookback: int = REGIME_LOOKBACK,
    veto_ratio: float = VETO_RATIO,
    cooldown_bars: int = COOLDOWN_BARS,
) -> dict[date, OverlayState]:
    """Date -> OverlayState for a vol-proxy bar series (e.g. VIXY).

    Causal by construction: the state stored for a date is computed
    from vol bars up to and including that date only, so truncating
    the vol series never changes an earlier date's state.

    ``vol_bars`` must be in chronological order — the ratio windows
    and the veto latch are sequence concepts, and silently accepting
    shuffled bars would manufacture nonsense states.
    """
    for prev, cur in zip(vol_bars, vol_bars[1:]):
        if cur.date < prev.date:
            raise ValueError("vol_bars must be in chronological order")
    ratios = vix_ratios([b.close for b in vol_bars], lookback)
    states = overlay_states(ratios, veto_ratio=veto_ratio, cooldown_bars=cooldown_bars)
    return {bar.date: state for bar, state in zip(vol_bars, states)}


def apply_overlay(signal: Signal, state: OverlayState) -> Signal:
    """Apply one day's overlay state to a base signal.

    Veto flattens any signal. Otherwise the regime scale multiplies
    BUY target weights only (BreakFree scales the long side only);
    SELL/HOLD signals pass through unscaled. The returned signal's
    reason and evidence always record what the overlay did.
    """
    evidence = dict(signal.evidence)
    evidence["vix_ratio"] = state.ratio
    evidence["vix_scale"] = state.scale
    evidence["vix_veto"] = 1.0 if state.vetoed else 0.0
    if state.vetoed:
        ratio_txt = f"{state.ratio:.2f}" if state.ratio is not None else "n/a"
        return replace(
            signal,
            action="SELL",
            target_weight=0.0,
            reason=f"{signal.reason} | vix veto: ratio {ratio_txt} spike/cooldown; forced flat",
            evidence=evidence,
        )
    if signal.action == "BUY" and state.scale < 1.0:
        return replace(
            signal,
            target_weight=signal.target_weight * state.scale,
            reason=f"{signal.reason} | vix regime scale x{state.scale:.2f} (ratio {state.ratio:.2f})",
            evidence=evidence,
        )
    return replace(signal, evidence=evidence)


def with_vix_overlay(
    signal_fn: Callable[[str, list[Bar]], Signal],
    vol_bars: Sequence[Bar],
    *,
    lookback: int = REGIME_LOOKBACK,
    veto_ratio: float = VETO_RATIO,
    cooldown_bars: int = COOLDOWN_BARS,
) -> Callable[[str, list[Bar]], Signal]:
    """Wrap a ``fn(symbol, bars) -> Signal`` with the vol-regime overlay.

    The wrapper stays a pure function of ``(symbol, bars)``: overlay
    states are precomputed from the vol series alone and looked up by
    the last bar's date (most recent vol date on or before it), so the
    repo's lookahead pre-flight and backtest engine treat it exactly
    like any other signal. Dates before the vol series starts — or an
    empty vol series — leave the base signal untouched.
    """
    states = overlay_by_date(vol_bars, lookback=lookback, veto_ratio=veto_ratio, cooldown_bars=cooldown_bars)
    state_dates = sorted(states)

    def wrapped(symbol: str, bars: list[Bar]) -> Signal:
        base = signal_fn(symbol, bars)
        if not bars or not state_dates:
            return base
        idx = bisect.bisect_right(state_dates, bars[-1].date) - 1
        if idx < 0:
            return base
        return apply_overlay(base, states[state_dates[idx]])

    wrapped.__name__ = f"{getattr(signal_fn, '__name__', 'signal')}+vix_overlay"
    return wrapped


__all__ = [
    "COOLDOWN_BARS",
    "REGIME_LOOKBACK",
    "VETO_RATIO",
    "OverlayState",
    "apply_overlay",
    "overlay_by_date",
    "overlay_states",
    "regime_scale",
    "vix_ratios",
    "with_vix_overlay",
]
