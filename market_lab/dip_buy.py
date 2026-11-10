"""Stream 7 — quality dip-buy signal.

Implements the price rule registered in
docs/dip-buy-preregistration.md, and nothing else:

- Buy a dip below the 200-day SMA **only while the SMA itself is
  rising** (slope vs 60 bars ago > 0) — a dip inside an intact
  long-term uptrend, not a falling knife.
- Target weight is a ladder on the *deepest* depth of the current
  dip episode (an episode runs from the last bar that closed at or
  above the SMA): >= 3% below -> 1/3, >= 7% -> 2/3, >= 12% -> 1.0.
  The ladder only steps up as a dip deepens; it never steps down
  mid-episode.
- Exit to flat when the close reclaims the SMA (close >= SMA200).
- The slope is a continuing condition: if it turns non-positive
  while price is below the line, the signal stands aside (HOLD,
  target weight 0 — under the repo's weighted harness HOLD targets
  0 exactly like SELL, so a slope break flattens the position).
- No stop-loss, no time stop (registered design; the ladder plus
  the quality gate is the risk control).

The signal is a pure function of (symbol, bars) — causal by
construction. The quality gate lives in fundamentals.py; combine
the two with :func:`with_quality_gate`, which evaluates the gate
as of the last bar's date (point-in-time by filed date), exactly
like the vol-regime overlay wraps a base signal elsewhere in this
repo.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .data import Bar
from .fundamentals import QualityReport, quality_as_of
from .indicators import rolling_volatility, rsi, sma
from .signals import Signal

STRATEGY = "dip_buy"

SMA_DAYS = 200
SLOPE_LOOKBACK = 60
MIN_BARS = SMA_DAYS + SLOPE_LOOKBACK  # 260

# (minimum episode depth, target weight, confidence), deepest first.
TRANCHES: tuple[tuple[float, float, float], ...] = (
    (0.12, 1.0, 0.70),
    (0.07, 2.0 / 3.0, 0.55),
    (0.03, 1.0 / 3.0, 0.40),
)


def _tranche_for_depth(episode_depth: float) -> tuple[int, float, float]:
    """(tranche number 0-3, target weight, confidence) for a depth."""
    for idx, (threshold, weight, confidence) in enumerate(TRANCHES):
        if episode_depth >= threshold:
            return len(TRANCHES) - idx, weight, confidence
    return 0, 0.0, 0.0


def generate_dip_buy_signal(symbol: str, bars: list[Bar]) -> Signal:
    if len(bars) < MIN_BARS:
        close = bars[-1].close if bars else 0.0
        return Signal(symbol, "HOLD", 0.0,
                      f"{STRATEGY}: insufficient history; need {MIN_BARS} bars",
                      close, None, None, None, None, STRATEGY)
    closes = [b.close for b in bars]
    close = closes[-1]
    sma200_series = sma(closes, SMA_DAYS)
    s200 = sma200_series[-1]
    s200_prev = sma200_series[-1 - SLOPE_LOOKBACK]
    assert s200 is not None and s200_prev is not None  # guaranteed by MIN_BARS
    slope = s200 / s200_prev - 1.0 if s200_prev > 0 else 0.0
    depth_below = (s200 - close) / s200 if s200 > 0 else 0.0

    # Current episode: bars since the most recent close >= SMA200.
    # episode_depth is the deepest dip inside it (0 when above).
    episode_depth = 0.0
    for i in range(len(closes) - 1, -1, -1):
        line = sma200_series[i]
        if line is None or closes[i] >= line:
            break
        episode_depth = max(episode_depth, (line - closes[i]) / line)

    tranche, weight, confidence = _tranche_for_depth(episode_depth)
    s20 = sma(closes, 20)[-1]
    s50 = sma(closes, 50)[-1]
    r14 = rsi(closes, 14)[-1]
    vol20 = rolling_volatility(closes, 20)[-1]
    evidence: dict[str, float | str | None] = {
        "sma200": s200,
        "slope": slope,
        "depth_below": depth_below,
        "episode_depth": episode_depth,
        "tranche": tranche,
    }

    if close >= s200:
        return Signal(symbol, "SELL", 0.5,
                      f"{STRATEGY}: close {close:.2f} at/above SMA200 {s200:.2f}; "
                      "dip episode over, exit to flat",
                      close, r14, s20, s50, vol20, STRATEGY, 0.0, evidence)
    if slope <= 0:
        return Signal(symbol, "HOLD", 0.0,
                      f"{STRATEGY}: below SMA200 but the average is not rising "
                      f"(slope {slope:.2%}); uptrend premise failed, stand aside",
                      close, r14, s20, s50, vol20, STRATEGY, 0.0, evidence)
    if tranche == 0:
        return Signal(symbol, "HOLD", 0.0,
                      f"{STRATEGY}: dip depth {depth_below:.1%} has not reached "
                      f"the first tranche ({TRANCHES[-1][0]:.0%} below SMA200)",
                      close, r14, s20, s50, vol20, STRATEGY, 0.0, evidence)
    return Signal(symbol, "BUY", confidence,
                  f"{STRATEGY}: tranche {tranche} — episode depth "
                  f"{episode_depth:.1%} below a rising SMA200 "
                  f"(slope {slope:.2%}); target weight {weight:.2f}",
                  close, r14, s20, s50, vol20, STRATEGY, weight, evidence)


def with_quality_gate(
    signal_fn: Callable[[str, list[Bar]], Signal],
    quality_fn: Callable[[str, object], QualityReport] = quality_as_of,
) -> Callable[[str, list[Bar]], Signal]:
    """Wrap a signal so BUYs fire only when the quality gate passes.

    The gate is evaluated as of the last bar's date; fundamentals
    are point-in-time by filed date, so the wrapper stays causal.
    Non-BUY signals pass through untouched (the gate never blocks
    an exit). A blocked BUY becomes HOLD at weight 0 with the
    gate's reasons in the signal reason and evidence.
    """

    def gated(symbol: str, bars: list[Bar]) -> Signal:
        base = signal_fn(symbol, bars)
        if base.action != "BUY" or not bars:
            return base
        report = quality_fn(symbol, bars[-1].date)
        evidence = dict(base.evidence)
        evidence["quality"] = "pass" if report.passes else "fail"
        if report.reasons:
            evidence["quality_reasons"] = "; ".join(report.reasons)
        if report.passes:
            return replace(base, evidence=evidence)
        return Signal(
            symbol, "HOLD", 0.0,
            f"{base.strategy}: BUY blocked by the quality gate: "
            + "; ".join(report.reasons),
            base.close, base.rsi14, base.sma20, base.sma50, base.volatility20,
            base.strategy, 0.0, evidence,
        )

    gated.__name__ = f"{signal_fn.__name__}+quality_gate"
    return gated
