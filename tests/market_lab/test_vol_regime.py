"""Baseline tests for the BreakFree-harvested vol-regime overlay."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from market_lab.data import Bar
from market_lab.signals import Signal
from market_lab.validation import check_lookahead, signal_series
from market_lab.vol_regime import (
    VETO_STALENESS_BARS,
    apply_overlay,
    overlay_by_date,
    overlay_states,
    regime_scale,
    vix_ratios,
    with_vix_overlay,
    OverlayState,
)

DAY = date(2024, 1, 2)


def make_bars(closes: list[float], start: date = DAY) -> list[Bar]:
    return [
        Bar(date=start + timedelta(days=i), open=c, high=c, low=c, close=c, volume=1)
        for i, c in enumerate(closes)
    ]


def make_signal(action: str = "BUY", weight: float = 0.8) -> Signal:
    return Signal(
        symbol="SPY", action=action, confidence=0.6, reason="base",
        close=100.0, rsi14=None, sma20=None, sma50=None, volatility20=None,
        strategy="vt_trend", target_weight=weight, evidence={},
    )


# ------------------------------------------------------------------ ratios

def test_ratios_warmup_and_value():
    closes = [10.0] * 20 + [14.0]
    ratios = vix_ratios(closes, lookback=20)
    assert ratios[:19] == [None] * 19
    assert ratios[19] == pytest.approx(1.0)
    # 21st bar: mean of last 20 = (19*10 + 14)/20 = 10.2; ratio = 14/10.2
    assert ratios[20] == pytest.approx(14.0 / 10.2)


def test_ratios_refuse_bad_data():
    with pytest.raises(ValueError):
        vix_ratios([10.0, float("nan")])
    with pytest.raises(ValueError):
        vix_ratios([10.0, 0.0])
    with pytest.raises(ValueError):
        vix_ratios([10.0, -3.0])
    with pytest.raises(ValueError):
        vix_ratios([10.0], lookback=0)


# ------------------------------------------------------------------ scaling

def test_regime_scale_mapping():
    assert regime_scale(None) == 1.0
    assert regime_scale(0.7) == 1.0
    assert regime_scale(1.0) == 1.0
    # Just above 1.0 the dissected mapping drops to ~0.8 (documented quirk).
    assert regime_scale(1.0001) == pytest.approx(0.8, abs=0.001)
    assert regime_scale(1.2) == pytest.approx(0.5)
    assert regime_scale(1.399) == pytest.approx(0.2, abs=0.01)  # floor
    assert regime_scale(1.4) == 0.0
    assert regime_scale(2.5) == 0.0
    with pytest.raises(ValueError):
        regime_scale(float("inf"))


# ------------------------------------------------------------------ veto machine

def test_veto_latches_three_bars_then_reevaluates():
    ratios = [1.0, 1.5, 1.0, 1.0, 1.0, 1.0]
    states = overlay_states(ratios, cooldown_bars=3)
    assert [s.vetoed for s in states] == [False, True, True, True, False, False]
    assert all(s.scale == 0.0 for s in states if s.vetoed)


def test_veto_does_not_extend_during_cooldown_but_retriggers_after():
    # Spike persists: latch covers bars 0-2; bar 3 re-evaluates, re-triggers,
    # and latches a fresh 3 bars (3-5); bar 6 is finally free.
    ratios = [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0]
    states = overlay_states(ratios, cooldown_bars=3)
    assert [s.vetoed for s in states] == [True, True, True, True, True, True, False]


def test_veto_threshold_is_strictly_above():
    states = overlay_states([1.4, 1.400001])
    assert states[0].vetoed is False and states[0].scale == 0.0  # at threshold: scale 0, no latch
    assert states[1].vetoed is True


def test_overlay_states_refuse_nonfinite_ratio_and_bad_cooldown():
    with pytest.raises(ValueError):
        overlay_states([float("nan")])
    with pytest.raises(ValueError):
        overlay_states([1.0], cooldown_bars=0)


# ------------------------------------------------------------------ apply / wrap

def test_apply_overlay_veto_flattens_any_signal():
    out = apply_overlay(make_signal("BUY", 0.8), OverlayState(1.6, 0.0, True))
    assert out.action == "SELL" and out.target_weight == 0.0
    assert out.evidence["vix_veto"] == 1.0


def test_apply_overlay_scales_buy_only():
    state = OverlayState(1.2, 0.5, False)
    bought = apply_overlay(make_signal("BUY", 0.8), state)
    assert bought.target_weight == pytest.approx(0.4)
    sold = apply_overlay(make_signal("SELL", 0.0), state)
    assert sold.target_weight == 0.0 and "regime scale" not in sold.reason
    held = apply_overlay(make_signal("HOLD", 0.0), OverlayState(0.9, 1.0, False))
    assert held.action == "HOLD"


def test_overlay_by_date_is_causal_under_truncation():
    closes = [10.0] * 20 + [30.0] + [12.0] * 10
    bars = make_bars(closes)
    full = overlay_by_date(bars)
    truncated = overlay_by_date(bars[:25])
    for d, state in truncated.items():
        assert full[d] == state
    # The spike day (index 20, ratio 3.0) and its cooldown are vetoed.
    spike_day = bars[20].date
    assert full[spike_day].vetoed is True
    assert full[bars[23].date].vetoed is False


def test_with_vix_overlay_wrapper_lookup_and_purity():
    closes = [10.0] * 20 + [30.0] + [12.0] * 30
    vix_bars = make_bars(closes)

    def base(symbol: str, bars: list[Bar]) -> Signal:
        return make_signal("BUY", 0.8)

    wrapped = with_vix_overlay(base, vix_bars)
    calm = wrapped("SPY", vix_bars[:20])
    assert calm.target_weight == pytest.approx(0.8)  # warmup ratio None -> neutral
    spike = wrapped("SPY", vix_bars[:21])
    assert spike.action == "SELL" and spike.target_weight == 0.0
    # Purity: same prefix, same answer, regardless of prior calls.
    assert wrapped("SPY", vix_bars[:21]) == spike
    # Dates before the vol series leave the base signal untouched.
    early = [Bar(date=date(2020, 1, 1), open=1, high=1, low=1, close=1, volume=1)]
    assert wrapped("SPY", early).target_weight == pytest.approx(0.8)
    assert wrapped.__name__.endswith("+vix_overlay")


# ------------------------------------------------------- stale-veto guard

def _mid_veto_vol_bars() -> list[Bar]:
    """Vol series that ENDS mid-veto: spike at index 20, cooldown 21-22."""
    bars = make_bars([10.0] * 20 + [30.0] + [12.0] * 2)
    assert overlay_by_date(bars)[bars[-1].date].vetoed is True
    return bars


def _buy(symbol: str, bars: list[Bar]) -> Signal:
    return make_signal("BUY", 0.8)


def test_stale_veto_lapses_after_the_bound():
    vol = _mid_veto_vol_bars()  # last vol bar: index 22, vetoed
    equity = make_bars([100.0] * 38)  # same day grid, runs 15 bars past
    wrapped = with_vix_overlay(_buy, vol)
    # Within the bound the (stale) veto still flattens.
    for i in (22, 23, 31):
        out = wrapped("SPY", equity[: i + 1])
        assert out.action == "SELL" and out.target_weight == 0.0, i
        assert "vix_veto_stale" not in out.evidence
    # Past the bound the veto lapses: the base signal passes through
    # with the stale ratio recorded and the lapse marked in evidence.
    for i in (33, 37):
        out = wrapped("SPY", equity[: i + 1])
        assert out.action == "BUY" and out.target_weight == pytest.approx(0.8), i
        assert out.evidence["vix_veto"] == 0.0
        assert out.evidence["vix_veto_stale"] == 1.0
        assert out.evidence["vix_ratio"] is not None


def test_stale_veto_boundary_exactly_at_the_bound():
    vol = _mid_veto_vol_bars()
    equity = make_bars([100.0] * 34)
    wrapped = with_vix_overlay(_buy, vol)
    assert VETO_STALENESS_BARS == 10
    # Exactly 10 trading days after the last vol observation: still fresh.
    out = wrapped("SPY", equity[: 22 + 10 + 1])
    assert out.target_weight == 0.0 and "vix_veto_stale" not in out.evidence
    # The 11th trading day: lapsed.
    out = wrapped("SPY", equity[: 22 + 11 + 1])
    assert out.target_weight == pytest.approx(0.8)
    assert out.evidence["vix_veto_stale"] == 1.0


def test_stale_non_veto_state_is_not_lapsed():
    # Vol series ends in a scaled (non-veto) regime: ratio ~1.28.
    vol = make_bars([10.0] * 20 + [13.0])
    last = overlay_by_date(vol)[vol[-1].date]
    assert last.vetoed is False and last.scale < 1.0
    equity = make_bars([100.0] * 36)  # 15 bars past the vol end
    wrapped = with_vix_overlay(_buy, vol)
    out = wrapped("SPY", equity)
    # The guard covers the veto only: the stale scale still applies.
    assert out.target_weight == pytest.approx(0.8 * last.scale)
    assert "vix_veto_stale" not in out.evidence


def test_fresh_series_never_marks_staleness():
    closes = [10.0] * 20 + [30.0] + [12.0] * 30
    vol = make_bars(closes)
    wrapped = with_vix_overlay(_buy, vol)
    for i in (20, 21, 25, 50):
        out = wrapped("SPY", vol[: i + 1])
        assert "vix_veto_stale" not in out.evidence
    spike = wrapped("SPY", vol[:21])
    assert spike.action == "SELL" and spike.target_weight == 0.0


def test_stale_veto_custom_bound_and_validation():
    vol = _mid_veto_vol_bars()
    equity = make_bars([100.0] * 30)
    wrapped = with_vix_overlay(_buy, vol, stale_veto_bars=3)
    assert wrapped("SPY", equity[: 22 + 3 + 1]).target_weight == 0.0
    assert wrapped("SPY", equity[: 22 + 4 + 1]).target_weight == pytest.approx(0.8)
    with pytest.raises(ValueError):
        with_vix_overlay(_buy, vol, stale_veto_bars=-1)


def test_stale_veto_guard_is_truncation_invariant():
    vol = _mid_veto_vol_bars()
    equity = make_bars([100.0] * 38)
    wrapped = with_vix_overlay(_buy, vol)
    report = check_lookahead(lambda b: signal_series(wrapped, "SPY")(b), equity)
    assert report.passed, report.violations
    # Non-vacuous: the full series really contains the veto AND the lapse.
    weights = signal_series(wrapped, "SPY")(equity)
    assert weights[32] == 0.0   # 10 days past the vol end: still vetoed
    assert weights[33] == pytest.approx(0.8)  # 11th day: lapsed
