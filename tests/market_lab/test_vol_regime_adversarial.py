"""Adversarial tests for market_lab.vol_regime (BreakFree harvest).

Attack stance: the module docstring is the contract. Where production
behavior matches the docstring — including the DELIBERATE quirks (the
regime_scale discontinuity at ratio == 1.0, the 3-BAR not 3-day
cooldown, neutral-on-warmup) — these tests pin the behavior as-is.
Where behavior contradicts the module's own stated discipline
(purity/causality of the precomputed series, ValueError for broken
data, no fabricated values), the tests assert the contract and are
expected to fail until the hole is fixed; see the pass report.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from statistics import fmean

import pytest

from market_lab.data import Bar
from market_lab.signals import Signal
from market_lab.vol_regime import (
    OverlayState,
    apply_overlay,
    overlay_by_date,
    overlay_states,
    regime_scale,
    vix_ratios,
    with_vix_overlay,
)

DAY = date(2024, 1, 2)


def make_bars(closes: list[float], start: date = DAY) -> list[Bar]:
    return [
        Bar(date=start + timedelta(days=i), open=c, high=c, low=c, close=c, volume=1)
        for i, c in enumerate(closes)
    ]


def make_signal(action: str = "BUY", weight: float = 0.8, evidence=None) -> Signal:
    return Signal(
        symbol="SPY", action=action, confidence=0.6, reason="base",
        close=100.0, rsi14=None, sma20=None, sma50=None, volatility20=None,
        strategy="vt_trend", target_weight=weight, evidence=evidence or {},
    )


def spike_series() -> list[Bar]:
    """20 flat bars, a 3x spike at index 20, then calm 12s."""
    return make_bars([10.0] * 20 + [30.0] + [12.0] * 30)


# ==================================================================== ratios

def test_ratios_match_bruteforce_reference_for_many_lookbacks():
    rng = random.Random(20261107)
    closes = [rng.uniform(5.0, 60.0) for _ in range(80)]
    for lookback in (1, 3, 7, 20, 79, 80, 81):
        expected = [
            None if i < lookback - 1 else closes[i] / fmean(closes[i - lookback + 1 : i + 1])
            for i in range(len(closes))
        ]
        got = vix_ratios(closes, lookback=lookback)
        assert len(got) == len(closes)
        for g, e in zip(got, expected):
            assert g == (None if e is None else pytest.approx(e, rel=1e-9))


def test_ratios_lookback_one_has_no_warmup_and_is_all_ones():
    assert vix_ratios([7.0, 42.0, 3.5], lookback=1) == [1.0, 1.0, 1.0]


def test_ratios_window_actually_slides_old_spike_drops_out():
    closes = [1000.0] + [10.0] * 24
    ratios = vix_ratios(closes, lookback=20)
    # Bar 19's window still contains the bar-0 spike; bar 20's does not.
    assert ratios[19] == pytest.approx(10.0 / ((1000.0 + 19 * 10.0) / 20))
    assert ratios[20] == pytest.approx(1.0)


def test_ratios_empty_series_is_empty():
    assert vix_ratios([]) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_ratios_refuse_bad_close_at_any_position(bad):
    with pytest.raises(ValueError):
        vix_ratios([bad] + [10.0] * 30)
    with pytest.raises(ValueError):
        vix_ratios([10.0] * 30 + [bad])  # late corruption is still corruption


def test_ratios_negative_lookback_refused():
    with pytest.raises(ValueError):
        vix_ratios([10.0] * 5, lookback=-3)


# ==================================================================== scaling

def test_scale_discontinuity_at_one_is_deliberate_and_pinned():
    # DOCUMENTED QUIRK (module docstring): the dissected BreakFree
    # mapping is discontinuous at 1.0 — 1.0 from below, ~0.8 just
    # above. Pinned as-is; trials are meant to measure the real rule.
    assert regime_scale(1.0) == 1.0
    assert regime_scale(1.0 + 1e-9) == pytest.approx(0.8, abs=1e-6)
    assert regime_scale(1.0 - 1e-9) == 1.0


def test_scale_is_monotone_nonincreasing_across_domain():
    xs = [i / 100 for i in range(0, 301)]
    ys = [regime_scale(x) for x in xs]
    assert all(a >= b for a, b in zip(ys, ys[1:]))


def test_scale_floor_and_veto_zone_boundaries():
    assert regime_scale(1.4 - 1e-9) == pytest.approx(0.2, abs=1e-6)
    assert regime_scale(1.4) == 0.0
    assert regime_scale(1.4 + 1e-9) == 0.0


def test_scale_with_custom_veto_ratio_generalizes_linearly():
    assert regime_scale(1.5, veto_ratio=2.0) == pytest.approx(0.5)
    assert regime_scale(2.0, veto_ratio=2.0) == 0.0
    assert regime_scale(1.1, veto_ratio=1.2) == pytest.approx(0.5)


def test_scale_nonpositive_ratio_reads_as_calm():
    # A ratio from vix_ratios can never be <= 0 (closes are refused
    # non-positive upstream); the pure function's <= 1.0 branch treats
    # it as calm. Pinned so a future validation change is a decision.
    assert regime_scale(0.0) == 1.0
    assert regime_scale(-2.5) == 1.0


def test_scale_nan_refused():
    with pytest.raises(ValueError):
        regime_scale(float("nan"))


# ================================================================ veto machine

def test_cooldown_one_means_trigger_bar_only():
    states = overlay_states([1.5, 1.5, 1.0], cooldown_bars=1)
    assert [s.vetoed for s in states] == [True, True, False]


def test_spike_on_the_exact_bar_cooldown_expires_retriggers():
    states = overlay_states([1.5, 1.0, 1.0, 1.5, 1.0], cooldown_bars=3)
    assert [s.vetoed for s in states] == [True, True, True, True, True]


def test_none_ratios_still_burn_cooldown_bars():
    states = overlay_states([1.5, None, None, None], cooldown_bars=3)
    assert [s.vetoed for s in states] == [True, True, True, False]
    assert states[1].ratio is None and states[1].scale == 0.0


def test_at_threshold_bar_sets_no_latch():
    states = overlay_states([1.4, 1.0])
    assert states[0].vetoed is False and states[0].scale == 0.0
    assert states[1].vetoed is False and states[1].scale == 1.0  # bar 2 fully live


def test_latched_state_records_live_ratio_verbatim():
    states = overlay_states([1.5, 1.1, 1.0, 1.0], cooldown_bars=3)
    assert states[1].vetoed is True and states[1].ratio == pytest.approx(1.1)


def test_overlay_states_empty_and_refusals():
    assert overlay_states([]) == []
    with pytest.raises(ValueError):
        overlay_states([float("inf")])
    with pytest.raises(ValueError):
        overlay_states([1.0], cooldown_bars=-2)


# ================================================================ by-date map

def test_by_date_out_of_order_refused():
    bars = make_bars([10.0] * 25)
    bars[10], bars[11] = bars[11], bars[10]
    with pytest.raises(ValueError):
        overlay_by_date(bars)


def test_by_date_empty_and_single_warmup_bar():
    assert overlay_by_date([]) == {}
    bars = make_bars([10.0])
    out = overlay_by_date(bars)
    assert out == {bars[0].date: OverlayState(None, 1.0, False)}


def test_by_date_extension_with_future_spike_changes_nothing_earlier():
    bars = make_bars([10.0] * 25)
    before = overlay_by_date(bars)
    after = overlay_by_date(bars + make_bars([90.0] * 5, start=bars[-1].date + timedelta(days=1)))
    for d, state in before.items():
        assert after[d] == state


def test_by_date_duplicate_dates_refused():
    # HOLE (see report): two bars sharing one date are corrupt daily
    # data. The chronological-order check uses `<`, so duplicates pass
    # silently and the dict keeps whichever state came last — the
    # module's own broken-data discipline says refuse, loudly.
    bars = make_bars([10.0] * 21 + [50.0] + [10.0] * 3)
    bars[21] = Bar(date=bars[20].date, open=50.0, high=50.0, low=50.0, close=50.0, volume=1)
    with pytest.raises(ValueError):
        overlay_by_date(bars)


def test_by_date_truncation_invariance_holds_even_with_duplicate_dates():
    # HOLE (see report): the docstring promises "truncating the vol
    # series never changes an earlier date's state". With a duplicated
    # date, truncating between the pair changes that date's state
    # (vetoed in the full series, calm in the truncated one).
    # RESOLVED by refusal (see the test above): a series containing a
    # duplicate date raises, so no truncation can pick a winner; and
    # the clean prefix before the duplicate is state-identical to the
    # same bars presented as a standalone series.
    bars = make_bars([10.0] * 21 + [50.0] + [10.0] * 3)
    bars[21] = Bar(date=bars[20].date, open=50.0, high=50.0, low=50.0, close=50.0, volume=1)
    with pytest.raises(ValueError):
        overlay_by_date(bars)
    truncated = overlay_by_date(bars[:21])
    clean = overlay_by_date(make_bars([10.0] * 21))
    assert truncated == clean


# ==================================================================== apply

def test_apply_veto_with_none_ratio_does_not_crash():
    out = apply_overlay(make_signal("BUY", 0.8), OverlayState(None, 0.0, True))
    assert out.action == "SELL" and out.target_weight == 0.0
    assert "n/a" in out.reason


def test_apply_scaled_buy_with_none_ratio_refused_cleanly():
    # HOLE, minor (see report): OverlayState(None, 0.5, False) is
    # hand-constructible through the public API. Every other invalid
    # input in this module raises ValueError; this path instead dies
    # with a bare TypeError from an f-string (None formatted as .2f).
    with pytest.raises(ValueError):
        apply_overlay(make_signal("BUY", 0.8), OverlayState(None, 0.5, False))


def test_apply_veto_preserves_identity_fields():
    base = make_signal("HOLD", 0.0)
    out = apply_overlay(base, OverlayState(1.7, 0.0, True))
    assert (out.symbol, out.close, out.strategy) == (base.symbol, base.close, base.strategy)
    assert out.action == "SELL"


def test_apply_sell_and_hold_weights_pass_through_unscaled():
    state = OverlayState(1.2, 0.5, False)
    assert apply_overlay(make_signal("SELL", 0.3), state).target_weight == pytest.approx(0.3)
    assert apply_overlay(make_signal("HOLD", 0.25), state).target_weight == pytest.approx(0.25)


def test_apply_buy_at_threshold_state_zeroes_weight_but_stays_buy():
    out = apply_overlay(make_signal("BUY", 0.8), OverlayState(1.4, 0.0, False))
    assert out.action == "BUY" and out.target_weight == 0.0


def test_apply_scale_one_buy_changes_evidence_only():
    base = make_signal("BUY", 0.8)
    out = apply_overlay(base, OverlayState(0.9, 1.0, False))
    assert out.target_weight == pytest.approx(0.8) and out.reason == base.reason
    assert out.evidence["vix_ratio"] == pytest.approx(0.9)
    assert out.evidence["vix_veto"] == 0.0


def test_apply_overlay_never_mutates_the_base_signal():
    base = make_signal("BUY", 0.8, evidence={"keep": 1.0})
    out = apply_overlay(base, OverlayState(1.2, 0.5, False))
    out.evidence["injected"] = 9.9  # mutating the copy must not leak back
    assert base.evidence == {"keep": 1.0}
    assert out.evidence["keep"] == 1.0  # pre-existing evidence survives


# ==================================================================== wrapper

def test_wrapper_rewrap_with_extended_vol_series_keeps_prefix_answers():
    vix = spike_series()
    base = lambda symbol, bars: make_signal("BUY", 0.8)  # noqa: E731
    short = with_vix_overlay(base, vix)
    extended = with_vix_overlay(base, vix + make_bars([99.0] * 10, start=vix[-1].date + timedelta(days=1)))
    for prefix in (vix[:20], vix[:21], vix[:23], vix[:30]):
        assert extended("SPY", prefix) == short("SPY", prefix)


def test_wrapper_ignores_mutation_of_the_vol_list_after_wrapping():
    vix = spike_series()
    wrapped = with_vix_overlay(lambda s, b: make_signal("BUY", 0.8), vix)
    before = wrapped("SPY", vix[:30])
    vix.append(Bar(date=vix[24].date, open=500.0, high=500.0, low=500.0, close=500.0, volume=1))
    assert wrapped("SPY", vix[:30]) == before  # states were precomputed at wrap time


def test_wrapper_out_of_order_calls_are_consistent():
    vix = spike_series()
    wrapped = with_vix_overlay(lambda s, b: make_signal("BUY", 0.8), vix)
    long_first = wrapped("SPY", vix[:24])
    _ = wrapped("SPY", vix[:5])          # shorter call in between
    _ = wrapped("QQQ", vix[:40])         # different symbol, longer prefix
    assert wrapped("SPY", vix[:24]) == long_first


def test_wrapper_empty_vol_series_returns_base_untouched():
    sentinel = make_signal("BUY", 0.8)
    wrapped = with_vix_overlay(lambda s, b: sentinel, [])
    assert wrapped("SPY", make_bars([1.0])) is sentinel


def test_wrapper_empty_target_bars_returns_base_untouched():
    sentinel = make_signal("HOLD", 0.0)
    wrapped = with_vix_overlay(lambda s, b: sentinel, spike_series())
    assert wrapped("SPY", []) is sentinel


def test_wrapper_refuses_bad_vol_series_at_wrap_time_not_call_time():
    base = lambda symbol, bars: make_signal()  # noqa: E731
    shuffled = spike_series()
    shuffled[3], shuffled[4] = shuffled[4], shuffled[3]
    with pytest.raises(ValueError):
        with_vix_overlay(base, shuffled)
    zeroed = spike_series()
    zeroed[7] = Bar(date=zeroed[7].date, open=0.0, high=0.0, low=0.0, close=0.0, volume=1)
    with pytest.raises(ValueError):
        with_vix_overlay(base, zeroed)


def test_wrapper_uses_most_recent_vol_date_on_or_before_target():
    vix = spike_series()[:21]  # vol series ENDS on the spike bar (index 20)
    wrapped = with_vix_overlay(lambda s, b: make_signal("BUY", 0.8), vix)
    # Target ends 2 days past the last vol date (a gap in the vol
    # series): the spike day's vetoed state is the most recent
    # on-or-before state, so the veto still lands.
    target = make_bars([100.0], start=vix[20].date + timedelta(days=2))
    out = wrapped("SPY", target)
    assert out.action == "SELL" and out.target_weight == 0.0


def test_wrapper_stale_veto_persists_past_vol_series_end_PINNED_SHARP_EDGE():
    # PINNED, flagged in the report: the documented lookup rule (most
    # recent vol state on or before the target date) means a vol
    # series that ENDS mid-veto flattens every later target bar,
    # however far ahead. Faithful to the documented rule; dangerous
    # if a caller ever feeds a truncated vol series.
    vix = spike_series()[:21]  # series ends ON the spike bar
    wrapped = with_vix_overlay(lambda s, b: make_signal("BUY", 0.8), vix)
    target = make_bars([100.0], start=vix[-1].date + timedelta(days=30))
    out = wrapped("SPY", target)
    assert out.action == "SELL" and out.target_weight == 0.0


def test_wrapper_cooldown_bars_resolve_by_date():
    vix = spike_series()
    wrapped = with_vix_overlay(lambda s, b: make_signal("BUY", 0.8), vix)
    third_cooldown = wrapped("SPY", vix[:23])  # indices 20,21,22 vetoed
    assert third_cooldown.action == "SELL"
    freed = wrapped("SPY", vix[:24])           # index 23: latch expired
    assert freed.action == "BUY" and 0.0 < freed.target_weight < 0.8


# ============================================== runner wiring (cheap, offline)

def test_runner_overlay_registry_is_wired_to_a_real_base_strategy():
    from scripts.run_trend_backtest import OVERLAY_STRATEGIES, STRATEGIES, VOL_SYMBOL

    assert OVERLAY_STRATEGIES == {"vt_trend_vix": "vt_trend"}
    assert OVERLAY_STRATEGIES["vt_trend_vix"] in STRATEGIES
    assert "vt_trend_vix" not in STRATEGIES  # overlay name must not shadow a base
    assert VOL_SYMBOL == "VIXY"


def test_runner_style_wrapping_of_real_vt_trend():
    from market_lab.signals import generate_vt_trend_signal

    vix = spike_series()
    wrapped = with_vix_overlay(generate_vt_trend_signal, vix)
    target = make_bars([100.0] * 30)  # too short for vt_trend: base is HOLD
    out = wrapped("SPY", target[:0] + make_bars([100.0] * 30, start=DAY))
    assert out.action in ("HOLD", "SELL")  # sanity: no crash on the real signal fn
    spiked = with_vix_overlay(generate_vt_trend_signal, vix)("SPY", make_bars([100.0] * 30, start=vix[20].date - timedelta(days=29)))
    assert spiked.action == "SELL" and spiked.target_weight == 0.0  # veto lands on spike date
    calm = with_vix_overlay(generate_vt_trend_signal, [])("SPY", make_bars([100.0] * 30))
    assert calm.action == "HOLD"  # empty vol series: base untouched
