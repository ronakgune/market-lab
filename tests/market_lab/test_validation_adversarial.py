"""Adversarial tests for market_lab.validation (hygiene layer).

The baseline suite proves the analyzers catch the classic leaks and that an
honest indicator passes. This suite tries to BREAK the hygiene layer itself:
subtle leaks the default cutoff grid can step over, non-finite values used
as camouflage, stateful functions that poison the checker, ledger files
that have been gapped / forged / torn by a crash, and Sharpe math pushed
into regions where the formulas stop meaning anything.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from market_lab.data import Bar
from market_lab.validation import (
    TrialLedger,
    check_lookahead,
    check_recursive,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
)


def make_bars(n: int, start: float = 100.0) -> list[Bar]:
    bars = []
    price = start
    day = date(2020, 1, 2)
    for i in range(n):
        price *= 1.0 + 0.001 * math.sin(i / 7.0) + (0.002 if i % 11 == 0 else 0.0)
        bars.append(
            Bar(date=day + timedelta(days=i), open=price, high=price * 1.01,
                low=price * 0.99, close=price, volume=1000 + i)
        )
    return bars


def flat_stretch_bars(n: int = 120) -> list[Bar]:
    """Bars with runs of identical consecutive closes (flat stretches)."""
    bars = []
    day = date(2020, 1, 2)
    price = 100.0
    for i in range(n):
        if i % 10 >= 3:  # move on 3 of every 10 bars, flat otherwise
            price *= 1.01
        bars.append(
            Bar(date=day + timedelta(days=i), open=price, high=price,
                low=price, close=price, volume=1000 + i)
        )
    return bars


# ===========================================================================
# 1. Lookahead evasion
# ===========================================================================

def global_zscore(bars):
    """LEAK: every value is normalised by the FULL-series mean/std."""
    closes = [b.close for b in bars]
    mean = sum(closes) / len(closes)
    var = sum((c - mean) ** 2 for c in closes) / len(closes)
    std = math.sqrt(var) or 1.0
    return [(c - mean) / std for c in closes]


def test_global_statistics_leak_is_caught():
    report = check_lookahead(global_zscore, make_bars(120))
    assert not report.passed
    assert report.violations


def single_index_peek(target_index: int, peek_index: int):
    """LEAK confined to ONE index: its value reads a later bar (clipped)."""
    def _fn(bars):
        out = []
        for i, b in enumerate(bars):
            if i == target_index and len(bars) > target_index:
                out.append(bars[min(peek_index, len(bars) - 1)].close)
            else:
                out.append(b.close)
        return out
    return _fn


def test_single_index_peek_between_old_grid_points_is_caught():
    # n=240: the ORIGINAL default grid stepped by 240//20 = 12 (cutoffs
    # 12, 24, ..., 228, 239). A peek at index 127 reading bar 131 is only
    # exposed by cutoffs in (127, 131] — the old grid has none (132 is the
    # first cutoff past it, and by then bar 131 is inside the prefix, so
    # the values agree). The exhaustive default grid must catch it.
    fn = single_index_peek(127, 131)
    report = check_lookahead(fn, make_bars(240))
    assert not report.passed
    assert any(v.index == 127 for v in report.violations)


def test_single_index_peek_caught_with_explicit_exhaustive_cutoffs():
    # Control: even where the default grid samples, the checker itself is
    # sound when the caller supplies every cutoff.
    fn = single_index_peek(605, 609)
    bars = make_bars(1200)
    report = check_lookahead(fn, bars, cutoffs=range(1, len(bars)))
    assert not report.passed
    assert any(v.index == 605 for v in report.violations)


def test_sampled_grid_residual_for_very_long_series():
    # Documents the residual limitation stated in check_lookahead's
    # docstring: for n > 512 the default grid samples ~n//100 cutoffs
    # (step 12 at n=1200). A 4-bar peek at index 605 is exposed only by
    # cutoffs in (605, 609]; the sampled grid (multiples of 12, plus 1199)
    # contains none. If this test ever fails, the grid got denser — update
    # the docstring too.
    fn = single_index_peek(605, 609)
    report = check_lookahead(fn, make_bars(1200))
    assert report.passed  # known residual: sampling, not soundness


def make_stateful_global_mean():
    """LEAK via state: the first call's full-history mean is cached and
    reused for every later call, so truncated reruns agree with the full
    run — the checker poisons its own control group."""
    cache: dict[str, float] = {}

    def _fn(bars):
        if "mean" not in cache:
            cache["mean"] = sum(b.close for b in bars) / len(bars)
        m = cache["mean"]
        return [b.close - m for b in bars]

    return _fn


@pytest.mark.xfail(
    reason="stateful first-call caching evades rerun comparison: the "
           "analyzer's contract assumes a pure SeriesFn but cannot verify "
           "purity black-box; documented limitation, not yet defended",
    strict=False,
)
def test_stateful_cached_global_statistic_is_caught():
    report = check_lookahead(make_stateful_global_mean(), make_bars(120))
    assert not report.passed


def nan_inf_series(bars):
    """Honest but non-finite: NaN where the close is unchanged from the
    prior bar (a 0/0-style degenerate), +inf on every 50th bar. Both are
    determined by data at or before each bar — no future information."""
    out = []
    for i, b in enumerate(bars):
        if i > 0 and b.close == bars[i - 1].close:
            out.append(float("nan"))
        elif i % 50 == 0:
            out.append(float("inf"))
        else:
            out.append(b.close)
    return out


def test_nan_and_inf_at_data_determined_positions_pass_lookahead():
    # Identical NaN in both runs is the same non-value, not a leak; same
    # for identical +inf. (Pre-fix, nan != nan made this a false violation.)
    report = check_lookahead(nan_inf_series, flat_stretch_bars(120))
    assert report.passed, report.violations


def test_peek_disguised_as_nan_is_still_caught():
    # A leak that HIDES behind NaN: honest bars get NaN at flat spots, but
    # the value at each flat spot secretly reads the NEXT bar's close when
    # one exists. NaN-alignment must not become a smuggling route.
    def _fn(bars):
        out = []
        for i, b in enumerate(bars):
            if i > 0 and b.close == bars[i - 1].close:
                if i + 1 < len(bars):
                    out.append(bars[i + 1].close)  # LEAK dressed as data
                else:
                    out.append(float("nan"))
            else:
                out.append(b.close)
        return out

    report = check_lookahead(_fn, flat_stretch_bars(120))
    assert not report.passed


def test_wrong_length_on_truncated_run_is_rejected():
    def bad(bars):
        if len(bars) < 200:
            return [0.0] * (len(bars) + 1)  # lies only on prefixes
        return [0.0] * len(bars)

    with pytest.raises(ValueError):
        check_lookahead(bad, make_bars(200))


def test_out_of_range_explicit_cutoffs_are_tolerated():
    def honest(bars):
        return [b.close for b in bars]

    report = check_lookahead(honest, make_bars(60), cutoffs=[0, -5, 10 ** 6, 50, 50])
    assert report.passed
    assert report.cutoffs_checked == 1


# ===========================================================================
# 2. Recursive analyzer evasion
# ===========================================================================

def reappearing_deviation(bars):
    """Cold-start error decays exponentially — then REAPPEARS at j == 77
    for any run that starts at original index >= 30 (marked by volume).
    An analyzer that stops at the first settle would under-report warmup;
    required_warmup must be driven by the LAST bad index."""
    start_idx = bars[0].volume - 1000
    out = []
    for j, b in enumerate(bars):
        err = 8.0 * math.exp(-j / 4.0)
        if start_idx >= 30 and j == 77:
            err += 25.0
        out.append(b.close + err)
    return out


def test_reappearing_deviation_extends_required_warmup():
    report = check_recursive(reappearing_deviation, make_bars(300), tol=1e-6)
    assert report.required_warmup >= 78  # the j==77 spike, not the early decay
    assert report.max_deviation >= 25.0


def cumsum_series(bars):
    """Infinite memory: cumulative sum of close changes. A cold start can
    NEVER reproduce full-history values (the missing head sum is a
    permanent offset), so warmup must report 'not settled in-window' —
    the largest observable candidate — never a small comfortable number."""
    out = []
    total = 0.0
    prev = None
    for b in bars:
        if prev is not None:
            total += b.close - prev
        prev = b.close
        out.append(total)
    return out


def test_infinite_memory_indicator_never_reports_settled():
    bars = make_bars(200)
    report = check_recursive(cumsum_series, bars, tol=1e-6)
    # smallest default offset is 20, so the longest shifted run is 180 bars
    # and every index of it deviates -> warmup candidate 180.
    assert report.required_warmup >= 150
    assert report.max_deviation > 1.0


def fixed_position_none(bars):
    """Returns None at position 25 of WHATEVER input it is given, values
    elsewhere. On shifted runs the None lands on a different absolute bar
    than the full run's None: value-vs-None mismatches in both directions,
    which must count as unsettled, never as agreement."""
    return [None if j == 25 else b.close for j, b in enumerate(bars)]


def test_none_value_mismatch_counts_as_unsettled():
    report = check_recursive(fixed_position_none, make_bars(200), tol=1e-9)
    assert report.max_deviation == math.inf
    assert report.required_warmup >= 26


def nan_only_series(bars):
    """NaN exactly where the data is flat (close unchanged) — a property
    of the DATA, so the NaNs sit at the same absolute bars in every run."""
    out = []
    for i, b in enumerate(bars):
        if i > 0 and b.close == bars[i - 1].close:
            out.append(float("nan"))
        else:
            out.append(b.close)
    return out


def test_data_aligned_nans_settle_immediately_in_recursive():
    # NaN at the same ABSOLUTE bars in full and shifted runs (flat spots
    # are a property of the data) is agreement, not deviation. Pre-fix,
    # nan-vs-nan scored as infinite deviation at every one of the ~84 flat
    # bars and warmup ballooned. The ONLY legitimate disagreement is each
    # shifted run's first bar: the function cannot know that bar was flat
    # (its predecessor is outside its input), so it emits the close where
    # the full run has NaN — a true 1-bar boundary effect.
    report = check_recursive(nan_only_series, flat_stretch_bars(120), tol=1e-9)
    assert report.required_warmup == 1


# ===========================================================================
# 3. TrialLedger integrity
# ===========================================================================

def _seed_line(trial: int, stream: str = "seeded") -> str:
    return json.dumps({
        "trial": trial, "ts": "2026-01-01T00:00:00+00:00", "stream": stream,
        "strategy": "forged", "run_id": f"seed{trial}", "params": {},
        "params_hash": "0" * 64, "dataset": {}, "metrics": {"sharpe": 0.1},
        "notes": "",
    }, sort_keys=True)


def _record(ledger: TrialLedger, **kw):
    args = dict(stream="s", strategy="x", params={}, dataset={}, metrics={"a": 1.0})
    args.update(kw)
    return ledger.record(**args)


def test_two_ledgers_same_path_alternate_cleanly(tmp_path):
    path = tmp_path / "trials.jsonl"
    a, b = TrialLedger(path), TrialLedger(path)
    trials = [_record(a).trial, _record(b).trial, _record(a).trial, _record(b).trial]
    assert trials == [1, 2, 3, 4]
    assert a.count() == 4
    assert len({r.run_id for r in a.records()}) == 4


def test_seeded_gap_and_duplicate_numbers_do_not_renumber_backwards(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join([_seed_line(1), _seed_line(7), _seed_line(7)]) + "\n")
    ledger = TrialLedger(path)
    nxt = _record(ledger)
    # Numbering continues from the MAXIMUM present (8), never backwards
    # into the gap and never onto an existing number.
    assert nxt.trial == 8
    assert ledger.count() == 4


def test_deleted_line_does_not_cause_duplicate_trial_number(tmp_path):
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    for _ in range(3):
        _record(ledger)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")  # delete trial 2
    nxt = _record(ledger)
    # Count-based numbering would reissue 3, duplicating the surviving
    # trial 3. Max-based numbering must give 4.
    assert nxt.trial == 4
    assert [r.trial for r in ledger.records()] == [1, 3, 4]


def test_torn_last_line_does_not_brick_the_ledger(tmp_path):
    path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(path)
    _record(ledger)
    _record(ledger)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"trial": 3, "ts": "2026-11-06T23:')  # crash mid-append
    assert ledger.count() == 2
    assert len(ledger.records()) == 2
    nxt = _record(ledger)  # recording still works after the torn write
    assert nxt.trial == 3
    assert ledger.count() == 3


def test_corrupt_middle_line_is_loud_not_silent(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text("\n".join([_seed_line(1), "NOT JSON AT ALL", _seed_line(2)]) + "\n")
    ledger = TrialLedger(path)
    # A torn TAIL is tolerated; corruption in the middle means history was
    # tampered with or the file is not a ledger — that must raise, because
    # silently skipping it would understate the trial count (the exact
    # failure this ledger exists to prevent).
    with pytest.raises(json.JSONDecodeError):
        ledger.count()


def test_nested_params_hash_is_key_order_stable(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    p1 = {"cfg": {"levels": [1, {"beta": 2, "alpha": 1}], "name": "x"}, "top_n": 3}
    p2 = {"top_n": 3, "cfg": {"name": "x", "levels": [1, {"alpha": 1, "beta": 2}]}}
    r1 = _record(ledger, params=p1)
    r2 = _record(ledger, params=p2)
    assert r1.params_hash == r2.params_hash
    p3 = {"top_n": 3, "cfg": {"name": "x", "levels": [1, {"alpha": 1, "beta": 3}]}}
    r3 = _record(ledger, params=p3)
    assert r3.params_hash != r1.params_hash  # a real nested change IS seen


def test_unicode_and_huge_int_params_roundtrip(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    params = {"label": "café ☕ ünïcode", "big": 10 ** 30, "neg": -(10 ** 25)}
    _record(ledger, params=params)
    back = ledger.records()[0]
    assert back.params == params  # exact ints, exact unicode
    again = _record(ledger, params=dict(params))
    assert again.params_hash == back.params_hash


def test_int_metrics_are_stored_as_floats(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    rec = _record(ledger, metrics={"trades": 100, "sharpe": 1})
    assert rec.metrics == {"trades": 100.0, "sharpe": 1.0}
    assert ledger.records()[0].metrics == {"trades": 100.0, "sharpe": 1.0}


# ===========================================================================
# 4. Sharpe math edges
# ===========================================================================

def test_deflated_sharpe_rejects_nonpositive_trial_counts():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.05, n_trials=0, n_obs=1000)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.05, n_trials=-3, n_obs=1000)
    with pytest.raises(ValueError):
        expected_max_sharpe(0)


def test_deflated_sharpe_extreme_moments_stay_a_probability():
    # skew=10, sr=0.5 makes the BLP variance factor negative; the module
    # clamps the denominator rather than taking sqrt of a negative. The
    # result must remain a probability — no crash, no NaN, no >1.
    for skew, kurt in [(10.0, 3.0), (-10.0, 30.0), (0.0, 100.0), (50.0, 3.0)]:
        p = deflated_sharpe_ratio(0.5, n_trials=10, n_obs=500, skew=skew, kurtosis=kurt)
        assert 0.0 <= p <= 1.0


def test_deflated_sharpe_huge_observed_sr_saturates_not_crashes():
    p = deflated_sharpe_ratio(5.0, n_trials=2, n_obs=100)
    assert p == pytest.approx(1.0)
    p0 = deflated_sharpe_ratio(-5.0, n_trials=2, n_obs=100)
    assert p0 == pytest.approx(0.0)


def test_expected_max_sharpe_monotone_to_100k_trials():
    ns = [1, 2, 10, 100, 1_000, 10_000, 100_000]
    vals = [expected_max_sharpe(n) for n in ns]
    assert vals[0] == 0.0
    assert all(math.isfinite(v) for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_min_track_record_sub_unit_kurtosis_sane_for_small_sr():
    # Kurtosis slightly below the Gaussian 3 (even below 1) with small SR
    # and zero skew keeps the variance factor positive and the answer sane.
    days = min_track_record_length(0.02, skew=0.0, kurtosis=0.5)
    assert math.isfinite(days)
    assert days >= 1.0


def test_min_track_record_impossible_moments_raise():
    # skew=3 with kurtosis=1 is mathematically impossible for any real
    # return series (kurtosis >= 1 + skew^2 = 10); at sr=2 the variance
    # factor goes negative and an unguarded formula returns a NEGATIVE
    # required track record. It must refuse instead.
    with pytest.raises(ValueError):
        min_track_record_length(2.0, skew=3.0, kurtosis=1.0)


# ===========================================================================
# 5. Controls — honest strategies must still pass
# ===========================================================================

def macd_series(fast: int = 12, slow: int = 26):
    def _ema(values, span):
        alpha = 2.0 / (span + 1.0)
        out, v = [], None
        for x in values:
            v = x if v is None else alpha * x + (1 - alpha) * v
            out.append(v)
        return out

    def _fn(bars):
        closes = [b.close for b in bars]
        f, s = _ema(closes, fast), _ema(closes, slow)
        return [a - b for a, b in zip(f, s)]

    return _fn


def test_honest_macd_passes_lookahead():
    report = check_lookahead(macd_series(), make_bars(200))
    assert report.passed
    assert report.violations == []


def test_slow_causal_ema_gets_finite_correct_warmup():
    def ema50(bars):
        alpha = 2.0 / 51.0
        out, v = [], None
        for b in bars:
            v = b.close if v is None else alpha * b.close + (1 - alpha) * v
            out.append(v)
        return out

    report = check_recursive(ema50, make_bars(600), tol=1e-4)
    assert 0 < report.required_warmup < 300  # settles, but not instantly
