"""Validation hygiene for strategy research (clean-room, program-wide).

Three tools every strategy stream must pass through before its numbers are
believed:

1. ``LookaheadAnalyzer`` — proves a per-bar series function only uses data
   available at each bar. It reruns the function on truncated histories and
   requires every overlapping value to match the full-history run. Any value
   that changes when future bars are removed was computed with future data.

2. ``RecursiveAnalyzer`` — measures indicator warmup. It reruns the function
   on histories that start later and reports where the values converge to the
   full-history values within tolerance. That index is the honest warmup: a
   backtest that trades before it is trading on unsettled indicator values.

3. ``TrialLedger`` — an append-only JSONL ledger counting EVERY experiment
   run across every stream, with the Bailey / Lopez de Prado deflated Sharpe
   and minimum-track-record math that turns a raw trial count into an honesty
   adjustment. With parallel streams, uncounted trials are how a program
   manufactures a fake winner, so runners are expected to record each run
   here before quoting its metrics.

Series functions have the shape ``fn(bars) -> list[float | None]`` with one
value per bar. ``signal_series`` adapts this repo's prefix-style signal
functions (``fn(symbol, bars) -> Signal``) into that shape using
``target_weight``.

Clean-room implementations of concepts popularised by Freqtrade's
lookahead-analysis / recursive-analysis (GPL — concepts only, no code) and
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .data import Bar

SeriesFn = Callable[[Sequence[Bar]], Sequence[Optional[float]]]

_EULER_GAMMA = 0.5772156649015329
_NORM = statistics.NormalDist()


# ---------------------------------------------------------------------------
# Shared comparison helpers
# ---------------------------------------------------------------------------

def _values_match(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) and math.isnan(b):
        return True  # same non-value in both runs (nan != nan, so check first)
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Lookahead analysis
# ---------------------------------------------------------------------------

@dataclass
class LookaheadViolation:
    cutoff: int          # truncated history length that exposed the leak
    index: int           # bar index whose value changed
    full_value: Optional[float]
    truncated_value: Optional[float]


@dataclass
class LookaheadReport:
    passed: bool
    bars_checked: int
    cutoffs_checked: int
    violations: list[LookaheadViolation] = field(default_factory=list)


def check_lookahead(
    series_fn: SeriesFn,
    bars: Sequence[Bar],
    *,
    tol: float = 1e-9,
    cutoffs: Optional[Iterable[int]] = None,
) -> LookaheadReport:
    """Verify ``series_fn`` never uses future bars.

    Runs the function on the full history, then on truncated prefixes. For
    each prefix of length ``c``, the first ``c`` values must match the full
    run exactly (within ``tol``); a mismatch at index ``i < c`` means the
    value at bar ``i`` depended on bars after ``i``.

    The default cutoff grid is exhaustive (every prefix length) for up to
    512 bars. For longer series it samples ~100 cutoffs: a leak confined to
    a single interior index whose reach is shorter than the sampling step
    can hide between sampled cutoffs, so pass explicit ``cutoffs`` when an
    exhaustive check of a long series is required.
    """
    bars = list(bars)
    n = len(bars)
    if n < 2:
        return LookaheadReport(passed=True, bars_checked=n, cutoffs_checked=0)

    full = list(series_fn(bars))
    if len(full) != n:
        raise ValueError(
            f"series_fn returned {len(full)} values for {n} bars; must be one per bar"
        )

    if cutoffs is None:
        if n <= 512:
            cutoffs = list(range(1, n))
        else:
            step = max(1, n // 100)
            cutoffs = list(range(step, n, step))
            if n - 1 not in cutoffs:
                cutoffs.append(n - 1)
    cutoffs = sorted({c for c in cutoffs if 1 <= c < n})

    violations: list[LookaheadViolation] = []
    for c in cutoffs:
        truncated = list(series_fn(bars[:c]))
        if len(truncated) != c:
            raise ValueError(
                f"series_fn returned {len(truncated)} values for a {c}-bar prefix"
            )
        for i in range(c):
            if not _values_match(full[i], truncated[i], tol):
                violations.append(
                    LookaheadViolation(
                        cutoff=c,
                        index=i,
                        full_value=full[i],
                        truncated_value=truncated[i],
                    )
                )
                break  # one violation per cutoff is enough evidence
    return LookaheadReport(
        passed=not violations,
        bars_checked=n,
        cutoffs_checked=len(cutoffs),
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Recursive (warmup) analysis
# ---------------------------------------------------------------------------

@dataclass
class RecursiveReport:
    required_warmup: int          # bars from a cold start before values settle
    max_deviation: float
    tol: float
    offsets_checked: int


def check_recursive(
    series_fn: SeriesFn,
    bars: Sequence[Bar],
    *,
    tol: float = 1e-6,
    offsets: Optional[Iterable[int]] = None,
) -> RecursiveReport:
    """Measure how many bars ``series_fn`` needs before its values settle.

    For each start offset ``o``, the function is rerun on ``bars[o:]`` and
    compared against the full-history run over the overlapping region. The
    required warmup is the largest, over all offsets, of the last index
    (counted from the cold start) whose deviation exceeds ``tol`` — i.e. how
    much history must be fed before the values can be trusted to ``tol``.
    """
    bars = list(bars)
    n = len(bars)
    if n < 8:
        return RecursiveReport(required_warmup=0, max_deviation=0.0, tol=tol, offsets_checked=0)

    full = list(series_fn(bars))
    if len(full) != n:
        raise ValueError(
            f"series_fn returned {len(full)} values for {n} bars; must be one per bar"
        )

    if offsets is None:
        step = max(1, n // 10)
        offsets = list(range(step, n - 4, step))
    offsets = sorted({o for o in offsets if 1 <= o < n - 1})

    required_warmup = 0
    max_deviation = 0.0
    for o in offsets:
        shifted = list(series_fn(bars[o:]))
        if len(shifted) != n - o:
            raise ValueError(
                f"series_fn returned {len(shifted)} values for a {n - o}-bar suffix"
            )
        last_bad = -1
        for j in range(n - o):
            a, b = full[o + j], shifted[j]
            if a is None or b is None:
                if a is None and b is None:
                    continue
                deviation = math.inf
            elif math.isfinite(a) and math.isfinite(b):
                deviation = abs(a - b)
            elif math.isnan(a) and math.isnan(b):
                deviation = 0.0  # same non-value at the same bar in both runs
            else:
                deviation = 0.0 if a == b else math.inf
            if deviation > max_deviation:
                max_deviation = deviation
            if deviation > tol:
                last_bad = j
        if last_bad >= 0:
            required_warmup = max(required_warmup, last_bad + 1)
    return RecursiveReport(
        required_warmup=required_warmup,
        max_deviation=max_deviation,
        tol=tol,
        offsets_checked=len(offsets),
    )


# ---------------------------------------------------------------------------
# Adapter for this repo's prefix-style signal functions
# ---------------------------------------------------------------------------

def signal_series(signal_fn, symbol: str) -> SeriesFn:
    """Adapt ``fn(symbol, bars) -> Signal`` to a per-bar target-weight series.

    The value at index ``i`` is the ``target_weight`` the signal produces
    when fed exactly ``bars[:i+1]`` — precisely the information available at
    that bar's close. O(n) signal calls; intended for daily bars.
    """

    def _series(bars: Sequence[Bar]) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        prefix: list[Bar] = []
        for bar in bars:
            prefix.append(bar)
            out.append(float(signal_fn(symbol, list(prefix)).target_weight))
        return out

    return _series


# ---------------------------------------------------------------------------
# Trial ledger + deflated Sharpe / minimum track record
# ---------------------------------------------------------------------------

def _canonical_params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_finite_metrics(metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"metric {key!r} is not finite: {value!r}")
        out[str(key)] = v
    return out


@dataclass
class TrialRecord:
    trial: int
    ts: str
    stream: str
    strategy: str
    run_id: str
    params: dict
    params_hash: str
    dataset: dict
    metrics: dict[str, float]
    notes: str = ""


class TrialLedger:
    """Append-only JSONL ledger of every experiment run, program-wide."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- writing -----------------------------------------------------------
    def record(
        self,
        *,
        stream: str,
        strategy: str,
        params: dict,
        dataset: dict,
        metrics: dict,
        notes: str = "",
    ) -> TrialRecord:
        clean_metrics = _require_finite_metrics(metrics)
        params_hash = _canonical_params_hash(params)
        # Number from the largest trial already present, not the line count:
        # if a line was ever removed (or numbers were seeded with gaps), a
        # count-based number would duplicate an existing trial.
        existing_max = max((r.trial for r in self._iter_records()), default=0)
        record = TrialRecord(
            trial=existing_max + 1,
            ts=datetime.now(timezone.utc).isoformat(),
            stream=stream,
            strategy=strategy,
            run_id=uuid.uuid4().hex[:16],
            params=params,
            params_hash=params_hash,
            dataset=dataset,
            metrics=clean_metrics,
            notes=notes,
        )
        line = json.dumps(
            {
                "trial": record.trial,
                "ts": record.ts,
                "stream": record.stream,
                "strategy": record.strategy,
                "run_id": record.run_id,
                "params": record.params,
                "params_hash": record.params_hash,
                "dataset": record.dataset,
                "metrics": record.metrics,
                "notes": record.notes,
            },
            sort_keys=True,
            allow_nan=False,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_append()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def _prepare_append(self) -> None:
        """Make the file safe to append to after a crash mid-write.

        A torn final write leaves a partial line with no trailing newline;
        appending directly would glue the new record onto the fragment and
        both would be lost. A partial tail is dropped; a complete record
        that lost only its newline gets the newline restored.
        """
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        head, _, tail = raw.rpartition(b"\n")
        try:
            json.loads(tail.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.path.write_bytes(head + b"\n" if head else b"")
        else:
            with open(self.path, "ab") as fh:
                fh.write(b"\n")

    # -- reading -----------------------------------------------------------
    def _iter_records(self) -> Iterable[TrialRecord]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh]
        lines = [line for line in lines if line]
        for idx, line in enumerate(lines):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                if idx == len(lines) - 1:
                    break  # torn final write (crash mid-append): ignore the partial tail
                raise
            yield TrialRecord(
                    trial=int(raw["trial"]),
                    ts=raw["ts"],
                    stream=raw["stream"],
                    strategy=raw["strategy"],
                    run_id=raw["run_id"],
                    params=raw["params"],
                    params_hash=raw["params_hash"],
                    dataset=raw["dataset"],
                    metrics={k: float(v) for k, v in raw["metrics"].items()},
                    notes=raw.get("notes", ""),
                )

    def count(self, *, stream: Optional[str] = None, strategy: Optional[str] = None) -> int:
        return sum(
            1
            for r in self._iter_records()
            if (stream is None or r.stream == stream)
            and (strategy is None or r.strategy == strategy)
        )

    def records(self, *, stream: Optional[str] = None, strategy: Optional[str] = None) -> list[TrialRecord]:
        return [
            r
            for r in self._iter_records()
            if (stream is None or r.stream == stream)
            and (strategy is None or r.strategy == strategy)
        ]


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """Expected maximum Sharpe under the null of zero true Sharpe.

    Bailey & Lopez de Prado's benchmark: with ``n_trials`` independent
    trials whose Sharpe estimates have standard deviation ``sr_std``, the
    expected best-of-N Sharpe from pure luck. A winner that does not clear
    this bar has not beaten the multiple-testing process that produced it.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_std: float = 1.0,
) -> float:
    """Probability (0..1) that the true Sharpe exceeds the luck benchmark.

    ``observed_sr`` is per-period (e.g. daily) Sharpe, ``n_obs`` the number
    of return observations behind it. Returns the DSR as a probability:
    values above 0.95 are the conventional 'survives deflation' bar.
    """
    if n_obs < 3:
        raise ValueError("n_obs must be >= 3")
    benchmark = expected_max_sharpe(n_trials, sr_std=sr_std)
    denom = math.sqrt(max(1e-12, 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2))
    z = (observed_sr - benchmark) * math.sqrt(n_obs - 1) / denom
    return _NORM.cdf(z)


def min_track_record_length(
    observed_sr: float,
    *,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed to reject ``benchmark_sr`` at ``confidence``.

    Bailey & Lopez de Prado's MinTRL. This is the math behind 'a Sharpe 0.5
    strategy needs ~16 years of daily data to prove itself': the count is in
    the same periods as ``observed_sr`` (daily SR -> trading days).
    """
    if observed_sr <= benchmark_sr:
        return math.inf
    z = _NORM.inv_cdf(confidence)
    factor = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if factor <= 0.0:
        # The Sharpe's variance factor went non-positive: the moments are
        # mutually impossible (real distributions need kurtosis >= 1 + skew^2)
        # and a negative "required length" would be nonsense.
        raise ValueError(
            "skew/kurtosis imply a non-positive Sharpe variance factor; "
            "the moments are inconsistent (kurtosis must exceed 1 + skew^2)"
        )
    return 1.0 + factor * (z / (observed_sr - benchmark_sr)) ** 2
