# Negative Results Registry — what doesn't work, and what it taught us

This is the program's memory for failure. Every falsified strategy,
broken implementation, dissected external system, and method trap
gets an entry here with its evidence and the rule it produced.
Positive results live in the trial ledger (`data/market-lab/trials.jsonl`);
this file exists so the negative ones are just as hard to lose —
and so nobody re-spends trials re-learning them.

Rules:
- A falsified strategy is closed. Reopening one requires a *different
  mechanism*, not a re-tuned parameter, and a new pre-registration.
- Every entry names its evidence (trial numbers, report, or commit)
  and ends with the rule going forward.
- External systems are dissected, never ported wholesale
  ("use what is good, discard the rest").

---

## FALSIFIED — tested here, killed by pre-registered rules

### Morning Catalyst / opening-range continuation (Stream 3) — CLOSED 2026-11-06
- **Evidence:** trials #11 (full) / #12 (holdout). 741 real |gap|≥5%
  events, 12 names, 2019–2026, entry at open, +0.5%/−0.5% race to
  12:00 ET, costs 0.28R/trade.
- **Result:** holdout win rate 34.4%, profit factor 0.30, mean
  −0.592R/trade, Sharpe −10.87 vs a survival bar of Sharpe ≥1.0 and
  PF ≥1.3. Gross expectancy was negative *before* costs.
- **Mechanism:** the post-gap morning range dwarfs a 0.5% stop, so the
  stop clips on noise; the "race" is decided by the range, not the gap.
- **Rule:** closed permanently. No retuning the target/stop and
  re-running. (Corroborated by the deep-research replication of the
  ORB family on QQQ: Sharpe −0.06 — the family is not a capital
  strategy; see research report 2026-11-06.)

### Generic model-vs-price edge on prediction markets — DROPPED 2026-11-06
- **Evidence:** deep strategy research (2026-11-06): average Kalshi
  taker −31.46% and maker −9.64% post-fee; no evidence a generic
  model beats venue prices.
- **Rule:** no generic prediction-market strategy. The only surviving
  design is the capped, pre-registered, maker-only favorite-side
  experiment (Stream 5), itself still gated on explicit approval.

## FALSIFIED IN RESEARCH — do not build

### Crypto cross-sectional momentum, long-short — FAILED REPLICATION
- **Evidence:** deep strategy research (2026-11-06): the published
  crypto XS-momentum long-short result failed a pre-registered
  replication.
- **Rule:** do not build it. (Equities XS momentum is a different
  result and is Stream 2 — long-only, vol-scaled, large-cap.)

## DOESN'T WORK AS BUILT — implementation failures, family not falsified

### TSMOM sizing cap (Stream 1) — diagnosed 2026-11-06
- **Evidence:** trials #1–8. SPY TSMOM CAGR 0.41% / Sharpe 0.27 with
  max drawdown only −4.3%: the sleeve was barely invested. Average
  target weight 0.098 (SPY and QQQ); BUY days ~49%.
- **Mechanism:** `signals.py` BUY weight =
  `clamp(target_vol / max(vol20, 0.05), 0.02, 0.20)` — the 0.20 cap
  binds on virtually every BUY day, so the strategy as coded is a
  20%-or-nothing switch, not vol-targeted momentum.
- **Rule:** a flat result from a capped implementation is evidence
  about the cap, not the family. Redesign (raise cap toward
  vt_trend-style sizing) is a design change generating NEW trials —
  pending Ronak's go-ahead; do not fold it into other streams.

### BreakFree TQQQ/SQQQ voting timer — dissected 2026-11-07, not ported
- **Provenance:** ronakgune/BreakFree @ 2f36051 (Ronak's Dec 2025–
  Jan 2026 build). Full evaluation:
  `~/workspace/goals/stock-and-market-analysis-system/files/breakfree-evaluation-2026-11-07.md`.
- **Evidence:** BreakFree's own `docs/StrategyResearch.md` verified
  run on 2023–2025: **−6.53% return, −30.02% max drawdown, 268
  trades** — while its README advertises +185% (5y) / +550% (10y).
- **Mechanism:** daily-rebalanced ensemble whipsaw — ~13% fee drag
  plus slippage over 268 trades; signals computed on TQQQ's own
  decay-distorted price series; fills at the same close the signal
  was computed from.
- **Lessons, each now a standing rule here:**
  1. *Verify claims against the system's own primary artifacts.*
     The README table and the research doc contradicted each other;
     the research doc was the truth.
  2. *Signals on the underlying, trades on the instrument.* SMAs on a
     3x leveraged ETF's price measure the decay, not the trend.
     (market-lab: daily_data provenance + signals computed on the
     traded series only when it IS the underlying.)
  3. *Same-close fills are a lookahead-flavored fiction.* Signals
     fill next-open here, always (`backtest.py`), with a truncation
     pre-flight (`validation.check_lookahead`).
  4. *Silent fallbacks are fabricated data.* BreakFree substitutes
     constant placeholder prices (SQQQ = 100.0, vix_close = 20.0)
     when data is missing, and its Alpaca driver returns $0.00 cash
     on any API error. Here: missing data refuses the run
     (`require_live`, provenanced sidecars); errors raise.
  5. *Absolute vol thresholds break across instruments/regimes.*
     BreakFree's hardcoded VIX>30 veto, fed VIXY (mean ~$47), pinned
     it in cash through a bull run — the "VIXY trap." Vol signals
     here are ratios to their own baseline (`vol_regime.py`).
  6. *Turnover is a cost center with a vote.* A daily timer that
     trades 268 times in 2 years donates its edge to fees. Slow
     sleeves, explicit cost models (5bps + $1), trade counts recorded
     per trial.
  7. *Duplicated implementations drift.* BreakFree carries the same
     strategies/backtest three times. One implementation per concept
     here; shared code lives in `market_lab/`, scripts stay thin.
- **Harvested (the parts that were good):** the adaptive VIX veto +
  cooldown and VIX-regime sizing → `market_lab/vol_regime.py`;
  the ensemble quorum fail-safe (isolate member failures; below
  quorum, go flat instead of trading on partial information) →
  `market_lab/ensemble.py`. Both rebuilt pure/precomputed to pass
  this repo's lookahead discipline.
- **Status:** the leveraged TQQQ sleeve itself remains UNBUILT — it
  has no credible evidence behind it. It earns trials only as a
  fresh pre-registered stream (signals on QQQ, next-open fills, a
  kill rule aimed at the whipsaw mechanism above).

## METHOD TRAPS — results that look better than they are

### Survivorship in the XS-momentum backtest — flagged 2026-11-06
- **Evidence:** trials #9–10 (CAGR 33.51%, Sharpe 1.20) use a
  universe of 2026-known winners (11 names). The numbers are an
  UPPER BOUND, recorded as such in the ledger.
- **Rule:** no capital belief from a survivorship-tainted universe.
  Belief requires a point-in-time universe (paid data decision) or
  forward paper evidence; the forward tracker (live since 2026-11-06)
  is the honest measurement.

### Local-green can be environment-masked — learned 2026-11-07
- **Evidence:** PR #19 CI: a stale test mock (missing a `deadline`
  kwarg added by a security fix) passed locally for weeks because
  sandbox DNS resolves example.com to a restricted address, so the
  code path that calls the mock never ran. GitHub's public DNS ran
  it and failed.
- **Rule:** tests that depend on DNS/network shape are guilty until
  proven hermetic — patch the classifier/transport boundary (as the
  security-fix tests do) instead of relying on ambient DNS. A green
  local suite is not a green CI suite for network-shaped code.
