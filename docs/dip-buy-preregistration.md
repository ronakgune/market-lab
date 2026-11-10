# Stream 7 — Quality Dip-Buy: Pre-Registration

Registered 2026-11-10, before any signal code was written and before
any backtest was run. Changing any number in this document after
results exist is a new pre-registration, not an edit.

## Hypothesis

For a company that passes a strict quality gate (profitable,
cash-generative, not over-levered), a price dip **below** its 200-day
moving average — while that average is still rising — is more often a
temporary dislocation than the start of permanent impairment. Buying
the dip in tranches, sized by depth, and holding until the price
reclaims the average should therefore beat buy-and-hold **on a
risk-adjusted basis** (Sharpe, drawdown) on the names that pass the
gate, while giving up some raw CAGR (the sleeve is in cash whenever
price is above the average).

This is Ronak's variant of the 200-day rule: the moving average is
only the entry timer. The quality gate does the work — it is the
"terms and conditions." Without the gate, below-the-200 is a
value-trap magnet (multi-year residents like GE and Intel are the
canonical counterexamples).

## The signal (price rule)

Computed per symbol from daily closes (Alpaca SIP, adjustment=all,
the provenanced `daily_data` cache). SMA = simple moving average.

- **Warmup:** fewer than 260 bars → HOLD, insufficient history
  (200 for the average + 60 for its slope).
- **Definitions at bar t:** `sma200` = SMA of closes over the last
  200 bars. `slope` = sma200(t) / sma200(t−60) − 1. `depth` =
  (sma200 − close) / sma200 (positive when price is below the line).
  A **dip episode** is the run of bars since the most recent bar with
  close ≥ sma200. `episode_depth` = the maximum depth reached inside
  the current episode (0 when the last bar is above the line).
- **Entry / standing target weight** (total weight for the name):
  - close < sma200, slope > 0, and episode_depth ≥ **3%** → tranche 1,
    target weight **1/3**
  - episode_depth ≥ **7%** → tranche 2, target weight **2/3**
  - episode_depth ≥ **12%** → tranche 3, target weight **1.00**
  Weight is set by the *deepest* depth of the current episode, so the
  ladder only steps up as a dip deepens; it never steps down
  mid-episode.
- **Exit:** close ≥ sma200 (reclaim) → SELL, target weight 0. The
  episode ends; the next dip starts a fresh ladder.
- **Slope is a continuing condition, not just an entry condition:**
  the sleeve only holds while sma200 is rising. If slope ≤ 0 while
  price is below the line (the "intact long-term uptrend" premise has
  failed), the signal is HOLD with target weight 0 — under the
  weighted harness, HOLD targets 0 exactly like SELL, so a slope
  break flattens the position at the next open.
- **No stop-loss, no time stop.** The tranche ladder plus the quality
  gate is the risk design. Adding a stop later is a different stream.
- Every signal carries evidence: `sma200`, `slope`, `depth_below`
  (current), `episode_depth`, `tranche` (0–3).
- The signal is a pure function of `(symbol, bars)`: causal by
  construction and verified against `check_lookahead`.

## The quality gate (the load-bearing piece)

`quality_as_of(symbol, date)` from SEC EDGAR company facts
(`market_lab/fundamentals.py`). **Point-in-time strictly by the
fact's `filed` date**: a query as of date D may only use facts with
`filed` ≤ D (and period end ≤ D). Restated values enter only when
the restatement was filed.

A name passes the gate as of D only if **all** of these hold, using
the most recent data filed by D:

1. **Profitability:** trailing net income / latest stockholders'
   equity (ROE) ≥ **10%**. Trailing net income = the most recent
   annual (fiscal-year) `NetIncomeLoss`; if no annual fact exists,
   the sum of the 4 most recent quarterly facts. Non-positive equity
   makes ROE undefined → gate fails.
2. **Cash generation:** free cash flow (operating cash flow −
   capital expenditures, annual facts matched by period end) is
   **positive in the latest fiscal year** AND **positive in at
   least 3 of the last 5 fiscal years**. A fiscal year with no
   matched annual OCF/capex pair counts as not positive. Fewer than
   3 fiscal years of matched data → gate fails (insufficient
   history).
3. **Leverage:** total liabilities / total assets ≤ **0.85** at the
   latest balance sheet filed by D. (The cap is deliberately
   generous: buyback-heavy quality names carry high *accounting*
   leverage — e.g. Apple's liabilities/assets has printed ~0.80–0.84
   in recent years. The binding screens are 1 and 2; this cap exists
   to exclude genuinely debt-funded balance sheets, not to relitigate
   capital-return policy.)

Missing concepts, stale filings, or any inability to evaluate a leg
→ the gate **fails with the reason recorded**. The gate never passes
on absent data.

The gated sleeve applies the price rule only when the gate passes as
of the signal date. Gate membership is therefore time-varying across
a backtest — that is intended, and it is why the gate must be
point-in-time honest.

## Universe

Candidate universe = the single-name operating companies in the
price cache: **AAPL, AMD, AMZN, AVGO, COIN, GOOGL, META, MSFT, MSTR,
NVDA, SMCI, TSLA** (12 names). Excluded, with reasons — never
silently:

- **SPY, QQQ** — index ETFs; they are this stream's benchmarks, and
  an ETF has no company fundamentals to gate on.
- **DIA, IWM** — index ETFs, same reason.
- **VIXY** — volatility futures ETP; no operating company.
- **IBIT, ETHA** — spot crypto ETPs (trusts); no company facts.
- **GLD, USO** — commodity trusts/pools; no company facts.
- **TLT** — bond ETF; no company facts.

Names enter evaluation when they have ≥260 bars of price history
(e.g. COIN only begins in 2021); there is no backfilling a name into
years before it traded.

## Evaluation (phase 2 — not part of this build)

- Harness: the repo's weighted backtest runner (the Stream 1 trend
  runner), per name standalone, **$25,000** starting cash per name,
  next-open fills, 5 bps slippage + $1 commission — identical
  conditions to trials #1–8.
- Windows: **full** (cache start 2016-11-07 → latest) and
  **later-half** (the runner's midpoint convention), per name.
- Benchmarks, same window, same starting cash: buy-and-hold of the
  same name, and buy-and-hold SPY.
- Reporting aggregates are **medians across the candidate names**
  (per-name results also reported in full).
- Every configuration run is recorded in the program-wide trial
  ledger — including failures. No configuration is re-run with
  tweaked parameters under this pre-registration.

## Survival bars (fixed now, before results)

The stream **survives** only if ALL of these hold on the full-period
medians across candidate names:

1. Median Sharpe ≥ median buy-and-hold Sharpe (same names).
2. Median max drawdown at least **25% shallower** than median
   buy-and-hold max drawdown.
3. Median CAGR ≥ **60%** of median buy-and-hold CAGR.

**Kill criteria** — any one kills the stream as designed:

1. Median Sharpe < buy-and-hold median Sharpe − 0.10.
2. Median max-drawdown improvement < 15% vs buy-and-hold.
3. Median CAGR < 50% of buy-and-hold.
4. Inertness: median entries per name over the full window < 2.
5. Falling-knife signature: in ≥ 1/3 of names, the strategy's max
   drawdown is > 10 percentage points *worse* than that name's
   buy-and-hold max drawdown while its CAGR also trails.

Later-half check: if kill criterion 1 or 2 triggers on the
later-half medians alone, the stream does not survive even if the
full-period bars pass.

## Caveats (read before believing any phase-2 number)

- **Survivorship is load-bearing.** The 12-name candidate universe
  is the 2026 cache — companies that survived and mattered enough to
  be cached. Every name in it is a known winner or a known survivor.
  Backtest results under this pre-registration are an **upper
  bound**, exactly like XS momentum trials #9–10. Belief requires
  the forward paper tracker (the honest measurement), which is a
  later phase and its own decision.
- EDGAR company facts are as-filed data: coverage is strong from
  ~2010 on; fiscal-year matching depends on filers tagging annual
  OCF and capex under the standard concepts. Where tagging differs,
  the gate fails closed (recorded), which biases the backtest
  conservatively — acceptable, and preferable to imputing data.
- The price cache starts 2016-11-07, so "full period" is one decade
  containing two sharp V-recoveries (2020, 2022) that flatter dip
  buying. The later-half check exists partly for this reason.

## What phase 1 builds (this change)

- This pre-registration.
- `market_lab/fundamentals.py` — EDGAR fetch + cache, point-in-time
  quality gate exactly as specified above.
- `market_lab/dip_buy.py` — the price signal exactly as specified
  above, plus the quality-gated wrapper.
- Baseline tests for both, including point-in-time correctness and
  causality. **No backtests, no trial-ledger entries in phase 1.**
