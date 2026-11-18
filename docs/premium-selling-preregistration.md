# Stream 4 — Defined-Risk Premium Selling: Pre-Registration

Registered 2026-11-17, before any backtest was run and before any
trial-ledger entry exists for this stream. Changing any number in
this document after results exist is a new pre-registration, not an
edit. Phase 1 (this change) builds the coverage map, the data layer,
and the strategy machinery only. **No evaluation runs under this
document until it has been reviewed and approved.**

## Hypothesis

Index options systematically price implied volatility above the
realized volatility that follows (the variance risk premium). A
seller who (a) only sells when implied vol is rich relative to its
own recent history, (b) always buys the wings — worst case fixed at
entry, no naked exposure — and (c) manages exits mechanically
should harvest that premium with a positively-skewed-after-costs
return stream on a small book, at the price of negative skew:
frequent small wins, occasional full defined losses. This is the US
form of the hedged option-selling structure Ronak pointed at on
2026-11-17; the long wings are the hedge, and "defined risk" is a
property of the structure, not a claim that losses are small or
rare. One max loss consumes roughly 3–5 typical wins by design.

## Vehicle and data (phase-1 reality)

- **Vehicle: SPY options.** The free dataset contains no index
  options (SPX/SPXW confirmed absent), so SPX/XSP are out for
  phase 1. SPY is American-style; early-assignment risk on short
  legs is real in live trading but cannot occur in an EOD-chain
  backtest and is named here as an unmodelled live-trading risk,
  not silently ignored.
- **Source:** DoltHub community dataset `post-no-preference/options`,
  table `option_chain`, via `market_lab/options_history.py`.
  Community-maintained, no SLA, no accuracy guarantee; provenance
  sidecars make every chain used in evaluation auditable to its
  fetch. The column named `vol` is implied volatility, not volume.
- **This data has no open interest and no volume.** There is
  therefore NO liquidity screen in this pre-registration — not
  because liquidity doesn't matter, but because it is unobservable
  here. Any live deployment decision requires a data source that
  has it (the paid upgrade path below).

## Coverage map (measured 2026-11-17, before registration of windows)

Method: one scoped COUNT query per date against the live dataset
(`scripts/probe_options_coverage.py`, resumable, checkpointed) —
every Friday 2019-01-01 → 2026-11-13, plus every weekday inside the
stress windows Feb–Apr 2020, Jan–Nov 2022, Jul–Aug 2024, Mar–Apr
2025: 725 dates mapped (706 probed individually, 19 resolved by the
weekday pattern documented below).

**Result: 501 chain days present of 725 dates mapped** (706 probed
individually; 19 Tue/Thu dates marked absent by pattern, below).
The dataset is a **Mon/Wed/Fri snapshot series**:

- In the two full-weekday censuses (Feb–Apr 2020 and Jan–Nov 2022,
  every weekday probed), chains exist on **Mon 59/61, Wed 60/61,
  Fri 55/59** probed days and on **Tue 0/61, Thu 0/60** — never on
  Tue/Thu. The few Mon/Wed/Fri misses are market holidays plus a
  handful of genuine gaps (11 absent Fridays total in 2020–2026,
  including 2024-08-02 and 2024-08-09 around the vol spike).
- Chain days found per year (probed set): 2019: **1** (a single
  stray, 2019-05-10, of 52 Fridays probed — the series effectively
  does not exist in 2019); 2020: 75; 2021: 51; 2022: 143; 2023: 52;
  2024: 65; 2025: 70; 2026: 44 (through Nov 13). Friday presence
  2021–2026 runs 96–100% per year; the median gap between Friday
  chain days is **7 days** (the max gap, 245 days, is the 2019
  desert).
- Stress windows are all covered: COVID Feb–Apr 2020: **36** chain
  days; 2022 bear Jan–Nov: **138**; Jul–Aug 2024: **22**; Mar–Apr
  2025: **26**.
- Implication of the cadence: the full chain-day census
  2020→2026 is roughly **3 days/week ≈ 1,000+ chain days**. The map
  probed every Friday plus stress-window weekdays; Mon/Wed chain
  days in ordinary weeks are established by the census pattern,
  and phase 2 fetches them on demand through the data layer.
- Method caveat: range/GROUP BY queries against this API silently
  return empty results (verified against per-date ground truth for
  Feb 2020) — every number above comes from per-date queries only.

Evaluation windows, frozen from the map:

- **Full window:** every chain day from **2020-01-10** (first chain
  day of the continuous series) → **2026-11-13** (latest probed).
  The stray 2019-05-10 chain is excluded: one isolated day cannot
  support entries, marks, or the IV gate's trailing history.
- **Stress read (reported separately, never cherry-picked away):**
  the four probed stress windows above.
- Years the map shows as too sparse to support entries are not
  dropped silently: they contribute chain days to the full window
  and the sparsity itself is reported.

## Structures

Both are built **only from strikes present in that day's chain**
(`market_lab/premium_selling.py`). If the registered structure
cannot be formed, construction returns an explicit skip with the
reason; no near-enough substitute is ever taken.

1. **Iron condor** (attempted first on an entry day): short put +
   long put wing below it, short call + long call wing above it,
   all in the registered expiry.
2. **Put credit spread** (attempted if the condor skips): short
   put + long put wing below it, registered expiry.

- **Expiry:** among the day's available expiries, the one whose
  DTE is nearest **37**, subject to **30 ≤ DTE ≤ 45**. No expiry in
  the band → skip (`no expiry with DTE in [30,45]`).
- **Short strikes:** the available strike whose vendor |delta| is
  nearest **0.13**, subject to **0.10 ≤ |delta| ≤ 0.16**, per side.
  No strike in the band → skip.
- **Wings:** the **next available strike beyond** each short
  (nearest strike strictly below the short put / above the short
  call) in the same expiry. No wing strike → skip. Wing width is
  whatever the sampled chain gives — see Deviations.

## Entry rules

- Entries are attempted **only on chain days**, at most **one new
  position per chain day**, condor first, then put spread.
- **IV-rich gate (self-contained in this data):** define the day's
  IV index as the mean vendor IV over the registered expiry's rows
  with **0.10 ≤ |delta| ≤ 0.20**. Entry requires today's index to
  sit at or above the **60th percentile** of the trailing **60
  chain-day** indices (strictly prior chain days), with at least
  **20** prior chain-day observations; otherwise no entry
  (`insufficient IV history` / `IV percentile < 60`).
- **Fill model:** short legs fill at the **bid**, long legs at the
  **ask** — natural sides, never mid-price. Commission **$0.65 per
  contract per leg per side**, charged on entry and on exit.
- **Sizing:** contracts = the largest count whose worst case fits
  the per-position cap: max loss per contract =
  (widest wing width − credit per share) × 100; position max loss
  ≤ **$250** (1% of the $25,000 book). Zero contracts → skip.
  **Aggregate open-risk cap: $1,250** (5% of book) across all open
  positions; an entry that would breach it is not taken.

## Management rules (evaluated on chain days only, in this order)

For each open position, on each chain day ≤ expiration:

1. **Hard exit:** DTE ≤ **7** → exit at this chain's closing cost.
2. **Stop:** closing cost − credit received ≥ **2× credit
   received** → exit.
3. **Take profit:** closing cost ≤ **50% of credit received** →
   exit.

Closing cost is always computed at conservative sides (buy back
shorts at ask, sell longs at bid) plus exit commissions. Realized
PnL = credit − entry commissions − closing cost.

- **Leg missing from today's chain** → no decision is possible;
  the position is held and the event is recorded. No interpolated
  marks, ever.
- **No chain day near a trigger:** triggers fire on the first
  chain day on which they are observed true, at that day's quotes
  — including the first chain day *after* a gap. If no chain day
  occurs in [expiry − 7, expiry], the position rides to expiry
  settlement. This is a known, accepted consequence of the data's
  granularity, and it is exactly what kill criterion 4 measures.
- **Expiry settlement:** a position still open after expiration
  settles at intrinsic value against SPY's close on the expiration
  date from the provenanced daily-bars cache (last prior trading
  day's close if the date itself has no bar). No exit commission
  on settlement.

## The chain-day limitation, stated plainly

All decisions and marks exist only on chain days. Excursions
between chain days — the underlying trading through a short strike
and recovering, a stop level being crossed mid-gap — are
**unobservable in this data** and are not modelled. The registered
conservative supplement is the **excursion check**: using SPY daily
bars, any bar between entry and exit whose low (for short puts) or
high (for short calls) crosses a short strike is recorded as a
breach event and reported alongside results. It changes no price
and forces no exit; it exists so the gap risk is measured and
visible rather than assumed away. A backtest on this data is
therefore an **upper bound** on the strategy's quality, in the
same way the XS-momentum trials were: the machinery is honest, the
observation grid is coarse.

## Survival bars (fixed now, before results)

Computed on the full window, all costs in, equity marked on chain
days. Sharpe is computed from returns between consecutive chain-day
equity snapshots, annualized by √(252 / median chain-day gap in
trading days). Monthly returns use first/last snapshots per
calendar month.

The stream **survives** only if ALL of these hold:

1. Net Sharpe **> 0** after costs across the full evaluation window.
2. Max drawdown ≤ **12%** of the $25,000 book.
3. No single calendar month worse than **−6%**.
4. At least **40 closed trades** in the full window — below that,
   the read does not count in either direction.

**Kill criteria** — any one kills the stream as designed:

1. Net Sharpe ≤ 0, or max drawdown > 12%, or a month worse than −6%.
2. **Stop-execution failure:** ≥ **25%** of closed trades realize
   ≥ **90%** of their position max loss — the signature that exits
   cannot execute at chain-day granularity and the "defined risk"
   is being realized in full instead of managed.
3. Excursion breaches (underlying trading through a short strike
   between chain days) on ≥ **25%** of closed trades — the grid is
   too coarse for the management rules to mean anything.
4. Fewer than 40 closed trades across the full window with no
   realistic prospect of more on this data → the free-data phase
   cannot decide the hypothesis; the stream does not survive *on
   this evidence* (the paid-data decision is then Ronak's, made
   with the coverage map and skip statistics in hand).

No configuration is re-run with tweaked parameters under this
pre-registration. Every phase-2 run is recorded in the program-wide
trial ledger, including failures.

## Deviations the data forced (named, not buried)

1. **SPY instead of SPX/XSP.** No index options exist in the free
   dataset. Consequences: American-style exercise (unmodelled
   here), no 60/40 tax treatment, SPY dividend effects embedded in
   the chains as published.
2. **Sampled strike grid.** ~26 strikes per expiry with spacing
   that widens away from the money (measured: 5–15pt near the
   registered short strikes in 2020 chains; 7–40pt in 2026 chains).
   Wings are therefore far wider than a real SPY chain offers
   (real SPY strikes are 1pt apart), so per-contract max loss is
   far larger and the $250 cap sizes most structures to **zero
   contracts**. Measured on the two embedded real fixtures: on
   2026-11-13 the registered put spread (short 695 / wing 680,
   credit 0.88) has max loss **$1,412/contract → skip**; the condor
   additionally finds **no call strike inside the delta band** in
   the 41-DTE expiry that day. On 2020-03-18 no expiry falls in
   the DTE band at all. **Consequence: on this data the strategy
   will skip most chain days, and the binding constraint is the
   sampled grid, not the strategy.** Phase 2 reports the full skip
   census by reason; if skips starve the trade count (kill
   criterion 4), that is a data verdict, not a strategy verdict.
3. **Chain-day-only observation.** See the limitation section:
   management granularity is days-to-weeks, not daily. Stops are
   evaluated late by construction.
4. **No liquidity dimension.** No open interest, no volume: no
   liquidity screen, no way to detect stale/wide quotes beyond the
   bid/ask as published. Fills at natural sides are the only
   protection, and they are already conservative.
5. **Vendor greeks and IV are taken as published.** Delta-band
   selection and the IV index inherit whatever model the dataset's
   contributor used; there is no independent recomputation in
   phase 1.
6. **Coverage starts in 2020 and runs on a Mon/Wed/Fri cadence.**
   2019 is a desert (1 stray chain day) and is excluded from the
   evaluation window. Within 2020–2026 the cadence is reliable
   (Friday presence 96–100%/yr), so "chain-day granularity" for
   this stream means a decision opportunity roughly every 2–3
   trading days — coarse versus daily management, far denser than
   the weekly-only grid the probe's first returns suggested. The
   map probed Fridays everywhere and all weekdays only inside the
   stress windows; Mon/Wed existence in ordinary weeks rests on
   the two full censuses (Mon 59/61, Wed 60/61, Tue 0/61, Thu
   0/60), and any Mon/Wed date phase 2 actually uses is fetched
   and provenanced individually — a date the pattern predicts but
   the dataset lacks returns NoData and is simply not a chain day.

## Paid upgrade path (phase 2+, Ronak's decision, not assumed)

If the free-data phase shows life — or shows the machinery works
but the data cannot decide — the priced option on record is
ThetaData Standard ($80/mo: SPX included, tick NBBO + greeks back
to 2016), which removes deviations 1, 2, 4 and most of 3/6. That
purchase is a separate decision with this document's skip census
as its evidence base.

## What phase 1 builds (this change)

- This pre-registration.
- `scripts/probe_options_coverage.py` + the coverage map it
  produced, summarized by `scripts/summarize_options_coverage.py`
  into `coverage_spy.csv` (data under `data/market-lab/options/`,
  gitignored like all market data; summary numbers recorded above).
- `market_lab/options_history.py` — DoltHub chain fetch/cache with
  provenance sidecars and explicit NoData results. (A new module:
  the legacy `options_data.py` serves the yfinance snapshot stack
  and is untouched.)
- `market_lab/premium_selling.py` — chain model, structure
  construction with explicit skips, IV gate, management engine,
  settlement, excursion check, book accounting.
- Tests on real probed chain fixtures plus a dense-grid logic
  chain, covering parsing, provenance refusal, NoData,
  construction and skip cases, fill-side conservatism, MTM signs,
  every management trigger, and no-lookahead. **No backtests, no
  trial-ledger entries in phase 1.**
