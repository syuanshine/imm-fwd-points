# Year-end turn premium — standalone module

Measures the **year-end funding turn** in the forward curves of the six
Asian currencies (THB offshore, IDR/INR/PHP/TWD/KRW NDFs) using
market-convention methods. Standalone from the IMM module: the turn is a
property of each currency's whole forward curve, not of IMM dates.

## 1. What the turn is, and why it's priced

Banks and dealers face **point-in-time regulatory reporting** at Dec 31
(G-SIB scores, leverage ratio snapshots). They shrink balance sheets into
that date, so cash lent *across* Dec 31 gets scarce and forward points
covering that crossing carry a one-off premium — anticipated and priced by
the market weeks-to-months in advance (BIS; CME). In NDF curves the
measured turn blends the local year-end, the *USD leg's* year-end funding
premium, and NDF-market positioning.

## 2. The market-convention measurement (what this module implements)

The standard descends from **Burghardt & Kirshner, "One Good Turn" (1994)**
and is how production curve builders (e.g. **LSEG turn-impact adjusted FX
forward curves**) treat turns:

> Assume the implied interest differential is constant *per day* within any
> forward-forward period between quoted tenors that contains no turn. A
> period containing the turn has that same daily accrual for its non-turn
> days **plus a one-off jump** on the turn date(s). The non-turn daily
> accrual in the turn period is taken from the **neighbouring turn-free
> periods**.

Key property: everything happens in **forward-forward space, day-count
matched** — whole tenors of different lengths are never compared directly,
so a non-flat curve does not masquerade as a turn. (This replaces the
first-pass IMM-space calculation, whose trailing-median baseline conflated
rate trends with the turn; see §6.)

### Methods in `turn_methods.py`

| | Method | What it does | Strengths / limits |
|---|---|---|---|
| A | **Direct turn quotes** | Dealer-quoted broken-date forward-forwards across Dec 31 | Ground truth; data exists mainly close to year-end — data-agent task, see `turn_data.py` |
| B | **Flanking forward-forward** (`turn_jump_flanking`) | Daily accrual of the Dec-31-containing interval minus the mean of its neighbours, × interval days | The Burghardt–Kirshner/LSEG convention; local, robust to curve slope; uses 3 intervals only |
| C | **Jump-dummy curve fit** (`turn_jump_regression`) | OLS of points on days + step dummy at Dec 31 across the whole curve | Uses all tenors (robust to one noisy quote); assumes locally linear accrual |
| D | **Kick-in tracker** (`kickin_frame`) | A fixed tenor's implied rate as its window rolls over Dec 31 | The BIS/CME diagnostic; shows *when* and *how hard* the premium enters |

Units: jumps are in **points**; `jump_to_ann_pct` converts to an annualized
% per turn day (ACT/360) for cross-year and cross-currency comparison.

## 3. Outputs (demo run: `python turn/run_turn_analysis.py`)

- **Validation table** (console): flanking & regression estimates vs the
  *known* jump embedded in the synthetic data — both recover truth to
  within ~2%. Re-run this check after any change to the extraction math.
- `T1_turn_jump_series.png` — estimated jump through each year's run-up
- `T2_decomposition_KRW.png` — the method made visible: per-interval daily
  accrual with the Dec-31 interval highlighted
- `T3_kickin.png` — 1M implied rate vs days-to-year-end, per year
- `T4_turn_by_year.png` — cross-currency turn premium by year
- `turn_flanking_*.csv`, `turn_regression_*.csv`

## 4. Roadmap to production

**Phase 1 — reuse existing curve pull.** Works as soon as the agent's
`BloombergProvider.fetch_curve_history` exists (`--bbg`). Monthly-tenor
resolution: the jump is identified but diluted 1:interval-days; estimates
sharpest in Q4.

**Phase 2 — data upgrades** (detail in `turn_data.py`): actual `SETTLE_DT`
per tenor (misplacing the turn interval by days is the main error source);
denser short end from October (ON/TN/SW/2W/3W); the year-end **value-date
gap** per year for correct annualization (Dec 31 on a Friday ≈ 3-4 days).

**Phase 3 — Method A calibration.** Where dealers quote the turn outright,
store those quotes and score B/C against them (exactly how LSEG calibrates
against FXall prints).

**Phase 4 — attribution.** Split local vs USD-leg turn: estimate the USD
year-end turn from SOFR futures (Dec contract vs neighbours — same
Burghardt–Kirshner method, rates space) and subtract from the NDF-curve
turn. Requires the rates leg already stubbed in `fetch_rate_diff_history`.

**Phase 5 — other turns** (the same machinery, different dummy date):
quarter-ends; **India's fiscal year-end (Mar 31)** — FBIL explicitly builds
its INR forward premia curve around both calendar and financial-year turns,
so a Mar 31 dummy for INR is a first-class citizen, not an extension;
Lunar New Year for TWD/KRW/THB short dates.

## 5. Caveats

- Monthly grid = diluted estimate between Oct-Dec observation windows;
  don't read the summer-time estimates as precise (wide identification
  intervals). The `days_to_turn` column is there to filter.
- Interpolated curve sources (including our Route-B IMM interpolation)
  *smear* the turn — always run turn extraction on the **quoted tenor
  curve**, never on already-interpolated points. This module does.
- NDF turn = local + USD + positioning blend until Phase 4 attribution.
- Synthetic demo embeds a constant per-currency turn: year-to-year bars
  are flat *by construction*; real data won't be.

## 6. Relation to the IMM module's turn gauge

`imm_fwd/analytics.turn_series` is now a **same-day** comparison (front
Dec-Mar pair minus same-day deferred Mar-Jun pair, both off one curve
snapshot) — the staleness bias of the original trailing-median version is
gone, and the demo chart (11_turn.png) went from noisy random-sign bars to
stable, correctly-ordered levels. It remains a *quick richness gauge* in
IMM space: the two quarters are different future windows, so residual
curve-shape differences still land in it. This module is the measurement
of record for the turn itself.

## Sources

- Burghardt & Kirshner, "One Good Turn" (1994) — via
  [Clarus FT, Year End Turn Rates](https://www.clarusft.com/year-end-turn-rates/)
- [LSEG turn-impact adjusted FX forward curves methodology](https://solutions.lseg.com/building-out-the-turn-impact-adjusted-curves)
- [BIS Quarterly Review (Mar 2019), year-end stress in FX swaps](https://www.bis.org/publ/qtrpdf/r_qt1903t.htm)
- [CME, Understanding year-end effects in the FX swap market](https://www.cmegroup.com/education/articles-and-reports/understanding-and-analyzing-year-end-effects-in-the-fx-swap-market.html)
- [FBIL forward premia curve methodology](https://www.fbil.org.in/uploads/REVISED_ETHODOLOGY_FOR_FORWARD_PREMIA_AND_MIFOR_CURVE_4b25a32468.pdf) (INR financial-year turn)
