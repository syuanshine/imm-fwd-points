# Asian NDF IMM Forward-Points Analyzer

Analyzes the evolution of **IMM-period forward points** for THB (offshore),
IDR NDF, INR NDF, PHP NDF, TWD NDF and KRW NDF over a 10-year lookback.

## Definitions

* **IMM dates**: 3rd Wednesday of Mar / Jun / Sep / Dec (CME convention).
* **IMM forward points** on observation date *t*: the calendar spread between
  the two nearest IMM dates after *t* — near leg = 1st IMM date, far leg = 2nd.
  E.g. "Sep–Dec IMM" between the Jun and Sep IMM dates. The pair **rolls on
  each IMM date** (Sep–Dec → Dec–Mar on the Sep IMM date).
* **Normalization**: raw points are not comparable across currencies, so
  analytics run on the annualized CIP-implied local-minus-USD yield gap:
  `ann_pct = ((S + far_pts)/(S + near_pts) − 1) × 360/days × 100` (ACT/360).

## Module layout

| File | Purpose |
|---|---|
| `imm_fwd/imm_dates.py` | IMM calendar: 3rd-Wed generator, front pair for any date, labels |
| `imm_fwd/config.py` | Currency universe, NDF ticker roots, points scales (**verify on terminal**) |
| `imm_fwd/data.py` | `DataProvider` interface, `BloombergProvider` **stub to implement**, `SyntheticProvider` demo |
| `imm_fwd/series.py` | Curve→IMM interpolation, rolling front series, roll-adjusted changes, per-contract alignment |
| `imm_fwd/analytics.py` | Z-scores, percentiles, seasonality, year-end turn premium, correlations, realized vol |
| `imm_fwd/charts.py` | All charts (matplotlib, small multiples, fixed ccy→color map) |
| `imm_fwd/events.py` | CB meeting calendar (demo) + event-window move attribution |
| `imm_fwd/run_analysis.py` | Entry point; `--bbg` switches to the Bloomberg provider |
| [`style/COLOR_STYLE_GUIDE.md`](style/COLOR_STYLE_GUIDE.md) | The currency color palette (validated colorblind-safe), theme tokens, and rules — **read before building a Streamlit/web dashboard on this data** |

## === TASK FOR THE DATA-PULLING AGENT ===

Implement **one method**: `BloombergProvider.fetch_curve_history(ccy, start, end)`
in `imm_fwd/data.py`. Return a DataFrame indexed by business date with columns
`spot`, `<tenor>_pts`, and optionally `<tenor>_days` for the tenors in
`config.CURVE_TENORS`. Points must be in **quote units** (outright = spot + pts),
i.e. raw Bloomberg points divided by the verified `FWD_SCALE`.

Guidance (verify all tickers via ALLQ/FRD on your terminal):
* Spot: `USDKRW Curncy` etc. (`config.CcyConfig.spot_ticker`)
* NDF points: typically `<root>+<tenor> Curncy` — KWN (KRW), NTN (TWD),
  IRN (INR), IHN (IDR), PPN (PHP), THB offshore forwards for THB.
* Fields: `PX_LAST` daily history; confirm scaling via `FWD_SCALE` / DES.
* Prefer pulling actual settlement dates per tenor (populate `<tenor>_days`);
  otherwise the `TENOR_DAYS` approximation is used.
* Optional Route A: if your source contributes IMM-dated points directly,
  implement `fetch_imm_points_history` instead and interpolation is skipped.

Everything downstream (series construction, analytics, charts) then runs via:

    python imm_fwd/run_analysis.py --bbg

Test the full pipeline first without Bloomberg: `python imm_fwd/run_analysis.py`
(synthetic data; charts land in `output/`).

## Analytics & charts produced

1. **01_rolling_imm.png** — 10y continuous front IMM–IMM series per currency
   (annualized %), small multiples: the core "evolution" view.
2. **02_seasonality.png** — distribution by quarter-pair (Mar–Jun … Dec–Mar);
   Dec–Mar embeds the **year-end funding turn** (classically TWD/KRW).
3. **03_zscores.png** — latest level vs 1y/3y/full history: rich/cheap monitor.
4. **04_corr.png** — cross-currency correlation of weekly roll-adjusted changes.
5. **05_vol.png** — rolling 3m realized vol of the spread (regime detection).
6. **06_evolution_<CCY>.png** — each Sep–Dec (front-quarter) vintage overlaid on
   a days-to-near-IMM axis vs the 10–90% envelope of prior contracts: shows how
   the current pair trades vs history at the same point in its life.
7. **07_deepdive_<CCY>.png** — per-currency one-pager in **raw points** (no
   annualization) for idiosyncratic analysis: 10y raw spread, seasonality,
   rolling vol of daily changes, 21d-change distribution, current vintage vs
   prior-contract envelope, and a snapshot + change-stats table by horizon
   (1d/5d/21d/63d: mean, std, skew, kurtosis, hit rate, worst/best, latest).
   Changes are computed within each named pair only, so roll jumps never
   contaminate the stats. Backing CSVs: `return_stats_*.csv`, `snapshot_*.csv`.
8. **08_vintage_<CCY>.png** — **vintage-path seasonality**: every Sep–Dec (or
   current quarter) vintage's path over its final 120 days, aligned on
   days-to-near-IMM and expressed as *cumulative change in raw points from the
   T-120 anchor*, plus the cross-vintage median and 25–75% band. Backing CSVs:
   `vintage_paths_*.csv`, `vintage_stats_*.csv` (checkpoints at T-90/60/30/10/0).
9. **09_fv_gap.png** — **fair-value gap**: IMM-implied differential minus the
   money-market rate differential over the same window. The residual is the
   basis / flow premium — the tradeable component not explained by policy
   paths. Requires the optional `fetch_rate_diff_history` (USD leg: SOFR IMM
   futures cover the exact same windows; local leg: best available IRS/OIS);
   skipped gracefully per currency when unavailable.
10. **10_vol_by_dtn.png** — stdev of daily point changes bucketed by
    days-to-near-IMM: does the spread get noisier into the roll?
11. **11_turn.png** — year-end turn premium by year: Dec–Mar level minus a
    trailing median of non-turn-quarter levels (`turn_series`).
12. **12_spot_beta.png** — rolling 126d beta of point changes to spot returns
    (points per 1% spot move): hedge-ratio input and flow/stress diagnostic.
13. **Mean-reversion & tail tables** — `mean_reversion.csv` (AR(1) phi,
    half-life, t-stat per currency), `fade_table_*.csv` (distribution of next
    21d change conditional on the starting z-bucket — the licence, or not, for
    fading extremes), `mae_*.csv` (max adverse excursion along each vintage's
    T-120→T-0 path, long and short side — what a stop-loss has to survive),
    and `event_share_*.csv` (share of |move| in CB-meeting windows vs quiet
    days; **bundled calendar is a demo approximation** — production should
    load a real ECO calendar via `events.load_calendar(csv)`).
14. **summary.csv / seasonality_*.csv / correlations.csv** — the tables behind them.

### Why cumulative change, not percentage returns

Forward points are a **spread, not a price**: they sit near zero and change
sign (KRW and TWD IMM points do exactly this). `P_t/P_0 − 1` therefore explodes
when the anchor is small and flips sign when the path crosses zero, so a
"−200% return" can mean a move from −0.1 to +0.1. `vintage_paths(...)` defaults
to `mode="change"` (cumulative change in points) and offers `mode="z"` /
`"common_z"` for vol-normalized comparison. `mode="pct"` exists but warns.

### Tracking a vintage beyond ~91 days

A vintage is only the *front* pair for ~91 days (the gap between IMM dates), so
T-120 analysis requires pricing the **deferred** pair too. `build_imm_points_from_curve`
takes `slots=(0, 1)`; `vintage_paths` splices the two so each contract is followed
for its full 120-day run-up. All front-pair analytics still filter `slot == 0`.

Natural extensions once live data is in: spread-vs-outright regression (beta of
IMM points to spot moves), event studies around central-bank dates, and a
cross-currency RV monitor (pairwise ann_pct differentials with z-scores).

## Caveats

* Linear interpolation of points between curve tenors is the standard first
  approximation, but it **smears the year-end turn** across Dec-spanning
  tenors; for precise Dec-pair work prefer direct IMM/turn quotes (Route A).
* NDF fixings (KRW KFTC18, TWD TAIFX1, INR RBI/FBIL, IDR JISDOR, PHP BAP) fix
  ~2 business days before value; we treat the IMM date as the leg value date.
* THB is offshore/deliverable-restricted rather than a pure NDF — onshore vs
  offshore basis can distort comparisons in stress periods (e.g. BOT measures).
* Bloomberg NDF points history >10y can be patchy for PHP/IDR — check for
  stale prints (repeated values) before trusting vol/correlation numbers.
