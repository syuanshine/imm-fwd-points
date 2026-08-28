"""Entry point: pull 10y of data, build IMM series, run analytics, save charts.

Usage:
    python run_analysis.py            # synthetic demo data (no Bloomberg needed)
    python run_analysis.py --bbg      # once BloombergProvider is implemented

Handing off to the data-pulling agent: implement
    data.BloombergProvider.fetch_curve_history
(see its docstring for tickers/fields/scaling) - nothing else needs to change.
"""
import argparse
import datetime as dt
import os

import pandas as pd

from config import UNIVERSE
from data import BloombergProvider, SyntheticProvider
from series import (add_derived, build_imm_points_from_curve, per_contract_series,
                    vintage_path_stats, vintage_paths)
from analytics import (ar1_half_life, conditional_changes, correlation_matrix,
                       fair_value_gap, horizon_return_stats, mae_stats, spot_beta,
                       seasonality_by_quarter, single_ccy_snapshot,
                       summary_table, turn_premium, turn_series, vol_by_days_to_imm)
from events import demo_calendar, event_share
import charts

LOOKBACK_YEARS = 10


def main(use_bbg: bool = False, outdir: str = "output"):
    os.makedirs(outdir, exist_ok=True)
    end = dt.date.today()
    start = end.replace(year=end.year - LOOKBACK_YEARS)
    provider = BloombergProvider() if use_bbg else SyntheticProvider()

    # ---- build tidy IMM frames per currency -------------------------------
    tidy_by_ccy = {}
    all_slots_by_ccy = {}
    for cfg in UNIVERSE:
        direct = provider.fetch_imm_points_history(cfg.code, start, end)
        if direct is not None:                     # Route A: direct IMM tickers
            tidy = direct.assign(ccy=cfg.code)
        else:                                      # Route B: interpolate curve
            curve = provider.fetch_curve_history(cfg.code, start, end)
            # slots=(0,1): front pair for the rolling series, deferred pair so
            # each vintage can be tracked back beyond ~91 days (T-120 analysis)
            tidy = build_imm_points_from_curve(curve, cfg.code, slots=(0, 1))
        full = add_derived(tidy)
        all_slots_by_ccy[cfg.code] = full
        # every legacy analytic uses the FRONT pair only
        tidy_by_ccy[cfg.code] = full[full["slot"] == 0].reset_index(drop=True)
        tidy_by_ccy[cfg.code].to_csv(os.path.join(outdir, "imm_tidy_{}.csv".format(cfg.code)), index=False)

    # ---- analytics --------------------------------------------------------
    summ = summary_table(tidy_by_ccy)
    summ.to_csv(os.path.join(outdir, "summary.csv"))
    print("\n=== Snapshot: front IMM-IMM annualized implied yield gap (%) ===")
    print(summ.round(2).to_string())

    print("\n=== Year-end turn premium (Dec-Mar mean minus other pairs, ann %) ===")
    for ccy, df in tidy_by_ccy.items():
        print("  {}: {:+.2f}".format(ccy, turn_premium(df)))

    corr = correlation_matrix(tidy_by_ccy)
    corr.to_csv(os.path.join(outdir, "correlations.csv"))

    # ---- charts -----------------------------------------------------------
    files = [
        charts.plot_rolling_panels(tidy_by_ccy, outfile=os.path.join(outdir, "01_rolling_imm.png")),
        charts.plot_seasonality(tidy_by_ccy, outfile=os.path.join(outdir, "02_seasonality.png")),
        charts.plot_zscore_bars(tidy_by_ccy, outfile=os.path.join(outdir, "03_zscores.png")),
        charts.plot_corr_heatmap(corr, outfile=os.path.join(outdir, "04_corr.png")),
        charts.plot_vol_panels(tidy_by_ccy, outfile=os.path.join(outdir, "05_vol.png")),
    ]
    for ccy, df in tidy_by_ccy.items():
        live_quarter = df["quarter_pair"].iloc[-1]     # e.g. "Sep-Dec" today
        piv = per_contract_series(df, quarter=live_quarter)
        files.append(charts.plot_contract_evolution(
            piv, ccy, os.path.join(outdir, "06_evolution_{}.png".format(ccy)),
            quarter=live_quarter))

    # ---- per-currency deep dives (RAW points, idiosyncratic view) ---------
    print("\n=== Per-currency raw-points change stats (1d/5d/21d/63d) ===")
    for ccy, df in tidy_by_ccy.items():
        live_quarter = df["quarter_pair"].iloc[-1]
        piv_pts = per_contract_series(df, field="imm_spread_pts", quarter=live_quarter)
        ret_stats = horizon_return_stats(df, field="imm_spread_pts")
        snap = single_ccy_snapshot(df, field="imm_spread_pts")
        files.append(charts.plot_ccy_deepdive(
            df, piv_pts, ret_stats, snap, ccy,
            os.path.join(outdir, "07_deepdive_{}.png".format(ccy))))
        ret_stats.to_csv(os.path.join(outdir, "return_stats_{}.csv".format(ccy)))
        snap.to_csv(os.path.join(outdir, "snapshot_{}.csv".format(ccy)))
        print("\n  -- {} ({}) --".format(ccy, snap["pair"]))
        print(ret_stats.round(4).to_string())

    # ---- vintage-path seasonality (T-120 -> IMM date, per currency) -------
    print("\n=== Vintage-path seasonality: cumulative change from T-120 (raw points) ===")
    for ccy, df in tidy_by_ccy.items():
        live_quarter = df["quarter_pair"].iloc[-1]
        paths = vintage_paths(all_slots_by_ccy[ccy], live_quarter,
                              field="imm_spread_pts", anchor_days=120, mode="change")
        if paths.empty:
            continue
        vstats = vintage_path_stats(paths)
        vstats.to_csv(os.path.join(outdir, "vintage_stats_{}.csv".format(ccy)))
        paths.to_csv(os.path.join(outdir, "vintage_paths_{}.csv".format(ccy)))
        files.append(charts.plot_vintage_paths(
            paths, vstats, ccy, live_quarter,
            os.path.join(outdir, "08_vintage_{}.png".format(ccy))))
        print("\n  -- {} {} --".format(ccy, live_quarter))
        print(vstats.round(4).to_string())

    # ---- PM analytics: fair value, mean reversion, tails, events, beta ----
    # 1) fair-value gap (skipped per-ccy when no rates leg is available)
    gaps = {}
    for ccy, df in tidy_by_ccy.items():
        rd = provider.fetch_rate_diff_history(ccy, start, end)
        if rd is not None:
            gaps[ccy] = fair_value_gap(df, rd)
            gaps[ccy].to_csv(os.path.join(outdir, "fv_gap_{}.csv".format(ccy)))
    if gaps:
        files.append(charts.plot_fair_value_gap(gaps, os.path.join(outdir, "09_fv_gap.png")))

    # 2) mean reversion: AR(1) half-life + conditional fade table
    print("\n=== Mean reversion (ann_pct): AR(1) half-life ===")
    mr = pd.DataFrame({c: ar1_half_life(df) for c, df in tidy_by_ccy.items()}).T
    mr.to_csv(os.path.join(outdir, "mean_reversion.csv"))
    print(mr.round(3).to_string())
    for ccy, df in tidy_by_ccy.items():
        conditional_changes(df).to_csv(os.path.join(outdir, "fade_table_{}.csv".format(ccy)))
    print("\n=== Conditional 21d change by starting z-bucket (KRW example) ===")
    print(conditional_changes(tidy_by_ccy["KRW"]).round(3).to_string())

    # 3) tails: MAE per vintage + vol vs days-to-IMM
    for ccy, df in tidy_by_ccy.items():
        live_quarter = df["quarter_pair"].iloc[-1]
        paths = vintage_paths(all_slots_by_ccy[ccy], live_quarter,
                              field="imm_spread_pts", anchor_days=120, mode="change")
        if not paths.empty:
            mae_stats(paths).to_csv(os.path.join(outdir, "mae_{}.csv".format(ccy)))
    vols_dtn = {c: vol_by_days_to_imm(df) for c, df in tidy_by_ccy.items()}
    files.append(charts.plot_vol_by_dtn(vols_dtn, os.path.join(outdir, "10_vol_by_dtn.png")))

    # 4) event decomposition + turn extractor
    cal = demo_calendar(start.year, end.year)   # replace with events.load_calendar(csv) in production
    print("\n=== Event-window share of |move| (demo calendar - replace for production) ===")
    for ccy, df in tidy_by_ccy.items():
        es = event_share(df, cal, ccy)
        es.to_csv(os.path.join(outdir, "event_share_{}.csv".format(ccy)))
        print("  {}: all-events share {:.0%} of |move| on {:.0%} of days (intensity {:.2f})".format(
            ccy, es.loc["all_events", "share_of_abs_move"],
            es.loc["all_events", "share_of_days"], es.loc["all_events", "intensity"]))
    turns = {c: turn_series(df) for c, df in tidy_by_ccy.items()}
    for c, ts in turns.items():
        ts.to_csv(os.path.join(outdir, "turn_series_{}.csv".format(c)))
    files.append(charts.plot_turn_series(turns, os.path.join(outdir, "11_turn.png")))

    # 5) spot beta
    betas = {c: spot_beta(df) for c, df in tidy_by_ccy.items()}
    files.append(charts.plot_spot_beta(betas, os.path.join(outdir, "12_spot_beta.png")))

    # per-ccy seasonality tables
    for ccy, df in tidy_by_ccy.items():
        seasonality_by_quarter(df).to_csv(os.path.join(outdir, "seasonality_{}.csv".format(ccy)))

    print("\nSaved {} charts + CSVs to {}/".format(len(files), outdir))
    return tidy_by_ccy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbg", action="store_true", help="use BloombergProvider (must be implemented)")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()
    main(use_bbg=args.bbg, outdir=args.outdir)
