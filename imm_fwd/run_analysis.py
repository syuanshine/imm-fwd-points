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
from series import add_derived, build_imm_points_from_curve, per_contract_series
from analytics import (correlation_matrix, horizon_return_stats,
                       seasonality_by_quarter, single_ccy_snapshot,
                       summary_table, turn_premium)
import charts

LOOKBACK_YEARS = 10


def main(use_bbg: bool = False, outdir: str = "output"):
    os.makedirs(outdir, exist_ok=True)
    end = dt.date.today()
    start = end.replace(year=end.year - LOOKBACK_YEARS)
    provider = BloombergProvider() if use_bbg else SyntheticProvider()

    # ---- build tidy IMM frames per currency -------------------------------
    tidy_by_ccy = {}
    for cfg in UNIVERSE:
        direct = provider.fetch_imm_points_history(cfg.code, start, end)
        if direct is not None:                     # Route A: direct IMM tickers
            tidy = direct.assign(ccy=cfg.code)
        else:                                      # Route B: interpolate curve
            curve = provider.fetch_curve_history(cfg.code, start, end)
            tidy = build_imm_points_from_curve(curve, cfg.code)
        tidy_by_ccy[cfg.code] = add_derived(tidy)
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
