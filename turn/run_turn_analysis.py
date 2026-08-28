"""Standalone year-end turn analysis runner.

    python turn/run_turn_analysis.py            # synthetic demo
    python turn/run_turn_analysis.py --bbg      # once BloombergProvider is done

The synthetic demo doubles as a VALIDATION HARNESS: the synthetic generator
embeds a known turn jump (spot x turn% x 10/360 / 100 points, constant per
currency), so the run prints estimated vs true jump per method. Estimates
should land close to truth - that is the check that the extraction math is
right before trusting it on real data.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from turn_data import CURVE_TENORS, TENOR_DAYS, UNIVERSE, BloombergProvider, SyntheticProvider
from turn_methods import kickin_frame, next_year_end, turn_jump_series
import charts_turn

LOOKBACK_YEARS = 10


def main(use_bbg=False, outdir=None):
    outdir = outdir or os.path.join(_HERE, "output")
    os.makedirs(outdir, exist_ok=True)
    end = dt.date.today()
    start = end.replace(year=end.year - LOOKBACK_YEARS)
    provider = BloombergProvider() if use_bbg else SyntheticProvider()

    series_fl, series_rg, kicks = {}, {}, {}
    print("=== Year-end turn extraction (flanking vs regression vs truth) ===")
    print("{:<5} {:>12} {:>12} {:>12}".format("ccy", "flank_med", "regr_med", "truth" if not use_bbg else "-"))
    for cfg in UNIVERSE:
        curve = provider.fetch_curve_history(cfg.code, start, end)
        fl = turn_jump_series(curve, CURVE_TENORS, TENOR_DAYS, method="flanking")
        rg = turn_jump_series(curve, CURVE_TENORS, TENOR_DAYS, method="regression")
        series_fl[cfg.code], series_rg[cfg.code] = fl, rg
        fl.to_csv(os.path.join(outdir, "turn_flanking_{}.csv".format(cfg.code)))
        rg.to_csv(os.path.join(outdir, "turn_regression_{}.csv".format(cfg.code)))
        kicks[cfg.code] = kickin_frame(curve, "1M", TENOR_DAYS)

        # identification window: last 45-5 days before year-end
        w = lambda df: df[(df["days_to_turn"] <= 45) & (df["days_to_turn"] >= 5)]["jump_pts"]
        if use_bbg:
            truth = np.nan
        else:  # embedded synthetic truth: spot x turn% x (10/360)/100 points
            p = SyntheticProvider.PARAMS[cfg.code]
            med_spot = curve["spot"].median()
            truth = med_spot * p[3] * (10.0 / 360.0) / 100.0
        print("{:<5} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            cfg.code, w(fl).median(), w(rg).median(), truth))

    # charts
    files = [
        charts_turn.plot_turn_jump_series(series_fl, os.path.join(outdir, "T1_turn_jump_series.png")),
        charts_turn.plot_kickin(kicks, "1M", os.path.join(outdir, "T3_kickin.png")),
        charts_turn.plot_turn_by_year(series_fl, os.path.join(outdir, "T4_turn_by_year.png")),
    ]
    # decomposition snapshot: KRW, ~30d before the most recent year-end in data
    curve = provider.fetch_curve_history("KRW", start, end)
    snap_date = None
    for ts in reversed(curve.index):
        dtt = (next_year_end(ts.date()) - ts.date()).days
        if 25 <= dtt <= 35:
            snap_date = ts
            break
    if snap_date is not None:
        row = curve.loc[snap_date]
        days = np.array([row.get(t + "_days", TENOR_DAYS[t]) for t in CURVE_TENORS], float)
        pts = np.array([row[t + "_pts"] for t in CURVE_TENORS], float)
        dtt = (next_year_end(snap_date.date()) - snap_date.date()).days
        files.append(charts_turn.plot_curve_decomposition(
            days, pts, dtt, "KRW", snap_date.strftime("%Y-%m-%d"),
            os.path.join(outdir, "T2_decomposition_KRW.png")))

    print("\nSaved {} charts + CSVs to {}/".format(len(files), outdir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbg", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    main(use_bbg=args.bbg, outdir=args.outdir)
