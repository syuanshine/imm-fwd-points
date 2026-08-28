"""Construction of IMM forward-point series.

Two constructions from the same tidy data:

1. build_imm_points_from_curve : per obs_date, interpolate curve points to
   the front pair's two IMM legs -> tidy frame (Route B).
2. rolling_front_series        : continuous 10y series of the front
   IMM-IMM spread (rolls each IMM date).
3. per_contract_series         : each named pair's history aligned on
   days-to-near-IMM (contract 'life').
"""
import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd

from config import CURVE_TENORS, TENOR_DAYS
from imm_dates import front_pair, pair_label

TIDY_COLS = ["obs_date", "ccy", "near_date", "far_date", "near_pts", "far_pts", "spot"]


def _interp_pts(days_grid: np.ndarray, pts_grid: np.ndarray, target_days: float) -> float:
    """Linear interpolation of forward points in days-to-settlement space.

    Points are ~linear in time for smooth curves; linear interp is the
    market-standard first approximation (turn-of-year kinks are the known
    exception - flagged in the README).
    Extrapolation is clamped to the nearest node.
    """
    order = np.argsort(days_grid)
    return float(np.interp(target_days, days_grid[order], pts_grid[order]))


def build_imm_points_from_curve(curve: pd.DataFrame, ccy: str) -> pd.DataFrame:
    """Route B: interpolate a standard-tenor curve to the front IMM pair.

    `curve` is the DataProvider.fetch_curve_history output.
    Returns the tidy frame (TIDY_COLS).
    """
    rows = []
    for ts, row in curve.iterrows():
        obs = ts.date()
        near, far = front_pair(obs)
        days_grid, pts_grid = [], []
        for tenor in CURVE_TENORS:
            pts = row.get(tenor + "_pts", np.nan)
            if pd.isna(pts):
                continue
            days = row.get(tenor + "_days", np.nan)
            if pd.isna(days):
                days = TENOR_DAYS[tenor]
            days_grid.append(float(days))
            pts_grid.append(float(pts))
        if len(days_grid) < 2 or pd.isna(row["spot"]):
            continue
        dg, pg = np.array(days_grid), np.array(pts_grid)
        near_pts = _interp_pts(dg, pg, (near - obs).days)
        far_pts = _interp_pts(dg, pg, (far - obs).days)
        rows.append((ts, ccy, near, far, near_pts, far_pts, row["spot"]))
    return pd.DataFrame(rows, columns=TIDY_COLS)


def add_derived(tidy: pd.DataFrame) -> pd.DataFrame:
    """Add spread, day counts, annualized implied yield differential, labels.

    ann_pct: CIP-implied local-minus-USD rate gap over the IMM window,
        ((S + far_pts) / (S + near_pts) - 1) * 360/days * 100
    (ACT/360, USD money-market convention). This is the cross-currency
    comparable "price" of the IMM period.
    """
    df = tidy.copy()
    df["imm_spread_pts"] = df["far_pts"] - df["near_pts"]
    df["leg_days"] = (pd.to_datetime(df["far_date"]) - pd.to_datetime(df["near_date"])).dt.days
    df["days_to_near"] = (pd.to_datetime(df["near_date"]) - pd.to_datetime(df["obs_date"])).dt.days
    near_out = df["spot"] + df["near_pts"]
    far_out = df["spot"] + df["far_pts"]
    df["ann_pct"] = (far_out / near_out - 1.0) * 360.0 / df["leg_days"] * 100.0
    df["pair"] = [pair_label(d) for d in df["near_date"]]
    df["quarter_pair"] = [d.strftime("%b") + "-" + f.strftime("%b")
                          for d, f in zip(df["near_date"], df["far_date"])]
    return df


def rolling_front_series(tidy: pd.DataFrame, field: str = "ann_pct") -> pd.Series:
    """Continuous front IMM-IMM series for one currency (rolls each IMM date)."""
    df = tidy.sort_values("obs_date")
    s = df.set_index("obs_date")[field]
    s.name = df["ccy"].iloc[0] if len(df) else field
    return s


def roll_adjusted_changes(tidy: pd.DataFrame, field: str = "ann_pct") -> pd.Series:
    """Daily changes of the rolling series with roll-date jumps removed
    (change set to NaN on the first day of each new pair) - use these for
    correlation / volatility analytics so roll gaps don't pollute them."""
    df = tidy.sort_values("obs_date")
    chg = df[field].diff()
    chg[df["pair"] != df["pair"].shift()] = np.nan
    chg.index = df["obs_date"]
    return chg


def per_contract_series(tidy: pd.DataFrame, field: str = "ann_pct",
                        quarter: str = None) -> pd.DataFrame:
    """Wide frame: index = days_to_near (contract life axis, decreasing to 0),
    columns = pair label in CHRONOLOGICAL order, values = field.
    One currency at a time. `quarter` (e.g. "Sep-Dec") restricts to one
    quarter-pair so vintages are like-for-like (Dec pairs carry the turn)."""
    df = tidy.copy()
    if quarter is not None:
        df = df[df["quarter_pair"] == quarter]
    piv = df.pivot_table(index="days_to_near", columns="pair", values=field)
    order = df.groupby("pair")["near_date"].first().sort_values().index
    return piv[list(order)].sort_index(ascending=False)
