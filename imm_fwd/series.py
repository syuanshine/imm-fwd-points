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
from imm_dates import front_pair, pair_at_slot, pair_label

TIDY_COLS = ["obs_date", "ccy", "slot", "near_date", "far_date", "near_pts", "far_pts", "spot"]


def _interp_pts(days_grid: np.ndarray, pts_grid: np.ndarray, target_days: float) -> float:
    """Linear interpolation of forward points in days-to-settlement space.

    Points are ~linear in time for smooth curves; linear interp is the
    market-standard first approximation (turn-of-year kinks are the known
    exception - flagged in the README).
    Extrapolation is clamped to the nearest node.
    """
    order = np.argsort(days_grid)
    return float(np.interp(target_days, days_grid[order], pts_grid[order]))


def build_imm_points_from_curve(curve: pd.DataFrame, ccy: str,
                                slots=(0,)) -> pd.DataFrame:
    """Route B: interpolate a standard-tenor curve to IMM pairs.

    `curve` is the DataProvider.fetch_curve_history output.
    `slots`: which IMM pairs to price each day. (0,) = front pair only
    (default, preserves the rolling-front series). Include 1 to also price
    the deferred pair, which extends each vintage's tracked life from ~91
    to ~180+ days - required for T-120 vintage-path analysis.
    Returns the tidy frame (TIDY_COLS).
    """
    rows = []
    for ts, row in curve.iterrows():
        obs = ts.date()
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
        for slot in slots:
            near, far = pair_at_slot(obs, slot)
            near_pts = _interp_pts(dg, pg, (near - obs).days)
            far_pts = _interp_pts(dg, pg, (far - obs).days)
            rows.append((ts, ccy, slot, near, far, near_pts, far_pts, row["spot"]))
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


def vintage_paths(tidy_all_slots: pd.DataFrame, quarter: str,
                  field: str = "imm_spread_pts", anchor_days: int = 120,
                  mode: str = "change") -> pd.DataFrame:
    """Seasonality frame: each vintage's path over the last `anchor_days`
    calendar days of its life, aligned on days-to-near-IMM.

    Requires `tidy_all_slots` built with slots=(0, 1) - a vintage is only the
    FRONT pair for ~91 days, so anything beyond that is spliced from the
    deferred slot (same near/far legs, priced one quarter earlier).

    `mode` chooses the normalization - this matters more than it looks:
      'change'   : cumulative CHANGE in raw points from the T-anchor level,
                   P(t) - P(anchor). The default and the safe choice.
      'z'        : that change divided by the vintage's OWN daily-change
                   stdev x sqrt(elapsed days) - strips each year's vol level
                   so only the SHAPE of the path is compared.
      'common_z' : change divided by the currency's FULL-SAMPLE daily-change
                   stdev - puts all years in common sigma units while still
                   showing that (e.g.) 2020 moved more than 2017.
      'pct'      : P(t)/P(anchor) - 1. Provided for completeness but UNSAFE
                   for this data - forward-point spreads are a spread, not a
                   price: they sit near zero and change sign (KRW/TWD IMM
                   points do exactly this), which makes percentage returns
                   explode and flip sign. A warning is emitted if the anchor
                   is small or the path crosses zero.

    Returns wide frame: index = days_to_near (descending, anchor -> 0),
    columns = vintage label, values = normalized path.
    """
    df = tidy_all_slots.copy()
    df["_dtn"] = df["days_to_near"]
    df = df[df["quarter_pair"] == quarter]
    df = df[(df["_dtn"] <= anchor_days) & (df["_dtn"] >= 0)]
    # one row per (vintage, day): prefer the front slot where both exist
    df = df.sort_values(["pair", "_dtn", "slot"])
    df = df.drop_duplicates(subset=["pair", "_dtn"], keep="first")

    out = {}
    for pair, g in df.groupby("pair"):
        g = g.sort_values("_dtn", ascending=False)      # anchor first
        s = g.set_index("_dtn")[field].dropna()
        if len(s) < 20:
            continue
        anchor = s.iloc[0]
        chg = s - anchor
        if mode == "change":
            out[pair] = chg
        elif mode == "pct":
            if abs(anchor) < 1e-9 or (s.min() < 0 < s.max()):
                import warnings
                warnings.warn("{} {}: percentage path is unreliable (anchor near "
                              "zero or path crosses zero)".format(pair, field))
            out[pair] = s / anchor - 1.0
        elif mode == "z":
            sd = s.diff().std()
            elapsed = np.abs(s.index.values - s.index.values[0])
            out[pair] = chg / (sd * np.sqrt(np.maximum(elapsed, 1))) if sd else chg * np.nan
        elif mode == "common_z":
            sd = tidy_all_slots.groupby("pair")[field].diff().std()
            elapsed = np.abs(s.index.values - s.index.values[0])
            out[pair] = chg / (sd * np.sqrt(np.maximum(elapsed, 1))) if sd else chg * np.nan
        else:
            raise ValueError("unknown mode: {}".format(mode))
    wide = pd.DataFrame(out)
    order = df.groupby("pair")["near_date"].first().sort_values().index
    wide = wide[[c for c in order if c in wide.columns]]
    return wide.sort_index(ascending=False)


def vintage_path_stats(paths: pd.DataFrame, checkpoints=(90, 60, 30, 10, 0)) -> pd.DataFrame:
    """Cross-vintage summary of the paths at selected days-to-IMM checkpoints:
    median, IQR, share of vintages positive, min/max, and n.

    NOTE ON SAMPLE SIZE: a 10y lookback gives ~10 vintages per quarter-pair.
    These are descriptive statistics, not evidence of a significant seasonal
    effect - read the dispersion, not just the median.
    """
    rows = {}
    for cp in checkpoints:
        if len(paths.index) == 0:
            continue
        nearest = paths.index[np.argmin(np.abs(paths.index.values - cp))]
        r = paths.loc[nearest].dropna()
        rows["T-{}d".format(cp)] = {
            "n": len(r), "median": r.median(),
            "q25": r.quantile(0.25), "q75": r.quantile(0.75),
            "share_up": (r > 0).mean() if len(r) else np.nan,
            "min": r.min(), "max": r.max(),
        }
    return pd.DataFrame(rows).T
