"""Year-end turn premium extraction - market-convention methods.

The market standard descends from Burghardt & Kirshner, "One Good Turn"
(1994, STIR futures) and is used today in production FX forward curve
building (e.g. LSEG turn-impact adjusted curves):

    * The interest differential implied by forward points is assumed
      CONSTANT PER DAY within any forward-forward period between quoted
      tenors that contains no turn.
    * A period containing the turn has the same constant daily accrual
      for its non-turn days PLUS a one-off jump on the turn date(s).
    * The baseline daily accrual for a turn period is taken from the
      NEIGHBOURING turn-free periods (average of preceding and following).

So the turn is measured in forward-forward space, day-count matched -
never by comparing whole tenors of different lengths, which conflates
the turn with curve slope.

Methods implemented (same estimand, different robustness trade-offs):

  A. direct turn quotes        - not computable from standard tenors; the
                                 data contract for it is in turn_data.py.
  B. flanking forward-forward  - the Burghardt-Kirshner/LSEG convention.
  C. jump-dummy curve fit      - regression generalisation: points vs days
                                 with a step at Dec 31; uses ALL tenors, so
                                 it is robust to a single noisy quote.
  D. kick-in tracker           - BIS/CME diagnostic: a fixed tenor's
                                 implied rate as its window rolls across
                                 year-end; shows when and how hard the
                                 premium enters the curve.

All functions work on one observation date's curve snapshot:
    days : ndarray of calendar days from obs_date to each tenor settlement
    pts  : ndarray of forward points in quote units (outright = spot + pts)
and return the jump J in POINTS (quote units). Use jump_to_ann_pct to
express it as an annualized % rate over the turn window.
"""
import datetime as dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def next_year_end(obs: dt.date) -> dt.date:
    return dt.date(obs.year, 12, 31)


def _marginals(days: np.ndarray, pts: np.ndarray):
    """Forward-forward daily point accrual per inter-tenor interval.
    Returns (interval_start_days, interval_end_days, daily_accrual)."""
    order = np.argsort(days)
    d, p = days[order], pts[order]
    dd = np.diff(d)
    return d[:-1], d[1:], np.diff(p) / dd


def turn_jump_flanking(days: np.ndarray, pts: np.ndarray,
                       days_to_turn: float) -> float:
    """Method B - Burghardt-Kirshner / LSEG flanking convention.

    Locate the inter-tenor interval containing the turn date; its total
    point accrual = (non-turn daily accrual x interval days) + jump.
    Non-turn daily accrual is proxied by the mean of the neighbouring
    turn-free intervals (falls back to the single available neighbour).
    Returns the jump in points; NaN when the turn is not bracketed.
    """
    lo, hi, m = _marginals(days, pts)
    inside = (lo < days_to_turn) & (days_to_turn <= hi)
    if not inside.any():
        return np.nan
    i = int(np.argmax(inside))
    neigh = [m[i - 1]] if i > 0 else []
    if i + 1 < len(m):
        neigh.append(m[i + 1])
    if not neigh:
        return np.nan
    base = float(np.mean(neigh))
    return float((m[i] - base) * (hi[i] - lo[i]))


def turn_jump_regression(days: np.ndarray, pts: np.ndarray,
                         days_to_turn: float, min_side: int = 2) -> float:
    """Method C - jump-dummy fit across the whole curve.

    OLS: pts_i = a + b*days_i + J*1(days_i > days_to_turn).
    b captures the (locally linear) carry accrual - the non-flat-curve
    control - and J is the turn jump in points. Requires >= min_side
    tenors strictly on each side of the turn for identification.
    """
    x = np.asarray(days, float)
    y = np.asarray(pts, float)
    dummy = (x > days_to_turn).astype(float)
    n_before, n_after = int((dummy == 0).sum()), int((dummy == 1).sum())
    if n_before < min_side or n_after < min_side:
        return np.nan
    X = np.column_stack([np.ones_like(x), x, dummy])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])


def jump_to_ann_pct(jump_pts: float, spot: float, turn_days: int = 1) -> float:
    """Express a jump in points as an annualized % rate premium over the
    turn window (ACT/360). turn_days: calendar days the turn actually
    covers (1 for a pure Dec31->Jan1 crossing; use the market's actual
    value-date gap, e.g. 4 for Dec 31 falling before a weekend)."""
    return jump_pts / spot * 360.0 / max(turn_days, 1) * 100.0


def turn_jump_series(curve: pd.DataFrame, tenors, tenor_days: dict,
                     method: str = "flanking",
                     min_days_before: int = 0) -> pd.DataFrame:
    """Daily history of the estimated NEXT year-end jump.

    curve: DataProvider.fetch_curve_history-style frame (spot, <tenor>_pts,
    optional <tenor>_days). Returns obs-date-indexed frame with columns
    jump_pts, jump_ann_pct (per turn day), days_to_turn, year (of the Dec 31
    being priced). Estimates are NaN whenever the turn is not identifiable
    from that day's curve (correct behaviour, not an error)."""
    fn = turn_jump_flanking if method == "flanking" else turn_jump_regression
    rows = []
    for ts, row in curve.iterrows():
        obs = ts.date()
        dtt = (next_year_end(obs) - obs).days
        days, pts = [], []
        for t in tenors:
            p = row.get(t + "_pts", np.nan)
            if pd.isna(p):
                continue
            d = row.get(t + "_days", np.nan)
            if pd.isna(d):
                d = tenor_days[t]
            days.append(float(d)); pts.append(float(p))
        if len(days) < 4 or pd.isna(row["spot"]) or dtt < min_days_before:
            continue
        j = fn(np.array(days), np.array(pts), dtt)
        rows.append((ts, j, jump_to_ann_pct(j, row["spot"]), dtt, obs.year))
    out = pd.DataFrame(rows, columns=["obs_date", "jump_pts", "jump_ann_pct",
                                      "days_to_turn", "year"])
    return out.set_index("obs_date")


def kickin_frame(curve: pd.DataFrame, tenor: str, tenor_days: dict,
                 window: Tuple[int, int] = (110, -20)) -> pd.DataFrame:
    """Method D - kick-in tracker. Implied annualized rate of ONE fixed
    tenor, indexed by days-to-year-end, one column per year. The step up
    when obs_date + tenor first crosses Dec 31 is the premium entering;
    the step down after Dec 31 passes out of the window is it leaving."""
    rows = []
    for ts, row in curve.iterrows():
        obs = ts.date()
        p = row.get(tenor + "_pts", np.nan)
        if pd.isna(p) or pd.isna(row["spot"]):
            continue
        d = row.get(tenor + "_days", np.nan)
        if pd.isna(d):
            d = tenor_days[tenor]
        dtt = (next_year_end(obs) - obs).days
        # measure relative to the NEAREST year-end so Jan shows the exit leg
        prev = (dt.date(obs.year - 1, 12, 31) - obs).days
        if abs(prev) < dtt:
            dtt, yr = prev, obs.year - 1
        else:
            yr = obs.year
        if not (window[1] <= dtt <= window[0]):
            continue
        ann = p / row["spot"] * 360.0 / d * 100.0
        rows.append((dtt, yr, ann))
    df = pd.DataFrame(rows, columns=["days_to_ye", "year", "ann_pct"])
    return df.pivot_table(index="days_to_ye", columns="year", values="ann_pct")
