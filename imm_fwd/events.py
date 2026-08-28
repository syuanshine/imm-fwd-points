"""Central-bank meeting calendar + event-window decomposition.

A PM pricing an IMM window cares which events sit inside it. This module
attributes each vintage's movement to event windows (meeting day +/- 1bd)
vs quiet days.

THE BUNDLED CALENDAR IS A DEMO APPROXIMATION (synthetic pipeline only).
For production, the data-pulling agent should export the real calendar
(Bloomberg ECO / central-bank websites) to CSV with columns
    event_date,label,ccy        # ccy='USD' for FOMC, else local ccy code
and load it with load_calendar(path).
"""
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

# rough FOMC rhythm (8/yr) - DEMO ONLY, dates are approximate
_FOMC_MD = [(1, 29), (3, 19), (4, 30), (6, 12), (7, 30), (9, 17), (11, 6), (12, 17)]
# local CBs approximated as monthly, 2nd Thursday - DEMO ONLY
_LOCAL_CCYS = ["THB", "IDR", "INR", "PHP", "TWD", "KRW"]


def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(days=7 * (n - 1))


def demo_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    """Approximate meeting calendar for demo runs. NOT for production."""
    rows = []
    for y in range(start_year, end_year + 1):
        for m, d in _FOMC_MD:
            rows.append((dt.date(y, m, d), "FOMC(demo)", "USD"))
        for month in range(1, 13):
            for ccy in _LOCAL_CCYS:
                rows.append((_nth_weekday(y, month, 3, 2), "CB(demo)", ccy))
    df = pd.DataFrame(rows, columns=["event_date", "label", "ccy"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def load_calendar(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["event_date"])
    return df[["event_date", "label", "ccy"]]


def event_share(tidy: pd.DataFrame, calendar: pd.DataFrame, ccy: str,
                field: str = "imm_spread_pts", window_bd: int = 1) -> pd.DataFrame:
    """Share of total |daily change| that occurred in event windows
    (meeting day +/- window_bd business days), split by event type,
    overall and per quarter-pair.

    Columns: share_of_abs_move, share_of_days, intensity (ratio of avg
    |change| on event days vs quiet days - >1 means meetings move points)."""
    df = tidy.sort_values("obs_date").copy()
    chg = df.groupby("pair")[field].diff()
    df["absmove"] = chg.abs()
    df = df.dropna(subset=["absmove"])
    dates = pd.DatetimeIndex(pd.to_datetime(df["obs_date"]))

    cal = calendar[calendar["ccy"].isin([ccy, "USD"])]
    out = {}
    for label, sub in [("all_events", cal),
                       ("USD_FOMC", cal[cal["ccy"] == "USD"]),
                       ("local_CB", cal[cal["ccy"] == ccy])]:
        mask = np.zeros(len(df), dtype=bool)
        for ed in pd.DatetimeIndex(sub["event_date"]).unique():
            lo = ed - pd.tseries.offsets.BDay(window_bd)
            hi = ed + pd.tseries.offsets.BDay(window_bd)
            mask |= ((dates >= lo) & (dates <= hi))
        ev, qt = df.loc[mask, "absmove"], df.loc[~mask, "absmove"]
        out[label] = {
            "share_of_abs_move": ev.sum() / df["absmove"].sum(),
            "share_of_days": mask.mean(),
            "intensity": (ev.mean() / qt.mean()) if len(qt) and qt.mean() > 0 else np.nan,
        }
    return pd.DataFrame(out).T
