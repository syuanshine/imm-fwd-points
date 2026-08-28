"""Analytics on IMM forward-point series (all cross-currency work uses
the annualized 'ann_pct' field so THB and IDR are comparable)."""
from typing import Dict

import numpy as np
import pandas as pd

from series import roll_adjusted_changes

TRADING_DAYS = 252


def zscore_snapshot(series: pd.Series, windows=(252, 756, None)) -> pd.Series:
    """Z-score of the latest value vs trailing windows (None = full history)."""
    out = {}
    for w in windows:
        hist = series.dropna() if w is None else series.dropna().iloc[-w:]
        label = "full" if w is None else "{}y".format(round(w / TRADING_DAYS))
        if len(hist) > 20:
            out[label] = (series.dropna().iloc[-1] - hist.mean()) / hist.std()
        else:
            out[label] = np.nan
    return pd.Series(out)


def percentile_snapshot(series: pd.Series, windows=(252, 756, None)) -> pd.Series:
    out = {}
    for w in windows:
        hist = series.dropna() if w is None else series.dropna().iloc[-w:]
        label = "full" if w is None else "{}y".format(round(w / TRADING_DAYS))
        out[label] = (hist <= series.dropna().iloc[-1]).mean() * 100 if len(hist) > 20 else np.nan
    return pd.Series(out)


def summary_table(tidy_by_ccy: Dict[str, pd.DataFrame], field: str = "ann_pct") -> pd.DataFrame:
    """One-row-per-currency dashboard: latest level, z-scores, percentiles,
    realized vol of daily changes, min/max over history."""
    rows = {}
    for ccy, df in tidy_by_ccy.items():
        s = df.set_index("obs_date")[field]
        chg = roll_adjusted_changes(df, field)
        z = zscore_snapshot(s)
        p = percentile_snapshot(s)
        rows[ccy] = {
            "latest": s.dropna().iloc[-1],
            "pair": df["pair"].iloc[-1],
            "z_1y": z.get("1y"), "z_3y": z.get("3y"), "z_full": z.get("full"),
            "pctile_full": p.get("full"),
            "vol_1y_daily": chg.iloc[-TRADING_DAYS:].std(),
            "min_full": s.min(), "max_full": s.max(),
        }
    return pd.DataFrame(rows).T


def seasonality_by_quarter(tidy: pd.DataFrame, field: str = "ann_pct") -> pd.DataFrame:
    """Distribution of the field grouped by quarter-pair (Mar-Jun, Jun-Sep,
    Sep-Dec, Dec-Mar). Dec-Mar embeds the year-end funding turn; TWD and
    KRW NDFs are the classic turn currencies in this universe."""
    order = ["Mar-Jun", "Jun-Sep", "Sep-Dec", "Dec-Mar"]
    g = tidy.groupby("quarter_pair")[field].describe()
    return g.reindex([q for q in order if q in g.index])


def turn_premium(tidy: pd.DataFrame, field: str = "ann_pct") -> float:
    """Year-end turn proxy: mean Dec-Mar level minus mean of the other pairs."""
    m = tidy.groupby("quarter_pair")[field].mean()
    if "Dec-Mar" not in m.index:
        return np.nan
    others = m.drop("Dec-Mar").mean()
    return m["Dec-Mar"] - others


def correlation_matrix(tidy_by_ccy: Dict[str, pd.DataFrame], field: str = "ann_pct",
                       freq: str = "W-FRI") -> pd.DataFrame:
    """Cross-currency correlation of (roll-adjusted) changes, resampled to
    weekly to reduce asynchronous-close noise across Asian markets."""
    chgs = {}
    for ccy, df in tidy_by_ccy.items():
        c = roll_adjusted_changes(df, field)
        chgs[ccy] = c.resample(freq).sum(min_count=1)
    return pd.DataFrame(chgs).corr()


def rolling_vol(tidy: pd.DataFrame, field: str = "ann_pct", window: int = 63) -> pd.Series:
    """Rolling realized vol (stdev of roll-adjusted daily changes, annualized)."""
    chg = roll_adjusted_changes(tidy, field)
    return chg.rolling(window, min_periods=int(window * 0.7)).std() * np.sqrt(TRADING_DAYS)


def horizon_return_stats(tidy: pd.DataFrame, field: str = "imm_spread_pts",
                         horizons=(1, 5, 21, 63)) -> pd.DataFrame:
    """Per-horizon stats of changes ('returns') in the field, in RAW units.

    Changes are computed WITHIN each named pair only (no cross-roll jumps),
    over h business days. One row per horizon:
        mean, std, skew, kurt (excess), hit_rate_up (share of positive moves),
        worst, best, latest - all in the field's own units (points).
    """
    df = tidy.sort_values("obs_date")
    rows = {}
    for h in horizons:
        chg = df.groupby("pair")[field].diff(h)
        c = chg.dropna()
        label = "{}d".format(h)
        if len(c) < 20:
            rows[label] = {k: np.nan for k in
                           ["mean", "std", "skew", "kurt", "hit_rate_up",
                            "worst", "best", "latest"]}
            continue
        rows[label] = {
            "mean": c.mean(), "std": c.std(),
            "skew": c.skew(), "kurt": c.kurt(),
            "hit_rate_up": (c > 0).mean(),
            "worst": c.min(), "best": c.max(),
            "latest": chg.iloc[-1],
        }
    return pd.DataFrame(rows).T


def single_ccy_snapshot(tidy: pd.DataFrame, field: str = "imm_spread_pts") -> pd.Series:
    """Level snapshot for one currency in raw units: latest, z-scores,
    percentiles, history range."""
    s = tidy.set_index("obs_date")[field]
    z = zscore_snapshot(s)
    p = percentile_snapshot(s)
    return pd.Series({
        "latest": s.dropna().iloc[-1],
        "pair": tidy["pair"].iloc[-1],
        "z_1y": z.get("1y"), "z_3y": z.get("3y"), "z_full": z.get("full"),
        "pctile_1y": p.get("1y"), "pctile_full": p.get("full"),
        "min_full": s.min(), "median_full": s.median(), "max_full": s.max(),
    })
