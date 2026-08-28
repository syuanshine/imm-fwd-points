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
                            "q05", "q25", "q75", "q95", "es95_down", "es95_up",
                            "worst", "best", "latest"]}
            continue
        rows[label] = {
            "mean": c.mean(), "std": c.std(),
            "skew": c.skew(), "kurt": c.kurt(),
            "hit_rate_up": (c > 0).mean(),
            "q05": c.quantile(0.05), "q25": c.quantile(0.25),
            "q75": c.quantile(0.75), "q95": c.quantile(0.95),
            "es95_down": c[c <= c.quantile(0.05)].mean(),
            "es95_up": c[c >= c.quantile(0.95)].mean(),
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


# ---------------------------------------------------------------------------
# Fair value vs money-market rates
# ---------------------------------------------------------------------------

def fair_value_gap(tidy: pd.DataFrame, rate_diff: pd.DataFrame) -> pd.Series:
    """NDF-IMM-implied differential minus the money-market differential over
    the same window, ann %. What's left after rates is basis / flow premium /
    intervention expectation - the component a PM can actually fade or own,
    since it is NOT explained by the local-vs-USD policy path."""
    s = tidy.set_index("obs_date")["ann_pct"]
    r = rate_diff["rate_diff_ann_pct"].reindex(s.index).ffill()
    gap = s - r
    gap.name = "fv_gap_ann_pct"
    return gap


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def ar1_half_life(tidy: pd.DataFrame, field: str = "ann_pct") -> pd.Series:
    """AR(1) evidence for the rolling front series: regress within-pair daily
    change on the lagged level. Returns phi (AR coefficient), half-life in
    business days, and the t-stat of the mean-reversion slope.
    A short half-life with a decent |t| is the statistical licence for fading
    extremes; phi ~ 1 (half-life -> inf) says extremes drift, don't fade."""
    df = tidy.sort_values("obs_date")
    lag = df.groupby("pair")[field].shift(1)
    chg = df[field] - lag
    m = pd.DataFrame({"chg": chg, "lag": lag}).dropna()
    x = m["lag"] - m["lag"].mean()
    b = (x * m["chg"]).sum() / (x ** 2).sum()
    resid = m["chg"] - b * x - m["chg"].mean()
    se = np.sqrt((resid ** 2).sum() / (len(m) - 2) / (x ** 2).sum())
    phi = 1.0 + b
    hl = np.log(2) / -np.log(phi) if 0 < phi < 1 else np.inf
    return pd.Series({"phi": phi, "half_life_bd": hl, "t_stat": b / se, "n": len(m)})


def conditional_changes(tidy: pd.DataFrame, field: str = "ann_pct",
                        horizon: int = 21, z_window: int = 504) -> pd.DataFrame:
    """Fade table: bucket each day by the level's trailing z-score, then look
    at the distribution of the NEXT `horizon`-day within-pair change.
    If the bottom bucket shows positive median forward change (and top bucket
    negative), extremes have historically reverted at that horizon."""
    df = tidy.sort_values("obs_date").reset_index(drop=True)
    s = df[field]
    mu = s.rolling(z_window, min_periods=250).mean()
    sd = s.rolling(z_window, min_periods=250).std()
    z = (s - mu) / sd
    fwd = df.groupby("pair")[field].shift(-horizon) - s
    m = pd.DataFrame({"z": z, "fwd": fwd}).dropna()
    edges = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    labels = ["z<-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", "z>1.5"]
    m["bucket"] = pd.cut(m["z"], edges, labels=labels)
    g = m.groupby("bucket")["fwd"]
    out = pd.DataFrame({
        "n": g.count(), "median": g.median(), "mean": g.mean(),
        "q05": g.quantile(0.05), "q95": g.quantile(0.95),
        "hit_rate_up": g.apply(lambda x: (x > 0).mean()),
    })
    return out


# ---------------------------------------------------------------------------
# Range-of-changes / tail upgrades
# ---------------------------------------------------------------------------

def mae_stats(paths: pd.DataFrame) -> pd.DataFrame:
    """Max adverse excursion per vintage from a vintage_paths(mode='change')
    frame: the worst mark-to-market dip along the T-anchor -> T-0 path, for a
    long-points and a short-points holder, plus the endpoint. The gap between
    MAE and the final change is what a stop-loss has to survive."""
    rows = {}
    for c in paths.columns:
        s = paths[c].dropna()
        if len(s) < 10:
            continue
        rows[c] = {"final_change": s.iloc[-1],
                   "long_MAE": s.min(), "short_MAE": -s.max()}
    df = pd.DataFrame(rows).T
    df.loc["MEDIAN"] = df.median()
    df.loc["WORST"] = [df["final_change"].min(), df["long_MAE"].min(), df["short_MAE"].min()]
    return df


def vol_by_days_to_imm(tidy: pd.DataFrame, field: str = "imm_spread_pts",
                       bucket: int = 10) -> pd.Series:
    """Stdev of within-pair daily changes bucketed by days-to-near-IMM,
    pooled across all vintages. Answers: does the spread get noisier as the
    roll approaches (hold through vs exit early)?"""
    df = tidy.sort_values("obs_date")
    chg = df.groupby("pair")[field].diff()
    b = (df["days_to_near"] // bucket) * bucket + bucket // 2
    out = chg.groupby(b).std()
    out.index.name = "days_to_near_mid"
    return out.sort_index()


# ---------------------------------------------------------------------------
# Year-end turn series & spot beta
# ---------------------------------------------------------------------------

def turn_series(all_slots: pd.DataFrame, field: str = "ann_pct") -> pd.Series:
    """IMM-space turn indicator, SAME-DAY version: on days when the front
    pair (slot 0) is Dec-Mar, its level minus the SAME DAY's deferred pair
    (slot 1, Mar-Jun). Both quotes come off the same curve snapshot, so a
    trending rate environment no longer contaminates the comparison (the
    flaw in the original trailing-median baseline).

    Remaining known limitation: Dec-Mar and Mar-Jun are different future
    windows, so genuine curve slope / meeting-calendar differences between
    the two quarters still land in this measure. For the market-convention
    extraction that controls for that (forward-forward flanking with a
    day-count-matched local baseline), use the standalone turn/ module -
    this IMM-space indicator is kept as a quick same-day richness gauge of
    the turn-carrying pair, not as the turn estimate."""
    df = all_slots.sort_values("obs_date")
    front = df[(df["slot"] == 0) & (df["quarter_pair"] == "Dec-Mar")]
    deferred = df[df["slot"] == 1].set_index("obs_date")[field]
    f = front.set_index("obs_date")[field]
    ts = f - deferred.reindex(f.index)
    ts.name = "turn_premium_ann_pct"
    return ts.dropna()


def spot_beta(tidy: pd.DataFrame, field: str = "imm_spread_pts",
              window: int = 126) -> pd.Series:
    """Rolling beta of daily point changes to spot %-returns (points per 1%
    spot move). Non-zero beta = points are trading directionally with USD
    (flow/stress regime) rather than as a pure rates instrument; it is also
    the PM's hedge-ratio input for a points position."""
    df = tidy.sort_values("obs_date")
    chg = df.groupby("pair")[field].diff()
    ret = df["spot"].pct_change() * 100.0
    m = pd.DataFrame({"chg": chg, "ret": ret})
    m.index = df["obs_date"]
    cov = m["chg"].rolling(window, min_periods=int(window * 0.7)).cov(m["ret"])
    var = m["ret"].rolling(window, min_periods=int(window * 0.7)).var()
    beta = cov / var
    beta.name = "beta_pts_per_1pct_spot"
    return beta
