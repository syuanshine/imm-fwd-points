"""Charts (matplotlib, static PNG output).

Design rules applied throughout:
  * fixed currency -> color mapping (color follows the entity, never rank)
  * small multiples instead of 6 overlapping lines
  * one y-axis per panel, recessive grid, direct labels where useful
  * diverging blue<->red with neutral gray midpoint for the correlation map
Palette: validated default categorical order (see dataviz reference palette).
"""
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics import rolling_vol, seasonality_by_quarter, zscore_snapshot

# fixed slot order - never re-assign when subsetting currencies
CCY_COLORS = {
    "THB": "#2a78d6",  # blue
    "IDR": "#eb6834",  # orange
    "INR": "#1baf7a",  # aqua
    "PHP": "#eda100",  # yellow
    "TWD": "#e87ba4",  # magenta
    "KRW": "#008300",  # green
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.dpi": 130,
})


def _style_ax(ax):
    ax.grid(True, axis="y", linewidth=0.6, color=GRID)
    ax.grid(False, axis="x")


def plot_rolling_panels(tidy_by_ccy: Dict[str, pd.DataFrame], field: str = "ann_pct",
                        outfile: str = "rolling_imm.png", title: str = None):
    """Small multiples: 10y rolling front IMM-IMM series, one panel per ccy."""
    ccys = list(tidy_by_ccy)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, ccy in zip(axes.ravel(), ccys):
        df = tidy_by_ccy[ccy]
        s = df.set_index("obs_date")[field]
        ax.plot(np.asarray(s.index), np.asarray(s.values), color=CCY_COLORS[ccy], linewidth=1.4)
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        # direct label: latest value
        last = s.dropna()
        if len(last):
            ax.annotate("{:+.2f}".format(last.iloc[-1]), xy=(last.index[-1], last.iloc[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        _style_ax(ax)
    fig.suptitle(title or "Front IMM-IMM forward points, annualized implied yield gap (%, local - USD, ACT/360)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_contract_evolution(piv: pd.DataFrame, ccy: str, outfile: str,
                            recent_n: int = 8, quarter: str = None):
    """Life of each named pair aligned on days-to-near-IMM. Older contracts as
    a gray envelope (10-90 pct band + median), `recent_n` most recent as lines
    in a single-hue ordinal ramp (oldest light -> newest dark)."""
    cols = list(piv.columns)
    recent = cols[-recent_n:]
    older = [c for c in cols if c not in recent]
    fig, ax = plt.subplots(figsize=(9, 5))
    if older:
        band = piv[older]
        ax.fill_between(np.asarray(piv.index), band.quantile(0.10, axis=1), band.quantile(0.90, axis=1),
                        color=GRID, alpha=0.8, label="prior contracts 10-90%")
        ax.plot(np.asarray(piv.index), np.asarray(band.median(axis=1)), color=INK2, linewidth=1.0,
                linestyle="--", label="prior contracts median")
    # ordinal single-hue ramp for recent contracts (light -> dark = old -> new)
    ramp = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
    steps = ramp[-len(recent):]
    for c, col in zip(recent, steps):
        s = piv[c].dropna()
        ax.plot(np.asarray(s.index), np.asarray(s.values), color=col, linewidth=1.5, label=c)
    ax.invert_xaxis()
    ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("days to near IMM date")
    ax.set_ylabel("annualized %")
    ax.set_title("{}: {} IMM pair evolution over contract life".format(
                     ccy, quarter or "front"),
                 loc="left", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_seasonality(tidy_by_ccy: Dict[str, pd.DataFrame], field: str = "ann_pct",
                     outfile: str = "seasonality.png"):
    """Small-multiple boxplots by quarter-pair; Dec-Mar highlights the year-end turn."""
    order = ["Mar-Jun", "Jun-Sep", "Sep-Dec", "Dec-Mar"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, df) in zip(axes.ravel(), tidy_by_ccy.items()):
        data = [df.loc[df["quarter_pair"] == q, field].dropna().values for q in order]
        bp = ax.boxplot(data, labels=order, widths=0.5, patch_artist=True,
                        showfliers=False, medianprops={"color": INK})
        for patch in bp["boxes"]:
            patch.set_facecolor(CCY_COLORS[ccy])
            patch.set_alpha(0.55)
            patch.set_edgecolor(CCY_COLORS[ccy])
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        _style_ax(ax)
    fig.suptitle("Seasonality by IMM quarter-pair (annualized %) - Dec-Mar embeds the year-end turn",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_zscore_bars(tidy_by_ccy: Dict[str, pd.DataFrame], field: str = "ann_pct",
                     outfile: str = "zscores.png"):
    """Current level vs history: z-scores over 1y / 3y / full, grouped by ccy."""
    windows = ["1y", "3y", "full"]
    shades = {"1y": 1.0, "3y": 0.65, "full": 0.35}  # alpha per window, hue = ccy
    ccys = list(tidy_by_ccy)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ccys))
    w = 0.26
    for j, win in enumerate(windows):
        vals = []
        for ccy in ccys:
            s = tidy_by_ccy[ccy].set_index("obs_date")[field]
            vals.append(zscore_snapshot(s).get(win, np.nan))
        pos = x + (j - 1) * w
        for xi, v, ccy in zip(pos, vals, ccys):
            ax.bar(xi, v, width=w - 0.03, color=CCY_COLORS[ccy], alpha=shades[win],
                   edgecolor=SURFACE, linewidth=1)
            if not np.isnan(v):
                ax.annotate("{:+.1f}".format(v), xy=(xi, v), fontsize=7, color=INK2,
                            ha="center", va="bottom" if v >= 0 else "top",
                            xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(ccys)
    ax.axhline(0, color=INK2, linewidth=0.8)
    for z in (-2, 2):
        ax.axhline(z, color=INK2, linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylabel("z-score of latest level")
    ax.set_title("How rich/cheap is the current front IMM-IMM level? (opacity: 1y / 3y / full history)",
                 loc="left", fontweight="bold")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_corr_heatmap(corr: pd.DataFrame, outfile: str = "corr.png"):
    """Diverging blue<->red heatmap (gray midpoint) of weekly-change correlations."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#8f1f1e"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, "{:.2f}".format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_title("Correlation of weekly changes in IMM-IMM annualized points",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_vol_panels(tidy_by_ccy: Dict[str, pd.DataFrame], outfile: str = "vol.png",
                    window: int = 63):
    """Rolling 3m realized vol of roll-adjusted changes (annualized), small multiples."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, df) in zip(axes.ravel(), tidy_by_ccy.items()):
        v = rolling_vol(df, window=window)
        ax.plot(np.asarray(v.index), np.asarray(v.values), color=CCY_COLORS[ccy], linewidth=1.4)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    fig.suptitle("Rolling {}d realized vol of IMM-IMM changes (annualized %)".format(window),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_ccy_deepdive(df: pd.DataFrame, piv_pts: pd.DataFrame, ret_stats: pd.DataFrame,
                      snapshot: pd.Series, ccy: str, outfile: str):
    """Single-currency one-pager, everything in RAW forward points (quote
    units, far leg minus near leg) - no annualization, no cross-ccy scaling.

    Panels: (1) 10y rolling raw spread  (2) seasonality by quarter-pair
            (3) rolling 63d vol of daily changes  (4) 21d change distribution
            (5) current vintage vs prior-contract envelope (contract life)
            (6) snapshot + horizon return stats table
    """
    col = CCY_COLORS[ccy]
    field = "imm_spread_pts"
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # (1) rolling raw spread
    s = df.set_index("obs_date")[field]
    ax1.plot(np.asarray(s.index), np.asarray(s.values), color=col, linewidth=1.2)
    ax1.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
    last = s.dropna()
    if len(last):
        ax1.annotate("{:+.3f}".format(last.iloc[-1]), xy=(last.index[-1], last.iloc[-1]),
                     xytext=(4, 0), textcoords="offset points", fontsize=8,
                     color=INK, va="center")
    ax1.set_title("Front IMM-IMM spread, raw points", loc="left", fontweight="bold")
    _style_ax(ax1)

    # (2) seasonality boxplot
    order = ["Mar-Jun", "Jun-Sep", "Sep-Dec", "Dec-Mar"]
    data = [df.loc[df["quarter_pair"] == q, field].dropna().values for q in order]
    bp = ax2.boxplot(data, labels=order, widths=0.5, patch_artist=True,
                     showfliers=False, medianprops={"color": INK})
    for patch in bp["boxes"]:
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    ax2.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
    ax2.set_title("Seasonality by quarter-pair (raw points)", loc="left", fontweight="bold")
    _style_ax(ax2)

    # (3) rolling vol of daily changes, raw points
    from series import roll_adjusted_changes
    chg = roll_adjusted_changes(df, field)
    v = chg.rolling(63, min_periods=45).std()
    ax3.plot(np.asarray(v.index), np.asarray(v.values), color=col, linewidth=1.2)
    ax3.set_title("Rolling 63d stdev of daily changes (points)", loc="left", fontweight="bold")
    _style_ax(ax3)

    # (4) 21d change distribution with latest marked
    c21 = df.sort_values("obs_date").groupby("pair")[field].diff(21).dropna()
    ax4.hist(c21.values, bins=40, color=col, alpha=0.65, edgecolor=SURFACE)
    if len(c21):
        latest21 = df.sort_values("obs_date").groupby("pair")[field].diff(21).iloc[-1]
        if not np.isnan(latest21):
            ax4.axvline(latest21, color=INK, linewidth=1.2, linestyle="--")
            ax4.annotate("latest {:+.3f}".format(latest21), xy=(latest21, ax4.get_ylim()[1]),
                         xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK)
    ax4.set_title("Distribution of 21d changes (points)", loc="left", fontweight="bold")
    _style_ax(ax4)

    # (5) current vintage vs prior-contract envelope, raw points
    cols = list(piv_pts.columns)
    if len(cols) > 1:
        prior = piv_pts[cols[:-1]]
        ax5.fill_between(np.asarray(piv_pts.index),
                         np.asarray(prior.quantile(0.10, axis=1)),
                         np.asarray(prior.quantile(0.90, axis=1)),
                         color=GRID, alpha=0.8, label="prior contracts 10-90%")
        ax5.plot(np.asarray(piv_pts.index), np.asarray(prior.median(axis=1)),
                 color=INK2, linewidth=1.0, linestyle="--", label="prior median")
    cur = piv_pts[cols[-1]].dropna()
    ax5.plot(np.asarray(cur.index), np.asarray(cur.values), color=col,
             linewidth=1.8, label=cols[-1])
    ax5.invert_xaxis()
    ax5.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
    ax5.set_xlabel("days to near IMM date")
    ax5.set_title("Current vintage vs history (contract life, points)", loc="left", fontweight="bold")
    ax5.legend(fontsize=7, frameon=False)
    _style_ax(ax5)

    # (6) snapshot + return-stats table
    ax6.axis("off")
    snap_txt = ("pair {}   latest {:+.3f} pts\n"
                "z: 1y {:+.1f}   3y {:+.1f}   full {:+.1f}\n"
                "pctile: 1y {:.0f}%   full {:.0f}%\n"
                "range: [{:+.3f}, {:+.3f}]  med {:+.3f}").format(
        snapshot["pair"], snapshot["latest"], snapshot["z_1y"], snapshot["z_3y"],
        snapshot["z_full"], snapshot["pctile_1y"], snapshot["pctile_full"],
        snapshot["min_full"], snapshot["max_full"], snapshot["median_full"])
    ax6.text(0.0, 0.98, snap_txt, va="top", fontsize=9, family="monospace", color=INK)
    keep = [c for c in ["mean", "std", "q05", "q95", "hit_rate_up",
                        "worst", "best", "latest"] if c in ret_stats.columns]
    tbl = ret_stats[keep].round(4)
    table = ax6.table(cellText=tbl.values, rowLabels=tbl.index, colLabels=tbl.columns,
                      loc="center", cellLoc="right", bbox=[0.02, 0.05, 0.96, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for cell in table.get_celld().values():
        cell.set_edgecolor(GRID)
    ax6.set_title("Snapshot & change stats by horizon (points)", loc="left", fontweight="bold")

    fig.suptitle("{} - IMM forward points deep dive (raw quote units)".format(ccy),
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=col)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_vintage_paths(paths: pd.DataFrame, stats: pd.DataFrame, ccy: str,
                       quarter: str, outfile: str, mode: str = "change",
                       units: str = "points"):
    """Seasonality view: every vintage's path from T-anchor to the near IMM
    date, aligned on days-to-IMM, plus the cross-vintage median/IQR band.

    Left: individual vintage paths (ordinal ramp, oldest light -> newest dark),
          current vintage highlighted in the currency colour.
    Right: median path with the 25-75% band across vintages - the actual
          'is there a seasonal drift?' answer, with dispersion visible.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
    cols = list(paths.columns)
    steps = [ramp[int(round(i * (len(ramp) - 1) / max(len(cols) - 1, 1)))]
             for i in range(len(cols))]
    for c, colr in zip(cols[:-1], steps[:-1]):
        s = paths[c].dropna()
        ax1.plot(np.asarray(s.index), np.asarray(s.values), color=colr, linewidth=1.2)
    if cols:
        s = paths[cols[-1]].dropna()
        ax1.plot(np.asarray(s.index), np.asarray(s.values), color=CCY_COLORS[ccy],
                 linewidth=2.2, label=cols[-1] + " (current)")
        ax1.legend(fontsize=8, frameon=False, loc="best")
    ax1.axhline(0, color=INK2, linewidth=0.9)
    ax1.invert_xaxis()
    ax1.set_xlabel("days to near IMM date")
    ax1.set_ylabel("cumulative change from anchor ({})".format(units))
    ax1.set_title("Vintage paths ({} vintages)".format(len(cols)),
                  loc="left", fontweight="bold")
    _style_ax(ax1)

    med = paths.median(axis=1)
    q25, q75 = paths.quantile(0.25, axis=1), paths.quantile(0.75, axis=1)
    ax2.fill_between(np.asarray(med.index), np.asarray(q25), np.asarray(q75),
                     color=CCY_COLORS[ccy], alpha=0.22, label="25-75% across vintages")
    ax2.plot(np.asarray(med.index), np.asarray(med.values), color=CCY_COLORS[ccy],
             linewidth=2.0, label="median path")
    ax2.axhline(0, color=INK2, linewidth=0.9)
    ax2.invert_xaxis()
    ax2.set_xlabel("days to near IMM date")
    ax2.set_title("Median path & dispersion", loc="left", fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)
    _style_ax(ax2)

    n = int(stats["n"].max()) if len(stats) and not stats["n"].isna().all() else 0
    fig.suptitle("{} - {} IMM points: seasonality of the run-up to the near IMM date "
                 "(n={} vintages, descriptive only)".format(ccy, quarter, n),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_fair_value_gap(gaps: dict, outfile: str = "fv_gap.png"):
    """Small multiples: NDF-IMM-implied differential minus money-market
    differential (ann %). The residual basis / flow premium a PM can trade."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, gap) in zip(axes.ravel(), gaps.items()):
        g = gap.dropna()
        ax.plot(np.asarray(g.index), np.asarray(g.values),
                color=CCY_COLORS[ccy], linewidth=1.2)
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        if len(g):
            ax.annotate("{:+.2f}".format(g.iloc[-1]), xy=(g.index[-1], g.iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")
        _style_ax(ax)
    fig.suptitle("Fair-value gap: IMM-implied minus money-market rate differential (ann %) "
                 "- the flow/basis premium", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_vol_by_dtn(vols: dict, outfile: str = "vol_by_dtn.png"):
    """Vol of daily point changes vs days-to-near-IMM, one panel per ccy.
    Rising bars toward 0 = the spread gets noisier into the roll."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, v) in zip(axes.ravel(), vols.items()):
        ax.bar(np.asarray(v.index), np.asarray(v.values), width=8,
               color=CCY_COLORS[ccy], alpha=0.75, edgecolor=SURFACE)
        ax.invert_xaxis()
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("days to near IMM date (bucket mid)")
    fig.suptitle("Stdev of daily point changes by days-to-near-IMM (pooled vintages, raw points)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_spot_beta(betas: dict, outfile: str = "spot_beta.png"):
    """Rolling beta of point changes to spot returns, one panel per ccy.
    Sustained departures from zero flag flow/stress regimes."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, b) in zip(axes.ravel(), betas.items()):
        s = b.dropna()
        ax.plot(np.asarray(s.index), np.asarray(s.values),
                color=CCY_COLORS[ccy], linewidth=1.2)
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    fig.suptitle("Rolling 126d beta of daily point changes to spot (points per 1% spot move)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_turn_series(turns: dict, outfile: str = "turn_series.png"):
    """Year-end turn premium per year (mean of the Dec-Mar turn series by
    the Dec year), one panel per ccy."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, ts) in zip(axes.ravel(), turns.items()):
        if len(ts.dropna()):
            yearly = ts.groupby(ts.index.year).mean()
            ax.bar(np.asarray(yearly.index.astype(int)), np.asarray(yearly.values),
                   color=CCY_COLORS[ccy], alpha=0.75, edgecolor=SURFACE)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    fig.suptitle("Year-end turn premium by year: Dec-Mar level minus trailing non-turn baseline (ann %)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile
