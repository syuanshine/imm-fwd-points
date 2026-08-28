"""Charts for the standalone turn module. Reuses the project palette and
style from imm_fwd/charts.py (fixed ccy -> color mapping, small multiples,
single axes, recessive grid)."""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "imm_fwd"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from charts import CCY_COLORS, GRID, INK, INK2, SURFACE, _style_ax  # noqa: E402


def plot_turn_jump_series(series_by_ccy, outfile, field="jump_ann_pct",
                          min_days=120):
    """Small multiples: estimated next-year-end jump through time, shown for
    the run-up window (last `min_days` days before each year-end, where the
    estimate is identified by liquid tenors)."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=False)
    for ax, (ccy, df) in zip(axes.ravel(), series_by_ccy.items()):
        d = df[df["days_to_turn"] <= min_days].dropna(subset=[field])
        ax.plot(np.asarray(d.index), np.asarray(d[field].values),
                color=CCY_COLORS[ccy], linewidth=1.0)
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    fig.suptitle("Estimated year-end turn jump (annualized % per turn day, flanking method), "
                 "final {}d run-up each year".format(min_days),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_curve_decomposition(days, pts, days_to_turn, ccy, obs_label, outfile):
    """One curve snapshot: forward-forward daily point accrual per interval,
    with the turn-containing interval highlighted. This is the method made
    visible - the excess height of the highlighted bar over its neighbours,
    times the interval length, IS the estimated jump."""
    order = np.argsort(days)
    d, p = np.asarray(days)[order], np.asarray(pts)[order]
    lo, hi = d[:-1], d[1:]
    m = np.diff(p) / np.diff(d)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for a, b, v in zip(lo, hi, m):
        spans = a < days_to_turn <= b
        ax.bar((a + b) / 2, v, width=(b - a) * 0.92,
               color=CCY_COLORS[ccy] if spans else GRID,
               edgecolor=CCY_COLORS[ccy] if spans else INK2, linewidth=0.6,
               alpha=0.9 if spans else 0.7)
    ax.axvline(days_to_turn, color=INK, linewidth=1.0, linestyle="--")
    ax.annotate("Dec 31", xy=(days_to_turn, ax.get_ylim()[1]), fontsize=8,
                color=INK, xytext=(3, -10), textcoords="offset points")
    ax.set_xlabel("days from observation date")
    ax.set_ylabel("forward-forward daily accrual (points/day)")
    ax.set_title("{} {}: daily point accrual per inter-tenor interval".format(ccy, obs_label),
                 loc="left", fontweight="bold")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_kickin(kick_by_ccy, tenor, outfile, recent_n=6):
    """Method D small multiples: the fixed tenor's implied ann rate vs
    days-to-year-end, one line per year (single-hue ordinal ramp, oldest
    light -> newest dark), demeaned per year so the turn kink is comparable
    across years with different rate levels."""
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, piv) in zip(axes.ravel(), kick_by_ccy.items()):
        years = list(piv.columns)[-recent_n:]
        steps = ramp[-len(years):]
        for yr, colr in zip(years, steps):
            s = piv[yr].dropna()
            if len(s) < 10:
                continue
            base = s[s.index > 45].mean()          # pre-kick-in level
            ax.plot(np.asarray(s.index), np.asarray(s.values - base),
                    color=colr, linewidth=1.2, label=str(yr))
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
        ax.axhline(0, color=INK2, linewidth=0.8, alpha=0.5)
        ax.invert_xaxis()
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        ax.legend(fontsize=6, ncol=3, frameon=False)
        _style_ax(ax)
    for ax in axes[-1]:
        ax.set_xlabel("days to Dec 31 (negative = past year-end)")
    fig.suptitle("Turn kick-in: {} implied rate vs its pre-turn level as the window "
                 "rolls across year-end (ann %)".format(tenor),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_turn_by_year(series_by_ccy, outfile, field="jump_ann_pct",
                      window=(45, 5)):
    """Cross-ccy comparison: median estimated jump per year, measured in the
    clean identification window (days_to_turn between `window` bounds)."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, (ccy, df) in zip(axes.ravel(), series_by_ccy.items()):
        d = df[(df["days_to_turn"] <= window[0]) & (df["days_to_turn"] >= window[1])]
        yearly = d.groupby("year")[field].median().dropna()
        ax.bar(np.asarray(yearly.index.astype(int)), np.asarray(yearly.values),
               color=CCY_COLORS[ccy], alpha=0.8, edgecolor=SURFACE)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(ccy, loc="left", fontweight="bold", color=CCY_COLORS[ccy])
        _style_ax(ax)
    fig.suptitle("Year-end turn premium by year: median estimate over the final {}-{}d "
                 "run-up (ann % per turn day)".format(window[0], window[1]),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile
