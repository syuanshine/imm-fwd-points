"""Data layer.

The pulling agent implements ONE of the two provider routes:

Route A - direct IMM tickers: if your source contributes IMM-dated NDF
    points directly, implement `fetch_imm_points_history` and skip the
    curve interpolation entirely.

Route B (recommended / most robust) - standard tenor curve: implement
    `fetch_curve_history` returning daily spot + forward points for the
    standard tenors; `build_imm_points_from_curve` (in series.py) then
    interpolates points to the two IMM legs on every history date.

Everything downstream consumes the same tidy DataFrame schema:

    obs_date | ccy | near_date | far_date | near_pts | far_pts | spot
    (points in QUOTE units, i.e. already divided by points_scale so that
     outright = spot + pts)

A SyntheticProvider is included so the whole pipeline runs end-to-end
without Bloomberg - replace it with BloombergProvider in production.
"""
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import CCY_MAP, CURVE_TENORS, TENOR_DAYS, curve_ticker


class DataProvider:
    """Interface the data-pulling agent must satisfy."""

    def fetch_curve_history(self, ccy: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily history of spot and forward points for the standard tenor curve.

        Returns a DataFrame indexed by obs_date (DatetimeIndex, business days)
        with columns:
            'spot'                       : spot rate
            '<tenor>_pts'  for each tenor: forward points in QUOTE units
                                           (outright = spot + pts)
            '<tenor>_days' for each tenor: OPTIONAL actual days from obs_date
                                           to that tenor's settlement date.
                                           If omitted, TENOR_DAYS fallback is used.
        """
        raise NotImplementedError

    def fetch_rate_diff_history(self, ccy: str, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
        """OPTIONAL (enables fair-value analytics): money-market implied
        local-minus-USD rate differential for the FRONT IMM window, ann %.

        Returns DataFrame indexed by obs_date with column 'rate_diff_ann_pct'.
        USD leg: SOFR IMM futures cover exactly the same IMM-to-IMM windows -
            100 - price of the front quarterly SFR contract (SFRU5 etc.).
        Local leg: cleanest available money-market curve - KRW IRS/CD, INR
            OIS/MIFOR, THB THOR/IRS, TWD/IDR/PHP interbank or implied yields.
        Return None if unavailable; fair-value analytics are then skipped.
        """
        return None

    def fetch_imm_points_history(self, ccy: str, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
        """OPTIONAL Route A: directly-quoted IMM points.

        Returns tidy frame: obs_date | near_date | far_date | near_pts | far_pts | spot
        or None if not available for this currency.
        """
        return None


class BloombergProvider(DataProvider):
    """=== TO BE IMPLEMENTED BY THE DATA-PULLING AGENT (blpapi) ===

    Guidance:
      * spot ticker:   CCY_MAP[ccy].spot_ticker           (field PX_LAST)
      * points ticker: curve_ticker(CCY_MAP[ccy].ndf_root, tenor)
                       e.g. "KWN+1M Curncy"               (field PX_LAST)
      * scale:         divide raw points by CCY_MAP[ccy].points_scale,
                       AFTER verifying the scale via the FWD_SCALE field.
      * days:          ideally also pull SETTLE_DT per tenor per day
                       (reference data with overrides, or derive from the
                       market's T+ convention + holiday calendars) and
                       populate '<tenor>_days'.
      * Use BDH-style HistoricalDataRequest, currency-by-currency,
        with non-trading-day fill = NIL and periodicity DAILY.
    """

    def fetch_curve_history(self, ccy, start, end):
        raise NotImplementedError("Data agent: implement blpapi pull here.")

    def fetch_rate_diff_history(self, ccy, start, end):
        # Data agent: optional but high value - see DataProvider docstring.
        return None


class SyntheticProvider(DataProvider):
    """Plausible random-walk curves so the pipeline can be demoed/tested.

    Points are generated from a mean-reverting annualized carry process per
    currency (roughly calibrated signs: INR/IDR/PHP positive carry, THB/TWD
    negative/turn-prone, KRW mildly negative recently) plus a Dec year-end
    turn premium so the seasonality analytics have something to find.
    """

    PARAMS = {  # ccy: (spot0, carry_mean_%, carry_vol_%, turn_%)
        "THB": (33.0, -1.0, 0.8, 0.6),
        "IDR": (14500.0, 4.5, 1.5, 0.8),
        "INR": (75.0, 4.0, 1.2, 0.5),
        "PHP": (52.0, 2.5, 1.0, 0.7),
        "TWD": (30.5, -1.5, 1.0, 1.5),
        "KRW": (1250.0, 0.5, 1.2, 0.9),
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._cache = {}

    def _paths(self, ccy, start, end):
        """Deterministic shared (spot, carry) paths so the points and the
        'money-market rates' legs are built from the same underlying carry."""
        key = (ccy, start, end)
        if key in self._cache:
            return self._cache[key]
        p = self.PARAMS[ccy]
        rng = np.random.RandomState(self.seed + sum(ord(c) for c in ccy))
        idx = pd.bdate_range(start, end)
        n = len(idx)
        spot = p[0] * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        carry = np.zeros(n)
        carry[0] = p[1]
        for i in range(1, n):
            carry[i] = carry[i-1] + 0.02 * (p[1] - carry[i-1]) + rng.normal(0, p[2] / 16.0)
        # persistent NDF flow/basis premium on top of 'clean' rates (OU, mean ~30bp)
        basis = np.zeros(n)
        basis[0] = 0.3
        for i in range(1, n):
            basis[i] = basis[i-1] + 0.01 * (0.3 - basis[i-1]) + rng.normal(0, 0.03)
        self._cache[key] = (idx, spot, carry, basis)
        return self._cache[key]

    def fetch_rate_diff_history(self, ccy, start, end):
        idx, _, carry, basis = self._paths(ccy, start, end)
        # 'clean' money-market differential = NDF-implied carry minus basis
        return pd.DataFrame({"rate_diff_ann_pct": carry - basis}, index=idx)

    def fetch_curve_history(self, ccy, start, end):
        p = self.PARAMS[ccy]
        idx, spot, carry, _ = self._paths(ccy, start, end)
        n = len(idx)
        out = pd.DataFrame(index=idx)
        out["spot"] = spot
        year_frac_turn = 10.0 / 360.0  # ~10-day turn window
        for tenor in CURVE_TENORS:
            days = TENOR_DAYS[tenor]
            settle = idx + pd.Timedelta(days=days)
            # year-end turn premium if the tenor window spans Dec 31
            spans_turn = (idx.year != settle.year).astype(float)
            ann = carry + spans_turn * p[3] * (year_frac_turn * 360.0 / np.maximum(days, 1))
            out[tenor + "_pts"] = spot * ann / 100.0 * days / 360.0
            out[tenor + "_days"] = days
        return out
