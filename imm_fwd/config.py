"""Currency universe and Bloomberg ticker configuration.

The data-pulling agent should verify/complete every ticker below against
their terminal (e.g. via FRD <GO> / NDF <GO> / ALLQ) before use - NDF
ticker roots vary by contributor source (BGN vs CMPN vs bank pages).

points_scale: forward points on Bloomberg are quoted such that
    outright = spot + points / points_scale
Verify per ticker via the FWD_SCALE field / DES page - defaults below are
the usual BGN conventions but MUST be confirmed by the data agent.
"""
from typing import Dict, List, NamedTuple


class CcyConfig(NamedTuple):
    code: str            # our internal code
    name: str
    spot_ticker: str     # e.g. "USDKRW Curncy" (KRW spot; NDFs fix vs onshore ref)
    ndf_root: str        # NDF forward-points ticker root, e.g. "KWN" for KRW NDF
    points_scale: float  # outright = spot + points/points_scale (VERIFY via FWD_SCALE)
    is_ndf: bool
    fixing_note: str


UNIVERSE: List[CcyConfig] = [
    CcyConfig("THB", "Thai baht (offshore)", "USDTHB Curncy", "THB",   100.0,   False,
              "Offshore deliverable/quasi; BOT restrictions apply to onshore access"),
    CcyConfig("IDR", "Indonesian rupiah NDF", "USDIDR Curncy", "IHN",  1.0,     True,
              "Fixing: JISDOR (IDR JISDOR Index)"),
    CcyConfig("INR", "Indian rupee NDF",      "USDINR Curncy", "IRN",  100.0,   True,
              "Fixing: RBI reference rate (FBIL)"),
    CcyConfig("PHP", "Philippine peso NDF",   "USDPHP Curncy", "PPN",  100.0,   True,
              "Fixing: BSP/BAP reference (PDSPESO)"),
    CcyConfig("TWD", "Taiwan dollar NDF",     "USDTWD Curncy", "NTN",  100.0,   True,
              "Fixing: TAIFX1"),
    CcyConfig("KRW", "Korean won NDF",        "USDKRW Curncy", "KWN",  1.0,     True,
              "Fixing: KFTC18 (KRW KFTC18 Index)"),
]

CCY_MAP: Dict[str, CcyConfig] = {c.code: c for c in UNIVERSE}

# Standard NDF curve tenors used when interpolating points to IMM dates.
# Bloomberg convention for NDF points tickers is typically
#   "<root>+<tenor> Curncy"  e.g. "KWN+1M Curncy", "IRN+3M Curncy"
# (data agent to confirm the exact form on their terminal).
CURVE_TENORS = ["1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M", "9M", "12M"]

# Approximate tenor -> days used only as a fallback if the agent cannot
# supply actual settlement dates per tenor (actual dates are preferred).
TENOR_DAYS = {"1W": 7, "2W": 14, "1M": 30, "2M": 61, "3M": 91, "4M": 122,
              "5M": 152, "6M": 182, "9M": 273, "12M": 365}


def curve_ticker(root: str, tenor: str) -> str:
    return "{}+{} Curncy".format(root, tenor)
