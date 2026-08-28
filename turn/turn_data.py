"""Data contract for the standalone year-end turn module.

The estimators in turn_methods.py consume the SAME curve frame as the IMM
module (spot, <tenor>_pts, optional <tenor>_days), so the demo reuses
imm_fwd.data.SyntheticProvider and production reuses the agent's
BloombergProvider unchanged.

=== PRODUCTION UPGRADES FOR THE DATA-PULLING AGENT (in priority order) ===

1. ACTUAL SETTLEMENT DATES per tenor per day ('<tenor>_days'). The whole
   method keys off which inter-tenor interval contains Dec 31; approximating
   settlement with fixed day counts can misplace the turn interval by a few
   days right when it matters most. Pull SETTLE_DT per tenor.

2. DENSER SHORT END from ~October: ON, TN, SW, 2W, 3W points. The closer
   the flanking tenors sit to Dec 31, the cleaner the extraction (the jump
   is diluted 1:interval_days inside its interval). By late December the
   turn sits inside the 1W/TN dates and is essentially directly observable.

3. DIRECT TURN QUOTES (Method A) where they exist: dealers quote the turn
   outright (e.g. broken-date forward-forwards Dec 30 -> Jan 4). If your
   source carries them (ALLQ broken dates / bank runs), store them as the
   ground truth against which B/C estimates are checked - this is exactly
   how LSEG calibrates its turn-adjusted curves against FXall trade prints.

4. TURN DAY COUNT: the number of calendar days the year-end value-date gap
   actually covers (Dec 31 on a Friday -> 3-4 days). Needed to annualize
   jumps comparably across years (jump_to_ann_pct(turn_days=...)).

NOTE ON NDFs: for KRW/TWD/INR/IDR/PHP the "turn" measured here is in the
offshore NDF curve - it blends the local year-end AND the year-end USD
funding premium (the USD leg's turn), plus any regulatory positioning in
the NDF market itself. For THB offshore the same applies. Attribution
between the two legs needs the fair-value machinery (SOFR futures carry
their own Dec turn) - see the roadmap README, Phase 4.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "imm_fwd"))

from config import CCY_MAP, CURVE_TENORS, TENOR_DAYS, UNIVERSE  # noqa: E402
from data import BloombergProvider, SyntheticProvider           # noqa: E402
