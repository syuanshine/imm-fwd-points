"""IMM date calendar utilities.

IMM dates are the third Wednesday of March, June, September and December
(CME convention, see https://en.wikipedia.org/wiki/IMM_dates).

An "IMM forward points" quote observed on date t refers to the calendar
spread between the two nearest IMM dates strictly after t:
    near leg = 1st IMM date after t
    far  leg = 2nd IMM date after t
e.g. any date between the Jun and Sep IMM dates sees "Sep-Dec" as the
front pair; on the Sep IMM date the pair rolls to "Dec-Mar".

NOTE on settlement: for NDFs the *fixing* is typically 2 business days
before the value date per each market's convention (KRW KFTC18, TWD
TAIFX1, INR RBIB, IDR JISDOR/DNDF conventions, PHP BAP, THB onshore ref).
For analytics we treat the IMM date itself as the leg's value date; if
the data-pulling agent has precise settlement dates from Bloomberg
(SETTLE_DT), those can be passed instead - nothing else changes.
"""
from datetime import date, timedelta
from typing import List, Tuple

IMM_MONTHS = (3, 6, 9, 12)
_MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}


def third_wednesday(year: int, month: int) -> date:
    """Third Wednesday of a given month."""
    d = date(year, month, 1)
    # weekday(): Mon=0 ... Wed=2
    first_wed = d + timedelta(days=(2 - d.weekday()) % 7)
    return first_wed + timedelta(days=14)


def imm_calendar(start_year: int, end_year: int) -> List[date]:
    """All IMM dates (3rd Wed of Mar/Jun/Sep/Dec) for the year range, inclusive."""
    return [third_wednesday(y, m)
            for y in range(start_year, end_year + 1)
            for m in IMM_MONTHS]


def next_imm_dates(d: date, count: int = 2) -> List[date]:
    """The `count` nearest IMM dates strictly after d."""
    cal = imm_calendar(d.year - 1, d.year + 2)
    future = [x for x in cal if x > d]
    return future[:count]


def front_pair(d: date) -> Tuple[date, date]:
    """(near, far) IMM legs of the front pair active on observation date d."""
    near, far = next_imm_dates(d, 2)
    return near, far


def pair_at_slot(d: date, slot: int = 0) -> Tuple[date, date]:
    """(near, far) legs of the `slot`-th IMM pair seen from date d.

    slot=0 is the front pair (the one that rolls each IMM date); slot=1 is
    the deferred pair, i.e. the SAME calendar spread one quarter before it
    becomes front. Tracking slot>=1 is what lets a vintage be followed for
    more than ~91 days of life (see series.vintage_paths).
    """
    legs = next_imm_dates(d, slot + 2)
    return legs[slot], legs[slot + 1]


def pair_label(near: date, style: str = "long") -> str:
    """Label a pair by its near leg. 'Sep25-Dec25' (long) or 'U5Z5' (code)."""
    far = next_imm_dates(near - timedelta(days=1), 2)[1]
    if style == "code":
        return "{}{}{}{}".format(_MONTH_CODE[near.month], near.year % 10,
                                 _MONTH_CODE[far.month], far.year % 10)
    return "{}{}-{}{}".format(near.strftime("%b"), near.strftime("%y"),
                              far.strftime("%b"), far.strftime("%y"))


def days_between_legs(near: date, far: date) -> int:
    return (far - near).days
