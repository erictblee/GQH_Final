"""Check that our roll dates in 2015-2023 are real CME trading days.

    python tools/check_roll_dates.py

A roll date must be a day on which CME publishes a settlement price. This
script lists every roll date that falls on a weekend or on a CME holiday.
It only REPORTS. It changes no rule and no data.

!! The holiday list below is built from the standard rules for US market
!! holidays, not copied from CME's published calendar. Before relying on this
!! check for real data, compare HOLIDAYS against CME's holiday calendar for
!! each year and correct any difference (for example one-off closures such as
!! the 5 December 2018 day of mourning, which is not in this list).
"""

import sys
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)

# Let this script find config.py and src/ when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import market

FIRST_YEAR, LAST_YEAR = 2015, 2023


class CmeHolidays(AbstractHolidayCalendar):
    """Days with no CME settlement for ES, ZN and GC (rule-based, see above)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def find_mismatches(m):
    """Return a list of (asset, roll date, reason) for every roll date in
    2015-2023 that is not a CME trading day."""
    holidays = CmeHolidays().holidays(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31", return_name=True)

    mismatches = []
    n_checked = 0
    for asset in m.assets:
        roll_days = m.dates[m.roll[asset].to_numpy()]
        for day in roll_days:
            if not FIRST_YEAR <= day.year <= LAST_YEAR:
                continue
            n_checked += 1
            if day.dayofweek >= 5:                      # 5 = Saturday, 6 = Sunday
                mismatches.append((asset, day.date(), "weekend"))
            elif day in holidays.index:
                mismatches.append((asset, day.date(), holidays[day]))
    return mismatches, n_checked


if __name__ == "__main__":
    # In-sample only. 2015-2023 lies entirely inside the in-sample period.
    m = market.load_market(config.IS_END)
    mismatches, n_checked = find_mismatches(m)

    print(f"DATA SOURCE: {m.source}")
    print(f"Checked {n_checked} roll dates from {FIRST_YEAR} to {LAST_YEAR}.")
    if not mismatches:
        print("All of them fall on CME trading days.")
    else:
        print(f"{len(mismatches)} roll date(s) do NOT fall on a CME trading day:")
        for asset, day, reason in mismatches:
            print(f"  {asset}  {day}  ({reason})")
    if m.source == "SYNTHETIC":
        print("NOTE: synthetic data uses a made-up calendar (weekdays minus three "
              "holidays), so mismatches here are expected and mean nothing.")
