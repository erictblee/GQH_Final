"""The ENSO signal: NOAA's monthly Nino-3.4 index, used point-in-time.

ENSO (El Nino / Southern Oscillation) is a swing in Pacific sea temperature
that shifts rainfall around the world, and so affects harvests.

Two things are done here (ENSO_SPEC.md sections 7 and 18):

  1. Z-SCORE. Each month's Nino-3.4 anomaly is compared with the previous 30
     years of anomalies:  z = (anomaly - average) / standard deviation.
     The 30 years END at that month, so no later month is used.

  2. AVAILABILITY. NOAA publishes month M's value early in month M+1. We
     allow it to be used only from the 15th of month M+1. Between releases
     the last available value stays in force.

State:  z >= +0.5 "el_nino",  z <= -0.5 "la_nina",  otherwise "neutral".
"""

import numpy as np
import pandas as pd

import config


def load_enso(path=None):
    """Read data/external/nino34.csv and add the z-score, the state and the
    date from which each month's value may be used. One row per month."""
    path = config.EXTERNAL_DIR / "nino34.csv" if path is None else path
    return add_signal(pd.read_csv(path))


def add_signal(table):
    """table: columns year, month, anomaly (oldest month first)."""
    table = table.copy()

    # rolling(360) looks at this month and the 359 months before it. Until
    # 360 months exist the result is empty (NaN) and the state is "neutral".
    window = table["anomaly"].rolling(config.ENSO_Z_MONTHS)
    table["z"] = (table["anomaly"] - window.mean()) / window.std(ddof=1)

    table["state"] = "neutral"
    table.loc[table["z"] >= config.ENSO_THRESHOLD, "state"] = "el_nino"
    table.loc[table["z"] <= -config.ENSO_THRESHOLD, "state"] = "la_nina"

    # First day of the month, plus one month, plus (15 - 1) days = the 15th
    # of the following month.
    first_of_month = pd.to_datetime(dict(year=table["year"], month=table["month"], day=1))
    table["available"] = (first_of_month + pd.offsets.MonthBegin(1)
                          + pd.Timedelta(days=config.ENSO_RELEASE_DAY - 1))
    return table


def on_dates(table, dates):
    """The ENSO state and z-score in force on each trading day.

    For each day we take the LAST month whose `available` date is on or
    before that day. Returns (list of states, array of z-scores).
    """
    available = table["available"].to_numpy()
    # searchsorted counts how many months are available by each day; minus 1
    # is the row number of the newest one.
    rows = np.searchsorted(available, pd.DatetimeIndex(dates).to_numpy(), side="right") - 1
    if (rows < 0).any():
        raise ValueError("a trading day comes before the first ENSO value")
    return list(table["state"].to_numpy()[rows]), table["z"].to_numpy()[rows]
