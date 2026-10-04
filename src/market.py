"""Market data: build the daily futures panel from individual contracts.

INPUT (in data/processed/, never committed):

    settlements.csv   one row per (date, contract):
                      date, asset, contract, settle, volume
    contracts.csv     one row per contract:
                      asset, contract, year, month, expiration
    source.txt        one word saying where the data came from, for example
                      SYNTHETIC or DATABENTO. It is printed on every output.

OUTPUT: a `Market` object. For every asset and every day it says which
contract we hold, its settlement prices, and the daily return.

The two price tables are the key idea (HYPOTHESIS.md, "Returns"):

    p_old[t] = today's settlement of the contract we held COMING INTO day t
    p_new[t] = today's settlement of the contract we hold AFTER day t

They are the same number on every day except a roll day. The daily return is

    r[t] = (p_old[t] - p_new[t-1]) / p_new[t-1]

Both prices in that formula belong to the SAME contract (the one held after
day t-1 is the one held coming into day t), so the price gap between the old
and the new contract on a roll day never shows up as a return.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class Market:
    """Everything the backtest needs. Tables have one row per day in `dates`
    and one column per asset."""

    dates: pd.DatetimeIndex   # master calendar (days on which any asset traded)
    assets: list              # asset names, in config.ASSETS order
    p_new: pd.DataFrame       # settlement of the contract held after day t
    p_old: pd.DataFrame       # settlement of the contract held coming into day t
    returns: pd.DataFrame     # daily return of the held contract
    roll: pd.DataFrame        # True on days we switch contract
    is_open: pd.DataFrame     # False when that market had no settlement that day
    volume: pd.DataFrame      # volume of the held contract
    held: pd.DataFrame        # name of the contract held after day t
    start: int                # row number of the common start date
    source: str               # "SYNTHETIC", "DATABENTO", ...


def find_roll_dates(asset, contracts, prices, last_day):
    """The roll date of every contract of one asset.

    contracts: this asset's rows of contracts.csv, sorted by expiration.
    prices:    table of settlements, one row per day, one column per contract
               (empty where that contract has no settlement record).
    last_day:  the last day of data we are allowed to see.

    Returns a dict {contract name: roll date}. The roll date is
      - a real date when we can work it out from the data we have,
      - None when it lies in the future (we cannot know it yet),
      - pd.Timestamp.min when the contract cannot be used at all (the data
        starts when the contract is already inside its roll window).
    """
    roll_dates = {}
    for row in contracts.itertuples():
        # "Business days" of a contract = days with a settlement record for it.
        business_days = prices[row.contract].dropna().index

        if config.ROLL_ANCHOR[asset] == "expiration":
            # ES: count back from the expiration date.
            anchor = row.expiration
            anchor_known = last_day >= anchor
            anchor_guess = anchor
        else:
            # ZN, GC: count back from first notice day, which is the last
            # business day of the month BEFORE the contract month.
            # (Two months before for crude oil and natural gas: config.ROLL_MONTHS_BEFORE.)
            months_back = getattr(config, "ROLL_MONTHS_BEFORE", {}).get(asset, 1)
            month_before = pd.Period(year=row.year, month=row.month, freq="M") - months_back
            days_in_month = business_days[business_days.to_period("M") == month_before]
            anchor = days_in_month[-1] if len(days_in_month) > 0 else None
            # We only know which day is the LAST business day of that month
            # once the month is over: that is, once the next weekday after
            # our last day of data falls in a later month. (This is a fact
            # about the calendar, not about future prices.)
            next_weekday = last_day + pd.offsets.BDay(1)
            anchor_known = next_weekday.to_period("M") > month_before
            anchor_guess = month_before.end_time

        if not anchor_known:
            # The roll is still in the future, as far as our data can tell.
            roll_dates[row.contract] = None
            # Honest limitation: if the data ends just before the anchor, the
            # roll date may already have passed without us being able to see
            # it. Warn loudly so nobody trusts those last few days.
            if 0 <= (anchor_guess - last_day).days <= 10:
                warnings.warn(
                    f"{row.contract}: data ends on {last_day.date()}, within 10 days "
                    f"of its roll anchor. The final roll may be missing."
                )
            continue

        # Business days strictly before the anchor; the 5th from the end is
        # "5 business days before the anchor".
        days_before = business_days[business_days < anchor] if anchor is not None else []
        if len(days_before) < config.ROLL_DAYS_BEFORE:
            roll_dates[row.contract] = pd.Timestamp.min   # unusable contract
        else:
            roll_dates[row.contract] = days_before[-config.ROLL_DAYS_BEFORE]

    return roll_dates


def build_asset(asset, contracts, prices, volumes, dates):
    """Walk through the master calendar for ONE asset, day by day, deciding
    which contract is held and recording its prices.

    Returns a table with columns p_old, p_new, roll, is_open, volume, held.
    """
    contracts = contracts.sort_values("expiration")
    roll_dates = find_roll_dates(asset, contracts, prices, dates[-1])
    order = list(contracts["contract"])   # nearest expiration first

    n = len(dates)
    p_old = np.zeros(n)
    p_new = np.zeros(n)
    roll = np.zeros(n, dtype=bool)
    is_open = np.zeros(n, dtype=bool)
    volume = np.zeros(n)
    held_names = []

    held = None        # contract we hold right now
    nearest = 0        # position in `order` of the nearest usable contract

    for t, day in enumerate(dates):
        # Which contract SHOULD we hold after today's settlement?
        # The nearest one whose roll date has not arrived yet.
        while roll_dates[order[nearest]] is not None and roll_dates[order[nearest]] <= day:
            nearest += 1
        wanted = order[nearest]

        # Today's settlement of a contract (NaN if it has no record today).
        def settle(contract):
            return prices.at[day, contract] if day in prices.index else np.nan

        if held is None:
            # Very first day: just start in the wanted contract.
            held = wanted
            if np.isnan(settle(held)):
                raise ValueError(f"{asset}: no settlement for {held} on first day {day.date()}")
            p_old[t] = p_new[t] = settle(held)
            is_open[t] = True

        elif np.isnan(settle(held)):
            # Market closed for our contract: hold the position, carry
            # yesterday's price forward, zero return. (Never use future data.)
            p_old[t] = p_new[t] = p_new[t - 1]
            is_open[t] = False

        else:
            is_open[t] = True
            p_old[t] = settle(held)          # price of the contract we came in with
            if wanted != held:
                # Roll day: close the old contract, open the new one, both at
                # today's settlement.
                if np.isnan(settle(wanted)):
                    raise ValueError(f"{asset}: cannot roll into {wanted} on {day.date()}: no settlement")
                held = wanted
                roll[t] = True
            p_new[t] = settle(held)          # price of the contract we leave with

        if is_open[t]:
            volume[t] = volumes.at[day, held]
        held_names.append(held)

    return pd.DataFrame(
        {"p_old": p_old, "p_new": p_new, "roll": roll, "is_open": is_open,
         "volume": volume, "held": held_names},
        index=dates,
    )


def load_market(cutoff=config.IS_END, data_dir=None, assets=None, warmup=None):
    """Load the processed data and build the Market.

    assets / warmup: left as None they mean the main strategy's
    config.ASSETS and config.WARMUP_DAYS. The ENSO strategy passes its own.

    cutoff:   last date to load. The default is the end of the in-sample
              period, so the out-of-sample holdout is never touched by
              accident. Pass cutoff=None ONLY for the final run.
    data_dir: folder holding the three input files (default: data/processed).
    """
    data_dir = config.PROCESSED_DIR if data_dir is None else data_dir
    assets = list(config.ASSETS) if assets is None else list(assets)
    warmup = config.WARMUP_DAYS if warmup is None else warmup

    settlements = pd.read_csv(data_dir / "settlements.csv", parse_dates=["date"])
    contracts = pd.read_csv(data_dir / "contracts.csv", parse_dates=["expiration"])
    source = (data_dir / "source.txt").read_text().strip()

    # Throw away everything after the cutoff BEFORE doing anything else, so
    # no later step can peek at it.
    if cutoff is not None:
        settlements = settlements[settlements["date"] <= pd.Timestamp(cutoff)]

    # Master calendar: every day on which at least one asset has a record,
    # starting from the first day on which ALL assets have data.
    first_day_of_each_asset = settlements.groupby("asset")["date"].min()
    first_common_day = first_day_of_each_asset[assets].max()
    all_days = np.sort(settlements["date"].unique())
    dates = pd.DatetimeIndex(all_days[all_days >= first_common_day])

    # Build each asset separately.
    built = {}
    for asset in assets:
        rows = settlements[settlements["asset"] == asset]
        # Turn the long list into a table: one row per day, one column per contract.
        prices = rows.pivot(index="date", columns="contract", values="settle")
        volumes = rows.pivot(index="date", columns="contract", values="volume").fillna(0.0)
        asset_contracts = contracts[
            (contracts["asset"] == asset)
            & contracts["month"].isin(config.CONTRACT_MONTHS[asset])   # eligible months only
            & contracts["contract"].isin(prices.columns)               # that we have data for
        ]
        built[asset] = build_asset(asset, asset_contracts, prices, volumes, dates)

    # Put the per-asset columns side by side: one table per quantity.
    def table(column):
        return pd.DataFrame({asset: built[asset][column] for asset in assets})

    p_old = table("p_old")
    p_new = table("p_new")

    # r[t] = (p_old[t] - p_new[t-1]) / p_new[t-1]. shift(1) means "yesterday".
    # The first day has no yesterday, so its return is set to 0 (it is inside
    # the warm-up and is never used).
    returns = ((p_old - p_new.shift(1)) / p_new.shift(1)).fillna(0.0)

    if len(dates) <= warmup:
        raise ValueError(f"not enough data: need more than {warmup} days for the warm-up")

    return Market(
        dates=dates,
        assets=assets,
        p_new=p_new,
        p_old=p_old,
        returns=returns,
        roll=table("roll"),
        is_open=table("is_open"),
        volume=table("volume"),
        held=table("held"),
        start=warmup,               # this many trading days after the first common day
        source=source,
    )
