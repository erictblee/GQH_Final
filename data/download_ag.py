"""Download corn, soybean and wheat futures data from Databento into data/raw_ag/.

    python data/download_ag.py            in-sample period only (the default)
    python data/download_ag.py --final    also the out-of-sample period.
                                          Run ONCE, at the very end of the project.

This is the data for the ENSO strategy (ENSO_SPEC.md). It is kept small on
purpose:

  - ONE file per period (one in-sample, one out-of-sample), not one per year.
  - Only the `statistics` schema (official settlement prices and volume).
    No `definition` files: grain contract months and expiry dates follow a
    fixed CME rule, which data/build_ag.py writes out itself.
  - Only the contracts we can actually hold, asked for by name (ZCZ1, ZSX1,
    ...). No calendar spreads and no far-away contracts.

The API key is read from `.env` in the repo folder, as for data/download.py.
A file that is already in data/raw_ag/ is skipped, so running this twice
does not pay twice.

After this, run  python data/build_ag.py
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

# Let this script find config.py when run from anywhere.
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import config

DATASET = "GLBX.MDP3"

# CME's letter for each contract month (1 = January).
MONTH_LETTER = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def contract_list(first_month, last_month):
    """Every grain contract with a delivery month from first_month to
    last_month (both are pandas Periods, e.g. 2021-11).

    Returns a table: asset, symbol, contract, year, month, expiration.
    """
    rows = []
    for asset in config.AG_ASSETS:
        for month in pd.period_range(first_month, last_month, freq="M"):
            if month.month not in config.CONTRACT_MONTHS[asset]:
                continue
            # CME rule for grains: the last trading day is the business day
            # before the 15th of the contract month. Here "business day"
            # means a weekday; the date is only used to put the contracts
            # in order, never to decide a trade.
            expiration = pd.Timestamp(year=month.year, month=month.month, day=15) - pd.offsets.BDay(1)
            rows.append({
                "asset": asset,
                # CME's name: root + month letter + last digit of the year, e.g. ZCZ1.
                "symbol": f"{asset}{MONTH_LETTER[month.month]}{month.year % 10}",
                # Our own name with the full year, e.g. ZC_2021_12.
                "contract": f"{asset}_{month.year}_{month.month:02d}",
                "year": month.year,
                "month": month.month,
                "expiration": expiration,
            })
    return pd.DataFrame(rows)


def periods(final):
    """The pieces to download: (start, end) with `end` NOT included. The
    in-sample piece stops the day after config.IS_END, so the in-sample file
    never contains an out-of-sample day."""
    in_sample_stop = pd.Timestamp(config.IS_END) + pd.Timedelta(days=1)
    pieces = [(pd.Timestamp(config.AG_DATA_START), in_sample_stop)]
    if final:
        pieces.append((in_sample_stop, pd.Timestamp(config.AG_OOS_END) + pd.Timedelta(days=1)))
    return pieces


def symbols_for(start, end):
    """Contracts we might hold between start and end: delivery months from
    the month after `start` to 8 months after `end` (we always hold a
    contract that has not reached its delivery month, and we need the next
    one to roll into)."""
    first = pd.Period(start, freq="M") + 1
    last = pd.Period(end, freq="M") + 8
    return list(contract_list(first, last)["symbol"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="also download the out-of-sample period (final run only)")
    args = parser.parse_args()

    load_dotenv(REPO_DIR / ".env")
    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("put DATABENTO_API_KEY=... in the file .env in the repo folder")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    config.AG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for start, end in periods(args.final):
        path = config.AG_RAW_DIR / f"statistics_{start.date()}_{end.date()}.dbn.zst"
        if path.exists():
            print(f"have     {path.name}")
            continue

        request = dict(dataset=DATASET, symbols=symbols_for(start, end), stype_in="raw_symbol",
                       schema="statistics", start=start.date().isoformat(), end=end.date().isoformat())
        # Ask the price first (this call is free), then download.
        cost = client.metadata.get_cost(**request)
        print(f"download {path.name}  (${cost:.2f}, {len(request['symbols'])} contracts)")
        # Download to a temporary name, then rename: a file with the final
        # name is therefore always complete.
        partial = path.with_suffix(".partial")
        for attempt in range(1, 6):
            try:
                client.timeseries.get_range(**request, path=str(partial))
                break
            except db.BentoError as error:
                print(f"  attempt {attempt} failed ({error}); trying again")
                if attempt == 5:
                    raise
        partial.rename(path)

        # Record what was downloaded and when (this small file IS committed).
        with open(REPO_DIR / "data" / "DOWNLOAD_DATE.txt", "a") as log:
            log.write(f"{date.today().isoformat()}  {DATASET}  raw_ag/{path.name}  ${cost:.2f}\n")


if __name__ == "__main__":
    main()
