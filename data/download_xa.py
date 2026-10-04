"""Download the futures data of the cross-asset strategy into data/raw_xa/.

    python data/download_xa.py              development + validation years (default)
    python data/download_xa.py --test       also the locked test years (2020 to 2024-09-30)
    python data/download_xa.py --final      also the final out-of-sample period.
    python data/download_xa.py --years 2012 2013     only pieces starting in those years

What is ALREADY on disk is not bought again:
    ES, ZN, GC          all of it (data/processed/, from the main strategy)
    ZC, ZS, ZW          2021-10-01 onward (data/raw_ag/, from the ENSO strategy)
So this script buys NQ, ZB, CL, NG and HG, plus the grains before 2021-10-01.

As in data/download_ag.py, only the `statistics` schema is downloaded and
only for contracts we can hold, asked for by name. One file per year.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import config

DATASET = "GLBX.MDP3"
MONTH_LETTER = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
                7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
NEW_ASSETS = ["NQ", "ZB", "CL", "NG", "HG"]
GRAINS_ALREADY_FROM = pd.Timestamp(config.AG_DATA_START)    # grains after this are in data/raw_ag/


def third_friday(year, month):
    """The third Friday of a month (the expiry day of equity index futures)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_friday = (4 - first.dayofweek) % 7          # Monday = 0 ... Friday = 4
    return first + pd.Timedelta(days=days_to_friday + 14)


def contract_list(assets, first_month, last_month):
    """Every contract of `assets` with a delivery month from first_month to
    last_month (pandas Periods). Columns: asset, symbol, contract, year,
    month, expiration, last_possible_day."""
    rows = []
    for asset in assets:
        for month in pd.period_range(first_month, last_month, freq="M"):
            if month.month not in config.CONTRACT_MONTHS[asset]:
                continue
            if config.ROLL_ANCHOR[asset] == "expiration":
                expiration = third_friday(month.year, month.month)
            else:
                # Only used to put contracts in order (rolls are counted
                # back from first notice day, see src/market.py).
                expiration = pd.Timestamp(year=month.year, month=month.month, day=15) - pd.offsets.BDay(1)
            rows.append({
                "asset": asset,
                "symbol": f"{asset}{MONTH_LETTER[month.month]}{month.year % 10}",   # e.g. CLZ5
                "contract": f"{asset}_{month.year}_{month.month:02d}",              # e.g. CL_2015_12
                "year": month.year, "month": month.month, "expiration": expiration,
                # No contract trades after its delivery month. Used to tell
                # apart two contracts that share a symbol ten years apart.
                "last_possible_day": month.end_time.normalize(),
            })
    return pd.DataFrame(rows)


def pieces(test, final):
    """(start, end) of each file, `end` NOT included: one per calendar year,
    never crossing the end of a sample period."""
    first = pd.Timestamp(config.DATA_START)
    stops = [pd.Timestamp(config.XA_TEST_START)]                          # default: up to end of validation
    if test or final:
        stops.append(pd.Timestamp(config.IS_END) + pd.Timedelta(days=1))
    if final:
        stops.append(pd.Timestamp(config.AG_OOS_END) + pd.Timedelta(days=1))
    last = stops[-1]
    cuts = {first, *stops}
    cuts.update(pd.Timestamp(year=y, month=1, day=1) for y in range(first.year + 1, last.year + 1))
    cuts.add(GRAINS_ALREADY_FROM)
    cuts = sorted(c for c in cuts if first <= c <= last)
    return list(zip(cuts[:-1], cuts[1:]))


def symbols_for(start, end):
    """Contracts we might hold between start and end: delivery months from
    the month after `start` to 8 months after `end`."""
    assets = list(NEW_ASSETS)
    if start < GRAINS_ALREADY_FROM:
        assets += config.AG_ASSETS
    table = contract_list(assets, pd.Period(start, freq="M") + 1, pd.Period(end, freq="M") + 8)
    return list(table["symbol"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="also the locked test years")
    parser.add_argument("--final", action="store_true", help="also the final out-of-sample period")
    parser.add_argument("--years", type=int, nargs="*", help="only pieces that start in these years")
    parser.add_argument("--price-only", action="store_true", help="print the price of each piece and stop")
    args = parser.parse_args()

    load_dotenv(REPO_DIR / ".env")
    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("put DATABENTO_API_KEY=... in the file .env in the repo folder")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    config.XA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for start, end in pieces(args.test, args.final):
        if args.years and start.year not in args.years:
            continue
        path = config.XA_RAW_DIR / f"statistics_{start.date()}_{end.date()}.dbn.zst"
        if path.exists():
            print(f"have     {path.name}")
            continue
        request = dict(dataset=DATASET, symbols=symbols_for(start, end), stype_in="raw_symbol",
                       schema="statistics", start=start.date().isoformat(), end=end.date().isoformat())
        cost = client.metadata.get_cost(**request)      # free call
        total += cost
        if args.price_only:
            size = client.metadata.get_billable_size(**request)
            print(f"{path.name}  ${cost:.2f}  {size / 1e6:.0f} MB before compression")
            continue
        print(f"download {path.name}  (${cost:.2f}, {len(request['symbols'])} contracts)", flush=True)
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
        with open(REPO_DIR / "data" / "DOWNLOAD_DATE.txt", "a") as log:
            log.write(f"{date.today().isoformat()}  {DATASET}  raw_xa/{path.name}  ${cost:.2f}\n")
    print(f"Total for this run: ${total:.2f}")


if __name__ == "__main__":
    main()
