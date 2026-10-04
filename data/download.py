"""Download raw CME futures data from Databento into data/raw/.

    python data/download.py            in-sample period only (the default)
    python data/download.py --final    also the out-of-sample period.
                                       Run ONCE, at the very end of the project.

What is downloaded (dataset GLBX.MDP3, every ES, ZN and GC futures contract):

    statistics   the exchange's daily statistics: official settlement prices
                 and cleared volume.
    definition   instrument definitions: each contract's expiration date.

The API key is read from the file `.env` in the repo folder (never committed):

    DATABENTO_API_KEY=db-...

Databento charges for every download. A file that is already in data/raw/ is
skipped, so running this script twice does not pay twice.

After this, run  python data/build_processed.py  to turn the raw files into
the tables the strategy reads.
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
# "ES.FUT" means "every futures contract whose root is ES" (Databento calls
# this a parent symbol). It includes calendar spreads; build_processed.py
# throws those away.
SYMBOLS = [f"{asset}.FUT" for asset in config.ASSETS]
SCHEMAS = ["statistics", "definition"]
RAW_DIR = REPO_DIR / "data" / "raw"


def date_chunks(final):
    """Split the period into pieces of at most one calendar year.

    Returns a list of (start, end) dates; `end` is NOT included in the piece.
    The in-sample pieces stop the day after config.IS_END, so that in-sample
    files never contain out-of-sample days.
    """
    first = pd.Timestamp(config.DATA_START)
    in_sample_stop = pd.Timestamp(config.IS_END) + pd.Timedelta(days=1)
    last = pd.Timestamp(date.today()) if final else in_sample_stop

    # Cut points: the first day, every 1 January, the in-sample stop, the last day.
    cuts = {first, last}
    cuts.update(pd.Timestamp(year=y, month=1, day=1) for y in range(first.year + 1, last.year + 1))
    if in_sample_stop < last:
        cuts.add(in_sample_stop)
    cuts = sorted(c for c in cuts if first <= c <= last)
    return list(zip(cuts[:-1], cuts[1:]))


def download_fills(client):
    """Download daily price bars (schema ohlcv-1d) for the few days on which
    Databento has no settlement record. The days and contracts are listed in
    data/gap_fills.csv. Three extra days on each side are downloaded too, so
    that closes can be compared with settlements on normal days.
    See the README's Deviations log."""
    gaps = pd.read_csv(REPO_DIR / "data" / "gap_fills.csv", parse_dates=["date"])
    for symbol, rows in gaps.groupby("symbol"):
        start = (rows["date"].min() - pd.Timedelta(days=3)).date().isoformat()
        end = (rows["date"].max() + pd.Timedelta(days=4)).date().isoformat()
        path = RAW_DIR / f"fill_{symbol}_{start}_{end}.dbn.zst"
        if path.exists():
            print(f"have     {path.name}")
            continue
        request = dict(dataset=DATASET, symbols=[symbol], stype_in="raw_symbol",
                       schema="ohlcv-1d", start=start, end=end)
        cost = client.metadata.get_cost(**request)
        print(f"download {path.name}  (${cost:.4f})")
        client.timeseries.get_range(**request, path=str(path))
        with open(REPO_DIR / "data" / "DOWNLOAD_DATE.txt", "a") as log:
            log.write(f"{date.today().isoformat()}  {DATASET}  {path.name}  ${cost:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="also download the out-of-sample period (final run only)")
    parser.add_argument("--fills", action="store_true",
                        help="download daily bars for the gap days listed in data/gap_fills.csv")
    parser.add_argument("--years", type=int, nargs="*",
                        help="download only pieces that start in these years")
    parser.add_argument("--only", choices=SCHEMAS,
                        help="download just one kind of data (lets two copies run side by side)")
    args = parser.parse_args()
    schemas = [args.only] if args.only else SCHEMAS

    load_dotenv(REPO_DIR / ".env")
    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("put DATABENTO_API_KEY=... in the file .env in the repo folder")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.fills:
        download_fills(client)
        return
    total_cost = 0.0
    for schema in schemas:
        for start, end in date_chunks(args.final):
            if args.years and start.year not in args.years:
                continue
            path = RAW_DIR / f"{schema}_{start.date()}_{end.date()}.dbn.zst"
            if path.exists():
                print(f"have     {path.name}")
                continue

            request = dict(dataset=DATASET, symbols=SYMBOLS, stype_in="parent",
                           schema=schema, start=start.date().isoformat(), end=end.date().isoformat())
            # Ask the price first (this call is free), then download.
            cost = client.metadata.get_cost(**request)
            total_cost += cost
            print(f"download {path.name}  (${cost:.2f})")
            # Download to a temporary name, then rename: a file with the final
            # name is therefore always complete. The connection sometimes
            # drops on big files, so try up to 5 times.
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
                log.write(f"{date.today().isoformat()}  {DATASET}  {path.name}  ${cost:.2f}\n")

    print(f"Done. Charged about ${total_cost:.2f} this run.")


if __name__ == "__main__":
    main()
