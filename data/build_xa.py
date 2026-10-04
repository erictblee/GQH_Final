"""Build the data tables of the cross-asset strategy in data/processed_xa/.

    python data/build_xa.py            everything up to config.IS_END (the default)
    python data/build_xa.py --final    including the final out-of-sample period

It puts three sources into one pair of tables (same layout as data/processed/):

    ES, ZN, GC                  copied from data/processed/ (main strategy)
    ZC, ZS, ZW from 2021-10     read from data/raw_ag/      (ENSO strategy)
    everything else             read from data/raw_xa/      (data/download_xa.py)

The settlement price of each contract and day is chosen exactly as in
data/build_processed.py. Nothing is filled in or smoothed.

NOTE: the tables may contain days after the validation period for some
assets. That is fine: run_xa.py cuts the data at the end of validation
unless it is given --test or --final.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "data"))
import config
import build_processed          # reader of Databento statistics files
import download_xa              # list of contracts, built from CME's rules

ALREADY_BUILT = ["ES", "ZN", "GC"]      # taken as they are from data/processed/


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the final out-of-sample period (final run only)")
    args = parser.parse_args()

    stats = pd.concat([
        build_processed.read_statistics(args.final, config.XA_RAW_DIR),
        build_processed.read_statistics(args.final, config.AG_RAW_DIR),
    ])
    if not args.final:
        stats = stats[stats["date"] <= pd.Timestamp(config.IS_END)]
    stats = stats[stats["date"].dt.dayofweek < 5]       # no settlements on weekends

    # One settlement per contract per day: final beats preliminary, rounded
    # beats unrounded, later beats earlier (the preferred record sorts LAST).
    settle = stats[stats["stat_type"] == build_processed.SETTLEMENT_PRICE]
    settle = settle.sort_values(["is_final", "is_rounded", "published"])
    settle = settle.groupby(["symbol", "date"], as_index=False).last()

    volume = stats[stats["stat_type"] == build_processed.CLEARED_VOLUME].sort_values("published")
    volume = volume.groupby(["symbol", "date"], as_index=False).last()
    volume = volume[["symbol", "date", "quantity"]].rename(columns={"quantity": "volume"})

    table = settle.merge(volume, on=["symbol", "date"], how="left")
    table["volume"] = table["volume"].fillna(0).astype(int)

    # Which contract is a symbol? A name like CLZ0 is reused every ten years
    # (December 2010, December 2020). For each record, take the contract
    # with that symbol whose delivery month ends first on or after the day.
    new_assets = [a for a in config.XA_ASSETS if a not in ALREADY_BUILT]
    contracts = download_xa.contract_list(
        new_assets, pd.Period(config.DATA_START, freq="M"), pd.Period(config.AG_OOS_END, freq="M") + 12)
    # (Both date columns must be stored the same way before they can be matched.)
    table["date"] = table["date"].astype("datetime64[ns]")
    contracts["last_possible_day"] = contracts["last_possible_day"].astype("datetime64[ns]")
    table = pd.merge_asof(
        table.sort_values("date"), contracts.sort_values("last_possible_day"),
        by="symbol", left_on="date", right_on="last_possible_day", direction="forward")
    table = table.dropna(subset=["contract"])

    new_rows = table[["date", "asset", "contract", "price", "volume"]].rename(columns={"price": "settle"})
    new_rows["price_type"] = "settlement"

    # Add ES, ZN and GC as the main strategy built them.
    old_rows = pd.read_csv(config.PROCESSED_DIR / "settlements.csv", parse_dates=["date"])
    old_contracts = pd.read_csv(config.PROCESSED_DIR / "contracts.csv", parse_dates=["expiration"])
    if not args.final:
        old_rows = old_rows[old_rows["date"] <= pd.Timestamp(config.IS_END)]
    settlements = pd.concat([old_rows, new_rows]).sort_values(["date", "asset", "contract"])
    columns = ["asset", "contract", "year", "month", "expiration"]
    all_contracts = pd.concat([old_contracts[columns], contracts[columns]])
    used = all_contracts[all_contracts["contract"].isin(settlements["contract"])]

    out = config.XA_PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    settlements.to_csv(out / "settlements.csv", index=False, date_format="%Y-%m-%d")
    used.to_csv(out / "contracts.csv", index=False, date_format="%Y-%m-%d")
    (out / "source.txt").write_text("DATABENTO\n")

    checksum = hashlib.sha256((out / "settlements.csv").read_bytes()).hexdigest()
    label = "xa_full" if args.final else "xa_in_sample"
    fingerprint = (f"{label}  rows={len(settlements)}  first={settlements['date'].min().date()}  "
                   f"last={settlements['date'].max().date()}  sha256={checksum}")
    fingerprint_file = REPO_DIR / "data" / "DATA_FINGERPRINT.txt"
    old_lines = fingerprint_file.read_text().splitlines() if fingerprint_file.exists() else []
    lines = [line for line in old_lines if not line.startswith(label + " ")] + [fingerprint]
    fingerprint_file.write_text("\n".join(lines) + "\n")

    print(f"DATA SOURCE: DATABENTO ({label})")
    print(settlements.groupby("asset")["date"].agg(["min", "max", "count"]))
    print(fingerprint)


if __name__ == "__main__":
    main()
