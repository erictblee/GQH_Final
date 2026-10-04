"""Turn the raw grain files into the tables the ENSO strategy reads.

    python data/build_ag.py            in-sample only (the default)
    python data/build_ag.py --final    full sample (final run only)

Reads   data/raw_ag/statistics_*.dbn.zst
Writes  data/processed_ag/settlements.csv, contracts.csv and source.txt
        (the same three files, in the same layout, as data/processed/)
        and one line in data/DATA_FINGERPRINT.txt.

The settlement price of each contract and day is chosen exactly as in
data/build_processed.py: a FINAL record if there is one, otherwise a rounded
preliminary one, otherwise the latest record. Nothing is filled in or
smoothed. A day with no record is left missing and the strategy treats it
as "market closed".
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
import build_processed          # reuse its reader of Databento statistics files
import download_ag              # reuse its list of grain contracts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the out-of-sample period (final run only)")
    args = parser.parse_args()

    stats = build_processed.read_statistics(args.final, config.AG_RAW_DIR)
    if not args.final:
        # Belt and braces: drop anything dated after the in-sample period.
        stats = stats[stats["date"] <= pd.Timestamp(config.IS_END)]
    stats = stats[stats["date"] >= pd.Timestamp(config.AG_DATA_START)]
    stats = stats[stats["date"].dt.dayofweek < 5]       # no settlements on weekends

    # One settlement price per contract per day. Sorting puts the preferred
    # record LAST in each group (final beats preliminary, rounded beats
    # unrounded, later beats earlier).
    settle = stats[stats["stat_type"] == build_processed.SETTLEMENT_PRICE]
    settle = settle.sort_values(["is_final", "is_rounded", "published"])
    settle = settle.groupby(["symbol", "date"], as_index=False).last()

    # Cleared volume: the latest figure published for that day.
    volume = stats[stats["stat_type"] == build_processed.CLEARED_VOLUME].sort_values("published")
    volume = volume.groupby(["symbol", "date"], as_index=False).last()
    volume = volume[["symbol", "date", "quantity"]].rename(columns={"quantity": "volume"})

    table = settle.merge(volume, on=["symbol", "date"], how="left")
    table["volume"] = table["volume"].fillna(0).astype(int)

    # Which contract is a symbol? A name like ZCZ1 is reused every ten years,
    # but our data covers five, so inside it each name means one contract.
    contracts = download_ag.contract_list(
        pd.Period(config.AG_DATA_START, freq="M"), pd.Period(config.AG_OOS_END, freq="M") + 12)
    if contracts["symbol"].duplicated().any():
        raise ValueError("a contract symbol is used twice; the data window is too long")
    table = table.merge(contracts, on="symbol", how="inner")

    settlements = table[["date", "asset", "contract", "price", "volume"]].rename(columns={"price": "settle"})
    settlements = settlements.sort_values(["date", "asset", "contract"])
    used = contracts[contracts["contract"].isin(settlements["contract"])]

    out = config.AG_PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    settlements.to_csv(out / "settlements.csv", index=False, date_format="%Y-%m-%d")
    used[["asset", "contract", "year", "month", "expiration"]].to_csv(
        out / "contracts.csv", index=False, date_format="%Y-%m-%d")
    (out / "source.txt").write_text("DATABENTO\n")

    # Fingerprint: a checksum that changes if a single price changes.
    checksum = hashlib.sha256((out / "settlements.csv").read_bytes()).hexdigest()
    label = "ag_full" if args.final else "ag_in_sample"
    fingerprint = (f"{label}  rows={len(settlements)}  first={settlements['date'].min().date()}  "
                   f"last={settlements['date'].max().date()}  sha256={checksum}")
    fingerprint_file = REPO_DIR / "data" / "DATA_FINGERPRINT.txt"
    old_lines = fingerprint_file.read_text().splitlines() if fingerprint_file.exists() else []
    lines = [line for line in old_lines if not line.startswith(label + " ")] + [fingerprint]
    fingerprint_file.write_text("\n".join(lines) + "\n")

    print(f"DATA SOURCE: DATABENTO ({label})")
    print(f"{len(settlements)} settlement rows, {len(used)} contracts")
    print(fingerprint)


if __name__ == "__main__":
    main()
