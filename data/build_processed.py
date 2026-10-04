"""Turn the raw Databento files into the tables the strategy reads.

    python data/build_processed.py            in-sample only (the default)
    python data/build_processed.py --final    full sample (final run only)

Reads   data/raw/statistics_*.dbn.zst  and  data/raw/definition_*.dbn.zst
Writes  data/processed/settlements.csv   date, asset, contract, settle, volume
        data/processed/contracts.csv     asset, contract, year, month, expiration
        data/processed/source.txt        DATABENTO
        data/DATA_FINGERPRINT.txt        a checksum of settlements.csv, so other
                                         people can confirm they have the same data

HOW ONE SETTLEMENT PRICE PER DAY IS CHOSEN
CME publishes several settlement records for the same contract and day:
preliminary ones in the afternoon and a final one before the next session.
Some are "rounded" to the contract's tradable tick and some are not. For
each contract and trading day we keep, in this order of preference:
    1. a record flagged FINAL;
    2. otherwise a preliminary record flagged ROUNDED;
    3. otherwise whatever record there is;
and within the preferred group, the latest one published.

OTHER CLEANING (nothing else is changed, filled in or smoothed)
  - Calendar spreads are dropped; only outright futures are kept.
  - Records dated on a Saturday or Sunday are dropped.
  - GAP FILLS: data/gap_fills.csv lists the few days on which Databento has
    no settlement record for the contract we hold. Where Databento has a
    daily price bar (schema ohlcv-1d) for that contract and day, its CLOSE is
    used in place of the missing settlement, and the row is marked
    price_type = daily_close. Where it has no bar either, the day stays
    missing and the strategy treats it as "market closed": position held,
    zero return (HYPOTHESIS.md, "Data handling").
  Every one of these choices is recorded in the README's Deviations log.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import databento as db
import pandas as pd

# Let this script find config.py when run from anywhere.
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import config

RAW_DIR = REPO_DIR / "data" / "raw"

# Databento's codes for the two statistics we need.
SETTLEMENT_PRICE = 3
CLEARED_VOLUME = 6
# Bits inside a settlement record's `stat_flags` number.
FLAG_FINAL = 1          # final (not preliminary) settlement
FLAG_ROUNDED = 4        # rounded to the contract's tradable tick

# CME's letter for each contract month, used to double-check the definitions.
MONTH_LETTER = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def raw_files(kind, final, raw_dir=RAW_DIR):
    """The raw files of one kind, oldest first. Without --final, files that
    start after the in-sample period are not even opened."""
    files = []
    for path in sorted(raw_dir.glob(f"{kind}_*.dbn.zst")):
        start = pd.Timestamp(path.name.split("_")[1])       # the file's first day
        if final or start <= pd.Timestamp(config.IS_END):
            files.append(path)
    if not files:
        raise SystemExit(f"no {kind} files in {raw_dir}: run the download script first")
    return files


def read_contracts(final):
    """One row per futures contract: its name, month and expiration date."""
    pieces = []
    for path in raw_files("definition", final):
        table = db.DBNStore.from_file(path).to_df().reset_index()
        # "F" = an outright future. (The files also hold calendar spreads.)
        table = table[table["instrument_class"] == "F"]
        pieces.append(table[["ts_recv", "raw_symbol", "asset", "expiration",
                             "maturity_year", "maturity_month"]])
    table = pd.concat(pieces).sort_values("ts_recv")

    # A contract is defined again every day. Keep its most recent definition.
    table = table.drop_duplicates(["raw_symbol", "maturity_year", "maturity_month"], keep="last")
    table = table[table["asset"].isin(config.ASSETS)]

    contracts = pd.DataFrame({
        "asset": table["asset"],
        "symbol": table["raw_symbol"],                  # CME's name, e.g. ESH5
        "year": table["maturity_year"].astype(int),
        "month": table["maturity_month"].astype(int),
        "expiration": table["expiration"].dt.tz_localize(None).dt.normalize(),
    })
    # CME reuses a name like ESH5 every ten years, so we make our own unique
    # name from the full year and month, e.g. ES_2025_03.
    contracts["contract"] = [f"{a}_{y}_{m:02d}" for a, y, m in
                             zip(contracts["asset"], contracts["year"], contracts["month"])]

    # DATA INTEGRITY CHECK: the month letter and year digit inside CME's
    # symbol must agree with the definition's month and year.
    for row in contracts.itertuples():
        letter, digit = row.symbol[len(row.asset)], row.symbol[len(row.asset) + 1:]
        if MONTH_LETTER[letter] != row.month or int(digit) != row.year % 10:
            raise ValueError(f"symbol {row.symbol} does not match its definition "
                             f"({row.year}-{row.month:02d})")

    return contracts.sort_values(["asset", "expiration"]).reset_index(drop=True)


def read_statistics(final, raw_dir=RAW_DIR):
    """Every settlement and cleared-volume record of outright futures."""
    pieces = []
    for path in raw_files("statistics", final, raw_dir):
        table = db.DBNStore.from_file(path).to_df(map_symbols=True).reset_index()
        keep = (
            table["stat_type"].isin([SETTLEMENT_PRICE, CLEARED_VOLUME])
            & ~table["symbol"].str.contains("-", na=True)   # a "-" marks a calendar spread
            & table["ts_ref"].notna()                       # ts_ref = the trading day it refers to
        )
        table = table[keep]
        pieces.append(pd.DataFrame({
            "symbol": table["symbol"],
            "date": table["ts_ref"].dt.tz_localize(None).dt.normalize(),
            "published": table["ts_event"],
            "stat_type": table["stat_type"],
            "price": table["price"],
            "quantity": table["quantity"],
            "is_final": (table["stat_flags"] & FLAG_FINAL) > 0,
            "is_rounded": (table["stat_flags"] & FLAG_ROUNDED) > 0,
        }))
    return pd.concat(pieces)


def read_gap_fills(settlements):
    """Rows to add for the gap days in data/gap_fills.csv: the daily close of
    the held contract, where Databento has a daily bar for that day."""
    gaps = pd.read_csv(REPO_DIR / "data" / "gap_fills.csv", parse_dates=["date"])
    rows = []
    for gap in gaps.itertuples():
        already_there = ((settlements["date"] == gap.date) & (settlements["contract"] == gap.contract)).any()
        files = sorted(RAW_DIR.glob(f"fill_{gap.symbol}_*.dbn.zst"))
        if already_there or not files:
            continue
        bars = db.DBNStore.from_file(files[0]).to_df().reset_index()
        bar_dates = bars["ts_event"].dt.tz_localize(None).dt.normalize()
        bar = bars[bar_dates == gap.date]
        if len(bar) == 0:
            print(f"gap {gap.date.date()} {gap.contract}: no daily bar either; left missing")
            continue
        print(f"gap {gap.date.date()} {gap.contract}: filled with daily close {bar['close'].iloc[0]}")
        rows.append({"date": gap.date, "asset": gap.asset, "contract": gap.contract,
                     "settle": float(bar["close"].iloc[0]), "volume": int(bar["volume"].iloc[0]),
                     "price_type": "daily_close"})
    return pd.DataFrame(rows, columns=["date", "asset", "contract", "settle", "volume", "price_type"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the out-of-sample period (final run only)")
    args = parser.parse_args()

    contracts = read_contracts(args.final)
    stats = read_statistics(args.final)
    if not args.final:
        # Belt and braces: drop anything dated after the in-sample period.
        stats = stats[stats["date"] <= pd.Timestamp(config.IS_END)]

    # CME has no settlement on Saturdays or Sundays. The raw data contains a
    # handful of stray records dated on a weekend; drop them, or each would
    # add a fake "day" on which every market looks closed.
    weekend = stats["date"].dt.dayofweek >= 5           # 5 = Saturday, 6 = Sunday
    n_weekend = int((weekend & (stats["stat_type"] == SETTLEMENT_PRICE)).sum())
    if n_weekend:
        print(f"note: {n_weekend} settlement records dated on a weekend are dropped")
    stats = stats[~weekend]

    # --- One settlement price per contract per day (see the top of this file).
    #     Sorting puts the preferred record LAST in each group.
    settle = stats[stats["stat_type"] == SETTLEMENT_PRICE]
    settle = settle.sort_values(["is_final", "is_rounded", "published"])
    settle = settle.groupby(["symbol", "date"], as_index=False).last()

    # --- Cleared volume: the latest figure published for that day.
    volume = stats[stats["stat_type"] == CLEARED_VOLUME].sort_values("published")
    volume = volume.groupby(["symbol", "date"], as_index=False).last()
    volume = volume[["symbol", "date", "quantity"]].rename(columns={"quantity": "volume"})

    table = settle.merge(volume, on=["symbol", "date"], how="left")
    table["volume"] = table["volume"].fillna(0).astype(int)

    # --- Which contract does a symbol mean on a given day? CME reuses symbols
    #     every ten years, so pick the contract with that symbol whose
    #     expiration is the first one on or after the day.
    table = pd.merge_asof(
        table.sort_values("date"), contracts.sort_values("expiration"),
        by="symbol", left_on="date", right_on="expiration", direction="forward",
    )
    unmatched = table[table["contract"].isna()]
    if len(unmatched):
        print(f"note: {len(unmatched)} settlement rows have no matching definition and are dropped "
              f"(symbols: {sorted(unmatched['symbol'].unique())})")
    table = table.dropna(subset=["contract"])

    settlements = table[["date", "asset", "contract", "price", "volume"]].rename(columns={"price": "settle"})
    settlements["price_type"] = "settlement"

    # --- Gap fills (see the top of this file).
    fills = read_gap_fills(settlements)
    settlements = pd.concat([settlements, fills]).sort_values(["date", "asset", "contract"])
    used = contracts[contracts["contract"].isin(settlements["contract"])]

    # --- Write the three files the strategy reads.
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    settlements.to_csv(out / "settlements.csv", index=False, date_format="%Y-%m-%d")
    used[["asset", "contract", "year", "month", "expiration"]].to_csv(
        out / "contracts.csv", index=False, date_format="%Y-%m-%d")
    (out / "source.txt").write_text("DATABENTO\n")

    # --- Fingerprint: a checksum that changes if a single price changes.
    checksum = hashlib.sha256((out / "settlements.csv").read_bytes()).hexdigest()
    label = "full" if args.final else "in_sample"
    fingerprint = (f"{label}  rows={len(settlements)}  first={settlements['date'].min().date()}  "
                   f"last={settlements['date'].max().date()}  sha256={checksum}")
    fingerprint_file = REPO_DIR / "data" / "DATA_FINGERPRINT.txt"
    # Keep one line per label (in_sample / full); replace it if it exists.
    old_lines = fingerprint_file.read_text().splitlines() if fingerprint_file.exists() else []
    lines = [line for line in old_lines if not line.startswith(label + " ")] + [fingerprint]
    fingerprint_file.write_text("\n".join(lines) + "\n")

    print(f"DATA SOURCE: DATABENTO ({label})")
    print(f"{len(settlements)} settlement rows, {len(used)} contracts, "
          f"{settlements['date'].min().date()} to {settlements['date'].max().date()}")
    print(fingerprint)


if __name__ == "__main__":
    main()
