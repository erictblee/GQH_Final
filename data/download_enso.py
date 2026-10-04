"""Download NOAA's monthly Nino-3.4 index into data/external/nino34.csv.

    python data/download_enso.py            months up to config.IS_END (default)
    python data/download_enso.py --final    every month published (final run only)

Source: NOAA Climate Prediction Center, file `sstoi.indices` (monthly sea
surface temperature in the Nino regions, from 1982). It is free and needs
no key. We keep two columns: the Nino-3.4 temperature and NOAA's anomaly
(temperature minus the 1991-2020 average for that calendar month).

IMPORTANT, AND HONEST: NOAA does not publish old "vintages" of this file.
What we download is today's version, in which past months may have been
revised slightly since they were first published. The strategy handles
WHEN a value became known (src/enso.py); it cannot undo revisions.
"""

import argparse
import io
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import config

URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="keep every published month (final run only)")
    args = parser.parse_args()

    request = urllib.request.Request(URL, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode()

    # The file is a table separated by spaces. Columns: year, month, then a
    # temperature and an anomaly for each of four regions; Nino-3.4 is last.
    raw = pd.read_csv(io.StringIO(text), sep=r"\s+")
    table = pd.DataFrame({
        "year": raw["YR"],
        "month": raw["MON"],
        "sst": raw.iloc[:, 8],          # Nino-3.4 temperature, degrees C
        "anomaly": raw.iloc[:, 9],      # Nino-3.4 anomaly, degrees C
    })

    if not args.final:
        # Throw away every month after the in-sample period, so the file on
        # disk cannot leak out-of-sample information.
        last = pd.Timestamp(config.IS_END)
        before_or_same_year = (table["year"] < last.year) | (
            (table["year"] == last.year) & (table["month"] <= last.month))
        table = table[before_or_same_year]

    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.EXTERNAL_DIR / "nino34.csv", index=False)
    label = "all months" if args.final else f"months through {config.IS_END[:7]}"
    (config.EXTERNAL_DIR / "ENSO_DOWNLOADED.txt").write_text(
        f"{date.today().isoformat()}  NOAA CPC sstoi.indices, {label}\n")
    print(f"NOAA Nino-3.4: {len(table)} months, {table['year'].iloc[0]}-{table['month'].iloc[0]:02d} "
          f"to {table['year'].iloc[-1]}-{table['month'].iloc[-1]:02d}")


if __name__ == "__main__":
    main()
