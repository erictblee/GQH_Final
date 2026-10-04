"""Download the public data used by the diagnostics into data/external/.

    python data/download_external.py

  - FRED (Federal Reserve Bank of St. Louis): 10-year breakeven inflation
    (T10YIE), VIX (VIXCLS), 10-year real yield (DFII10), high-yield credit
    spread (BAMLH0A0HYM2) and the 3-month T-bill rate (DTB3).
  - Kenneth French data library: monthly market, size, value and momentum
    factors.

No key is needed and nothing is charged. FRED series are requested only up
to config.IS_END unless --final is given. None of this data is ever used as
a trading signal.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import config

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2010-01-01&coed={end}"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRENCH_FILES = ["F-F_Research_Data_Factors", "F-F_Momentum_Factor"]


def fetch(url):
    """Download one web address and return its bytes (up to 4 attempts)."""
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0", "Accept": "*/*"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError as error:
            print(f"  attempt {attempt} failed ({error})")
            if attempt == 4:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="download FRED data up to today (final run only)")
    args = parser.parse_args()
    end = date.today().isoformat() if args.final else config.IS_END

    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for series in config.FRED_SERIES:
        text = fetch(FRED_URL.format(series=series, end=end))
        (config.EXTERNAL_DIR / f"fred_{series}.csv").write_bytes(text)
        print(f"FRED {series}: {len(text.splitlines()) - 1} rows")

    for name in FRENCH_FILES:
        archive = zipfile.ZipFile(io.BytesIO(fetch(FRENCH_URL.format(name=name))))
        inner = archive.namelist()[0]                   # the zip holds one CSV file
        (config.EXTERNAL_DIR / f"french_{name}.csv").write_bytes(archive.read(inner))
        print(f"Kenneth French {name}: saved")

    (config.EXTERNAL_DIR / "DOWNLOADED.txt").write_text(
        f"{date.today().isoformat()}  FRED series through {end}; Kenneth French monthly factors\n")


if __name__ == "__main__":
    main()
