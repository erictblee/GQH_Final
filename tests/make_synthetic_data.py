"""Make FAKE (synthetic) futures data so the code can be developed and tested
without touching real data.

    python tests/make_synthetic_data.py

It writes the same three files that real data would use (see src/market.py)
and marks them with source.txt = SYNTHETIC. The prices are random numbers.
Nothing produced from them says anything about the real strategy.

The fake data is deliberately awkward, so the tests exercise every rule:
  - several contracts per asset with price gaps between them (tests rolls);
  - a calm period (the 4x leverage cap binds);
  - a period where everything falls together (the drawdown brake triggers);
  - holidays, and a few days on which only one market is closed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script find config.py when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SYNTHETIC_END = config.IS_END  # no fake data is made beyond the in-sample period

# Made-up starting price, yearly volatility, yearly drift, and yearly "carry"
# (how much more a far contract costs than a near one) for each asset.
START_PRICE = {"ES": 1100.0, "ZN": 120.0, "GC": 1200.0}
VOLATILITY = {"ES": 0.16, "ZN": 0.06, "GC": 0.15}
DRIFT = {"ES": 0.07, "ZN": 0.02, "GC": 0.02}
CARRY = {"ES": 0.015, "ZN": -0.015, "GC": 0.03}


def make_calendar():
    """Weekdays, minus three fixed holidays a year."""
    days = pd.bdate_range(config.DATA_START, SYNTHETIC_END)
    is_holiday = (
        ((days.month == 1) & (days.day == 1))
        | ((days.month == 7) & (days.day == 4))
        | ((days.month == 12) & (days.day == 25))
    )
    return days[~is_holiday]


def make_spot_prices(days, rng):
    """One random price path per asset, with three market "regimes"."""
    n = len(days)
    assets = config.ASSETS
    daily_returns = np.zeros((n, len(assets)))

    for t, day in enumerate(days):
        if day.year in (2013, 2014):
            # Calm years: low volatility, so target weights get large and the
            # 4x leverage cap binds.
            vol_scale, corr_es_zn, extra_drift = 0.35, -0.3, 0.0
        elif day.year == 2022:
            # Inflation shock: stocks and bonds fall together.
            vol_scale, corr_es_zn, extra_drift = 1.3, 0.6, -0.30
        else:
            vol_scale, corr_es_zn, extra_drift = 1.0, -0.3, 0.0

        # Correlation matrix in ES, ZN, GC order.
        corr = np.array([
            [1.0, corr_es_zn, 0.05],
            [corr_es_zn, 1.0, 0.20],
            [0.05, 0.20, 1.0],
        ])
        vol = np.array([VOLATILITY[a] for a in assets]) * vol_scale
        drift = np.array([DRIFT[a] for a in assets]) + extra_drift * vol / 0.16
        # Daily covariance = correlation scaled by volatilities, divided by 252.
        cov = corr * np.outer(vol, vol) / config.TRADING_DAYS
        daily_returns[t] = rng.multivariate_normal(drift / config.TRADING_DAYS, cov)

    start = np.array([START_PRICE[a] for a in assets])
    # Compound the daily returns into a price path.
    spot = start * np.cumprod(1.0 + daily_returns, axis=0)
    return pd.DataFrame(spot, index=days, columns=assets)


def expiration_date(asset, year, month):
    """A made-up but realistic expiration date for each contract."""
    month_days = pd.bdate_range(f"{year}-{month:02d}-01", periods=31)
    month_days = month_days[month_days.month == month]
    if asset == "ES":
        fridays = month_days[month_days.dayofweek == 4]
        return fridays[2]          # third Friday
    if asset == "ZN":
        return month_days[-8]      # 7 business days before the last one
    return month_days[-3]          # GC: third-to-last business day


def make(out_dir, seed=0):
    """Write settlements.csv, contracts.csv and source.txt into out_dir."""
    rng = np.random.default_rng(seed)
    days = make_calendar()
    spot = make_spot_prices(days, rng)

    settlement_rows = []
    contract_rows = []
    for asset in config.ASSETS:
        asset_days = days
        if asset == "GC":
            # Gold data starts two weeks late, to test "first date on which
            # all three markets have data".
            asset_days = asset_days[10:]
        if asset != "ES":
            # A dozen days on which only this market is closed.
            closed = rng.choice(len(asset_days), size=12, replace=False)
            asset_days = asset_days.delete(closed)

        for year in range(days[0].year, days[-1].year + 2):
            for month in config.CONTRACT_MONTHS[asset]:
                name = f"{asset}_{year}_{month:02d}"
                expiration = expiration_date(asset, year, month)
                # A contract has settlement records for its last 300 days.
                alive = asset_days[
                    (asset_days > expiration - pd.Timedelta(days=300))
                    & (asset_days <= expiration)
                ]
                if len(alive) == 0:
                    continue
                contract_rows.append(
                    {"asset": asset, "contract": name, "year": year,
                     "month": month, "expiration": expiration.date()}
                )
                for day in alive:
                    years_left = (expiration - day).days / 365.0
                    # Far contracts are priced away from spot by the carry:
                    # this is what creates the price gap on roll days.
                    price = spot.at[day, asset] * (1.0 + CARRY[asset] * years_left)
                    settlement_rows.append(
                        {"date": day.date(), "asset": asset, "contract": name,
                         "settle": price, "volume": 100_000}
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(settlement_rows).to_csv(out_dir / "settlements.csv", index=False)
    pd.DataFrame(contract_rows).to_csv(out_dir / "contracts.csv", index=False)
    (out_dir / "source.txt").write_text("SYNTHETIC\n")


if __name__ == "__main__":
    source_file = config.PROCESSED_DIR / "source.txt"
    # Safety: never overwrite real data with fake data.
    if source_file.exists() and source_file.read_text().strip() != "SYNTHETIC":
        raise SystemExit(f"{config.PROCESSED_DIR} holds real data. Refusing to overwrite it.")
    make(config.PROCESSED_DIR)
    print(f"SYNTHETIC data written to {config.PROCESSED_DIR}")
