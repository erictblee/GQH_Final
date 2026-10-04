"""The diagnostics added in HYPOTHESIS.md, Amendment 1.

Everything here is reporting only. Nothing changes how the strategy trades,
and none of the outside data (FRED, Kenneth French) is used as a signal.
"""

import numpy as np
import pandas as pd

import config
from src import analysis


# ---------------------------------------------------------------------------
# Outside data
# ---------------------------------------------------------------------------

def load_fred(market_dates):
    """The FRED series, lined up with our trading days and LAGGED ONE DAY:
    the value shown for day t is the latest value published before day t.
    Returns None if data/download_external.py has not been run."""
    columns = {}
    for series in config.FRED_SERIES:
        path = config.EXTERNAL_DIR / f"fred_{series}.csv"
        if not path.exists():
            return None
        table = pd.read_csv(path, parse_dates=["observation_date"], index_col="observation_date")
        # FRED writes "." or leaves a blank on days with no value.
        values = pd.to_numeric(table[series], errors="coerce").dropna()
        # Line up with our calendar (carrying the last value forward over
        # FRED's holidays), then shift by one day for the lag.
        columns[series] = values.reindex(market_dates, method="ffill").shift(1)
    return pd.DataFrame(columns)


def load_french_factors(last_month):
    """Monthly Kenneth French factors (market, size, value, momentum), as
    decimals, up to and including last_month. None if not downloaded."""
    tables = []
    for name in ["F-F_Research_Data_Factors", "F-F_Momentum_Factor"]:
        path = config.EXTERNAL_DIR / f"french_{name}.csv"
        if not path.exists():
            return None
        rows = []
        header = None
        for line in path.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                # The column names are on the first line that starts with a comma.
                if line.startswith(",") and len(parts) > 1:
                    header = parts[1:]
                continue
            # Monthly rows start with a 6-digit month like 201011. The file
            # switches to yearly rows afterwards; stop there.
            if len(parts[0]) != 6 or not parts[0].isdigit():
                break
            rows.append([parts[0]] + [float(v) for v in parts[1:]])
        table = pd.DataFrame(rows, columns=["month"] + header).set_index("month")
        tables.append(table)
    factors = pd.concat(tables, axis=1).dropna() / 100.0        # percent -> decimal
    factors.index = pd.PeriodIndex(factors.index, freq="M")
    factors = factors.rename(columns=lambda c: c.strip())
    return factors[factors.index <= last_month][["Mkt-RF", "SMB", "HML", "Mom"]]


# ---------------------------------------------------------------------------
# 1. Cost justification
# ---------------------------------------------------------------------------

def cost_justification(market, first, last):
    """Estimated real cost per side (half a tick + fees) against the 2 bps
    we charge, for one contract at the period's average price."""
    rows = {}
    for asset in market.assets:
        price = market.p_new[asset].loc[first:last].mean()
        notional = price * config.MULTIPLIER[asset]
        half_tick = 0.5 * config.TICK_SIZE[asset] * config.MULTIPLIER[asset]
        dollars = half_tick + config.FEE_PER_CONTRACT
        rows[asset] = {
            "average_price": price,
            "notional_per_contract": notional,
            "half_tick_dollars": half_tick,
            "fee_dollars": config.FEE_PER_CONTRACT,
            "estimated_cost_bps": dollars / notional * 10_000,
            "charged_bps": config.COST_BPS,
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 2. Capacity
# ---------------------------------------------------------------------------

def load_contract_volume(cutoff, data_dir=None):
    """Cleared volume of every contract: one row per day, one column per
    contract. Read straight from settlements.csv in data/processed (or in
    data_dir, which the tests use)."""
    data_dir = config.PROCESSED_DIR if data_dir is None else data_dir
    table = pd.read_csv(data_dir / "settlements.csv", parse_dates=["date"])
    if cutoff is not None:
        table = table[table["date"] <= pd.Timestamp(cutoff)]
    volume = table.pivot(index="date", columns="contract", values="volume")
    # Before Databento's volume records begin, the files hold 0, meaning
    # "no figure", not "nothing traded". Mark those days as missing so they
    # are not averaged in as zeros.
    first_day_with_volume = table.loc[table["volume"] > 0, "date"].min()
    volume[volume.index < first_day_with_volume] = np.nan
    return volume


def capacity(market, daily, contract_volume, first, last):
    """Square-root market-impact model (Amendment 1, item 2).

    For every trade:  impact, as a fraction of the traded notional
                      = 1.0 x daily volatility x sqrt(contracts traded / average daily volume)

    daily volatility      = standard deviation of the asset's 60 daily returns before the trade
    average daily volume  = the traded contract's average cleared volume over
                            the 20 days before the trade

    Trades are those of `daily` (run at $10M), scaled up linearly to each fund
    size. Rolls count as trades: the old contract count is sold in the old
    contract and the new count bought in the new contract.

    Databento's cleared-volume records do not cover the whole history, so
    the test is run only from the first day that has a 20-day average volume.
    Returns (table with one row per fund size, capacity in dollars, coverage),
    where coverage says which days were used and how many trade legs had to
    be left out for lack of volume.
    """
    dates = daily.index
    # Average volume over the 20 days BEFORE each day (shift(1) leaves today out).
    average_volume = contract_volume.rolling(config.IMPACT_VOLUME_WINDOW, min_periods=5).mean().shift(1)
    daily_vol = market.returns.rolling(config.IMPACT_VOL_WINDOW).std().shift(1)

    # For a $10M fund: the impact cost of each day as a fraction of equity is
    #   sum over trades of  vol x sqrt(n / volume) x (n x multiplier x price) / equity.
    # At a fund f times bigger, n becomes f x n and equity f x equity, so the
    # whole thing is simply multiplied by sqrt(f).
    # The first day on which any contract has an average volume.
    volume_start = average_volume.index[(average_volume > 0).any(axis=1)][0]
    legs_without_volume = 0

    drag_at_10m = pd.Series(0.0, index=dates)
    participation = []          # contracts traded / average volume, at $10M
    position_share = []         # contracts held / average volume, at $10M
    equity_before = daily["equity"].shift(1)

    for asset in market.assets:
        mult = config.MULTIPLIER[asset]
        held = market.held[asset]
        position = daily[f"pos_{asset}"]
        previous = position.shift(1).fillna(0.0)
        for t in range(1, len(dates)):
            day = dates[t]
            new_contract, old_contract = held[day], held[dates[t - 1]]
            n_old, n_new = previous.iloc[t], position.iloc[t]

            # The legs traded today: (contracts, which contract, price).
            if market.roll.at[day, asset]:
                legs = [(abs(n_old), old_contract, market.p_old.at[day, asset]),
                        (abs(n_new), new_contract, market.p_new.at[day, asset])]
            else:
                legs = [(abs(n_new - n_old), new_contract, market.p_new.at[day, asset])]

            for contracts, contract, price in legs:
                volume = average_volume.at[day, contract] if day in average_volume.index else np.nan
                if contracts == 0:
                    continue
                if not volume > 0:
                    # No volume figure for this contract: the leg cannot be priced.
                    legs_without_volume += day >= volume_start
                    continue
                impact = config.IMPACT_COEFFICIENT * daily_vol.at[day, asset] * np.sqrt(contracts / volume)
                drag_at_10m[day] += impact * contracts * mult * price / equity_before[day]
                if pd.Timestamp(first) <= day <= pd.Timestamp(last):
                    participation.append(contracts / volume)

            volume = average_volume.at[day, new_contract] if day in average_volume.index else np.nan
            if volume > 0 and pd.Timestamp(first) <= day <= pd.Timestamp(last):
                position_share.append(abs(n_new) / volume)

    start = max(pd.Timestamp(first), volume_start)
    in_period = (dates >= start) & (dates <= pd.Timestamp(last))
    ret = daily["ret"][in_period]
    drag = drag_at_10m[in_period]
    base_sharpe = analysis.sharpe(ret)

    rows = {}
    for size in config.FUND_SIZES:
        scale = size / config.CAPITAL
        net = ret - drag * np.sqrt(scale)
        rows[size] = {
            "sharpe": analysis.sharpe(net),
            "sharpe_kept": analysis.sharpe(net) / base_sharpe,
            "impact_cost_per_year": float(drag.mean() * np.sqrt(scale) * config.TRADING_DAYS),
            "max_participation": float(max(participation) * scale),
            "max_position_to_volume": float(max(position_share) * scale),
        }
    table = pd.DataFrame(rows).T
    within = table[table["sharpe_kept"] >= config.CAPACITY_SHARPE_KEPT]
    capacity_dollars = within.index.max() if len(within) else None
    coverage = {"first": start, "last": pd.Timestamp(last), "base_sharpe": base_sharpe,
                "legs_without_volume": int(legs_without_volume)}
    return table, capacity_dollars, coverage


# ---------------------------------------------------------------------------
# 3. Stress windows
# ---------------------------------------------------------------------------

def stress_windows(results, first, last):
    """Return and worst drawdown of each named result inside each stress
    window that lies wholly within [first, last]."""
    rows = []
    for label, (start, end) in config.STRESS_WINDOWS.items():
        if pd.Timestamp(start) < pd.Timestamp(first) or pd.Timestamp(end) > pd.Timestamp(last):
            continue                                    # not in this sample period
        row = {"window": label, "from": start, "to": end}
        for name, daily in results.items():
            ret = daily["ret"].loc[start:end]
            row[f"{name}_return"] = float((1.0 + ret).prod() - 1.0)
            row[f"{name}_drawdown"] = analysis.chained_drawdown(ret)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Position risk
# ---------------------------------------------------------------------------

def position_risk(daily, assets):
    """Worst one-day and five-day loss of each leg, worst portfolio day,
    longest brake spell and maximum actual gross leverage."""
    legs = {}
    for asset in assets:
        contribution = daily[f"leg_{asset}"]
        legs[asset] = {
            "worst_1_day": float(contribution.min()),
            "worst_5_day": float(contribution.rolling(5).sum().min()),
        }

    # Longest run of consecutive days with the brake on.
    longest = current = 0
    for braking in daily["braking"]:
        current = current + 1 if braking else 0
        longest = max(longest, current)

    return {
        "legs": pd.DataFrame(legs).T,
        "worst_day": float(daily["ret"].min()),
        "worst_day_date": daily["ret"].idxmin(),
        "longest_brake_spell_days": longest,
        "max_gross_leverage": float(daily["gross"].max()),
    }


# ---------------------------------------------------------------------------
# 6. Regime diagnostics (FRED)   7. Asynchronous settlement   8. Total return
# ---------------------------------------------------------------------------

def regime_diagnostics(market, fred, first, last):
    """How the stock-bond correlation regime lines up with macro series."""
    trailing_corr = market.returns["ES"].rolling(analysis.REGIME_WINDOW).corr(market.returns["ZN"])
    positive = analysis.positive_correlation_days(market)
    window = slice(pd.Timestamp(first), pd.Timestamp(last))

    averages = {}
    for series in ["T10YIE", "VIXCLS", "DFII10", "BAMLH0A0HYM2"]:
        values = fred[series].loc[window]
        is_positive = positive.loc[window]
        averages[series] = {
            "positive_correlation_days": float(values[is_positive].mean()),
            "other_days": float(values[~is_positive].mean()),
            "days_with_data": int(values.notna().sum()),
        }
    return {
        "corr_of_correlation_with_breakeven": float(
            trailing_corr.loc[window].corr(fred["T10YIE"].loc[window])),
        "averages": pd.DataFrame(averages).T,
    }


def asynchronous_settlement(market, first, last):
    """Daily versus weekly correlations. If settlement-time differences
    mattered a lot, weekly correlations would differ clearly from daily ones."""
    daily = market.returns.loc[first:last]
    weekly = analysis.compound(daily, "W-FRI")
    rows = {}
    for other in ["ZN", "GC"]:
        rows[f"ES-{other}"] = {"daily": float(daily["ES"].corr(daily[other])),
                               "weekly": float(weekly["ES"].corr(weekly[other]))}
    return pd.DataFrame(rows).T


def total_return(daily, fred, first, last):
    """Excess return plus T-bill interest (FRED DTB3, lagged one day), as an
    approximate total return. One day's interest = annual rate / 252."""
    ret = daily["ret"].loc[first:last]
    interest = (fred["DTB3"].reindex(ret.index) / 100.0 / config.TRADING_DAYS).fillna(0.0)
    return {
        "excess_return_per_year": float(ret.mean() * config.TRADING_DAYS),
        "tbill_per_year": float(interest.mean() * config.TRADING_DAYS),
        "total_return_per_year": float((ret + interest).mean() * config.TRADING_DAYS),
    }
