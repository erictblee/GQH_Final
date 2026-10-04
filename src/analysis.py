"""Analysis used to check the predictions in HYPOTHESIS.md.

Everything here takes tables of daily results (from src/backtest.py) and
returns numbers. Nothing here changes how the strategy trades.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

import config

REGIME_WINDOW = 60      # fixed 60-day window for "positive-correlation" days


def positive_correlation_days(market):
    """True on day t if the trailing 60-day correlation between ES and ZN
    daily returns, measured through day t-1, is above zero.

    (HYPOTHESIS.md, "Positive-correlation periods".)
    """
    es, zn = market.returns["ES"], market.returns["ZN"]
    trailing_corr = es.rolling(REGIME_WINDOW).corr(zn)
    # shift(1): the value known at the end of YESTERDAY classifies today.
    return trailing_corr.shift(1) > 0


def sharpe(returns):
    """Annualized Sharpe ratio of daily excess returns (risk-free rate 0)."""
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def volatility(returns):
    """Annualized volatility of daily returns."""
    return float(returns.std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def chained_drawdown(returns):
    """Max drawdown of the equity curve made by chaining these daily returns
    together in date order (used when the days are not contiguous)."""
    curve = (1.0 + returns).cumprod()
    # Start the running peak at 1.0, the value before the first return.
    peak = np.maximum(curve.cummax(), 1.0)
    return float((1.0 - curve / peak).max())


def volatility_matched_drawdown(returns, target_volatility):
    """Max drawdown after multiplying every return by one constant chosen so
    that realized volatility equals target_volatility. Scaling by a constant
    does not change the Sharpe ratio; it makes drawdowns comparable."""
    scale = target_volatility / volatility(returns)
    return chained_drawdown(returns * scale)


def leg_sharpes(daily, assets):
    """Sharpe ratio of each leg's daily P&L contribution, net of that leg's
    own costs, as a fraction of equity (prediction 3)."""
    return {asset: sharpe(daily[f"leg_{asset}"].iloc[1:]) for asset in assets}


def rescaled_to_full_target(daily):
    """Each day's return rescaled to a 10% target: multiplied by
    10% / sigma* in force at day t-1 (prediction 5). This undoes the effect
    of the drawdown brake halving the target."""
    sigma_in_force = daily["sigma_star"].shift(1)
    return (daily["ret"] * config.VOL_TARGET / sigma_in_force).iloc[1:]


def regime_table(results, positive_days):
    """For each named result: Sharpe, chained drawdown and realized
    volatility on positive-correlation days and on the other days."""
    rows = {}
    for name, daily in results.items():
        ret = daily["ret"].iloc[1:]
        is_positive = positive_days.reindex(ret.index).fillna(False).astype(bool)
        rescaled = rescaled_to_full_target(daily)
        rows[name] = {
            "share_positive_days": float(is_positive.mean()),
            "sharpe_positive": sharpe(ret[is_positive]),
            "sharpe_other": sharpe(ret[~is_positive]),
            "drawdown_positive": chained_drawdown(ret[is_positive]),
            "drawdown_other": chained_drawdown(ret[~is_positive]),
            # Realized volatility of returns rescaled to the 10% target.
            "vol_positive_rescaled": volatility(rescaled[is_positive]),
            "vol_other_rescaled": volatility(rescaled[~is_positive]),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Regressions with Newey-West standard errors
# ---------------------------------------------------------------------------

def compound(daily_returns, frequency):
    """Turn daily returns into weekly ("W-FRI") or monthly ("ME") returns by
    compounding the days inside each week or month."""
    return (1.0 + daily_returns).resample(frequency).prod() - 1.0


def newey_west_regression(y, x):
    """Ordinary least squares of y on the columns of x, plus an intercept,
    with Newey-West standard errors (which stay valid when returns are
    autocorrelated or change in volatility).

    The number of lags follows the usual rule of thumb
    floor(4 x (n / 100)^(2/9)); HYPOTHESIS.md does not fix it.

    Returns a table with one row per term: coefficient and t-statistic.
    """
    import statsmodels.api as sm

    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    n = len(data)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    design = sm.add_constant(data.drop(columns="y"))        # adds the intercept column
    fit = sm.OLS(data["y"], design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    table = pd.DataFrame({"coefficient": fit.params, "t_stat": fit.tvalues})
    table = table.rename(index={"const": "intercept"})
    table.attrs["n"] = n
    table.attrs["lags"] = lags
    table.attrs["r_squared"] = float(fit.rsquared)
    return table


# ---------------------------------------------------------------------------
# Stationary block bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)       # remember the answer: it is the same every time for a given n_days
def bootstrap_day_numbers(n_days):
    """Which days make up each bootstrap sample.

    Returns an array with config.BOOTSTRAP_SAMPLES rows and n_days columns.
    Each row is a made-up "history" built from blocks of consecutive real
    days: start on a random day, then each next day is either the following
    real day (probability 1 - 1/20) or a fresh random day (probability
    1/20). Blocks therefore average 20 days, which keeps the day-to-day
    patterns of the real data. The seed is fixed, so results are repeatable.
    """
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    samples = config.BOOTSTRAP_SAMPLES
    jump_probability = 1.0 / config.BOOTSTRAP_MEAN_BLOCK

    random_days = rng.integers(0, n_days, size=(samples, n_days))
    jumps = rng.random(size=(samples, n_days)) < jump_probability

    days = np.empty((samples, n_days), dtype=np.int64)
    days[:, 0] = random_days[:, 0]
    for t in range(1, n_days):
        next_day = (days[:, t - 1] + 1) % n_days            # wrap around at the end
        days[:, t] = np.where(jumps[:, t], random_days[:, t], next_day)
    return days


def _sharpe_of_each_sample(returns_array, days):
    """Sharpe ratio of every bootstrap sample (one per row of `days`)."""
    sampled = returns_array[days]
    return sampled.mean(axis=1) / sampled.std(axis=1, ddof=1) * np.sqrt(config.TRADING_DAYS)


def bootstrap_sharpe(returns):
    """95% confidence interval for a Sharpe ratio. Returns (low, high)."""
    days = bootstrap_day_numbers(len(returns))
    values = _sharpe_of_each_sample(returns.to_numpy(), days)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_sharpe_difference(returns_a, returns_b):
    """95% confidence interval for Sharpe(a) - Sharpe(b).

    Both series are resampled with the SAME days, so the comparison is
    paired. Returns (difference, low, high). If the interval excludes zero
    the difference is statistically distinguishable from zero.
    """
    both = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1).dropna()
    days = bootstrap_day_numbers(len(both))
    differences = (_sharpe_of_each_sample(both["a"].to_numpy(), days)
                   - _sharpe_of_each_sample(both["b"].to_numpy(), days))
    return (sharpe(both["a"]) - sharpe(both["b"]),
            float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5)))
