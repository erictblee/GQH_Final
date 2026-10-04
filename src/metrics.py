"""Performance numbers computed from a table of daily results.

All returns here are EXCESS returns (futures P&L net of costs, no T-bill
interest), so the Sharpe ratio uses a risk-free rate of zero.
"""

import numpy as np
import pandas as pd

import config


def max_drawdown(equity):
    """Largest fall from a previous peak, as a positive fraction (0.12 = 12%)."""
    running_peak = equity.cummax()
    return float((1.0 - equity / running_peak).max())


def summary(daily, first=None, last=None):
    """Headline numbers for one backtest, over the whole run or over the
    part of it between the dates `first` and `last`.

    daily: the table returned by backtest.run (needs columns equity, ret,
    trades, cap_binding, braking).
    """
    part = daily.loc[first:last]            # None means "no limit" on that side
    ret = part["ret"]
    if part.index[0] == daily.index[0]:
        # The run's first row is the start date itself: we are flat and its
        # return is 0 by construction, so leave it out of the statistics.
        ret = ret.iloc[1:]

    ann_return = ret.mean() * config.TRADING_DAYS
    ann_vol = ret.std(ddof=1) * np.sqrt(config.TRADING_DAYS)
    # Drawdown inside the period: rebuild the equity curve from the returns,
    # starting at 1, and measure the largest fall from a previous peak.
    curve = pd.concat([pd.Series([1.0]), (1.0 + ret).cumprod()])
    mdd = max_drawdown(curve)

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_return / ann_vol),
        "max_drawdown": mdd,
        "calmar": float(ann_return / mdd) if mdd > 0 else float("nan"),
        "trades": int(part["trades"].sum()),
        "final_equity": float(part["equity"].iloc[-1]),
        "cap_share": float(part["cap_binding"].mean()),    # share of days cap binds
        "brake_share": float(part["braking"].mean()),      # share of days brake is on
    }
