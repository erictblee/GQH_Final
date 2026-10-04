"""Performance numbers of the ENSO strategy (ENSO_SPEC.md section 31).

All returns are EXCESS returns (futures P&L, no interest on cash), so the
Sharpe ratio uses a risk-free rate of zero.
"""

import numpy as np
import pandas as pd

import config
from src import analysis, metrics


def summary(daily, first=None, last=None):
    """Headline numbers for one backtest, over the whole run or over the
    part of it between the dates `first` and `last`."""
    part = daily.loc[first:last]
    if part.index[0] == daily.index[0]:
        # The run's first row is the start date itself (flat, return 0).
        part = part.iloc[1:]
    ret = part["ret"]
    years = len(ret) / config.TRADING_DAYS

    def per_year(daily_mean):
        return float(daily_mean * config.TRADING_DAYS)

    growth = float((1.0 + ret).prod())                  # $1 becomes this much
    gross_growth = float((1.0 + part["gross_ret"]).prod())
    ann_vol = float(ret.std(ddof=1) * np.sqrt(config.TRADING_DAYS))
    # Sortino: like Sharpe, but only days with a loss count as "risk".
    downside = float(np.sqrt((ret.clip(upper=0.0) ** 2).mean()) * np.sqrt(config.TRADING_DAYS))
    curve = pd.concat([pd.Series([1.0]), (1.0 + ret).cumprod()])
    mdd = metrics.max_drawdown(curve)
    cagr = growth ** (1.0 / years) - 1.0

    # Compound the daily returns inside each calendar month and year.
    monthly = (1.0 + ret).groupby(ret.index.to_period("M")).prod() - 1.0
    yearly = (1.0 + ret).groupby(ret.index.year).prod() - 1.0

    # A strategy that never trades has zero volatility: no Sharpe ratio.
    has_risk = ann_vol > 0
    if has_risk:
        ci_low, ci_high = analysis.bootstrap_sharpe(ret)
    else:
        ci_low = ci_high = float("nan")

    average_equity = float(part["equity"].mean())
    return {
        "first_day": part.index[0].date().isoformat(),
        "last_day": part.index[-1].date().isoformat(),
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": per_year(ret.mean()) / ann_vol if has_risk else float("nan"),
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
        "sortino": per_year(ret.mean()) / downside if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": cagr / mdd if mdd > 0 else float("nan"),
        "worst_month": float(monthly.min()),
        "worst_year": float(yearly.min()),
        "positive_months": float((monthly > 0).mean()),
        "net_return": growth - 1.0,
        "gross_return": gross_growth - 1.0,
        "costs_dollars": float(part["cost"].sum()),
        "roll_costs_dollars": float(part["roll_cost"].sum()),
        # Turnover: dollars traded by rebalancing per year, in multiples of equity.
        "turnover_per_year": float(part["traded_notional"].sum() / average_equity / years),
        "time_invested": float((part["leverage"] > 0).mean()),
        "avg_leverage": float(part["leverage"].mean()),
        "max_leverage": float(part["leverage"].max()),
        "trades": int(part["trades"].sum()),
        "rolls": int(part["rolls"].sum()),
        "risk_limit_share": float(part["risk_limit_active"].mean()),
    }


def by_commodity(daily, assets, first=None, last=None):
    """Each commodity's contribution: average yearly return it added to the
    portfolio (net of its own costs) and the Sharpe ratio of that stream."""
    part = daily.loc[first:last]
    rows = []
    for a in assets:
        leg = part[f"leg_{a}"]
        vol = leg.std(ddof=1) * np.sqrt(config.TRADING_DAYS)
        mean = leg.mean() * config.TRADING_DAYS
        rows.append({"asset": a, "ann_return": float(mean),
                     "sharpe": float(mean / vol) if vol > 0 else float("nan"),
                     "days_held": int((part[f"pos_{a}"] != 0).sum())})
    return pd.DataFrame(rows)


def by_enso_state(daily, first=None, last=None):
    """Performance on the days of each ENSO state."""
    part = daily.loc[first:last]
    rows = []
    for state in ["el_nino", "neutral", "la_nina"]:
        ret = part.loc[part["enso_state"] == state, "ret"]
        vol = ret.std(ddof=1) * np.sqrt(config.TRADING_DAYS) if len(ret) > 1 else 0.0
        mean = ret.mean() * config.TRADING_DAYS if len(ret) else float("nan")
        rows.append({"state": state, "days": len(ret), "ann_return": float(mean),
                     "sharpe": float(mean / vol) if vol > 0 else float("nan")})
    return pd.DataFrame(rows)


def position_episodes(daily, assets):
    """Win / loss statistics that mean something for futures.

    An "episode" is one unbroken stretch in which we are long (or short) an
    asset: from the day the position is opened or flips sign to the day it
    is closed or flips again. Its P&L is every dollar that asset earned or
    lost during the stretch, net of its trading and roll costs. One row per
    episode: asset, direction, first_day, last_day, days, pnl.
    """
    rows = []
    for a in assets:
        position = daily[f"pos_{a}"]
        # Day t's P&L was earned by the position held COMING INTO day t. On
        # the day a position is opened there is no earlier position, so that
        # day (its opening cost) belongs to the new position.
        earned_by = np.sign(position.shift(1).fillna(0.0))
        earned_by = earned_by.where(earned_by != 0, np.sign(position))
        # A new episode starts whenever the direction changes.
        episode_number = (earned_by != earned_by.shift(1)).cumsum()
        for _, days in daily.groupby(episode_number):
            direction = earned_by.loc[days.index[0]]
            if direction == 0:
                continue                        # flat: not an episode
            rows.append({"asset": a, "direction": "long" if direction > 0 else "short",
                         "first_day": days.index[0].date(), "last_day": days.index[-1].date(),
                         "days": len(days), "pnl": float(days[f"pnl_{a}"].sum())})
    return pd.DataFrame(rows, columns=["asset", "direction", "first_day", "last_day", "days", "pnl"])


# ---------------------------------------------------------------------------
# Extra reports asked for by ABCD_PLAN.md section 12
# ---------------------------------------------------------------------------

def total_pnl(daily, assets, first=None, last=None):
    """Dollars earned in the period, net of costs."""
    part = daily.loc[first:last]
    return float(part[[f"pnl_{a}" for a in assets]].to_numpy().sum())


def annual_pnl(daily, assets):
    """Dollars earned in each calendar year, net of costs."""
    dollars = daily[[f"pnl_{a}" for a in assets]].sum(axis=1)
    return dollars.groupby(daily.index.year).sum()


def monthly_distribution(daily, first=None, last=None):
    """The shape of the monthly returns: where the middle is, how wide they
    spread and how bad the tails are."""
    ret = daily.loc[first:last, "ret"]
    monthly = (1.0 + ret).groupby(ret.index.to_period("M")).prod() - 1.0
    return {
        "months": len(monthly),
        "mean": float(monthly.mean()),
        "median": float(monthly.median()),
        "std": float(monthly.std(ddof=1)),
        "worst": float(monthly.min()),
        "pct_5": float(monthly.quantile(0.05)),
        "pct_25": float(monthly.quantile(0.25)),
        "pct_75": float(monthly.quantile(0.75)),
        "pct_95": float(monthly.quantile(0.95)),
        "best": float(monthly.max()),
        "skew": float(monthly.skew()),
        "positive_months": float((monthly > 0).mean()),
    }


def episode_counts(daily, assets, first=None, last=None):
    """Number of position episodes ("trades") in the period and the share
    of them that made money."""
    episodes = position_episodes(daily.loc[first:last], assets)
    return {"episodes": len(episodes),
            "win_rate": float((episodes["pnl"] > 0).mean()) if len(episodes) else float("nan")}


def by_contract(daily, assets, first=None, last=None):
    """Each futures market's part in the result. One row per asset.

    Returns and volatility are of the stream this asset added to the
    portfolio (its P&L net of its own costs, as a fraction of equity).
    max_drawdown is the largest peak-to-trough fall of that stream's running
    total, in the same units. "trades" are position episodes.
    """
    part = daily.loc[first:last]
    episodes = position_episodes(part, assets)
    total = total_pnl(part, assets)
    rows = []
    for a in assets:
        leg = part[f"leg_{a}"]
        mean = leg.mean() * config.TRADING_DAYS
        vol = leg.std(ddof=1) * np.sqrt(config.TRADING_DAYS)
        running = leg.cumsum()
        fall = (running.cummax().clip(lower=0.0) - running).max()
        own = episodes[episodes["asset"] == a]
        pnl = float(part[f"pnl_{a}"].sum())
        rows.append({
            "asset": a,
            "pnl_dollars": pnl,
            "share_of_pnl": pnl / total if total != 0 else float("nan"),
            "ann_return": float(mean),
            "ann_vol": float(vol),
            "sharpe": float(mean / vol) if vol > 0 else float("nan"),
            "max_drawdown": float(fall),
            "trades": len(own),
            "win_rate": float((own["pnl"] > 0).mean()) if len(own) else float("nan"),
            "long_pnl_dollars": float(own.loc[own["direction"] == "long", "pnl"].sum()),
            "short_pnl_dollars": float(own.loc[own["direction"] == "short", "pnl"].sum()),
        })
    return pd.DataFrame(rows)
