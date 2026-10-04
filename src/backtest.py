"""The backtest engine (pandas / numpy). This is the reference implementation.

It walks through the calendar one day at a time. On every day t, in order:

    1. P&L        positions held coming into today earn today's price change.
    2. Rolls      if we switch contract today: close the old contract count,
                  open the new count, pay costs on both legs.
    3. Fills      orders decided YESTERDAY are filled at TODAY's settlement.
    4. Equity     yesterday's equity + P&L - costs.
    5. Brake      update the drawdown brake with today's equity.
    6. Targets    compute target weights from returns up to today.
    7. Orders     decide what to trade; it will be filled TOMORROW (step 3).

Steps 5-7 use only information available at today's settlement, and the
trade happens at the next settlement: "decide at t, fill at t+1".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src import signals


@dataclass
class Spec:
    """One thing to backtest. The defaults are the main strategy (version B).
    Only the fields named in HYPOTHESIS.md's variants and diagnostics exist."""

    name: str = "B"
    version: str = "B"                 # "A", "B", "FIXED" (benchmark) or "CL" (constant leverage)
    assets: tuple = tuple(config.ASSETS)
    window: int = config.MAIN_WINDOW
    cost_bps: float = config.COST_BPS
    brake: bool = True
    integer_contracts: bool = True
    fixed_weights: dict = None         # only for version "FIXED" (benchmarks)
    leverage: float = None             # only for version "CL": the fixed multiplier
    demean_asset: str = None           # diagnostic: remove this asset's average return


def run(market, spec):
    """Run one backtest. Returns a table with one row per day, from the
    common start date to the last day."""
    assets = list(spec.assets)
    n_assets = len(assets)

    # Pull the tables out as plain arrays: rows = days, columns = assets.
    p_new = market.p_new[assets].to_numpy()
    p_old = market.p_old[assets].to_numpy()
    returns = market.returns[assets].to_numpy()
    roll = market.roll[assets].to_numpy()
    is_open = market.is_open[assets].to_numpy()
    mult = np.array([config.MULTIPLIER[a] for a in assets])
    dates = market.dates

    cost_rate = spec.cost_bps / 10_000.0     # 2 bps = 0.0002 of traded notional

    # Demeaned-gold diagnostic (HYPOTHESIS.md, "Diagnostics"): subtract the
    # asset's average daily return "for the period being evaluated" from each
    # day's P&L. The in-sample days use the in-sample average; out-of-sample
    # days (only present in the final run) use the out-of-sample average.
    # The returns used for volatilities and correlations are NOT touched, so
    # target weights stay the same.
    return_to_remove = np.zeros_like(returns)
    if spec.demean_asset is not None:
        i = assets.index(spec.demean_asset)
        evaluated = np.arange(len(dates)) > market.start
        in_sample = evaluated & (dates <= pd.Timestamp(config.IS_END))
        out_of_sample = evaluated & (dates > pd.Timestamp(config.IS_END))
        for period in (in_sample, out_of_sample):
            if period.any():
                return_to_remove[period, i] = returns[period, i].mean()

    # --- State that is carried from one day to the next ---
    equity = config.CAPITAL                  # start with $10M ...
    positions = np.zeros(n_assets)           # ... and no contracts
    orders = positions.copy()                # contract counts wanted after the next fill
    peak = equity                            # highest equity so far (for the brake)
    braking = False

    rows = []
    for t in range(market.start, len(dates)):
        equity_before = equity
        leg_pnl = np.zeros(n_assets)
        roll_cost = np.zeros(n_assets)       # cost of rolling, per asset
        trade_cost = np.zeros(n_assets)      # cost of rebalancing, per asset
        n_trades = 0

        if t > market.start:
            # 1. P&L of the contracts we came in with. p_old[t] and p_new[t-1]
            #    are prices of the SAME contract, so no roll gap sneaks in.
            leg_pnl = positions * mult * (p_old[t] - p_new[t - 1])
            if spec.demean_asset is not None:
                # Diagnostic only: take out the asset's average return.
                leg_pnl = leg_pnl - positions * mult * p_new[t - 1] * return_to_remove[t]

            # 2 and 3. Rolls, and fills of yesterday's orders, both at today's
            #    settlement. If a market is closed today its order is dropped
            #    (we hold the prior position).
            for i in range(n_assets):
                old_count = positions[i]
                new_count = orders[i] if is_open[t, i] else old_count

                if roll[t, i]:
                    # Roll day: close the old contract count at the old
                    # contract's price, open the new count at the new
                    # contract's price. Cost on both legs. If no rebalance is
                    # due, new_count equals old_count.
                    total = cost_rate * mult[i] * (abs(old_count) * p_old[t, i]
                                                   + abs(new_count) * p_new[t, i])
                    # For reporting only: the part caused by a rebalance is
                    # the cost of the EXTRA contracts opened (if any); the
                    # rest is the cost of the roll.
                    extra = max(abs(new_count) - abs(old_count), 0.0)
                    trade_cost[i] = cost_rate * mult[i] * extra * p_new[t, i]
                    roll_cost[i] = total - trade_cost[i]
                else:
                    # Ordinary day: only the change in contracts is traded.
                    trade_cost[i] = cost_rate * mult[i] * abs(new_count - old_count) * p_new[t, i]

                if new_count != old_count:
                    positions[i] = new_count
                    n_trades += 1

            # 4. Equity.
            equity = equity + leg_pnl.sum() - (roll_cost + trade_cost).sum()

        # 5. Drawdown brake, using equity through today.
        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak
        if spec.brake and spec.version in ("A", "B"):   # benchmarks and CL have no brake
            if not braking and drawdown > config.BRAKE_ON:
                braking = True               # fell more than 15% below the peak
            elif braking and drawdown < config.BRAKE_OFF:
                braking = False              # recovered to under 7.5%
        sigma_star = config.BRAKE_VOL_TARGET if braking else config.VOL_TARGET

        # What our positions are worth now, as a fraction of equity.
        current = positions * p_new[t] * mult / equity

        if spec.version == "FIXED":
            # Benchmark: constant target weights, no brake, no cap, no band.
            target = np.array([spec.fixed_weights[a] for a in assets])
            cap_binding = False
            # Rebalance on the start date and on the last trading day of each
            # month (i.e. when tomorrow is in a different month).
            is_last_day = t == len(dates) - 1
            month_end = (not is_last_day) and dates[t + 1].month != dates[t].month
            must_trade = np.full(n_assets, t == market.start or month_end)
        else:
            # 6. Target weights from the last `window` daily returns.
            window_returns = returns[t - spec.window + 1 : t + 1]
            cov = signals.annualized_cov(window_returns)
            if spec.version == "CL":
                # Constant leverage (prediction 2): B's ERC weights scaled to
                # add up to 1, times a fixed multiplier. No volatility target.
                target, cap_binding = signals.constant_leverage_weights(cov, spec.leverage)
            else:
                target, cap_binding = signals.target_weights(cov, spec.version, sigma_star)
            # Rebalance band: trade an asset only if it has drifted more than
            # 20% (relative) away from its target.
            must_trade = np.abs(current - target) > config.BAND * target

        # 7. Orders. Contract count from TODAY's equity and settlement price.
        wanted = target * equity / (p_new[t] * mult)
        if spec.integer_contracts:
            wanted = np.round(wanted)        # futures trade in whole contracts
        # Where we trade, go all the way to the wanted count; elsewhere keep
        # the position we have.
        orders = np.where(must_trade, wanted, positions)

        # Record the day.
        row = {
            "date": dates[t],
            "equity": equity,
            "ret": equity / equity_before - 1.0,
            "sigma_star": sigma_star,
            "cap_binding": cap_binding,
            "braking": braking,
            "trades": n_trades,
            "cost": (roll_cost + trade_cost).sum(),
            "gross": current.sum(),
        }
        for i, a in enumerate(assets):
            row[f"pos_{a}"] = positions[i]
            row[f"target_{a}"] = target[i]
            row[f"rollcost_{a}"] = roll_cost[i]
            row[f"tradecost_{a}"] = trade_cost[i]
            # This leg's P&L net of its own costs, as a fraction of equity
            # (used for prediction 3: Sharpe ratio of each leg).
            row[f"leg_{a}"] = (leg_pnl[i] - roll_cost[i] - trade_cost[i]) / equity_before
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")
