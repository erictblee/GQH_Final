"""The backtest engine of the ENSO strategy (pandas / numpy).

It walks through the calendar one day at a time. On every day t, in order:

    1. P&L        positions held coming into today earn today's price change.
    2. Rolls      if we switch contract today: close the old contract, open
                  the same number of the new one, pay costs on both legs.
    3. Fills      an order decided earlier is filled at TODAY's settlement.
    4. Equity     yesterday's equity + P&L - costs.
    5. Decision   ONLY on the first trading day of a week: compute new target
                  weights from data up to today and turn them into an order.
                  It is filled at the NEXT day's settlement (step 3).

So a decision never uses a price it could not have seen, and between weekly
decisions the contract counts do not change (rolls keep the same count).
"""

import numpy as np
import pandas as pd

import config
from src import alpha, enso


def is_decision_day(dates, t, start):
    """True on the start date and on the first trading day of each week.

    We compare today's (year, week number) with the previous trading day's.
    This needs no knowledge of future holidays.
    """
    if t == start:
        return True
    today, yesterday = dates[t].isocalendar(), dates[t - 1].isocalendar()
    return (today.year, today.week) != (yesterday.year, yesterday.week)


def run(market, enso_table, spec, decide=alpha.decide, ledger=None):
    """Run one backtest. Returns a table with one row per day, from the
    start date to the last day.

    decide: the function that makes the weekly decision. The ENSO grain
            strategy uses alpha.decide; the cross-asset strategy passes
            xa_alpha.decide. The mechanics below are the same for both.
    ledger: pass an empty list to get the TRANSACTION LEDGER: one entry is
            added for every fill (date, asset, contract, kind, contracts,
            price, notional, cost).
    """
    assets = list(market.assets)
    n_assets = len(assets)

    # Pull the tables out as plain arrays: rows = days, columns = assets.
    p_new = market.p_new[assets].to_numpy()
    p_old = market.p_old[assets].to_numpy()
    returns = market.returns[assets].to_numpy()
    roll = market.roll[assets].to_numpy()
    is_open = market.is_open[assets].to_numpy()
    held = market.held[assets].to_numpy()            # name of the contract held after each day
    mult = np.array([config.MULTIPLIER[a] for a in assets])
    dates = market.dates

    # The ENSO state in force on every trading day (point-in-time).
    enso_states, enso_z = enso.on_dates(enso_table, dates)

    cost_rate = spec.cost_bps / 10_000.0     # 2 bps = 0.0002 of traded notional

    # --- State that is carried from one day to the next ---
    equity = config.CAPITAL
    positions = np.zeros(n_assets)           # contracts held
    orders = positions.copy()                # contract counts wanted after the next fill
    decision = None                          # the latest weekly decision (a dict)
    margin_active = False

    rows = []
    for t in range(market.start, len(dates)):
        equity_before = equity
        leg_pnl = np.zeros(n_assets)
        roll_cost = np.zeros(n_assets)
        trade_cost = np.zeros(n_assets)
        traded_notional = 0.0                # dollars traded by rebalancing (not rolls)
        n_trades = 0
        n_rolls = 0

        if t > market.start:
            # 1. P&L. p_old[t] and p_new[t-1] are prices of the SAME
            #    contract, so no roll gap sneaks in.
            leg_pnl = positions * mult * (p_old[t] - p_new[t - 1])

            # 2 and 3. Rolls and fills, both at today's settlement. If a
            #    market is closed today its order waits for the next open day.
            for i in range(n_assets):
                old_count = positions[i]
                new_count = orders[i] if is_open[t, i] else old_count

                if roll[t, i]:
                    # Close the old count in the old contract, open the new
                    # count in the new contract. Cost on both legs.
                    total = cost_rate * mult[i] * (abs(old_count) * p_old[t, i]
                                                   + abs(new_count) * p_new[t, i])
                    # For reporting: the rebalance part is the cost of the
                    # CHANGE in contracts; the rest is the cost of rolling.
                    trade_cost[i] = cost_rate * mult[i] * abs(new_count - old_count) * p_new[t, i]
                    roll_cost[i] = total - trade_cost[i]
                    if old_count != 0:
                        n_rolls += 1
                    if ledger is not None:
                        for kind, contract, count, price in [
                                ("roll_close", held[t - 1, i], -old_count, p_old[t, i]),
                                ("roll_open", held[t, i], new_count, p_new[t, i])]:
                            if count != 0:
                                notional = mult[i] * abs(count) * price
                                ledger.append({"date": dates[t], "asset": assets[i], "contract": contract,
                                               "kind": kind, "contracts": count, "price": price,
                                               "notional": notional, "cost": cost_rate * notional})
                else:
                    trade_cost[i] = cost_rate * mult[i] * abs(new_count - old_count) * p_new[t, i]
                    if ledger is not None and new_count != old_count:
                        notional = mult[i] * abs(new_count - old_count) * p_new[t, i]
                        ledger.append({"date": dates[t], "asset": assets[i], "contract": held[t, i],
                                       "kind": "rebalance", "contracts": new_count - old_count,
                                       "price": p_new[t, i], "notional": notional,
                                       "cost": cost_rate * notional})

                if new_count != old_count:
                    traded_notional += mult[i] * abs(new_count - old_count) * p_new[t, i]
                    positions[i] = new_count
                    n_trades += 1

            # 4. Equity.
            equity = equity + leg_pnl.sum() - (roll_cost + trade_cost).sum()

        # 5. Weekly decision, from returns up to and including today.
        decided_today = is_decision_day(dates, t, market.start)
        if decided_today:
            decision = decide(spec, returns[: t + 1], enso_states[t], assets)
            orders, margin_active = alpha.whole_contracts(
                spec, decision["target"], equity, p_new[t], mult, assets)

        # What we actually hold now, as weights, and the risk that implies
        # under the covariance of the latest decision.
        actual = positions * p_new[t] * mult / equity
        actual_shares = alpha.risk_shares(actual, decision["cov_final"])
        actual_vol = float(np.sqrt(actual @ decision["cov_final"] @ actual))
        corr = decision["cov"] / np.outer(decision["sigma"], decision["sigma"])

        # Record the day (the diagnostic time series of ENSO_SPEC.md section 32).
        row = {
            "date": dates[t],
            "equity": equity,
            "ret": equity / equity_before - 1.0,
            # The same day's return before any cost.
            "gross_ret": (equity + (roll_cost + trade_cost).sum()) / equity_before - 1.0,
            "decision_day": decided_today,
            "enso_state": enso_states[t],
            "enso_z": enso_z[t],
            "trades": n_trades,
            "rolls": n_rolls,
            "cost": (roll_cost + trade_cost).sum(),
            "roll_cost": roll_cost.sum(),
            "traded_notional": traded_notional,
            "leverage": np.abs(actual).sum(),               # longs + shorts, / equity
            "net_exposure": actual.sum(),                   # longs - shorts, / equity
            "predicted_vol": decision["predicted_vol"],     # of the continuous target
            "actual_vol": actual_vol,                       # of the whole contracts held
            "cap_binding": decision["cap_binding"],
            "risk_limit_active": decision["risk_limit_active"],
            "position_limit_active": decision.get("position_limit_active", False),
            "margin_limit_active": margin_active,
        }
        for i, a in enumerate(assets):
            row[f"alpha_{a}"] = decision["alpha"][i]
            row[f"mom_{a}"] = decision["momentum"][i]
            row[f"vol_{a}"] = decision["sigma"][i]
            row[f"budget_{a}"] = decision["budgets"][i]
            row[f"riskshare_target_{a}"] = decision["risk_shares"][i]
            row[f"riskshare_actual_{a}"] = actual_shares[i]
            row[f"target_{a}"] = decision["target"][i]      # continuous target weight
            row[f"order_{a}"] = orders[i]                   # target contracts
            row[f"pos_{a}"] = positions[i]                  # actual contracts
            # This leg's P&L net of its own costs, as a fraction of equity.
            row[f"leg_{a}"] = (leg_pnl[i] - roll_cost[i] - trade_cost[i]) / equity_before
            row[f"pnl_{a}"] = leg_pnl[i] - roll_cost[i] - trade_cost[i]     # the same, in dollars
            for j in range(i + 1, n_assets):
                row[f"corr_{a}_{assets[j]}"] = corr[i, j]
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")
