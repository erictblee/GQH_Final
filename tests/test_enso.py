"""Tests of the ENSO strategy: the invariants of ENSO_SPEC.md section 33.

Like the other tests, these NEVER read real data. They build a small fake
market and a fake ENSO table in memory. Nothing here says anything about
how the real strategy performs.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import config
from src import ag_backtest, alpha, enso, market, signals

ASSETS = config.AG_ASSETS
MULT = np.array([config.MULTIPLIER[a] for a in ASSETS])


def fake_market(n_days=260, seed=1, closed_day=None):
    """Random prices for the three grains, with a roll (and a 10% price gap
    between the old and the new contract) every 40 days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(config.AG_DATA_START, periods=n_days)
    # Correlated random daily returns with an upward drift, so that momentum
    # is often positive and the strategy holds positions.
    common = rng.normal(0.0, 0.010, size=(n_days, 1))
    own = rng.normal(0.0008, 0.012, size=(n_days, 3))
    returns = common + own
    returns[0] = 0.0

    start_price = np.array([550.0, 1300.0, 750.0])
    p_old = np.zeros((n_days, 3))
    p_new = np.zeros((n_days, 3))
    roll = np.zeros((n_days, 3), dtype=bool)
    p_old[0] = p_new[0] = start_price
    for t in range(1, n_days):
        p_old[t] = p_new[t - 1] * (1.0 + returns[t])
        p_new[t] = p_old[t]
        if t % 40 == 0:
            roll[t] = True
            p_new[t] = p_old[t] * 1.10          # the new contract costs 10% more
    is_open = np.ones((n_days, 3), dtype=bool)
    if closed_day is not None:
        # Corn is closed that day: no price change, no fill.
        is_open[closed_day, 0] = False
        p_old[closed_day, 0] = p_new[closed_day, 0] = p_new[closed_day - 1, 0]
        returns[closed_day, 0] = 0.0
        p_old[closed_day + 1, 0] = p_new[closed_day, 0] * (1.0 + returns[closed_day + 1, 0])
        if not roll[closed_day + 1, 0]:
            p_new[closed_day + 1, 0] = p_old[closed_day + 1, 0]

    def frame(values):
        return pd.DataFrame(values, index=dates, columns=ASSETS)

    return market.Market(
        dates=dates, assets=list(ASSETS), p_new=frame(p_new), p_old=frame(p_old),
        returns=((frame(p_old) - frame(p_new).shift(1)) / frame(p_new).shift(1)).fillna(0.0),
        roll=frame(roll), is_open=frame(is_open), volume=frame(np.zeros((n_days, 3))),
        held=frame(np.full((n_days, 3), "X")), start=config.AG_WARMUP_DAYS, source="SYNTHETIC",
    )


def fake_enso(last="2024-09"):
    """A made-up ENSO history: a slow wave, so all three states occur."""
    months = pd.period_range("1982-01", last, freq="M")
    anomaly = 1.5 * np.sin(np.arange(len(months)) * 2 * np.pi / 14.0)
    return pd.DataFrame({"year": months.year, "month": months.month, "anomaly": anomaly})


@pytest.fixture(scope="module")
def result():
    return ag_backtest.run(fake_market(), enso.add_signal(fake_enso()), alpha.AgSpec())


# --- No look-ahead -----------------------------------------------------------

def test_cutting_off_the_future_changes_nothing(result):
    """Run again on a market that ends 50 days earlier. Every number up to
    that day must be identical: so no day used data from after it."""
    full = fake_market()
    cut = 210

    def shorten(table):
        return table.iloc[:cut]

    short = market.Market(
        dates=full.dates[:cut], assets=full.assets, p_new=shorten(full.p_new), p_old=shorten(full.p_old),
        returns=shorten(full.returns), roll=shorten(full.roll), is_open=shorten(full.is_open),
        volume=shorten(full.volume), held=shorten(full.held), start=full.start, source="SYNTHETIC")
    short_result = ag_backtest.run(short, enso.add_signal(fake_enso()), alpha.AgSpec())
    pd.testing.assert_frame_equal(short_result, result.iloc[:len(short_result)])


def test_trades_happen_only_the_day_after_a_weekly_decision(result):
    decided_yesterday = result["decision_day"].shift(1, fill_value=False)
    assert (result.loc[~decided_yesterday, "trades"] == 0).all()
    # A decision is made on the start date and whenever the week number changes.
    weeks = result.index.isocalendar().week.to_numpy()
    new_week = np.r_[True, weeks[1:] != weeks[:-1]]
    assert (result["decision_day"].to_numpy() == new_week).all()


# --- ENSO availability -------------------------------------------------------

def test_enso_value_is_not_used_before_its_release_date():
    table = enso.add_signal(fake_enso())
    # Give June 2023 and July 2023 recognisable z-scores.
    june = table.index[(table["year"] == 2023) & (table["month"] == 6)][0]
    table.loc[june, "z"], table.loc[june + 1, "z"] = 111.0, 222.0
    days = pd.to_datetime(["2023-08-14", "2023-08-15"])
    _, z = enso.on_dates(table, days)
    assert z[0] == 111.0        # 14 August: July's value is not out yet, June's is in force
    assert z[1] == 222.0        # 15 August: July's value may be used


def test_enso_zscore_ignores_later_months():
    short = enso.add_signal(fake_enso("2023-06"))
    longer = fake_enso("2024-09")
    longer.loc[longer.index[-10:], "anomaly"] = 9.0        # wild future values
    longer = enso.add_signal(longer)
    pd.testing.assert_series_equal(short["z"], longer["z"].iloc[:len(short)])


def test_enso_states_follow_the_threshold():
    table = enso.add_signal(fake_enso())
    known = table.dropna(subset=["z"])
    assert (known.loc[known["z"] >= config.ENSO_THRESHOLD, "state"] == "el_nino").all()
    assert (known.loc[known["z"] <= -config.ENSO_THRESHOLD, "state"] == "la_nina").all()
    assert set(known["state"]) == {"el_nino", "neutral", "la_nina"}


# --- Risk budget and volatility target ---------------------------------------

COV = np.array([[0.09, 0.03, 0.02], [0.03, 0.04, 0.01], [0.02, 0.01, 0.0625]])


def test_risk_contributions_match_the_budgets():
    budgets = np.array([0.6, 0.3, 0.1])
    weights = signals.risk_budget_weights(COV, budgets)
    assert weights.sum() == pytest.approx(1.0)
    assert signals.risk_contributions(weights, COV) == pytest.approx(budgets, abs=1e-8)


def test_equal_budgets_give_the_erc_weights():
    equal = signals.risk_budget_weights(COV, np.full(3, 1 / 3))
    erc = signals.erc_weights(COV)
    assert equal == pytest.approx(erc / erc.sum(), abs=1e-9)


def test_budgets_grow_with_alpha_and_respect_the_70_percent_limit():
    budgets, limited = alpha.risk_budgets(np.array([1.0, 1.0, 1.0]), k=1.0)
    assert budgets == pytest.approx([1 / 3, 1 / 3, 1 / 3]) and not limited
    budgets, limited = alpha.risk_budgets(np.array([1.5, 1.0]), k=1.0)
    assert budgets[0] > budgets[1] and budgets.sum() == pytest.approx(1.0) and not limited
    budgets, limited = alpha.risk_budgets(np.array([3.0, 0.1, 0.1]), k=1.0)
    assert budgets == pytest.approx([0.70, 0.15, 0.15]) and limited


@pytest.mark.parametrize("sizing", ["EW_VT", "ERC", "TILT"])
def test_predicted_volatility_is_the_target(sizing):
    full = fake_market()
    spec = alpha.AgSpec(sizing=sizing)
    decision = alpha.decide(spec, full.returns.to_numpy()[:150], "la_nina", ASSETS)
    assert decision["predicted_vol"] == pytest.approx(config.VOL_TARGET)
    if sizing != "EW_VT":
        held = decision["target"] > 0
        assert decision["risk_shares"][held] == pytest.approx(decision["budgets"][held], abs=1e-8)


def test_all_cash_when_no_alpha_is_positive():
    falling = np.full((60, 3), -0.002) + np.random.default_rng(0).normal(0, 0.001, (60, 3))
    decision = alpha.decide(alpha.AgSpec(), falling, "neutral", ASSETS)
    assert (decision["alpha"] <= 0).all() and (decision["target"] == 0).all()
    # In La Nina the ENSO effect (+1) can outweigh weak negative momentum,
    # but never creates a negative weight.
    assert (alpha.decide(alpha.AgSpec(), falling, "la_nina", ASSETS)["target"] >= 0).all()


# --- Positions, rolls and costs ----------------------------------------------

def test_positions_are_whole_and_never_negative(result):
    positions = result[[f"pos_{a}" for a in ASSETS]].to_numpy()
    assert (positions >= 0).all()
    assert (positions == np.round(positions)).all()
    assert (positions > 0).any()            # the test would be empty otherwise
    assert (result[[f"target_{a}" for a in ASSETS]].to_numpy() >= 0).all()


def test_pnl_and_costs_add_up_and_rolls_create_no_pnl(result):
    """Each day: change in equity = contracts x multiplier x price change of
    the HELD contract - costs. The 10% gap between old and new contract on a
    roll day must not appear anywhere."""
    full = fake_market()
    rate = config.COST_BPS / 10_000.0
    positions = result[[f"pos_{a}" for a in ASSETS]].to_numpy()
    equity = result["equity"].to_numpy()
    first = full.start
    saw_roll_with_position = False
    for k in range(1, len(result)):
        t = first + k
        p_old, p_new, p_before = full.p_old.iloc[t].to_numpy(), full.p_new.iloc[t].to_numpy(), full.p_new.iloc[t - 1].to_numpy()
        before, after = positions[k - 1], positions[k]
        pnl = np.sum(before * MULT * (p_old - p_before))
        cost = 0.0
        for i in range(3):
            if full.roll.iloc[t, i]:
                cost += rate * MULT[i] * (before[i] * p_old[i] + after[i] * p_new[i])
                saw_roll_with_position |= before[i] > 0
            else:
                cost += rate * MULT[i] * abs(after[i] - before[i]) * p_new[i]
        assert result["cost"].iloc[k] == pytest.approx(cost, abs=1e-6)
        assert equity[k] - equity[k - 1] == pytest.approx(pnl - cost, abs=1e-5)
    assert saw_roll_with_position


def test_gross_return_is_net_return_plus_costs(result):
    assert (result["gross_ret"] >= result["ret"] - 1e-15).all()
    assert result["cost"].sum() > 0


def test_order_waits_when_the_market_is_closed():
    plain = fake_market()
    base = ag_backtest.run(plain, enso.add_signal(fake_enso()), alpha.AgSpec())
    # Find a fill day for corn, and close the corn market on it.
    changed = base["pos_ZC"].diff().fillna(0) != 0
    fill_day = base.index[changed][3]
    t = plain.dates.get_loc(fill_day)
    closed = ag_backtest.run(fake_market(closed_day=t), enso.add_signal(fake_enso()), alpha.AgSpec())
    assert closed["pos_ZC"].iloc[t - plain.start] == closed["pos_ZC"].iloc[t - plain.start - 1]
    assert closed["pos_ZC"].iloc[t - plain.start + 1] == closed["order_ZC"].iloc[t - plain.start]


# --- No future parameter fitting ---------------------------------------------

def test_settings_come_from_config_and_the_holdout_is_the_default():
    spec = alpha.AgSpec()
    assert (spec.mom_lookback, spec.shrinkage, spec.tilt_k, spec.cost_bps) == (
        config.MOM_LOOKBACK, config.SHRINKAGE, config.TILT_K, config.COST_BPS)
    assert pd.Timestamp(config.AG_DEV_END) < pd.Timestamp(config.IS_END) < pd.Timestamp(config.OOS_START)
    # At most five years of futures data in total.
    assert (pd.Timestamp(config.AG_OOS_END) - pd.Timestamp(config.AG_DATA_START)).days <= 5 * 366
    # market.load_market's default cutoff is the end of the in-sample period.
    assert market.load_market.__defaults__[0] == config.IS_END


def test_cost_setting_scales_costs():
    full, table = fake_market(), enso.add_signal(fake_enso())
    free = ag_backtest.run(full, table, replace(alpha.AgSpec(), cost_bps=0.0))
    assert free["cost"].sum() == 0.0
    assert (free["ret"] == free["gross_ret"]).all()
