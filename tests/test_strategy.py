"""Tests of the core rules: rolls, ERC, look-ahead, costs, leverage cap,
drawdown brake and closed markets. Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import config
from conftest import tiny_market
from src import backtest, market, signals


# ---------------------------------------------------------------------------
# 1. Rolls
# ---------------------------------------------------------------------------

def test_roll_gaps_never_appear_as_returns(data_dir, synthetic_market):
    """On a roll day the old and new contracts have different prices, but the
    return must be the OLD contract's own price change."""
    m = synthetic_market
    raw = pd.read_csv(data_dir / "settlements.csv", parse_dates=["date"])
    raw = raw.set_index(["contract", "date"])["settle"]

    for asset in m.assets:
        roll_rows = np.flatnonzero(m.roll[asset].to_numpy())
        assert len(roll_rows) > 20                      # there really are rolls
        for t in roll_rows:
            today, yesterday = m.dates[t], m.dates[t - 1]
            old_contract = m.held[asset].iloc[t - 1]    # held coming into today
            assert m.held[asset].iloc[t] != old_contract
            assert m.p_old[asset].iloc[t] != m.p_new[asset].iloc[t]   # a gap exists
            # Skip the rare case where yesterday was a closed day for this asset.
            if not m.is_open[asset].iloc[t - 1]:
                continue
            same_contract_return = raw[old_contract, today] / raw[old_contract, yesterday] - 1.0
            assert abs(m.returns[asset].iloc[t] - same_contract_return) < 1e-12


def test_roll_happens_five_business_days_before_the_anchor(data_dir, synthetic_market):
    m = synthetic_market
    raw = pd.read_csv(data_dir / "settlements.csv", parse_dates=["date"])
    contracts = pd.read_csv(data_dir / "contracts.csv", parse_dates=["expiration"]).set_index("contract")

    for asset in m.assets:
        t = np.flatnonzero(m.roll[asset].to_numpy())[5]     # some roll in the middle
        roll_day = m.dates[t]
        old = m.held[asset].iloc[t - 1]
        days = raw[raw["contract"] == old]["date"].sort_values().reset_index(drop=True)

        if config.ROLL_ANCHOR[asset] == "expiration":
            anchor = contracts.loc[old, "expiration"]
        else:
            # First notice day: last business day of the month before the contract month.
            month_before = pd.Period(year=contracts.loc[old, "year"], month=contracts.loc[old, "month"], freq="M") - 1
            anchor = days[days.dt.to_period("M") == month_before].iloc[-1]

        position_of_anchor = days[days == anchor].index[0]
        assert days[position_of_anchor - config.ROLL_DAYS_BEFORE] == roll_day


# ---------------------------------------------------------------------------
# 2. ERC weights
# ---------------------------------------------------------------------------

def random_cov(seed):
    """A random but valid 3x3 covariance matrix."""
    x = np.random.default_rng(seed).normal(size=(200, 3)) * [0.01, 0.004, 0.009]
    x[:, 1] += 0.3 * x[:, 0]                        # add some correlation
    return signals.annualized_cov(x)


def test_erc_gives_equal_risk_contributions_and_hits_the_vol_target():
    for seed in range(5):
        cov = random_cov(seed)
        w_hat = signals.erc_weights(cov)
        assert np.all(w_hat > 0)
        assert np.allclose(signals.risk_contributions(w_hat, cov), 1 / 3, atol=1e-9)

        weights, _ = signals.target_weights(cov, "B", config.VOL_TARGET)
        if weights.sum() < config.GROSS_CAP:        # only when the cap is not binding
            assert abs(np.sqrt(weights @ cov @ weights) - config.VOL_TARGET) < 1e-12


def test_versions_a_and_b_agree_when_correlations_are_zero():
    cov = np.diag([0.16, 0.06, 0.15]) ** 2          # no correlation at all
    a, _ = signals.target_weights(cov, "A", config.VOL_TARGET)
    b, _ = signals.target_weights(cov, "B", config.VOL_TARGET)
    assert np.allclose(a, b, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. No look-ahead
# ---------------------------------------------------------------------------

def test_no_look_ahead(data_dir, synthetic_market):
    """Cutting the data off earlier must not change anything before the cut.
    If it did, some number would depend on data from the future."""
    long = synthetic_market
    short = market.load_market(cutoff="2019-07-15", data_dir=data_dir)
    n = len(short.dates)
    assert n < len(long.dates)

    for name in ["p_new", "p_old", "returns", "roll", "is_open", "held"]:
        assert getattr(short, name).equals(getattr(long, name).iloc[:n]), name

    for version in ["A", "B"]:
        spec = backtest.Spec(name=version, version=version)
        result_short = backtest.run(short, spec)
        result_long = backtest.run(long, spec).iloc[: len(result_short)]
        assert result_short.equals(result_long), version


def test_warning_when_data_ends_inside_a_roll_window(data_dir):
    """Known limitation: a roll date is '5 business days before the anchor',
    and we only learn which days were business days once they have passed.
    If the data ends in those last few days the roll cannot be seen, so the
    loader must warn. (2019-06-14 is an ES roll date in the synthetic data.)"""
    with pytest.warns(UserWarning, match="final roll may be missing"):
        market.load_market(cutoff="2019-06-14", data_dir=data_dir)


# ---------------------------------------------------------------------------
# 4. Costs
# ---------------------------------------------------------------------------

def test_trade_and_roll_costs_are_charged_exactly():
    """Prices never move, so the only thing that can change equity is cost.
    Benchmark weights 60% ES / 40% ZN; ES rolls on the last day."""
    # The ES contract we hold settles at 3990 every day. On the last day we
    # roll into the next contract, which settles at 4000 (a 10-point gap).
    p_old = {"ES": [3990.0] * 4, "ZN": [110.0] * 4, "GC": [2000.0] * 4}
    p_new = {"ES": [3990.0, 3990.0, 3990.0, 4000.0], "ZN": [110.0] * 4, "GC": [2000.0] * 4}
    roll = {"ES": [False, False, False, True], "ZN": [False] * 4, "GC": [False] * 4}
    m = tiny_market(p_new, p_old=p_old, roll=roll, start=1)

    spec = backtest.Spec(name="6040", version="FIXED", assets=("ES", "ZN"),
                         fixed_weights=config.SIXTY_FORTY)
    result = backtest.run(m, spec)

    rate = config.COST_BPS / 10_000
    capital = config.CAPITAL
    # Day 1 (start): decide. Contracts = round(weight x equity / (price x multiplier)).
    es = round(0.60 * capital / (3990.0 * 50))
    zn = round(0.40 * capital / (110.0 * 1000))
    # Day 2: fill. Cost = rate x contracts x multiplier x price.
    trade_cost = rate * (es * 50 * 3990.0 + zn * 1000 * 110.0)
    assert abs(result["equity"].iloc[1] - (capital - trade_cost)) < 1e-6
    assert result["trades"].iloc[1] == 2
    # Day 3: ES rolls from 3990 (old) to 4000 (new). Both legs pay.
    roll_cost = rate * es * 50 * (3990.0 + 4000.0)
    assert abs(result["equity"].iloc[2] - (capital - trade_cost - roll_cost)) < 1e-6
    assert result["pos_ES"].iloc[2] == es               # same number of contracts


def test_roll_day_with_a_rebalance_charges_old_count_and_new_count():
    """A rebalance is filled on the same day ES rolls. We must pay for closing
    the OLD contract count at the old price and opening the NEW count at the
    new price.

    Days: Wed 29 Jan, Thu 30 Jan (start: decide), Fri 31 Jan (fill; month-end,
    so decide again), Mon 3 Feb (fill + ES roll)."""
    p_old = {"ES": [3990.0] * 4, "ZN": [110.0, 110.0, 121.0, 121.0], "GC": [2000.0] * 4}
    p_new = {"ES": [3990.0, 3990.0, 3990.0, 4000.0], "ZN": p_old["ZN"], "GC": p_old["GC"]}
    roll = {"ES": [False, False, False, True], "ZN": [False] * 4, "GC": [False] * 4}
    m = tiny_market(p_new, p_old=p_old, roll=roll, start=1, first_day="2020-01-29")

    spec = backtest.Spec(name="6040", version="FIXED", assets=("ES", "ZN"),
                         fixed_weights=config.SIXTY_FORTY)
    result = backtest.run(m, spec)
    rate = config.COST_BPS / 10_000

    # Thu: decide the first position. Fri: fill it. ZN was still flat on Thu,
    # so its jump from 110 to 121 on Fri earns nothing.
    es1 = round(0.60 * config.CAPITAL / (3990.0 * 50))
    zn1 = round(0.40 * config.CAPITAL / (110.0 * 1000))
    equity_fri = config.CAPITAL - rate * (es1 * 50 * 3990.0 + zn1 * 1000 * 121.0)
    assert abs(result["equity"].iloc[1] - equity_fri) < 1e-6

    # Fri is month-end: rebalance back to 60/40 using Friday's equity and prices.
    es2 = round(0.60 * equity_fri / (3990.0 * 50))
    zn2 = round(0.40 * equity_fri / (121.0 * 1000))
    assert zn2 != zn1                                   # a real rebalance happens

    # Mon: prices do not move, so only costs change equity.
    es_cost = rate * 50 * (es1 * 3990.0 + es2 * 4000.0)         # roll: old count out, new count in
    zn_cost = rate * 1000 * abs(zn2 - zn1) * 121.0              # ordinary trade
    assert abs(result["equity"].iloc[2] - (equity_fri - es_cost - zn_cost)) < 1e-6
    assert result["pos_ES"].iloc[2] == es2 and result["pos_ZN"].iloc[2] == zn2


def test_no_costs_are_charged_when_the_cost_rate_is_zero(synthetic_market):
    base = backtest.run(synthetic_market, backtest.Spec())
    assert base["cost"].sum() > 0
    free = backtest.run(synthetic_market, backtest.Spec(cost_bps=0.0))
    assert free["cost"].sum() == 0


# ---------------------------------------------------------------------------
# 5. Leverage cap
# ---------------------------------------------------------------------------

def test_leverage_cap_scales_weights_down_proportionally():
    cov = np.diag([0.03, 0.01, 0.03]) ** 2          # very calm markets
    for version in ["A", "B"]:
        uncapped_sum = config.VOL_TARGET / np.sqrt(3) * (1 / 0.03 + 1 / 0.01 + 1 / 0.03)
        assert uncapped_sum > config.GROSS_CAP
        weights, cap_binding = signals.target_weights(cov, version, config.VOL_TARGET)
        assert cap_binding
        assert abs(weights.sum() - config.GROSS_CAP) < 1e-12
        # Proportions are unchanged: ZN is 3x calmer, so its weight is 3x larger.
        assert abs(weights[1] / weights[0] - 3.0) < 1e-8


def test_cap_binds_on_some_days_of_the_synthetic_run(synthetic_market):
    result = backtest.run(synthetic_market, backtest.Spec())
    assert result["cap_binding"].any()


# ---------------------------------------------------------------------------
# 6. Drawdown brake
# ---------------------------------------------------------------------------

def test_brake_switches_on_above_15_percent_and_off_below_7_5_percent(synthetic_market):
    result = backtest.run(synthetic_market, backtest.Spec())
    assert result["braking"].any() and not result["braking"].all()

    # Replay the brake rule from the equity curve alone and compare.
    drawdown = 1.0 - result["equity"] / result["equity"].cummax()
    braking = False
    for day in result.index:
        if not braking and drawdown[day] > config.BRAKE_ON:
            braking = True
        elif braking and drawdown[day] < config.BRAKE_OFF:
            braking = False
        assert result.at[day, "braking"] == braking
        expected_target = config.BRAKE_VOL_TARGET if braking else config.VOL_TARGET
        assert result.at[day, "sigma_star"] == expected_target

    no_brake = backtest.run(synthetic_market, backtest.Spec(brake=False))
    assert not no_brake["braking"].any()


# ---------------------------------------------------------------------------
# 7. Closed markets
# ---------------------------------------------------------------------------

def test_closed_market_has_zero_return_and_no_trade(synthetic_market):
    m = synthetic_market
    result = backtest.run(m, backtest.Spec())
    checked = 0
    for asset in ["ZN", "GC"]:
        closed_days = m.dates[~m.is_open[asset].to_numpy()]
        for day in closed_days:
            assert m.returns.at[day, asset] == 0.0
            if day in result.index[1:]:
                before = result[f"pos_{asset}"].shift(1).at[day]
                assert result.at[day, f"pos_{asset}"] == before     # position held
                checked += 1
    assert checked > 10
