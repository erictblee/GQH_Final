"""Tests of the A/B/C/D attribution models (ABCD_PLAN.md), on fake data."""

import numpy as np
import pandas as pd
import pytest

import config
from src import abcd_alpha, ag_backtest, ag_metrics, enso, market
from src.abcd_alpha import AbcdSpec
from test_enso import fake_enso
from test_xa import ASSETS, fake_market

GRAINS = tuple(config.AG_ASSETS)
IS_GRAIN = np.array([a in GRAINS for a in ASSETS])


def decision(state="neutral", **changes):
    returns = fake_market().returns.to_numpy()[:300]
    return abcd_alpha.decide(AbcdSpec(**changes), returns, state, ASSETS)


def test_model_a_is_equal_notional_in_the_direction_of_the_trend():
    d = decision(sizing="equal", enso="none")
    assert (d["target"] > 0).any() and (d["target"] < 0).any()
    assert (np.sign(d["target"]) == np.sign(d["momentum"])).all()
    assert np.abs(d["target"]) == pytest.approx(np.full(len(ASSETS), config.ABCD_EQUAL_GROSS / len(ASSETS)))


def test_model_b_is_signal_over_volatility_scaled_without_correlations():
    d = decision(sizing="invvol", enso="none")
    assert d["target"] * d["sigma"] == pytest.approx(
        np.sign(d["momentum"]) * config.VOL_TARGET / np.sqrt(len(ASSETS)))
    own_risk_only = np.sqrt(np.sum(d["target"] ** 2 * d["sigma"] ** 2))
    assert own_risk_only == pytest.approx(config.VOL_TARGET)


def test_model_c_keeps_b_positions_and_only_rescales_with_the_covariance():
    b, c = decision(sizing="invvol", enso="none"), decision(sizing="cov", enso="none")
    ratio = c["target"] / b["target"]
    assert ratio == pytest.approx(np.full(len(ASSETS), ratio[0]))       # one common factor
    assert ratio[0] != pytest.approx(1.0)
    assert np.sqrt(c["target"] @ c["cov"] @ c["target"]) == pytest.approx(config.VOL_TARGET)


def test_enso_changes_only_the_grains():
    for state in ["el_nino", "neutral", "la_nina"]:
        without = decision(state, sizing="cov", enso="none")["alpha"]
        added = decision(state, sizing="cov", enso="add")["alpha"]
        multiplied = decision(state, sizing="cov", enso="mult")["alpha"]
        effect = np.array([config.ENSO_EFFECT.get(a, {}).get(state, 0.0) for a in ASSETS])
        assert (added[~IS_GRAIN] == without[~IS_GRAIN]).all()
        assert (multiplied[~IS_GRAIN] == without[~IS_GRAIN]).all()
        assert added == pytest.approx(without + effect)
        agree, disagree = without * effect > 0, without * effect < 0
        assert multiplied[agree] == pytest.approx(without[agree] * config.ABCD_ENSO_AGREE)
        assert multiplied[disagree] == pytest.approx(without[disagree] * config.ABCD_ENSO_DISAGREE)
        # The multiplier never flips or closes a position.
        assert (np.sign(multiplied) == np.sign(without)).all()


def test_enso_only_ignores_the_trend():
    d = decision("la_nina", sizing="cov", enso="only", universe=GRAINS)
    assert (d["target"][IS_GRAIN] > 0).all() and (d["target"][~IS_GRAIN] == 0).all()
    assert (decision("neutral", sizing="cov", enso="only", universe=GRAINS)["target"] == 0).all()


def test_universe_and_long_only_switches():
    grains = decision(sizing="equal", enso="none", universe=GRAINS)
    assert (grains["target"][~IS_GRAIN] == 0).all()
    assert np.abs(grains["target"]).sum() == pytest.approx(config.ABCD_EQUAL_GROSS)
    long_only = decision(sizing="cov", enso="none", long_only=True)
    assert (long_only["target"] >= 0).all()
    assert ((long_only["target"] > 0) == (long_only["momentum"] > 0)).all()


def test_robustness_settings_reach_the_calculation():
    base = decision(sizing="cov", enso="none")
    assert decision(sizing="cov", enso="none", vol_target=0.12)["target"] == pytest.approx(base["target"] * 1.2)
    assert not np.allclose(decision(sizing="cov", enso="none", half_life=40)["sigma"], base["sigma"])


@pytest.fixture(scope="module")
def run_result():
    ledger = []
    daily = ag_backtest.run(fake_market(), enso.add_signal(fake_enso()), AbcdSpec(),
                            decide=abcd_alpha.decide, ledger=ledger)
    return daily, pd.DataFrame(ledger)


def test_cutting_off_the_future_changes_nothing(run_result):
    full = fake_market()
    cut = 380
    short = market.Market(
        dates=full.dates[:cut], assets=full.assets, p_new=full.p_new.iloc[:cut], p_old=full.p_old.iloc[:cut],
        returns=full.returns.iloc[:cut], roll=full.roll.iloc[:cut], is_open=full.is_open.iloc[:cut],
        volume=full.volume.iloc[:cut], held=full.held.iloc[:cut], start=full.start, source="SYNTHETIC")
    again = ag_backtest.run(short, enso.add_signal(fake_enso()), AbcdSpec(), decide=abcd_alpha.decide)
    pd.testing.assert_frame_equal(again, run_result[0].iloc[:len(again)])


def test_limits_hold_and_costs_are_all_in_the_ledger(run_result):
    daily, ledger = run_result
    weights = daily[[f"target_{a}" for a in ASSETS]].to_numpy()
    assert (np.abs(weights) <= config.XA_MAX_WEIGHT + 1e-12).all()
    assert (np.abs(weights).sum(axis=1) <= config.GROSS_CAP + 1e-9).all()
    assert ledger["cost"].sum() == pytest.approx(daily["cost"].sum())


def test_rebalancing_every_second_week_decides_half_as_often(run_result):
    weekly, _ = run_result
    spec = AbcdSpec(rebalance_weeks=2)
    slow = ag_backtest.run(fake_market(), enso.add_signal(fake_enso()), spec, decide=abcd_alpha.decide)
    n_weekly, n_slow = weekly["decision_day"].sum(), slow["decision_day"].sum()
    assert n_slow == (n_weekly + 1) // 2
    # Every decision of the slow run falls on a decision day of the weekly run.
    assert weekly.loc[slow.index[slow["decision_day"]], "decision_day"].all()


def test_market_by_market_report_adds_up(run_result):
    daily, _ = run_result
    table = ag_metrics.by_contract(daily, ASSETS)
    total = ag_metrics.total_pnl(daily, ASSETS)
    assert total == pytest.approx(daily["equity"].iloc[-1] - config.CAPITAL)
    assert table["pnl_dollars"].sum() == pytest.approx(total)
    assert table["share_of_pnl"].sum() == pytest.approx(1.0)
    assert (table["long_pnl_dollars"] + table["short_pnl_dollars"]).to_numpy() == pytest.approx(
        table["pnl_dollars"].to_numpy())
    assert ag_metrics.annual_pnl(daily, ASSETS).sum() == pytest.approx(total)
    assert ag_metrics.episode_counts(daily, ASSETS)["episodes"] == table["trades"].sum()
