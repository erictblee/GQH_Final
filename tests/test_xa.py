"""Tests of the cross-asset long/short strategy (XA_SPEC.md), on fake data."""

import numpy as np
import pandas as pd
import pytest

import config
from src import ag_backtest, ag_metrics, enso, market, xa_alpha
from test_enso import fake_enso

ASSETS = config.XA_ASSETS
MULT = np.array([config.MULTIPLIER[a] for a in ASSETS])


def fake_market(n_days=420, seed=3):
    """Random prices for the eleven futures: some trend up, some trend
    down, with a roll (and a 5% price gap) every 45 days."""
    rng = np.random.default_rng(seed)
    n = len(ASSETS)
    dates = pd.bdate_range(config.DATA_START, periods=n_days)
    drift = np.linspace(-0.0012, 0.0012, n)             # falling ... rising assets
    returns = drift + rng.normal(0.0, 0.006, size=(n_days, 1)) + rng.normal(0.0, 0.010, size=(n_days, n))
    returns[0] = 0.0
    start_price = 100.0 * MULT.max() / MULT / 250.0 + 50.0
    p_old, p_new = np.zeros((n_days, n)), np.zeros((n_days, n))
    roll = np.zeros((n_days, n), dtype=bool)
    p_old[0] = p_new[0] = start_price
    for t in range(1, n_days):
        p_old[t] = p_new[t - 1] * (1.0 + returns[t])
        p_new[t] = p_old[t]
        if t % 45 == 0:
            roll[t] = True
            p_new[t] = p_old[t] * 1.05

    def frame(values):
        return pd.DataFrame(values, index=dates, columns=ASSETS)

    names = np.array([[f"{a}_{t // 45}" for a in ASSETS] for t in range(n_days)])
    return market.Market(
        dates=dates, assets=list(ASSETS), p_new=frame(p_new), p_old=frame(p_old),
        returns=((frame(p_old) - frame(p_new).shift(1)) / frame(p_new).shift(1)).fillna(0.0),
        roll=frame(roll), is_open=frame(np.ones((n_days, n), dtype=bool)),
        volume=frame(np.zeros((n_days, n))), held=frame(names),
        start=config.XA_WARMUP_DAYS, source="SYNTHETIC")


@pytest.fixture(scope="module")
def run_result():
    ledger = []
    daily = ag_backtest.run(fake_market(), enso.add_signal(fake_enso()), xa_alpha.XaSpec(),
                            decide=xa_alpha.decide, ledger=ledger)
    return daily, pd.DataFrame(ledger)


def decision(**changes):
    returns = fake_market().returns.to_numpy()[:300]
    return xa_alpha.decide(xa_alpha.XaSpec(**changes), returns, "la_nina", ASSETS)


def test_sign_of_alpha_sets_direction_and_size_sets_risk_share():
    d = decision()
    held = d["target"] != 0
    assert (d["target"] > 0).any() and (d["target"] < 0).any()      # both longs and shorts
    assert (np.sign(d["target"][held]) == np.sign(d["alpha"][held])).all()
    # Every held asset carries the share of risk its signal strength asks for.
    assert d["risk_shares"][held] == pytest.approx(np.abs(d["budgets"][held]), abs=1e-8)
    assert np.abs(d["budgets"]).sum() == pytest.approx(1.0)
    assert d["predicted_vol"] == pytest.approx(config.VOL_TARGET)


def test_tiny_signals_get_no_position():
    d = decision()
    tiny = np.abs(d["alpha"]) < config.XA_MIN_SIGNAL
    assert (d["target"][tiny] == 0).all()
    assert (d["target"][~tiny] != 0).all()


def test_without_covariance_it_is_inverse_volatility():
    d = decision(use_cov=False)
    held = d["target"] != 0
    off_diagonal = d["cov_final"] - np.diag(np.diag(d["cov_final"]))
    assert (off_diagonal == 0).all()
    # With zero correlations the weight is sqrt(risk share) / volatility.
    expected = np.sign(d["alpha"][held]) * np.sqrt(np.abs(d["budgets"][held])) / d["sigma"][held]
    ratio = d["target"][held] / expected
    assert ratio == pytest.approx(np.full(held.sum(), ratio[0]))


def test_enso_changes_only_the_grains():
    without, with_enso = decision(use_enso=False), decision(use_enso=True)
    change = with_enso["alpha"] - without["alpha"]
    for i, asset in enumerate(ASSETS):
        expected = config.ENSO_EFFECT[asset]["la_nina"] if asset in config.AG_ASSETS else 0.0
        assert change[i] == pytest.approx(expected)


def test_recent_days_count_more_in_the_covariance():
    returns = fake_market().returns.to_numpy()[:300]
    calm_then_wild = returns.copy()
    calm_then_wild[-20:] *= 3.0                 # the last 20 days are three times as volatile
    wild_then_calm = returns.copy()
    wild_then_calm[-252:-232] *= 3.0            # the same, but long ago
    recent = np.diag(xa_alpha.ewma_cov(calm_then_wild))
    old = np.diag(xa_alpha.ewma_cov(wild_then_calm))
    assert (recent > old).all()


def test_cutting_off_the_future_changes_nothing(run_result):
    full = fake_market()
    cut = 380
    short = market.Market(
        dates=full.dates[:cut], assets=full.assets, p_new=full.p_new.iloc[:cut], p_old=full.p_old.iloc[:cut],
        returns=full.returns.iloc[:cut], roll=full.roll.iloc[:cut], is_open=full.is_open.iloc[:cut],
        volume=full.volume.iloc[:cut], held=full.held.iloc[:cut], start=full.start, source="SYNTHETIC")
    again = ag_backtest.run(short, enso.add_signal(fake_enso()), xa_alpha.XaSpec(), decide=xa_alpha.decide)
    pd.testing.assert_frame_equal(again, run_result[0].iloc[:len(again)])


def test_short_positions_earn_when_prices_fall_and_accounts_add_up(run_result):
    daily, _ = run_result
    full = fake_market()
    rate = config.COST_BPS / 10_000.0
    positions = daily[[f"pos_{a}" for a in ASSETS]].to_numpy()
    assert (positions < 0).any() and (positions > 0).any()
    assert (positions == np.round(positions)).all()
    equity = daily["equity"].to_numpy()
    for k in range(1, len(daily)):
        t = full.start + k
        p_old, p_new, p_before = full.p_old.iloc[t].to_numpy(), full.p_new.iloc[t].to_numpy(), full.p_new.iloc[t - 1].to_numpy()
        before, after = positions[k - 1], positions[k]
        pnl = np.sum(before * MULT * (p_old - p_before))        # negative count x falling price = profit
        rolling = full.roll.iloc[t].to_numpy()
        cost = rate * np.sum(MULT * np.where(
            rolling, np.abs(before) * p_old + np.abs(after) * p_new, np.abs(after - before) * p_new))
        assert equity[k] - equity[k - 1] == pytest.approx(pnl - cost, abs=1e-4)


def test_ledger_agrees_with_the_daily_table(run_result):
    daily, ledger = run_result
    assert set(ledger["kind"]) == {"rebalance", "roll_close", "roll_open"}
    # Every dollar of cost appears in the ledger, once.
    assert ledger["cost"].sum() == pytest.approx(daily["cost"].sum())
    assert np.allclose(ledger["cost"], ledger["notional"] * config.COST_BPS / 10_000.0)
    # Adding up every fill of an asset gives the position we end with.
    for a in ASSETS:
        assert ledger.loc[ledger["asset"] == a, "contracts"].sum() == daily[f"pos_{a}"].iloc[-1]
    # A roll closes one contract and opens a DIFFERENT one.
    closes = ledger[ledger["kind"] == "roll_close"].set_index(["date", "asset"])["contract"]
    opens = ledger[ledger["kind"] == "roll_open"].set_index(["date", "asset"])["contract"]
    both = closes.index.intersection(opens.index)
    assert len(both) > 0 and (closes[both] != opens[both]).all()


def test_position_episodes_account_for_all_pnl(run_result):
    daily, _ = run_result
    episodes = ag_metrics.position_episodes(daily, ASSETS)
    total = sum(daily[f"pnl_{a}"].sum() for a in ASSETS)
    assert episodes["pnl"].sum() == pytest.approx(total)
    assert set(episodes["direction"]) == {"long", "short"}


def test_limits_hold(run_result):
    daily, _ = run_result
    weights = daily[[f"target_{a}" for a in ASSETS]].to_numpy()
    assert (np.abs(weights) <= config.XA_MAX_WEIGHT + 1e-12).all()
    assert (np.abs(weights).sum(axis=1) <= config.GROSS_CAP + 1e-9).all()
    budgets = daily[[f"budget_{a}" for a in ASSETS]].to_numpy()
    assert (np.abs(budgets) <= config.MAX_RISK_SHARE + 1e-12).all()


def test_periods_are_in_order_and_locked_by_default():
    assert (pd.Timestamp(config.XA_DEV_END) < pd.Timestamp(config.XA_VALIDATION_END)
            < pd.Timestamp(config.XA_TEST_START) < pd.Timestamp(config.IS_END) < pd.Timestamp(config.OOS_START))
    spec = xa_alpha.XaSpec()
    assert (spec.mom_lookback, spec.cost_bps) == (config.XA_MOM_LOOKBACK, config.COST_BPS)
