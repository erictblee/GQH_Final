"""Tests of the pieces added for the predictions and diagnostics:
the constant-leverage portfolio, the demeaned-gold run, the bootstrap,
the regression helper and period slicing. Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

import config
import run_all
from src import analysis, backtest, metrics, signals


# ---------------------------------------------------------------------------
# Constant-leverage portfolio (prediction 2)
# ---------------------------------------------------------------------------

def test_constant_leverage_weights_keep_the_erc_mix_at_a_fixed_size():
    x = np.random.default_rng(1).normal(size=(200, 3)) * [0.01, 0.004, 0.009]
    cov = signals.annualized_cov(x)
    w_hat = signals.erc_weights(cov)

    weights, cap_binding = signals.constant_leverage_weights(cov, 2.5)
    assert not cap_binding
    assert abs(weights.sum() - 2.5) < 1e-12                     # fixed total size
    assert np.allclose(weights / weights.sum(), w_hat / w_hat.sum())   # ERC mix

    # Twice as volatile markets: an ERC vol-target portfolio would halve its
    # size, but the constant-leverage portfolio must not change at all.
    calm, _ = signals.constant_leverage_weights(cov, 2.5)
    wild, _ = signals.constant_leverage_weights(cov * 4.0, 2.5)
    assert np.allclose(calm, wild)

    # The 4x gross cap still applies.
    capped, cap_binding = signals.constant_leverage_weights(cov, 6.0)
    assert cap_binding and abs(capped.sum() - config.GROSS_CAP) < 1e-12


def test_constant_leverage_run_has_fixed_gross_target_and_no_brake(synthetic_market):
    spec = backtest.Spec(name="cl", version="CL", brake=False, leverage=2.0)
    result = backtest.run(synthetic_market, spec)
    gross_target = result[[f"target_{a}" for a in config.ASSETS]].sum(axis=1)
    assert np.allclose(gross_target, 2.0)
    assert not result["braking"].any()


def test_constant_leverage_multiplier_matches_the_target_volatility(synthetic_market):
    target = 0.08
    spec = run_all.constant_leverage_spec(synthetic_market, target)
    realized = analysis.volatility(backtest.run(synthetic_market, spec)["ret"].iloc[1:])
    assert spec.leverage <= config.GROSS_CAP
    assert abs(realized - target) < 0.002


# ---------------------------------------------------------------------------
# Demeaned gold (prediction 4)
# ---------------------------------------------------------------------------

def test_demeaning_gold_leaves_target_weights_unchanged(synthetic_market):
    """With the brake off, target weights depend only on past returns, which
    the demeaning does not touch. So they must be IDENTICAL; only gold's
    profit and loss changes."""
    normal = backtest.run(synthetic_market, backtest.Spec(brake=False))
    demeaned = backtest.run(synthetic_market, backtest.Spec(brake=False, demean_asset="GC"))

    for asset in config.ASSETS:
        assert normal[f"target_{asset}"].equals(demeaned[f"target_{asset}"])
    assert not normal["equity"].equals(demeaned["equity"])

    # Gold's average contribution moves in the direction of the average
    # return that was removed.
    gold_mean = synthetic_market.returns["GC"].iloc[synthetic_market.start + 1:].mean()
    change = normal["leg_GC"].mean() - demeaned["leg_GC"].mean()
    assert gold_mean != 0 and np.sign(change) == np.sign(gold_mean)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_is_reproducible_with_the_fixed_seed():
    rng = np.random.default_rng(7)
    a = pd.Series(rng.normal(0.0004, 0.006, 1500))
    b = pd.Series(rng.normal(0.0002, 0.006, 1500))

    assert analysis.bootstrap_sharpe(a) == analysis.bootstrap_sharpe(a)
    first = analysis.bootstrap_sharpe_difference(a, b)
    second = analysis.bootstrap_sharpe_difference(a, b)
    assert first == second

    days = analysis.bootstrap_day_numbers(1500)
    assert days.shape == (config.BOOTSTRAP_SAMPLES, 1500)
    assert days.min() >= 0 and days.max() < 1500
    # Blocks average about 20 days: a "jump" happens when the next day is not
    # simply the previous day + 1.
    jumps = (np.diff(days, axis=1) % 1500 != 1).mean()
    assert abs(1.0 / jumps - config.BOOTSTRAP_MEAN_BLOCK) < 1.0

    low, high = analysis.bootstrap_sharpe(a)
    assert low < analysis.sharpe(a) < high
    difference, low, high = first
    assert low < difference < high
    # A series compared with itself has no Sharpe difference at all.
    assert analysis.bootstrap_sharpe_difference(a, a) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Regression helper and period slicing
# ---------------------------------------------------------------------------

def test_newey_west_regression_recovers_a_known_relationship():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"x": rng.normal(size=800)})
    y = 0.5 + 2.0 * x["x"] + rng.normal(scale=0.1, size=800)
    table = analysis.newey_west_regression(y, x)
    assert abs(table.loc["intercept", "coefficient"] - 0.5) < 0.02
    assert abs(table.loc["x", "coefficient"] - 2.0) < 0.02
    assert table.loc["x", "t_stat"] > 50


def test_summary_of_a_period_uses_only_that_period(synthetic_market):
    daily = backtest.run(synthetic_market, backtest.Spec())
    whole = metrics.summary(daily)
    part = metrics.summary(daily, "2015-01-01", "2018-12-31")
    by_hand = daily["ret"].loc["2015-01-01":"2018-12-31"]
    assert abs(part["ann_return"] - by_hand.mean() * config.TRADING_DAYS) < 1e-12
    assert part["trades"] < whole["trades"]
    # The whole-run drawdown equals the drawdown of the equity column.
    assert abs(whole["max_drawdown"] - metrics.max_drawdown(daily["equity"])) < 1e-9


def test_every_braked_run_has_a_no_brake_twin():
    names = {spec.name for spec in run_all.NO_BRAKE_TWINS}
    assert names == {
        "diag_A_no_brake", "diag_B_no_brake", "diag_B_no_gold_no_brake",
        "diag_B_window_40_no_brake", "diag_B_window_90_no_brake", "diag_B_window_120_no_brake",
        "diag_A_cost_4bps_no_brake", "diag_B_cost_4bps_no_brake",
        "diag_B_fractional_no_brake", "diag_B_demeaned_gold_no_brake",
    }
    assert all(not spec.brake for spec in run_all.NO_BRAKE_TWINS)
    # The variant count is frozen at six, plus the two cost stress tests.
    assert len(run_all.VARIANTS) == 6 and len(run_all.STRESS) == 2


# ---------------------------------------------------------------------------
# The report, for a period that does not start at the run's first day
# ---------------------------------------------------------------------------

def test_report_can_be_written_for_a_later_period(synthetic_market, data_dir):
    """The out-of-sample report is a report on a LATER slice of one long run.
    That slice cannot be tested on real out-of-sample data (it must stay
    untouched), so here the same code is run on a later slice of the
    synthetic in-sample data."""
    pytest.importorskip("statsmodels")
    import sys
    sys.path.insert(0, str(config.REPO_DIR / "tools"))
    import make_report
    from src import diagnostics

    results = run_all.run_everything(synthetic_market)
    period = make_report.Period(synthetic_market, results, "2019-01-01", config.IS_END, "Later slice")
    volume = diagnostics.load_contract_volume(config.IS_END, data_dir=data_dir)

    lines = []
    make_report.write_period(lines.append, period, None, volume)
    text = "\n".join(lines)

    assert "# Later slice: 2019-01-01" in text
    for heading in ["## Results table", "### Prediction 1", "### Prediction 6",
                    "## Brake-noise diagnostics", "## Direct costs of B", "### 2. Capacity"]:
        assert heading in text
    # The slice's returns start on its first day, not at the run's first row.
    assert period.ret("B").index[0] >= pd.Timestamp("2019-01-01")
    assert len(period.ret("B")) < len(results["B"])
