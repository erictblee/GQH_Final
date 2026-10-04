"""Run every planned variant, benchmark and diagnostic and print one table.

    python run_all.py            in-sample only (the default)
    python run_all.py --final    the full sample, including out-of-sample.
                                 Run this ONCE, at the very end.

The list of variants is fixed in HYPOTHESIS.md ("Planned variants"). Do not
add a variant here without also adding it to the count in the note. Anything
that is not one of the six variants or the two cost stress tests is named
`bench_*` (benchmark) or `diag_*` (diagnostic).
"""

import argparse
from dataclasses import replace

import pandas as pd

import config
from src import analysis, backtest, market, metrics

# The six planned variants, in the order HYPOTHESIS.md lists them.
VARIANTS = [
    backtest.Spec(name="A", version="A"),
    backtest.Spec(name="B", version="B"),
    backtest.Spec(name="B_no_gold", version="B", assets=("ES", "ZN")),
    backtest.Spec(name="B_window_40", version="B", window=40),
    backtest.Spec(name="B_window_90", version="B", window=90),
    backtest.Spec(name="B_window_120", version="B", window=120),
]

# Cost stress tests (4 bps instead of 2).
STRESS = [
    backtest.Spec(name="A_cost_4bps", version="A", cost_bps=config.STRESS_COST_BPS),
    backtest.Spec(name="B_cost_4bps", version="B", cost_bps=config.STRESS_COST_BPS),
]

# Benchmarks. NOT counted as strategy variants.
BENCHMARKS = [
    backtest.Spec(name="bench_equal_weight", version="FIXED", fixed_weights=config.EQUAL_WEIGHT),
    backtest.Spec(name="bench_60_40", version="FIXED", assets=("ES", "ZN"), fixed_weights=config.SIXTY_FORTY),
]

# Diagnostics named in HYPOTHESIS.md. NOT counted as strategy variants.
DIAGNOSTICS = [
    backtest.Spec(name="diag_B_fractional", version="B", integer_contracts=False),
    backtest.Spec(name="diag_B_demeaned_gold", version="B", demean_asset="GC"),
]

# Brake-noise diagnostics: every run above that has a drawdown brake, run
# again with the brake switched off. Reporting only; the strategy's rule is
# unchanged. They show which conclusions depend on when the brake fires.
NO_BRAKE_TWINS = [
    replace(spec, name=f"diag_{spec.name.removeprefix('diag_')}_no_brake", brake=False)
    for spec in VARIANTS + STRESS + DIAGNOSTICS
]


def constant_leverage_spec(m, target_volatility):
    """The constant-leverage comparison portfolio (prediction 2), with its
    multiplier chosen so that its realized volatility matches
    target_volatility (B without the brake), up to the 4x cap.

    Volatility is very nearly proportional to the multiplier, so we start at
    1, rescale, and repeat a few times."""
    leverage = 1.0
    for _ in range(4):
        spec = backtest.Spec(name="diag_constant_leverage", version="CL", brake=False, leverage=leverage)
        realized = analysis.volatility(backtest.run(m, spec)["ret"].iloc[1:])
        leverage = min(leverage * target_volatility / realized, config.GROSS_CAP)
    return backtest.Spec(name="diag_constant_leverage", version="CL", brake=False, leverage=leverage)


def run_everything(m):
    """Run every spec on market m. Returns {name: daily results table}."""
    results = {}
    for spec in VARIANTS + STRESS + BENCHMARKS + DIAGNOSTICS + NO_BRAKE_TWINS:
        results[spec.name] = backtest.run(m, spec)

    # The constant-leverage portfolio needs B-without-brake's volatility first.
    target = analysis.volatility(results["diag_B_no_brake"]["ret"].iloc[1:])
    spec = constant_leverage_spec(m, target)
    results[spec.name] = backtest.run(m, spec)
    results[spec.name].attrs["leverage"] = spec.leverage
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the out-of-sample period (run once, at the end)")
    args = parser.parse_args()

    # In-sample by default: nothing after config.IS_END is even loaded.
    cutoff = None if args.final else config.IS_END
    label = "full" if args.final else "in_sample"
    m = market.load_market(cutoff)

    print(f"DATA SOURCE: {m.source}")
    print(f"Period: {m.dates[m.start].date()} to {m.dates[-1].date()} ({label})")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results = run_everything(m)
    table = {}
    for name, daily in results.items():
        daily = daily.copy()
        daily["source"] = m.source          # SYNTHETIC or real, on every row
        daily.to_csv(config.RESULTS_DIR / f"pandas_{name}_{label}.csv")
        table[name] = metrics.summary(daily)

    table = pd.DataFrame(table).T           # one row per run
    table["source"] = m.source
    table.to_csv(config.RESULTS_DIR / f"summary_{label}.csv")

    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(f"\nDATA SOURCE: {m.source}. Saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
