"""Run the ENSO strategy, its benchmarks, ablations and robustness tests.

    python run_enso.py            in-sample only (the default)
    python run_enso.py --final    the full sample, including out-of-sample.
                                  Run this ONCE, at the very end.

The list of runs is fixed in ENSO_SPEC.md (sections 23 and 24, and the
robustness tests of sections 6, 12, 13 and 20). Do not add a run here
without adding it to that document first.

Writes into results/:
    enso_<name>_<label>.csv     one row per day for each run
    enso_summary_<label>.csv    the headline numbers of every run and period
    ENSO_REPORT.md              the same, as tables to read
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

import config
from src import ag_backtest, ag_metrics, analysis, enso, market
from src.alpha import AgSpec

# The seven ablation steps of section 24. Each one changes ONE thing.
# Steps 5, 6 and 7 are also benchmarks B, C and D of section 23.
ABLATIONS = [
    AgSpec(name="1_mom_EW", signal="MOM", sizing="EW"),
    AgSpec(name="2_enso_EW", signal="ENSO", sizing="EW"),
    AgSpec(name="3_combo_EW", signal="MOM+ENSO", sizing="EW"),
    AgSpec(name="4_combo_EW_voltarget", signal="MOM+ENSO", sizing="EW_VT"),
    AgSpec(name="5_combo_invvol_B", signal="MOM+ENSO", sizing="IV"),
    AgSpec(name="6_combo_ERC_C", signal="MOM+ENSO", sizing="ERC"),
    AgSpec(name="7_combo_tilt_D", signal="MOM+ENSO", sizing="TILT"),
]
MAIN = ABLATIONS[-1]            # the proposed strategy (benchmark D)

# Benchmark A: the proposed strategy with the ENSO part taken out.
BENCHMARK_A = AgSpec(name="A_mom_tilt", signal="MOM", sizing="TILT")

# Robustness: the proposed strategy with ONE setting changed.
ROBUSTNESS = (
    [replace(MAIN, name=f"rob_shrinkage_{value:g}", shrinkage=value) for value in config.SHRINKAGE_ROBUST]
    + [replace(MAIN, name=f"rob_tilt_k_{value:g}", tilt_k=value) for value in config.TILT_K_ROBUST]
    + [replace(MAIN, name=f"rob_momentum_{days}", mom_lookback=days) for days in config.MOM_ROBUST_LOOKBACKS]
    + [replace(MAIN, name="rob_cost_4bps", cost_bps=config.STRESS_COST_BPS),
       replace(MAIN, name="rob_fractional", integer_contracts=False)]
)

# Which pairs answer which question: (question, run, compared with).
COMPARISONS = [
    ("Does ENSO add to momentum? (D - A)", "7_combo_tilt_D", "A_mom_tilt"),
    ("Does ENSO add to momentum, equal weights? (3 - 1)", "3_combo_EW", "1_mom_EW"),
    ("Does volatility targeting help? (4 - 3)", "4_combo_EW_voltarget", "3_combo_EW"),
    ("Does covariance beat volatility-only? (C - B)", "6_combo_ERC_C", "5_combo_invvol_B"),
    ("Does the alpha tilt beat equal risk? (D - C)", "7_combo_tilt_D", "6_combo_ERC_C"),
]


def sharpe_difference(daily_a, daily_b):
    """Sharpe(a) - Sharpe(b) with a 95% bootstrap interval. Empty (NaN) if
    either run never took a position in the period."""
    a, b = daily_a["ret"], daily_b["ret"]
    if a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan"), float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return analysis.bootstrap_sharpe_difference(a, b)


def markdown_table(table, percent=(), decimals=2):
    """A pandas table as markdown text. Columns named in `percent` are
    shown as percentages."""
    shown = table.copy()
    for column in shown.columns:
        if column in percent:
            shown[column] = [f"{v:.1%}" for v in shown[column]]
        elif shown[column].dtype.kind == "f":
            shown[column] = [f"{v:.{decimals}f}" for v in shown[column]]
    header = "| " + " | ".join(shown.columns) + " |"
    line = "|" + "---|" * len(shown.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.to_numpy()]
    return "\n".join([header, line] + body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the out-of-sample period (final run only)")
    args = parser.parse_args()

    cutoff = None if args.final else config.IS_END
    label = "full" if args.final else "in_sample"
    m = market.load_market(cutoff, data_dir=config.AG_PROCESSED_DIR,
                           assets=config.AG_ASSETS, warmup=config.AG_WARMUP_DAYS)
    enso_table = enso.load_enso()
    if args.final and m.dates[-1] <= pd.Timestamp(config.IS_END):
        raise SystemExit("--final given, but the data on disk is in-sample only. Run the "
                         "three data scripts with --final first.")

    # The periods to report. Out-of-sample exists only in the final run.
    day_after = pd.Timedelta(days=1)
    periods = {
        "development": (None, pd.Timestamp(config.AG_DEV_END)),
        "validation": (pd.Timestamp(config.AG_DEV_END) + day_after, pd.Timestamp(config.IS_END)),
        "in_sample": (None, pd.Timestamp(config.IS_END)),
    }
    if args.final:
        periods["OUT_OF_SAMPLE"] = (pd.Timestamp(config.OOS_START), None)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for spec in ABLATIONS + [BENCHMARK_A] + ROBUSTNESS:
        daily = ag_backtest.run(m, enso_table, spec)
        daily["source"] = m.source
        daily.to_csv(config.RESULTS_DIR / f"enso_{spec.name}_{label}.csv")
        results[spec.name] = daily

    # --- Headline numbers of every run in every period.
    rows = []
    for period, (first, last) in periods.items():
        for name, daily in results.items():
            rows.append({"period": period, "run": name, **ag_metrics.summary(daily, first, last)})
    summary = pd.DataFrame(rows)
    summary.insert(0, "source", m.source)
    summary.to_csv(config.RESULTS_DIR / f"enso_summary_{label}.csv", index=False)

    # --- The report.
    main_daily = results[MAIN.name]
    out = [
        f"# ENSO strategy on corn, soybeans and wheat: {label.replace('_', '-')} results",
        "",
        f"DATA SOURCE: {m.source}. Futures data {m.dates[0].date()} to {m.dates[-1].date()}; "
        f"first trading decision {m.dates[m.start].date()}. Generated by `run_enso.py`"
        + (" --final." if args.final else "; the out-of-sample period has NOT been loaded."),
        "",
        "Every setting was fixed in ENSO_SPEC.md before the first run. Returns are excess "
        "returns net of costs; Sharpe ratios use a risk-free rate of zero; the brackets are "
        "95% bootstrap intervals. **The periods are short (one to three years), so the "
        "intervals are wide: read differences between runs with that in mind.**",
    ]
    shown = ["run", "cagr", "ann_vol", "sharpe", "sharpe_ci_low", "sharpe_ci_high", "sortino",
             "max_drawdown", "calmar", "worst_month", "worst_year", "positive_months"]
    shown_2 = ["run", "gross_return", "net_return", "costs_dollars", "roll_costs_dollars",
               "turnover_per_year", "time_invested", "avg_leverage", "max_leverage", "trades",
               "rolls", "risk_limit_share"]
    percent = {"cagr", "ann_vol", "max_drawdown", "worst_month", "worst_year", "positive_months",
               "gross_return", "net_return", "time_invested", "risk_limit_share", "ann_return"}
    for period in periods:
        part = summary[summary["period"] == period]
        out += ["", f"## {period.replace('_', ' ').capitalize()}: {part['first_day'].iloc[0]} to {part['last_day'].iloc[0]}",
                "", "Risk and return:", "", markdown_table(part[shown], percent),
                "", "Costs, trading and leverage (costs in dollars; turnover in multiples of equity per year):",
                "", markdown_table(part[shown_2], percent)]

        first, last = periods[period]
        comparison_rows = []
        for question, a, b in COMPARISONS:
            difference, low, high = sharpe_difference(results[a].loc[first:last], results[b].loc[first:last])
            comparison_rows.append({"question": question, "sharpe_difference": difference,
                                    "ci_low": low, "ci_high": high})
        out += ["", "What each layer adds (difference in Sharpe ratio; an interval that "
                    "includes zero means the difference cannot be told apart from chance):",
                "", markdown_table(pd.DataFrame(comparison_rows))]
        out += ["", f"Proposed strategy ({MAIN.name}) by commodity:", "",
                markdown_table(ag_metrics.by_commodity(main_daily, m.assets, first, last), percent),
                "", f"Proposed strategy ({MAIN.name}) by ENSO state:", "",
                markdown_table(ag_metrics.by_enso_state(main_daily, first, last), percent)]

    # ENSO history actually used, so the timing can be checked by eye.
    changes = main_daily["enso_state"] != main_daily["enso_state"].shift(1)
    out += ["", "## ENSO state used by the strategy", "",
            "Each row is a day on which the state in force changed.", "",
            markdown_table(main_daily.loc[changes, ["enso_state", "enso_z"]].reset_index().assign(
                date=lambda t: t["date"].dt.date))]

    # Concentration: how much of the result comes from a few days (failure condition 8).
    ret = main_daily["ret"].iloc[1:]
    best_ten = ret.nlargest(10).sum()
    out += ["", "## Concentration", "",
            f"Proposed strategy, {label.replace('_', '-')}: the sum of all daily returns is {ret.sum():.2%}; "
            f"the ten best days add up to {best_ten:.2%} and the ten worst to {ret.nsmallest(10).sum():.2%}."]
    if not config.AG_INITIAL_MARGIN:
        out += ["", "Margin limit: SKIPPED (no CME initial margins entered in config.AG_INITIAL_MARGIN)."]

    report = "\n".join(out) + "\n"
    (config.RESULTS_DIR / ("ENSO_REPORT.md" if not args.final else "ENSO_REPORT_FINAL.md")).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
