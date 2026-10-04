"""Run the A/B/C/D attribution experiment (ABCD_PLAN.md).

    python run_abcd.py            development + validation periods (the default)
    python run_abcd.py --test     also the locked test period (2020 to 2024-09-30)
    python run_abcd.py --final    also the final out-of-sample period. Run ONCE.

The four models, each adding ONE thing to the one before:

    A trend                 sign of the 12-month trend, equal notional
    B trend + inv vol       the same signal, sized by 1 / volatility
    C trend + covariance    B's positions, scaled with the full covariance
    D trend + ENSO + cov    C, with the ENSO overlay on the three grains

Plus the attribution runs (grains only, long-only), the grains-only ENSO
test, the ENSO overlay as a multiplier, and the robustness tests of D.
run_xa.py (the earlier M1 to M4) is unchanged.

Writes into results/:
    abcd_<name>_<label>.csv           one row per day (the four models, D_mult_overlay)
    abcd_<name>_<label>_ledger.csv    the transaction ledger of the same runs
    abcd_summary_<label>.csv          the headline numbers of every run and period
    abcd_by_contract_<label>.csv      every model's result market by market
    ABCD_REPORT.md                    the same, as tables to read
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

import config
from run_enso import markdown_table, sharpe_difference
from src import abcd_alpha, ag_backtest, ag_metrics, enso, market
from src.abcd_alpha import AbcdSpec

GRAINS = tuple(config.AG_ASSETS)

MODELS = [
    AbcdSpec(name="A_trend", sizing="equal", enso="none"),
    AbcdSpec(name="B_trend_invvol", sizing="invvol", enso="none"),
    AbcdSpec(name="C_trend_cov", sizing="cov", enso="none"),
    AbcdSpec(name="D_trend_enso_cov", sizing="cov", enso="add"),
]
A, B, C, D = MODELS
TITLES = {"A_trend": "A Trend", "B_trend_invvol": "B Trend + InvVol",
          "C_trend_cov": "C Trend + Covariance", "D_trend_enso_cov": "D Trend + ENSO + Covariance"}

# The ENSO overlay in the plan's preferred form (a multiplier on the trend).
OVERLAY = [replace(D, name="D_mult_overlay", enso="mult")]

# Attribution (plan section 13): the same models on the grains alone, and
# without the right to go short.
ATTRIBUTION = [
    replace(A, name="A_grains", universe=GRAINS),
    replace(A, name="A_long_only", long_only=True),
    replace(A, name="A_grains_long_only", universe=GRAINS, long_only=True),
    replace(C, name="C_long_only", long_only=True),
]

# The ENSO-specific test (plan section 14): grains only, Model C's sizing.
ENSO_TEST = [
    replace(C, name="grains_trend", universe=GRAINS),
    replace(C, name="grains_enso_only", universe=GRAINS, enso="only"),
    replace(D, name="grains_trend_enso", universe=GRAINS),
]

# Robustness: Model D with ONE setting changed. Reported, never selected from.
ROBUSTNESS = (
    [replace(D, name=f"rob_trend_{days}d", mom_lookback=days) for days in config.ABCD_ROBUST_LOOKBACKS]
    + [replace(D, name=f"rob_half_life_{days}d", half_life=days) for days in config.ABCD_ROBUST_HALF_LIVES]
    + [replace(D, name=f"rob_vol_target_{target:.0%}", vol_target=target) for target in config.ABCD_ROBUST_VOL_TARGETS]
    + [replace(D, name=f"rob_every_{weeks}_weeks", rebalance_weeks=weeks) for weeks in config.ABCD_ROBUST_REBALANCE_WEEKS]
    + [replace(D, name="rob_cost_4bps", cost_bps=config.STRESS_COST_BPS)]
)

GROUPS = [
    ("The four models", MODELS),
    ("ENSO overlay as a multiplier", OVERLAY),
    ("Attribution runs: grains only, long-only", ATTRIBUTION),
    ("ENSO test on the grains alone", ENSO_TEST),
    ("Robustness of Model D (one setting changed)", ROBUSTNESS),
]
SAVED_IN_FULL = {spec.name for spec in MODELS + OVERLAY}

COMPARISONS = [
    ("Does inverse volatility help? (B - A)", "B_trend_invvol", "A_trend"),
    ("Does covariance help? (C - B)", "C_trend_cov", "B_trend_invvol"),
    ("Does ENSO help? (D - C)", "D_trend_enso_cov", "C_trend_cov"),
    ("Does ENSO help as a multiplier? (D_mult - C)", "D_mult_overlay", "C_trend_cov"),
    ("Does diversification help? (A on 11 markets - A on grains)", "A_trend", "A_grains"),
    ("Does shorting help? Model A (long/short - long-only)", "A_trend", "A_long_only"),
    ("Does shorting help? Model A, grains (long/short - long-only)", "A_grains", "A_grains_long_only"),
    ("Does shorting help? Model C (long/short - long-only)", "C_trend_cov", "C_long_only"),
    ("Grains alone: does ENSO add to trend? (trend + ENSO - trend)", "grains_trend_enso", "grains_trend"),
]

# The pairs whose differences the plan asks for in full (section 14).
ENSO_DELTAS = [
    ("Grains alone: trend + ENSO minus trend", "grains_trend_enso", "grains_trend"),
    ("Grains alone: ENSO only minus trend", "grains_enso_only", "grains_trend"),
    ("All 11 markets: D minus C", "D_trend_enso_cov", "C_trend_cov"),
    ("All 11 markets: D_mult_overlay minus C", "D_mult_overlay", "C_trend_cov"),
]

PERCENT = {"cagr", "ann_vol", "max_drawdown", "worst_month", "worst_year", "positive_months",
           "gross_return", "net_return", "time_invested", "ann_return", "win_rate", "share_of_pnl",
           "mean", "median", "std", "worst", "pct_5", "pct_25", "pct_75", "pct_95", "best",
           "d_cagr", "d_max_drawdown"}
SHOWN = ["run", "cagr", "net_return", "ann_vol", "sharpe", "sharpe_ci_low", "sharpe_ci_high",
         "sortino", "max_drawdown", "calmar", "worst_month", "positive_months"]
SHOWN_2 = ["run", "total_pnl_dollars", "gross_return", "net_return", "costs_dollars",
           "roll_costs_dollars", "turnover_per_year", "avg_leverage", "max_leverage",
           "trades", "rolls", "episodes", "win_rate"]


def dollars(table, columns):
    """Show dollar columns as whole dollars with thousands separators."""
    shown = table.copy()
    for column in columns:
        shown[column] = [f"{v:,.0f}" for v in shown[column]]
    return shown


def grain_states_table(m, daily):
    """The measured relationship (plan section 4): what each grain's own
    futures return was, per year, on the days of each ENSO state. This is a
    plain long position, no strategy. `t` is the mean divided by its standard
    error, treating days as independent (they are not: read it loosely)."""
    rows = []
    for state in ["el_nino", "neutral", "la_nina"]:
        days = daily.index[daily["enso_state"] == state]
        row = {"state": state, "days": len(days)}
        for a in GRAINS:
            r = m.returns.loc[days, a]
            row[f"{a}_ann_return"] = float(r.mean() * config.TRADING_DAYS) if len(r) else float("nan")
            row[f"{a}_t"] = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def enso_episode_table(results):
    """One row per unbroken stretch of one ENSO state: what the three grains
    earned, in dollars, without ENSO (C), with it (D) and as a multiplier."""
    base = results["C_trend_cov"]
    stretch = (base["enso_state"] != base["enso_state"].shift(1)).cumsum()
    grain_columns = [f"pnl_{a}" for a in GRAINS]
    rows = []
    for _, days in base.groupby(stretch):
        row = {"state": days["enso_state"].iloc[0], "first_day": days.index[0].date(),
               "last_day": days.index[-1].date(), "days": len(days)}
        for label, name in [("C", "C_trend_cov"), ("D", "D_trend_enso_cov"), ("D_mult", "D_mult_overlay")]:
            row[f"grain_pnl_{label}"] = float(results[name].loc[days.index, grain_columns].to_numpy().sum())
        row["D_minus_C"] = row["grain_pnl_D"] - row["grain_pnl_C"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="include the locked test period")
    parser.add_argument("--final", action="store_true", help="include the final out-of-sample period")
    args = parser.parse_args()

    if args.final:
        cutoff, label = None, "full"
    elif args.test:
        cutoff, label = config.IS_END, "with_test"
    else:
        cutoff, label = config.XA_VALIDATION_END, "dev_val"

    m = market.load_market(cutoff, data_dir=config.XA_PROCESSED_DIR,
                           assets=config.XA_ASSETS, warmup=config.XA_WARMUP_DAYS)
    enso_table = enso.load_enso()
    if (args.test or args.final) and m.dates[-1] <= pd.Timestamp(config.XA_VALIDATION_END):
        raise SystemExit("the data on disk stops at the end of validation. Download and build "
                         "the later period first (data/download_xa.py and data/build_xa.py).")
    if (args.test or args.final) and (~m.is_open.loc[config.XA_TEST_START:]).all().any():
        raise SystemExit("an asset has no data after the validation period: run data/download_xa.py --test")

    day_after = pd.Timedelta(days=1)
    periods = {
        "whole_run": (None, None),
        "development": (None, pd.Timestamp(config.XA_DEV_END)),
        "validation": (pd.Timestamp(config.XA_DEV_END) + day_after, pd.Timestamp(config.XA_VALIDATION_END)),
    }
    if args.test or args.final:
        periods["TEST"] = (pd.Timestamp(config.XA_TEST_START), pd.Timestamp(config.IS_END))
    if args.final:
        periods["FINAL_OUT_OF_SAMPLE"] = (pd.Timestamp(config.OOS_START), None)
    # The "OOS Sharpe" column of the final table: the latest period that was run.
    oos_period = list(periods)[-1]

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for _, specs in GROUPS:
        for spec in specs:
            ledger = []
            daily = ag_backtest.run(m, enso_table, spec, decide=abcd_alpha.decide, ledger=ledger)
            daily["source"] = m.source
            if spec.name in SAVED_IN_FULL:
                daily.to_csv(config.RESULTS_DIR / f"abcd_{spec.name}_{label}.csv")
                pd.DataFrame(ledger).to_csv(config.RESULTS_DIR / f"abcd_{spec.name}_{label}_ledger.csv", index=False)
            results[spec.name] = daily

    rows = []
    for period, (first, last) in periods.items():
        for name, daily in results.items():
            rows.append({"period": period, "run": name, **ag_metrics.summary(daily, first, last),
                         "total_pnl_dollars": ag_metrics.total_pnl(daily, m.assets, first, last),
                         **ag_metrics.episode_counts(daily, m.assets, first, last)})
    summary = pd.DataFrame(rows)
    summary.insert(0, "source", m.source)
    summary.to_csv(config.RESULTS_DIR / f"abcd_summary_{label}.csv", index=False)
    numbers = summary.set_index(["period", "run"])

    locked = ("" if args.final else
              " The final out-of-sample period has NOT been loaded." if args.test else
              " The test period (2020 onward) and the final out-of-sample period have NOT been loaded.")
    out = [
        f"# Cross-asset trend: the A/B/C/D attribution experiment ({label})",
        "",
        f"DATA SOURCE: {m.source}. {len(m.assets)} futures ({', '.join(m.assets)}), "
        f"{m.dates[0].date()} to {m.dates[-1].date()}; first decision {m.dates[m.start].date()}." + locked,
        "",
        "The plan is ABCD_PLAN.md; every setting was fixed before the first run (README, "
        "\"Fourth iteration\"). Returns are excess returns net of costs; Sharpe ratios use a "
        "risk-free rate of zero; brackets are 95% bootstrap intervals. A \"trade\" in the "
        "episodes column is one unbroken long or short stretch in one market.",
        "",
        "## Final comparison table",
        "",
        f"Whole run, except the last two columns. \"OOS Sharpe\" is the {oos_period.replace('_', ' ').lower()} "
        "period" + (": the latest period run so far, NOT the untouched final period." if not args.final else "."),
        "",
    ]
    final_rows = []
    for spec in MODELS:
        whole = numbers.loc[("whole_run", spec.name)]
        final_rows.append({
            "Model": TITLES[spec.name], "CAGR": whole["cagr"], "Vol": whole["ann_vol"],
            "Sharpe": whole["sharpe"], "Max DD": whole["max_drawdown"], "Trades": int(whole["episodes"]),
            "Costs": f"${whole['costs_dollars']:,.0f}",
            "Development Sharpe": numbers.loc[("development", spec.name), "sharpe"],
            "OOS Sharpe": numbers.loc[(oos_period, spec.name), "sharpe"]})
    out += [markdown_table(pd.DataFrame(final_rows), {"CAGR", "Vol", "Max DD"})]

    for period, (first, last) in periods.items():
        part = summary[summary["period"] == period]
        out += ["", f"## {period.replace('_', ' ').capitalize()}: {part['first_day'].iloc[0]} to {part['last_day'].iloc[0]}"]
        for title, specs in GROUPS:
            group = part[part["run"].isin([spec.name for spec in specs])]
            out += ["", f"### {title}", "", "Risk and return:", "", markdown_table(group[SHOWN], PERCENT),
                    "", "P&L, costs and trading (leverage = longs + shorts, as a multiple of equity; "
                        "trades = fills, episodes = positions):",
                    "", markdown_table(dollars(group[SHOWN_2], ["total_pnl_dollars", "costs_dollars", "roll_costs_dollars"]), PERCENT)]
        comparison_rows = []
        for question, a, b in COMPARISONS:
            difference, low, high = sharpe_difference(results[a].loc[first:last], results[b].loc[first:last])
            comparison_rows.append({"question": question, "sharpe_difference": difference,
                                    "ci_low": low, "ci_high": high})
        out += ["", "### What each layer adds", "",
                "Difference in Sharpe ratio; an interval that includes zero cannot be told apart from chance.",
                "", markdown_table(pd.DataFrame(comparison_rows))]

        delta_rows = []
        for question, a, b in ENSO_DELTAS:
            with_, without = numbers.loc[(period, a)], numbers.loc[(period, b)]
            delta_rows.append({"comparison": question,
                               "d_sharpe": with_["sharpe"] - without["sharpe"],
                               "d_cagr": with_["cagr"] - without["cagr"],
                               "d_max_drawdown": with_["max_drawdown"] - without["max_drawdown"],
                               "d_pnl_dollars": with_["total_pnl_dollars"] - without["total_pnl_dollars"],
                               "d_turnover": with_["turnover_per_year"] - without["turnover_per_year"]})
        out += ["", "### What ENSO changes", "",
                "Each number is the run with ENSO minus the run without (a positive d_max_drawdown is a DEEPER drawdown).",
                "", markdown_table(dollars(pd.DataFrame(delta_rows), ["d_pnl_dollars"]), PERCENT)]

    # --- ENSO, cycle by cycle ---
    c_daily = results["C_trend_cov"]
    states = grain_states_table(m, c_daily)
    episodes = enso_episode_table(results)
    out += ["", "## ENSO in detail (whole run)", "",
            "The measured relationship: each grain's own futures return (a plain long position, per year) "
            "on the days of each ENSO state. The overlay assumes La Nina is bullish for all three and "
            "El Nino bearish for soybeans (config.ENSO_EFFECT, from crop-yield studies, not from these prices).",
            "", markdown_table(states, {f"{a}_ann_return" for a in GRAINS}),
            "", "The grains' P&L in dollars during each unbroken stretch of one ENSO state, "
                "without ENSO (C), with it (D) and with the multiplier form:",
            "", markdown_table(dollars(episodes, ["grain_pnl_C", "grain_pnl_D", "grain_pnl_D_mult", "D_minus_C"])),
            "", f"Stretches in which D's grains beat C's: {(episodes['D_minus_C'] > 0).sum()} of "
                f"{(episodes['D_minus_C'] != 0).sum()} in which ENSO changed anything "
                f"({len(episodes)} stretches in total; in a neutral stretch D and C differ only through the "
                "portfolio-wide scaling and rounding)."]

    # --- Section 12: yearly P&L, monthly distribution, market by market ---
    yearly = pd.DataFrame({spec.name: ag_metrics.annual_pnl(results[spec.name], m.assets) for spec in MODELS})
    yearly = yearly.reset_index().rename(columns={"date": "year", "index": "year"})
    yearly["year"] = yearly["year"].astype(str)
    monthly = pd.DataFrame([{"run": spec.name, **ag_metrics.monthly_distribution(results[spec.name].iloc[1:])}
                            for spec in MODELS])
    out += ["", "## Annual P&L in dollars (starting equity $10,000,000)", "",
            markdown_table(dollars(yearly, [spec.name for spec in MODELS])),
            "", "## Monthly return distribution (whole run)", "", markdown_table(monthly, PERCENT)]

    contract_tables = []
    out += ["", "## Market by market (whole run)", "",
            "Returns, volatility and drawdown are of the stream each market added to the portfolio, "
            "as a fraction of equity. Trades are position episodes."]
    for spec in MODELS:
        table = ag_metrics.by_contract(results[spec.name], m.assets)
        contract_tables.append(table.assign(run=spec.name))
        out += ["", f"### {TITLES[spec.name]}", "",
                markdown_table(dollars(table, ["pnl_dollars", "long_pnl_dollars", "short_pnl_dollars"]), PERCENT)]
    for period, (first, last) in periods.items():
        if period != "whole_run":
            contract_tables += [ag_metrics.by_contract(results[spec.name], m.assets, first, last)
                                .assign(run=spec.name, period=period) for spec in MODELS]
    by_contract = pd.concat(contract_tables).fillna({"period": "whole_run"})
    by_contract.to_csv(config.RESULTS_DIR / f"abcd_by_contract_{label}.csv", index=False)

    # --- Concentration and limits ---
    out += ["", "## Concentration and limits (whole run)", ""]
    for spec in MODELS:
        daily = results[spec.name]
        ret = daily["ret"].iloc[1:]
        out += [f"- {TITLES[spec.name]}: sum of daily returns {ret.sum():.2%}; ten best days {ret.nlargest(10).sum():.2%}, "
                f"ten worst {ret.nsmallest(10).sum():.2%}. Position limit active on "
                f"{daily['position_limit_active'].mean():.1%} of days, gross cap on {daily['cap_binding'].mean():.1%}. "
                f"Realized volatility by year: "
                + ", ".join(f"{year} {vol:.1%}" for year, vol in
                            (ret.groupby(ret.index.year).std(ddof=1) * np.sqrt(config.TRADING_DAYS)).items()) + "."]
    if not config.AG_INITIAL_MARGIN:
        out += ["", "Margin limit: SKIPPED (no CME initial margins entered)."]

    report = "\n".join(out) + "\n"
    (config.RESULTS_DIR / f"ABCD_REPORT{'' if label == 'dev_val' else '_' + label.upper()}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
