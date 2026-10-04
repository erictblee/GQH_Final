"""Run the cross-asset strategy: four models, plus robustness tests.

    python run_xa.py            development + validation periods (the default)
    python run_xa.py --test     also the locked test period (2020 to 2024-09-30)
    python run_xa.py --final    also the final out-of-sample period. Run ONCE.

The four models (XA_SPEC.md) differ in exactly two switches:

                 ENSO on grains    covariance
    M1 trend          no               no     (inverse volatility)
    M2 trend+ENSO     yes              no
    M3 trend+cov      no               yes
    M4 everything     yes              yes

Writes into results/:
    xa_<name>_<label>.csv           one row per day
    xa_<name>_<label>_ledger.csv    the transaction ledger: one row per fill
    xa_summary_<label>.csv          the headline numbers
    XA_REPORT.md                    the same, as tables to read
"""

import argparse
from dataclasses import replace

import pandas as pd

import config
from run_enso import markdown_table, sharpe_difference
from src import ag_backtest, ag_metrics, enso, market, xa_alpha
from src.xa_alpha import XaSpec

MODELS = [
    XaSpec(name="M1_trend", use_enso=False, use_cov=False),
    XaSpec(name="M2_trend_enso", use_enso=True, use_cov=False),
    XaSpec(name="M3_trend_cov", use_enso=False, use_cov=True),
    XaSpec(name="M4_trend_enso_cov", use_enso=True, use_cov=True),
]
MAIN = MODELS[-1]

# Robustness: Model 4 with ONE setting changed. Reported, never selected from.
ROBUSTNESS = (
    [replace(MAIN, name=f"rob_trend_{days}d", mom_lookback=days) for days in config.XA_ROBUST_LOOKBACKS]
    + [replace(MAIN, name="rob_cost_4bps", cost_bps=config.STRESS_COST_BPS),
       replace(MAIN, name="rob_fractional", integer_contracts=False)]
)

COMPARISONS = [
    ("Does ENSO add information? Without covariance (M2 - M1)", "M2_trend_enso", "M1_trend"),
    ("Does ENSO add information? With covariance (M4 - M3)", "M4_trend_enso_cov", "M3_trend_cov"),
    ("Does covariance improve construction? Without ENSO (M3 - M1)", "M3_trend_cov", "M1_trend"),
    ("Does covariance improve construction? With ENSO (M4 - M2)", "M4_trend_enso_cov", "M2_trend_enso"),
    ("Does the complete architecture add value? (M4 - M1)", "M4_trend_enso_cov", "M1_trend"),
]

PERCENT = {"cagr", "ann_vol", "max_drawdown", "worst_month", "worst_year", "positive_months",
           "gross_return", "net_return", "time_invested", "risk_limit_share", "ann_return", "win_rate"}


def group_table(daily, first, last):
    """What each asset group added: yearly return and Sharpe ratio."""
    part = daily.loc[first:last]
    rows = []
    for group, assets in config.XA_GROUPS.items():
        stream = part[[f"leg_{a}" for a in assets]].sum(axis=1)
        mean = stream.mean() * config.TRADING_DAYS
        vol = stream.std(ddof=1) * config.TRADING_DAYS ** 0.5
        rows.append({"group": group, "ann_return": float(mean),
                     "sharpe": float(mean / vol) if vol > 0 else float("nan")})
    return pd.DataFrame(rows)


def grain_enso_table(results, first, last):
    """The grains' combined contribution on the days of each ENSO state, for
    the model without ENSO (M3) and the model with it (M4)."""
    rows = []
    for state in ["el_nino", "neutral", "la_nina"]:
        row = {"state": state}
        for name in ["M3_trend_cov", "M4_trend_enso_cov"]:
            part = results[name].loc[first:last]
            days = part[part["enso_state"] == state]
            grains = days[[f"leg_{a}" for a in config.AG_ASSETS]].sum(axis=1)
            row["days"] = len(days)
            row[f"grain_return_{name[:2]}"] = float(grains.mean() * config.TRADING_DAYS) if len(days) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def episode_summary(episodes):
    """Wins and losses, counted per position episode (see ag_metrics)."""
    if len(episodes) == 0:
        return pd.DataFrame()
    rows = []
    for direction, part in [("long", episodes[episodes["direction"] == "long"]),
                            ("short", episodes[episodes["direction"] == "short"]),
                            ("all", episodes)]:
        rows.append({"direction": direction, "episodes": len(part),
                     "wins": int((part["pnl"] > 0).sum()), "losses": int((part["pnl"] <= 0).sum()),
                     "win_rate": float((part["pnl"] > 0).mean()) if len(part) else float("nan"),
                     "total_pnl_dollars": float(part["pnl"].sum()),
                     "avg_days": float(part["days"].mean()) if len(part) else float("nan")})
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
        "development": (None, pd.Timestamp(config.XA_DEV_END)),
        "validation": (pd.Timestamp(config.XA_DEV_END) + day_after, pd.Timestamp(config.XA_VALIDATION_END)),
    }
    if args.test or args.final:
        periods["TEST"] = (pd.Timestamp(config.XA_TEST_START), pd.Timestamp(config.IS_END))
    if args.final:
        periods["FINAL_OUT_OF_SAMPLE"] = (pd.Timestamp(config.OOS_START), None)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    results, ledgers = {}, {}
    for spec in MODELS + ROBUSTNESS:
        ledger = []
        daily = ag_backtest.run(m, enso_table, spec, decide=xa_alpha.decide, ledger=ledger)
        daily["source"] = m.source
        daily.to_csv(config.RESULTS_DIR / f"xa_{spec.name}_{label}.csv")
        pd.DataFrame(ledger).to_csv(config.RESULTS_DIR / f"xa_{spec.name}_{label}_ledger.csv", index=False)
        results[spec.name], ledgers[spec.name] = daily, pd.DataFrame(ledger)

    rows = []
    for period, (first, last) in periods.items():
        for name, daily in results.items():
            rows.append({"period": period, "run": name, **ag_metrics.summary(daily, first, last)})
    summary = pd.DataFrame(rows)
    summary.insert(0, "source", m.source)
    summary.to_csv(config.RESULTS_DIR / f"xa_summary_{label}.csv", index=False)

    main_daily = results[MAIN.name]
    locked = ("" if args.final else
              " The final out-of-sample period has NOT been loaded." if args.test else
              " The test period (2020 onward) and the final out-of-sample period have NOT been loaded.")
    out = [
        f"# Cross-asset signal-weighted risk budgeting: results ({label})",
        "",
        f"DATA SOURCE: {m.source}. {len(m.assets)} futures ({', '.join(m.assets)}), "
        f"{m.dates[0].date()} to {m.dates[-1].date()}; first decision {m.dates[m.start].date()}." + locked,
        "",
        "Every setting was fixed in XA_SPEC.md before the first run. Returns are excess returns "
        "net of costs; Sharpe ratios use a risk-free rate of zero; brackets are 95% bootstrap intervals.",
    ]
    shown = ["run", "cagr", "ann_vol", "sharpe", "sharpe_ci_low", "sharpe_ci_high", "sortino",
             "max_drawdown", "calmar", "worst_month", "worst_year", "positive_months"]
    shown_2 = ["run", "gross_return", "net_return", "costs_dollars", "roll_costs_dollars",
               "turnover_per_year", "time_invested", "avg_leverage", "max_leverage", "trades",
               "rolls", "risk_limit_share"]
    for period, (first, last) in periods.items():
        part = summary[summary["period"] == period]
        out += ["", f"## {period.replace('_', ' ').capitalize()}: {part['first_day'].iloc[0]} to {part['last_day'].iloc[0]}",
                "", "Risk and return:", "", markdown_table(part[shown], PERCENT),
                "", "Costs, trading and leverage (leverage = longs + shorts, as a multiple of equity):",
                "", markdown_table(part[shown_2], PERCENT)]
        comparison_rows = []
        for question, a, b in COMPARISONS:
            difference, low, high = sharpe_difference(results[a].loc[first:last], results[b].loc[first:last])
            comparison_rows.append({"question": question, "sharpe_difference": difference,
                                    "ci_low": low, "ci_high": high})
        out += ["", "What each layer adds (an interval that includes zero cannot be told apart from chance):",
                "", markdown_table(pd.DataFrame(comparison_rows)),
                "", f"{MAIN.name} by asset group:", "", markdown_table(group_table(main_daily, first, last), PERCENT),
                "", f"{MAIN.name} by asset:", "",
                markdown_table(ag_metrics.by_commodity(main_daily, m.assets, first, last), PERCENT),
                "", "Grains' contribution (yearly return added to the portfolio) by ENSO state, without ENSO (M3) and with it (M4):",
                "", markdown_table(grain_enso_table(results, first, last), PERCENT | {"grain_return_M3", "grain_return_M4"})]

    # Wins and losses from the ledger-level view, over the whole run.
    episodes = ag_metrics.position_episodes(main_daily, m.assets)
    episodes.to_csv(config.RESULTS_DIR / f"xa_{MAIN.name}_{label}_episodes.csv", index=False)
    ledger = ledgers[MAIN.name]
    out += ["", f"## Trades of {MAIN.name} (whole run)", "",
            f"The transaction ledger has {len(ledger)} fills: "
            + ", ".join(f"{count} {kind}" for kind, count in ledger["kind"].value_counts().items())
            + f". Total cost ${ledger['cost'].sum():,.0f}.",
            "", "A position episode is one unbroken long (or short) stretch in one asset; its P&L is net of costs.",
            "", markdown_table(episode_summary(episodes), PERCENT)]

    ret = main_daily["ret"].iloc[1:]
    out += ["", "## Concentration", "",
            f"{MAIN.name}, whole run: the sum of all daily returns is {ret.sum():.2%}; the ten best days "
            f"add up to {ret.nlargest(10).sum():.2%} and the ten worst to {ret.nsmallest(10).sum():.2%}. "
            f"Position limit active on {main_daily['position_limit_active'].mean():.1%} of days, "
            f"gross cap on {main_daily['cap_binding'].mean():.1%}."]
    if not config.AG_INITIAL_MARGIN:
        out += ["", "Margin limit: SKIPPED (no CME initial margins entered)."]

    report = "\n".join(out) + "\n"
    (config.RESULTS_DIR / f"XA_REPORT{'' if label == 'dev_val' else '_' + label.upper()}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
