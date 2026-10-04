"""Write the test report.

    python tools/make_report.py            in-sample report  -> results/REPORT.md
    python tools/make_report.py --final    in-sample AND out-of-sample
                                           -> results/REPORT_full.md
                                           (final run only; see the checklist
                                           at the end of the in-sample report)

Run run_all.py, the backtrader runs and tools/compare_engines.py first (the
exact commands are in README.md).

How the file is organised: `Period` holds everything about one sample period
(which days, which results). Each `prediction_N` function returns a list of
claims, and each claim says whether it holds in that period. The same
functions are used for the runs with the drawdown brake and for their
no-brake twins, by passing a different table of names (`roles`).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script find config.py, run_all.py and src/ when run from anywhere.
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "tools"))
import check_roll_dates
import config
import run_all
from src import analysis, diagnostics, market, metrics

# Which run plays which part in the predictions: with the brake (the
# pre-registered runs) and without it (brake-noise diagnostics).
WITH_BRAKE = {"A": "A", "B": "B", "no_gold": "B_no_gold", "demeaned": "diag_B_demeaned_gold",
              "w40": "B_window_40", "w90": "B_window_90", "w120": "B_window_120"}
NO_BRAKE = {"A": "diag_A_no_brake", "B": "diag_B_no_brake", "no_gold": "diag_B_no_gold_no_brake",
            "demeaned": "diag_B_demeaned_gold_no_brake", "w40": "diag_B_window_40_no_brake",
            "w90": "diag_B_window_90_no_brake", "w120": "diag_B_window_120_no_brake"}
BENCHMARKS = ["bench_equal_weight", "bench_60_40"]


def pct(x):
    return f"{x:.2%}"


def verdict(holds):
    return "**holds**" if holds else "**does not hold**"


class Period:
    """One sample period: the market, every run's results, and the first and
    last day to evaluate."""

    def __init__(self, m, results, first, last, label):
        self.m, self.results, self.label = m, results, label
        self.first, self.last = pd.Timestamp(first), pd.Timestamp(last)
        self.positive = analysis.positive_correlation_days(m)

    def daily(self, name):
        """A run's daily table, cut to this period."""
        return self.results[name].loc[self.first:self.last]

    def ret(self, name):
        """A run's daily returns in this period (without the run's very
        first row, which is the flat start date)."""
        full = self.results[name]["ret"]
        part = full.loc[self.first:self.last]
        return part.iloc[1:] if part.index[0] == full.index[0] else part

    def summary(self, name):
        return metrics.summary(self.results[name], self.first, self.last)


def sharpe_claim(period, text, name_a, name_b):
    """Claim: run a has a higher Sharpe ratio than run b. Comes with the
    bootstrap 95% confidence interval of the difference."""
    a, b = period.ret(name_a), period.ret(name_b)
    difference, low, high = analysis.bootstrap_sharpe_difference(a, b)
    detail = (f"{analysis.sharpe(a):.2f} against {analysis.sharpe(b):.2f}; difference {difference:+.2f}, "
              f"95% interval [{low:+.2f}, {high:+.2f}]"
              + ("" if low > 0 or high < 0 else ", which includes zero"))
    return {"text": text, "holds": difference > 0, "detail": detail}


# ---------------------------------------------------------------------------
# The predictions. Each returns a list of claims.
# ---------------------------------------------------------------------------

def prediction_1(period, roles):
    """B versus the benchmarks: higher Sharpe, smaller volatility-matched drawdown."""
    b = period.ret(roles["B"])
    b_vol = analysis.volatility(b)
    b_drawdown = analysis.chained_drawdown(b)
    claims = []
    for bench in BENCHMARKS:
        claims.append(sharpe_claim(period, f"B has a higher Sharpe ratio than {bench}", roles["B"], bench))
        matched = analysis.volatility_matched_drawdown(period.ret(bench), b_vol)
        claims.append({"text": f"B has a smaller volatility-matched drawdown than {bench}",
                       "holds": b_drawdown < matched,
                       "detail": f"{pct(b_drawdown)} against {pct(matched)}"})
    return claims


def prediction_2(period):
    """Volatility management: regress B-no-brake on the constant-leverage
    portfolio; the intercept should be positive."""
    y = period.ret("diag_B_no_brake")
    x = period.ret("diag_constant_leverage").rename("constant_leverage").to_frame()
    table = analysis.newey_west_regression(y, x)
    intercept = table.loc["intercept", "coefficient"] * config.TRADING_DAYS
    leverage = period.results["diag_constant_leverage"].attrs.get("leverage", float("nan"))
    detail = (f"intercept {pct(intercept)} per year, Newey-West t-statistic {table.loc['intercept', 't_stat']:.2f} "
              f"({table.attrs['lags']} lags, {table.attrs['n']} days); slope {table.loc['constant_leverage', 'coefficient']:.2f}; "
              f"constant-leverage multiplier {leverage:.2f}x"
              + (" (at the 4x cap)" if leverage >= config.GROSS_CAP else "")
              + f"; volatility {pct(analysis.volatility(x['constant_leverage']))} against {pct(analysis.volatility(y))}")
    t_stat = table.loc["intercept", "t_stat"]
    if abs(t_stat) < 1.96:
        detail += "; the t-statistic is below 1.96, so the intercept is not statistically distinguishable from zero"
    return [{"text": "The intercept of B-no-brake on the constant-leverage portfolio is positive",
             "holds": intercept > 0, "detail": detail}]


def prediction_3(period):
    """Low-risk premium: the bond leg's Sharpe is at least the stock leg's."""
    legs = period.daily("B")
    zn, es = legs["leg_ZN"], legs["leg_ES"]
    difference, low, high = analysis.bootstrap_sharpe_difference(zn, es)
    detail = (f"ZN {analysis.sharpe(zn):.2f}, ES {analysis.sharpe(es):.2f}, GC {analysis.sharpe(legs['leg_GC']):.2f}; "
              f"ZN minus ES {difference:+.2f}, 95% interval [{low:+.2f}, {high:+.2f}]")
    return [{"text": "Within B, the bond leg's Sharpe ratio is at least the stock leg's",
             "holds": difference >= 0, "detail": detail}]


def prediction_4(period, roles):
    """Gold as a diversifier."""
    with_gold, no_gold, demeaned = roles["B"], roles["no_gold"], roles["demeaned"]
    b_vol = analysis.volatility(period.ret(with_gold))
    claims = [sharpe_claim(period, "B with gold has a higher Sharpe ratio than B without gold", with_gold, no_gold)]

    gold_dd = analysis.chained_drawdown(period.ret(with_gold))
    no_gold_dd = analysis.volatility_matched_drawdown(period.ret(no_gold), b_vol)
    claims.append({"text": "B with gold has a smaller volatility-matched drawdown than B without gold",
                   "holds": gold_dd < no_gold_dd, "detail": f"{pct(gold_dd)} against {pct(no_gold_dd)}"})

    # Is the improvement concentrated in positive-correlation periods?
    table = analysis.regime_table({k: period.daily(k) for k in (with_gold, no_gold)}, period.positive)
    gain_positive = table.loc[with_gold, "sharpe_positive"] - table.loc[no_gold, "sharpe_positive"]
    gain_other = table.loc[with_gold, "sharpe_other"] - table.loc[no_gold, "sharpe_other"]
    claims.append({"text": "The improvement from gold is concentrated in positive-correlation periods",
                   "holds": gain_positive > 0 and gain_positive > gain_other,
                   "detail": f"Sharpe change from adding gold: {gain_positive:+.2f} on positive-correlation days, "
                             f"{gain_other:+.2f} on other days"})

    # The demeaned-gold diagnostic must show the same thing.
    claim = sharpe_claim(period, "B with DEMEANED gold still has a higher Sharpe ratio than B without gold "
                                 "(diagnostic)", demeaned, no_gold)
    claims.append(claim)
    demeaned_dd = analysis.volatility_matched_drawdown(period.ret(demeaned), b_vol)
    claims.append({"text": "B with DEMEANED gold still has a smaller volatility-matched drawdown than B "
                           "without gold (diagnostic)",
                   "holds": demeaned_dd < no_gold_dd, "detail": f"{pct(demeaned_dd)} against {pct(no_gold_dd)}"})
    return claims


def prediction_5(period, roles):
    """Correlation awareness: A versus B on positive-correlation days."""
    a, b = roles["A"], roles["B"]
    table = analysis.regime_table({k: period.daily(k) for k in (a, b)}, period.positive)
    a_vol, b_vol = table.loc[a, "vol_positive_rescaled"], table.loc[b, "vol_positive_rescaled"]
    share = pct(table.loc[b, "share_positive_days"])
    return [
        {"text": "On positive-correlation days B has a smaller drawdown than A",
         "holds": table.loc[b, "drawdown_positive"] < table.loc[a, "drawdown_positive"],
         "detail": f"{pct(table.loc[b, 'drawdown_positive'])} against {pct(table.loc[a, 'drawdown_positive'])} "
                   f"(chained; positive-correlation days are {share} of the period)"},
        {"text": "On those days B's realized volatility is closer to the 10% target than A's",
         "holds": abs(b_vol - config.VOL_TARGET) < abs(a_vol - config.VOL_TARGET),
         "detail": f"B {pct(b_vol)}, A {pct(a_vol)} (returns rescaled to the 10% target)"},
        {"text": "On those days A's realized volatility exceeds its target",
         "holds": a_vol > config.VOL_TARGET, "detail": f"A {pct(a_vol)} against a target of {pct(config.VOL_TARGET)}"},
    ]


def prediction_6(period, roles):
    """Plateau: B is similar with 40-, 90- and 120-day windows."""
    details, all_include_zero = [], True
    for role, window in [("w40", 40), ("w90", 90), ("w120", 120)]:
        difference, low, high = analysis.bootstrap_sharpe_difference(period.ret(roles[role]), period.ret(roles["B"]))
        all_include_zero = all_include_zero and (low <= 0 <= high)
        details.append(f"{window}-day {analysis.sharpe(period.ret(roles[role])):.2f} "
                       f"(minus 60-day: {difference:+.2f}, 95% interval [{low:+.2f}, {high:+.2f}])")
    return [{"text": "B's Sharpe ratio is similar with 40-, 90- and 120-day windows (read here as: no window's "
                     "Sharpe differs from the 60-day window's by more than its 95% interval)",
             "holds": all_include_zero,
             "detail": f"60-day {analysis.sharpe(period.ret(roles['B'])):.2f}; " + "; ".join(details)}]


# ---------------------------------------------------------------------------
# Factor regressions
# ---------------------------------------------------------------------------

def factor_regressions(period):
    """(i) weekly, on ES, ZN and GC returns. (ii) monthly, on Kenneth French
    factors. Returns two tables (the second is None without the data)."""
    b = period.ret("B")
    assets = period.m.returns.loc[b.index[0]:b.index[-1]]
    weekly = analysis.newey_west_regression(analysis.compound(b, "W-FRI"), analysis.compound(assets, "W-FRI"))

    factors = diagnostics.load_french_factors(period.last.to_period("M"))
    monthly = None
    if factors is not None:
        b_monthly = analysis.compound(b, "ME")
        b_monthly.index = b_monthly.index.to_period("M")
        monthly = analysis.newey_west_regression(b_monthly, factors)
    return weekly, monthly


def write_regression(add, table, periods_per_year):
    add("| Term | Coefficient | Newey-West t-statistic |")
    add("|---|---:|---:|")
    for term, row in table.iterrows():
        if term == "intercept":
            add(f"| intercept (alpha), per year | {pct(row['coefficient'] * periods_per_year)} | {row['t_stat']:.2f} |")
        else:
            add(f"| {term} | {row['coefficient']:.3f} | {row['t_stat']:.2f} |")
    add("")
    add(f"{table.attrs['n']} observations, {table.attrs['lags']} Newey-West lags, R-squared {table.attrs['r_squared']:.2f}.")
    add("")


# ---------------------------------------------------------------------------
# One sample period
# ---------------------------------------------------------------------------

def write_claims(add, claims):
    for claim in claims:
        add(f"- {claim['text']}: {verdict(claim['holds'])}. {claim['detail']}.")
    add("")


def write_period(add, period, fred, contract_volume):
    m, results = period.m, period.results
    n_days = len(period.ret("B"))
    add(f"# {period.label}: {period.first.date()} to {period.last.date()} ({n_days} trading days)")
    add("")

    # ------------------------------------------------------------- results
    add("## Results table")
    add("")
    add("The first eight rows are the six pre-registered variants and the two cost stress tests. "
        "`bench_*` rows are benchmarks and `diag_*` rows are diagnostics; neither counts as a variant.")
    add("")
    add("| Run | Ann. return | Ann. vol | Sharpe | Max drawdown | Calmar | Trades | Cap binds | Brake on |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in results:
        row = period.summary(name)
        add(f"| {name} | {pct(row['ann_return'])} | {pct(row['ann_vol'])} | {row['sharpe']:.2f} | "
            f"{pct(row['max_drawdown'])} | {row['calmar']:.2f} | {row['trades']} | "
            f"{pct(row['cap_share'])} | {pct(row['brake_share'])} |")
    add("")
    low, high = analysis.bootstrap_sharpe(period.ret("B"))
    add(f"B's Sharpe ratio is {analysis.sharpe(period.ret('B')):.2f}, with a bootstrap 95% interval of "
        f"[{low:.2f}, {high:.2f}] (stationary block bootstrap, {config.BOOTSTRAP_SAMPLES} samples, "
        f"mean block {config.BOOTSTRAP_MEAN_BLOCK} days, seed {config.BOOTSTRAP_SEED}).")
    add("")

    # --------------------------------------------------------- predictions
    add("## Predictions")
    add("")
    add("A claim is marked *holds* or *does not hold* from its point estimate. Where a claim compares two "
        "Sharpe ratios, the bootstrap 95% interval of the difference is given: if it includes zero, the "
        "data cannot tell the two apart.")
    add("")
    blocks = [
        ("1. Versus benchmarks", prediction_1(period, WITH_BRAKE)),
        ("2. Volatility management adds value", prediction_2(period)),
        ("3. Low-risk premium", prediction_3(period)),
        ("4. Gold as diversifier", prediction_4(period, WITH_BRAKE)),
        ("5. Correlation awareness", prediction_5(period, WITH_BRAKE)),
        ("6. Plateau", prediction_6(period, WITH_BRAKE)),
    ]
    for title, claims in blocks:
        add(f"### Prediction {title}")
        add("")
        write_claims(add, claims)
        whole = all(claim["holds"] for claim in claims)
        add(f"Prediction {title.split('.')[0]} as a whole: {verdict(whole)} in this period.")
        add("")

    add("The Sharpe difference between B and A (not a prediction on its own, but listed in Amendment 1): "
        + sharpe_claim(period, "", "B", "A")["detail"] + ".")
    add("")

    # ------------------------------------------------- factor regressions
    add("## Factor regressions")
    add("")
    weekly, monthly = factor_regressions(period)
    add("**(i) B's weekly returns on ES, ZN and GC weekly returns.** The intercept is the value of dynamic "
        "sizing beyond static exposure to the three assets.")
    add("")
    write_regression(add, weekly, 52)
    if monthly is None:
        add("**(ii)** Kenneth French factors not downloaded (run `python data/download_external.py`).")
        add("")
    else:
        add("**(ii) B's monthly returns on the Kenneth French market, size, value and momentum factors.**")
        add("")
        write_regression(add, monthly, 12)

    # ----------------------------------------------------- failure criteria
    add("## \"What would make it fail\" (HYPOTHESIS.md)")
    add("")
    add("The criteria are written for the out-of-sample period; this is how they read in this period.")
    add("")
    b_sharpe = analysis.sharpe(period.ret("B"))
    sixty_forty = analysis.sharpe(period.ret("bench_60_40"))
    add(f"- Sharpe at or below zero, or no better than 60/40 after costs: B {b_sharpe:.2f}, 60/40 {sixty_forty:.2f}. "
        + ("This failure condition is met." if b_sharpe <= 0 or b_sharpe <= sixty_forty else "Not met."))
    # Profits from one regime: split at the end of 2020 if the period spans it.
    split = pd.Timestamp("2020-12-31")
    if period.first < split < period.last:
        add("- Profits almost entirely from one regime. Average yearly contribution to B's return (net of costs):")
        add("")
        add("  | Span | B | ES leg | ZN leg | GC leg |")
        add("  |---|---:|---:|---:|---:|")
        for label, lo, hi in [("through 2020", period.first, split), ("2021 onward", split + pd.Timedelta(days=1), period.last)]:
            part = results["B"].loc[lo:hi]
            legs = " | ".join(pct(part[f"leg_{a}"].mean() * config.TRADING_DAYS) for a in m.assets)
            add(f"  | {label} | {pct(part['ret'].mean() * config.TRADING_DAYS)} | {legs} |")
        add("")
    add(f"- Regression (ii) explains the returns fully, or regression (i) shows no value from dynamic sizing: "
        f"regression (i) intercept {pct(weekly.loc['intercept', 'coefficient'] * 52)} per year "
        f"(t = {weekly.loc['intercept', 't_stat']:.2f})"
        + ("" if monthly is None else f"; regression (ii) intercept {pct(monthly.loc['intercept', 'coefficient'] * 12)} "
                                      f"per year (t = {monthly.loc['intercept', 't_stat']:.2f})") + ".")
    no_gold, demeaned = analysis.sharpe(period.ret("B_no_gold")), analysis.sharpe(period.ret("diag_B_demeaned_gold"))
    add(f"- B without gold matches or beats B with demeaned gold: without gold {no_gold:.2f}, demeaned gold "
        f"{demeaned:.2f}. " + ("This failure condition is met." if no_gold >= demeaned else "Not met."))
    windows = {w: analysis.sharpe(period.ret(f"B_window_{w}")) for w in (40, 90, 120)}
    add(f"- Only the 60-day window works: Sharpe is {b_sharpe:.2f} at 60 days and "
        + ", ".join(f"{v:.2f} at {w}" for w, v in windows.items()) + ". "
        + ("This failure condition is met." if b_sharpe > 0 and all(v <= 0 for v in windows.values()) else "Not met."))
    add("")

    # ------------------------------------------------------- brake noise
    add("## Brake-noise diagnostics (reporting only; the brake rule is unchanged)")
    add("")
    add("Every run with a drawdown brake was run again with the brake switched off (`diag_*_no_brake` in "
        "the results table). The brake halves the volatility target when drawdown passes 15%, so a small "
        "change in equity can switch it on a different day and change the whole path that follows. This "
        "table shows which conclusions depend on that.")
    add("")
    add("| Claim | With the brake (pre-registered) | Without the brake (diagnostic) |")
    add("|---|---|---|")
    for function in (prediction_1, prediction_4, prediction_5, prediction_6):
        for with_brake, without in zip(function(period, WITH_BRAKE), function(period, NO_BRAKE)):
            add(f"| {with_brake['text']} | {verdict(with_brake['holds'])}. {with_brake['detail']} | "
                f"{verdict(without['holds'])}. {without['detail']} |")
    add("")
    changed = [w["text"] for f in (prediction_1, prediction_4, prediction_5, prediction_6)
               for w, n in zip(f(period, WITH_BRAKE), f(period, NO_BRAKE)) if w["holds"] != n["holds"]]
    add("Claims whose verdict changes when the brake is removed: " + ("; ".join(changed) if changed else "none") + ".")
    add("")

    # ----------------------------------------------------- cost breakdown
    add("## Direct costs of B, by source")
    add("")
    b_daily = period.daily("B")
    years = len(b_daily) / config.TRADING_DAYS
    add("| Asset | Rolls, $ | Rebalances, $ | Rolls, % of equity per year | Rebalances, % of equity per year |")
    add("|---|---:|---:|---:|---:|")
    total_roll = total_trade = 0.0
    for asset in m.assets:
        roll, trade = b_daily[f"rollcost_{asset}"], b_daily[f"tradecost_{asset}"]
        total_roll += roll.sum()
        total_trade += trade.sum()
        add(f"| {asset} | {roll.sum():,.0f} | {trade.sum():,.0f} | "
            f"{(roll / b_daily['equity']).sum() / years:.3%} | {(trade / b_daily['equity']).sum() / years:.3%} |")
    add(f"| Total | {total_roll:,.0f} | {total_trade:,.0f} | "
        f"{(b_daily[[f'rollcost_{a}' for a in m.assets]].sum(axis=1) / b_daily['equity']).sum() / years:.3%} | "
        f"{(b_daily[[f'tradecost_{a}' for a in m.assets]].sum(axis=1) / b_daily['equity']).sum() / years:.3%} |")
    add("")
    zn_roll = b_daily["rollcost_ZN"].sum()
    largest = max(((b_daily[f"{kind}cost_{a}"].sum(), f"{a} {label}") for a in m.assets
                   for kind, label in (("roll", "rolls"), ("trade", "rebalances"))))
    add(f"ZN rolls are {zn_roll / (total_roll + total_trade):.0%} of all direct costs; "
        f"the largest single source is {largest[1]}.")
    add("")

    # ------------------------------------------------ Amendment 1 diagnostics
    add("## Amendment 1 diagnostics")
    add("")

    add("### 1. Cost justification")
    add("")
    table = diagnostics.cost_justification(m, period.first, period.last)
    add("| Asset | Average price | Half a tick, $ | Fees, $ | Estimated cost per side, bps | Charged, bps |")
    add("|---|---:|---:|---:|---:|---:|")
    for asset, row in table.iterrows():
        add(f"| {asset} | {row['average_price']:,.2f} | {row['half_tick_dollars']:.2f} | {row['fee_dollars']:.2f} | "
            f"{row['estimated_cost_bps']:.2f} | {row['charged_bps']:.1f} |")
    add("")
    add("Estimated cost = half a tick plus an assumed $2.50 per contract, as basis points of one contract's "
        "notional at the period's average price.")
    add("")

    add("### 2. Capacity")
    add("")
    table, capacity, coverage = diagnostics.capacity(m, results["B"], contract_volume, period.first, period.last)
    add(f"Databento's cleared-volume records start in late 2015, so this test covers "
        f"{coverage['first'].date()} to {coverage['last'].date()} only. B's Sharpe ratio over those days, "
        f"before impact, is {coverage['base_sharpe']:.2f}. {coverage['legs_without_volume']} trade legs in "
        f"that span had no volume figure and are left out.")
    add("")
    add("| Fund size | Sharpe after impact | Share of B's Sharpe kept | Impact cost per year | "
        "Largest trade / average daily volume | Largest position / average daily volume |")
    add("|---|---:|---:|---:|---:|---:|")
    for size, row in table.iterrows():
        add(f"| ${size / 1e6:,.0f}M | {row['sharpe']:.2f} | {pct(row['sharpe_kept'])} | "
            f"{pct(row['impact_cost_per_year'])} | {pct(row['max_participation'])} | {pct(row['max_position_to_volume'])} |")
    add("")
    add("Capacity (largest size tested that keeps at least "
        f"{config.CAPACITY_SHARPE_KEPT:.0%} of B's Sharpe): "
        + (f"**${capacity / 1e9:g} billion**." if capacity else "below the smallest size tested."))
    add("Square-root impact model; trades scaled linearly from the $10M run; rolls counted as trades in the "
        "old and the new contract. The largest trade-to-volume ratios are roll legs into the NEW contract, "
        "whose volume over the previous 20 days is still low on the roll date. The last column is the "
        "largest position held relative to the held contract's average daily volume.")
    add("")

    add("### 3. Stress windows")
    add("")
    names = ["A", "B", "bench_equal_weight", "bench_60_40"]
    table = diagnostics.stress_windows({k: results[k] for k in names}, period.first, period.last)
    if len(table) == 0:
        add("No stress window lies inside this period.")
    else:
        add("| Window | Dates | " + " | ".join(f"{n} return" for n in names) + " | B worst drawdown |")
        add("|---|---|" + "---:|" * (len(names) + 1))
        for row in table.itertuples():
            add(f"| {row.window} | {row._2} to {row.to} | "
                + " | ".join(pct(getattr(row, f"{n}_return")) for n in names) + f" | {pct(row.B_drawdown)} |")
    add("")

    add("### 4. Position risk (B)")
    add("")
    risk = diagnostics.position_risk(b_daily, m.assets)
    add("| Leg | Worst one-day loss | Worst five-day loss |")
    add("|---|---:|---:|")
    for asset, row in risk["legs"].iterrows():
        add(f"| {asset} | {pct(row['worst_1_day'])} | {pct(row['worst_5_day'])} |")
    add("")
    add(f"Worst portfolio day: {pct(risk['worst_day'])} on {risk['worst_day_date'].date()}. "
        f"Longest continuous spell with the brake on: {risk['longest_brake_spell_days']} trading days. "
        f"Maximum actual gross leverage: {risk['max_gross_leverage']:.2f}x (losses are fractions of equity).")
    add("")

    add("### 5. Statistical uncertainty (stationary block bootstrap)")
    add("")
    add("| Comparison | Sharpe difference | 95% interval |")
    add("|---|---:|---|")
    for other in ["A", "B_no_gold", "bench_equal_weight", "bench_60_40"]:
        difference, low, high = analysis.bootstrap_sharpe_difference(period.ret("B"), period.ret(other))
        add(f"| B minus {other} | {difference:+.2f} | [{low:+.2f}, {high:+.2f}] |")
    add("")

    add("### 6. Regime diagnostics (FRED, each series lagged one day; explanatory only)")
    add("")
    if fred is None:
        add("FRED data not downloaded (run `python data/download_external.py`).")
        add("")
    else:
        regime = diagnostics.regime_diagnostics(m, fred, period.first, period.last)
        add(f"Correlation between the trailing 60-day ES-ZN correlation and 10-year breakeven inflation: "
            f"{regime['corr_of_correlation_with_breakeven']:.2f}.")
        add("")
        add("| Series | Average on positive-correlation days | Average on other days | Days with data |")
        add("|---|---:|---:|---:|")
        labels = {"T10YIE": "10-year breakeven inflation, %", "VIXCLS": "VIX", "DFII10": "10-year real yield, %",
                  "BAMLH0A0HYM2": "High-yield credit spread, %"}
        for series, row in regime["averages"].iterrows():
            add(f"| {labels[series]} | {row['positive_correlation_days']:.2f} | {row['other_days']:.2f} | "
                f"{int(row['days_with_data'])} |")
        add("")
        add("FRED now publishes only about three years of history for the high-yield spread (an ICE BofA "
            "series), so that row covers only the days shown.")
        add("")

    add("### 7. Asynchronous settlement check")
    add("")
    table = diagnostics.asynchronous_settlement(m, period.first, period.last)
    add("| Pair | Daily correlation | Weekly correlation |")
    add("|---|---:|---:|")
    for pair, row in table.iterrows():
        add(f"| {pair} | {row['daily']:.2f} | {row['weekly']:.2f} |")
    add("")

    add("### 8. Total return (B)")
    add("")
    if fred is None:
        add("FRED data not downloaded.")
    else:
        total = diagnostics.total_return(results["B"], fred, period.first, period.last)
        add(f"Excess return {pct(total['excess_return_per_year'])} per year plus T-bill interest "
            f"{pct(total['tbill_per_year'])} gives an approximate total return of "
            f"{pct(total['total_return_per_year'])} per year (FRED DTB3, lagged one day, annual rate / 252 per day).")
    add("")

    add("### 9. Margin")
    add("")
    add("Skipped: no CME initial margins have been entered (`INITIAL_MARGIN` in config.py is empty). "
        "Amendment 1 says this is reported only if margins are entered." if not config.INITIAL_MARGIN
        else "Margins entered; margin-to-equity is not implemented yet.")
    add("")


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="also report the out-of-sample period (final run only)")
    args = parser.parse_args()

    cutoff = None if args.final else config.IS_END
    m = market.load_market(cutoff)
    results = run_all.run_everything(m)
    fred = diagnostics.load_fred(m.dates)
    contract_volume = diagnostics.load_contract_volume(cutoff)
    label = "full" if args.final else "in_sample"

    lines = []
    add = lines.append

    add("# Test report" + (" (in-sample and out-of-sample)" if args.final else " (in-sample only)"))
    add("")
    add(f"**DATA SOURCE: {m.source}**"
        + (" (random numbers; nothing here says anything about the real strategy)" if m.source == "SYNTHETIC" else ""))
    add("")
    add(f"- First day with data for all three markets: {m.dates[0].date()}. The first {config.WARMUP_DAYS} trading "
        f"days are warm-up; every run starts flat with ${config.CAPITAL:,.0f} on {m.dates[m.start].date()}.")
    if args.final:
        add(f"- In-sample ends {config.IS_END}; out-of-sample runs from {config.OOS_START} to {m.dates[-1].date()}.")
    else:
        add(f"- The out-of-sample period (from {config.OOS_START}) has not been downloaded, loaded or run.")
    add("- All returns are excess returns (no T-bill interest), net of costs.")
    add("- No strategy setting was changed in response to any number in this report.")
    fingerprint = REPO_DIR / "data" / "DATA_FINGERPRINT.txt"
    if m.source != "SYNTHETIC" and fingerprint.exists():
        for line in fingerprint.read_text().strip().splitlines():
            add(f"- Data fingerprint: `{line}`")
    add("")

    # ---------------------------------------------------- software checks
    add("# Software checks")
    add("")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, capture_output=True, text=True)
    add(f"- Automated tests (on synthetic data): `{tests.stdout.strip().splitlines()[-1]}`. They cover roll "
        "handling, ERC risk contributions, look-ahead, cost charging, the leverage cap, the drawdown brake, "
        "closed markets, the constant-leverage portfolio, the demeaned-gold run, the bootstrap, and "
        "backtrader against the pandas engine.")
    comparison = config.RESULTS_DIR / ("engine_comparison_full.md" if args.final else "engine_comparison.md")
    comparison_text = comparison.read_text() if comparison.exists() else ""
    if f"DATA SOURCE: {m.source}" in comparison_text:
        verdicts = [line[3:] for line in comparison_text.splitlines() if line.startswith("## Version")]
        add(f"- pandas engine against backtrader (the organizers' kit), on {m.source} data: " + "; ".join(verdicts)
            + ". Details at the end.")
    else:
        add("- Engine comparison: not run on this data yet.")
    mismatches, n_checked = check_roll_dates.find_mismatches(m)
    add(f"- Roll dates {check_roll_dates.FIRST_YEAR}-{check_roll_dates.LAST_YEAR}: {n_checked} checked, "
        f"{len(mismatches)} on a weekend or CME holiday (rule-based holiday list, not yet compared with CME's "
        "published calendar)." + "".join(f" {a} {d} ({r})." for a, d, r in mismatches))
    add("- Data integrity: every contract's month letter and year digit was checked against its Databento "
        "definition when the data was built (the build stops on a mismatch). The optional check of settlements "
        "against daily closes was not run for the whole sample; see the Deviations log in README.md for what "
        "the seven-contract fill download showed.")
    add("")

    # --------------------------------------------------------- the periods
    is_period = Period(m, results, m.dates[m.start], config.IS_END, "In-sample")
    write_period(add, is_period, fred, contract_volume)
    if args.final:
        oos = Period(m, results, config.OOS_START, m.dates[-1], "Out-of-sample")
        write_period(add, oos, fred, contract_volume)

    # ------------------------------------------------------------ the data
    add("# Data")
    add("")
    add("| Asset | Rolls | Days with no price for the held contract | Days filled with a daily close | "
        "Largest daily move | Daily vol, annualized |")
    add("|---|---:|---:|---:|---:|---:|")
    gaps_file = REPO_DIR / "data" / "gap_fills.csv"
    gaps = pd.read_csv(gaps_file, parse_dates=["date"]) if gaps_file.exists() and m.source != "SYNTHETIC" else None
    evaluated = m.returns.iloc[m.start:]
    for asset in m.assets:
        closed = m.dates[m.start:][~m.is_open[asset].to_numpy()[m.start:]]
        n_filled = 0 if gaps is None else int(((gaps["asset"] == asset) & ~gaps["date"].isin(closed)).sum())
        add(f"| {asset} | {int(m.roll[asset].iloc[m.start:].sum())} | {len(closed)} | {n_filled} | "
            f"{pct(evaluated[asset].abs().max())} | {pct(analysis.volatility(evaluated[asset]))} |")
    add("")
    for asset in m.assets:
        closed = m.dates[m.start:][~m.is_open[asset].to_numpy()[m.start:]]
        if len(closed):
            add(f"- {asset}, no price (treated as market closed: position held, zero return): "
                + ", ".join(str(d.date()) for d in closed))
    add("")

    # ------------------------------------------- changes since last report
    previous = config.RESULTS_DIR / "previous_summary_in_sample.csv"
    if not args.final and previous.exists() and m.source != "SYNTHETIC":
        old = pd.read_csv(previous, index_col=0)
        add("# What changed since the previous report")
        add("")
        add("The data changed in one way: four asset-days with no settlement record (ZN and GC on 2020-02-27 "
            "and 2020-06-30) are now filled with Databento daily closes. The seven missing asset-days in 2014 "
            "could not be filled, because Databento has no data of any kind for them. No trading rule or "
            "setting changed. The table shows every run that was in the previous report.")
        add("")
        add("| Run | Sharpe before | Sharpe now | Max drawdown before | Max drawdown now | Brake on before | Brake on now |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for name, row in old.iterrows():
            if name in results:
                now = is_period.summary(name)
                add(f"| {name} | {row['sharpe']:.2f} | {now['sharpe']:.2f} | {pct(row['max_drawdown'])} | "
                    f"{pct(now['max_drawdown'])} | {pct(row['brake_share'])} | {pct(now['brake_share'])} |")
        add("")
        add("New in this report: the constant-leverage portfolio (prediction 2), the demeaned-gold run, the "
            "factor regressions, the Amendment 1 diagnostics, bootstrap intervals, the no-brake twins and the "
            "cost breakdown. The full list of data and implementation choices is in the Deviations log in "
            "README.md.")
        add("")

    # ------------------------------------------------- engine comparison
    add("# Engine comparison details")
    add("")
    for line in comparison_text.splitlines()[1:]:
        add("#" + line if line.startswith("#") else line)
    add("")

    # ------------------------------------------------------ OOS readiness
    if not args.final:
        add("# Out-of-sample readiness")
        add("")
        add("**Frozen** (nothing below changes before or after the out-of-sample run):")
        add("")
        add("- HYPOTHESIS.md: trading rules, settings, the six variants, predictions and failure criteria.")
        add("- `config.py`: every number.")
        add("- The data-building rules in `data/build_processed.py` and the gap list in `data/gap_fills.csv`.")
        add("- The engine, the backtrader strategy, and this report's code, as committed.")
        add("")
        add("**How the single run is executed.** One command does everything, in order: download the "
            "out-of-sample data, rebuild the tables, rerun every variant in both engines, compare the engines, "
            "and write `results/REPORT_full.md`. It refuses to run a second time.")
        add("")
        add("```bash")
        add("cd gqh-erc-strategy")
        add("bash tools/run_out_of_sample.sh --i-have-the-go-ahead")
        add("```")
        add("")
        add("**Before running it:**")
        add("")
        add("- [ ] Put the new (rotated) Databento key in `.env`.")
        add("- [ ] Commit everything, so the commit time shows the code was frozen first.")
        add("- [ ] Confirm the team accepts the Deviations log in README.md.")
        add("- [ ] Know that the out-of-sample code path has never been executed on data after "
            f"{config.IS_END} (by design). If it stops with an error, fix only the error; do not change a rule.")
        add("- [ ] Know that new data gaps in the out-of-sample period are not in the gap list and will be "
            "treated as market-closed days.")
        add("")
        add("**Not run.** This checklist is preparation only.")
        add("")

    out = config.RESULTS_DIR / ("REPORT_full.md" if args.final else "REPORT.md")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"DATA SOURCE: {m.source}. Report written to {out}")


if __name__ == "__main__":
    main()
