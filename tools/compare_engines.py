"""Compare our pandas engine with the backtrader implementation, in-sample.

Run these first (see README.md):

    python tools/export_backtrader_feeds.py
    BT_VERSION=A python <starter kit>/examples/backtest/main_futures.py
    BT_VERSION=B python <starter kit>/examples/backtest/main_futures.py

then:

    python tools/compare_engines.py

It writes results/engine_comparison.md and results/engine_comparison.png and
exits with an error if any tolerance fails.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # draw to a file; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Let this script find config.py, run_all.py and src/ when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import run_all
from src import backtest, market, metrics

# Acceptance tolerances for "the two engines agree". These are checks on the
# software, not strategy settings.
MIN_CORRELATION = 0.999         # daily-return correlation
MAX_SHARPE_DIFF = 0.02
MAX_EQUITY_DIFF = 0.005         # 0.5% of final equity
MAX_DRAWDOWN_DIFF = 0.005       # 0.5 percentage points


def compare(pandas_daily, bt_daily):
    """Compare two tables of daily results. Returns a dict of numbers and a
    pass/fail flag for each tolerance."""
    if not pandas_daily.index.equals(bt_daily.index):
        raise ValueError("the two engines do not cover the same dates")

    ours = metrics.summary(pandas_daily)
    theirs = metrics.summary(bt_daily)
    diff = (bt_daily["ret"] - pandas_daily["ret"]).abs()

    result = {
        "pandas": ours,
        "backtrader": theirs,
        "correlation": float(pandas_daily["ret"].iloc[1:].corr(bt_daily["ret"].iloc[1:])),
        "largest_diff": float(diff.max()),
        "largest_diff_date": diff.idxmax(),
        "worst_dates": diff.sort_values(ascending=False).head(10),
    }
    result["checks"] = {
        "correlation": result["correlation"] >= MIN_CORRELATION,
        "sharpe": abs(ours["sharpe"] - theirs["sharpe"]) <= MAX_SHARPE_DIFF,
        "final_equity": abs(theirs["final_equity"] / ours["final_equity"] - 1.0) <= MAX_EQUITY_DIFF,
        "max_drawdown": abs(ours["max_drawdown"] - theirs["max_drawdown"]) <= MAX_DRAWDOWN_DIFF,
    }
    result["passed"] = all(result["checks"].values())
    return result


def markdown_report(results, m):
    """The text of results/engine_comparison.md."""
    lines = [
        "# Engine comparison: pandas engine vs backtrader",
        "",
        f"**DATA SOURCE: {m.source}**"
        + (" (random numbers; these figures say nothing about the real strategy)" if m.source == "SYNTHETIC" else ""),
        "",
        f"Period: {m.dates[m.start].date()} to {m.dates[-1].date()}.",
        "",
    ]
    for version, r in results.items():
        ours, theirs = r["pandas"], r["backtrader"]
        lines += [
            f"## Version {version}: {'PASS' if r['passed'] else 'FAIL'}",
            "",
            "| Metric | pandas | backtrader |",
            "|---|---:|---:|",
            f"| Annualized return | {ours['ann_return']:.4%} | {theirs['ann_return']:.4%} |",
            f"| Annualized volatility | {ours['ann_vol']:.4%} | {theirs['ann_vol']:.4%} |",
            f"| Sharpe ratio | {ours['sharpe']:.4f} | {theirs['sharpe']:.4f} |",
            f"| Max drawdown | {ours['max_drawdown']:.4%} | {theirs['max_drawdown']:.4%} |",
            f"| Number of trades | {ours['trades']} | {theirs['trades']} |",
            f"| Final equity | {ours['final_equity']:,.2f} | {theirs['final_equity']:,.2f} |",
            "",
            "| Check | Value | Tolerance | Result |",
            "|---|---:|---:|---|",
            f"| Daily-return correlation | {r['correlation']:.10f} | >= {MIN_CORRELATION} | {'pass' if r['checks']['correlation'] else 'FAIL'} |",
            f"| Sharpe difference | {abs(ours['sharpe'] - theirs['sharpe']):.6f} | <= {MAX_SHARPE_DIFF} | {'pass' if r['checks']['sharpe'] else 'FAIL'} |",
            f"| Final equity difference | {abs(theirs['final_equity'] / ours['final_equity'] - 1):.8%} | <= {MAX_EQUITY_DIFF:.1%} | {'pass' if r['checks']['final_equity'] else 'FAIL'} |",
            f"| Max drawdown difference | {abs(ours['max_drawdown'] - theirs['max_drawdown']) * 100:.6f} pp | <= {MAX_DRAWDOWN_DIFF * 100} pp | {'pass' if r['checks']['max_drawdown'] else 'FAIL'} |",
            "",
            f"Largest daily-return difference: {r['largest_diff']:.3e} on {pd.Timestamp(r['largest_diff_date']).date()}.",
            "",
        ]
        if not r["passed"]:
            # Help with the diagnosis: show the worst days, and whether a roll
            # or a closed market happened on each of them.
            lines += ["Ten dates with the largest differences:", "",
                      "| Date | Difference | Rolls that day | Closed markets that day |", "|---|---:|---|---|"]
            for day, value in r["worst_dates"].items():
                rolls = ", ".join(a for a in m.assets if m.roll.at[day, a]) or "-"
                closed = ", ".join(a for a in m.assets if not m.is_open.at[day, a]) or "-"
                lines.append(f"| {pd.Timestamp(day).date()} | {value:.3e} | {rolls} | {closed} |")
            lines.append("")
    return "\n".join(lines)


def plot(daily_tables, results, m, path):
    """Overlay chart. Top row: equity of both engines. Bottom row: the gap
    between them, on its own axis."""
    blue, orange = "#2a78d6", "#eb6834"
    ink, muted, surface = "#0b0b0b", "#898781", "#fcfcfb"

    versions = list(daily_tables)
    fig, axes = plt.subplots(2, len(versions), figsize=(6.5 * len(versions), 6.5),
                             sharex=True, squeeze=False, facecolor=surface,
                             gridspec_kw={"height_ratios": [3, 1]})
    for column, version in enumerate(versions):
        ours, theirs = daily_tables[version]
        top, bottom = axes[0, column], axes[1, column]

        # The two curves sit on top of each other when the engines agree, so
        # pandas is drawn wide and backtrader thin: both stay visible.
        top.plot(ours.index, ours["equity"] / 1e6, color=blue, linewidth=4.5,
                 alpha=0.45, label="pandas engine (wide line)")
        top.plot(theirs.index, theirs["equity"] / 1e6, color=orange, linewidth=1.3,
                 label="backtrader (thin line)")
        verdict = "PASS" if results[version]["passed"] else "FAIL"
        top.set_title(f"Version {version}: equity, $ millions ({verdict})",
                      loc="left", color=ink, fontsize=11)
        top.legend(frameon=False, loc="upper left", labelcolor=ink)

        gap = theirs["equity"] - ours["equity"]
        bottom.plot(gap.index, gap, color=ink, linewidth=1.0)
        bottom.set_title("backtrader equity minus pandas equity, $",
                         loc="left", color=ink, fontsize=10)
        # Never let the axis zoom in further than +/- $1, so rounding noise
        # of a millionth of a dollar is not drawn as if it were a difference.
        limit = max(1.0, float(gap.abs().max()) * 1.2)
        bottom.set_ylim(-limit, limit)

        for ax in (top, bottom):
            ax.set_facecolor(surface)
            ax.grid(True, color=muted, alpha=0.2, linewidth=0.6)
            ax.tick_params(colors=muted, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(muted)

    fig.suptitle(f"Engine comparison.  DATA SOURCE: {m.source}",
                 x=0.01, ha="left", color=ink, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="compare over the full sample (final run only)")
    args = parser.parse_args()

    # In-sample by default: the out-of-sample period is not loaded.
    m = market.load_market(None if args.final else config.IS_END)
    label = "full" if args.final else "in_sample"

    # The same Specs that run_all.py uses for versions A and B.
    specs = {spec.name: spec for spec in run_all.VARIANTS if spec.name in ("A", "B")}

    results, daily_tables = {}, {}
    for version, spec in specs.items():
        pandas_daily = backtest.run(m, spec)
        bt_file = config.RESULTS_DIR / f"backtrader_{version}_{label}.csv"
        if not bt_file.exists():
            raise SystemExit(f"missing {bt_file}: run main_futures.py with BT_VERSION={version} first")
        bt_daily = pd.read_csv(bt_file, index_col="date", parse_dates=True)
        if bt_daily["source"].iloc[0] != m.source:
            raise SystemExit(f"{bt_file} was made from {bt_daily['source'].iloc[0]} data but the "
                             f"data on disk is {m.source}. Re-export the feeds and rerun backtrader.")
        results[version] = compare(pandas_daily, bt_daily)
        daily_tables[version] = (pandas_daily, bt_daily)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    report = markdown_report(results, m)
    suffix = "_full" if args.final else ""
    (config.RESULTS_DIR / f"engine_comparison{suffix}.md").write_text(report)
    plot(daily_tables, results, m, config.RESULTS_DIR / f"engine_comparison{suffix}.png")
    print(report)

    if not all(r["passed"] for r in results.values()):
        raise SystemExit("ENGINE COMPARISON FAILED: see results/engine_comparison.md")
    print("All tolerances passed.")


if __name__ == "__main__":
    main()
