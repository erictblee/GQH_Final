"""Does the backtrader implementation reproduce the pandas engine?

Runs both on synthetic in-sample data and asserts the acceptance tolerances.
The test is skipped if backtrader or the organizers' starter kit is missing.

The starter kit is looked for next to this repo, in a folder called
gqh-webull-backtrader-starter. Set the STARTER_KIT environment variable if
yours is somewhere else.
"""

import os
import sys
from pathlib import Path

import pytest

import config
from src import backtest

REPO_DIR = Path(__file__).resolve().parent.parent
STARTER_KIT = Path(os.environ.get("STARTER_KIT", REPO_DIR.parent / "gqh-webull-backtrader-starter"))
ENTRY_DIR = STARTER_KIT / "examples" / "backtest"

pytest.importorskip("backtrader")
if not (ENTRY_DIR / "main_futures.py").exists():
    pytest.skip("starter kit with main_futures.py not found", allow_module_level=True)

# Tell the backtrader strategy where this repo is, then import its entry point.
os.environ["STRATEGY_REPO"] = str(REPO_DIR)
sys.path.insert(0, str(ENTRY_DIR))
sys.path.insert(0, str(REPO_DIR / "tools"))
import main_futures
from compare_engines import compare
from export_backtrader_feeds import export_feeds


@pytest.fixture(scope="module")
def feed_dir(synthetic_market, tmp_path_factory):
    """Export the synthetic in-sample market as backtrader feeds."""
    folder = tmp_path_factory.mktemp("bt_feeds")
    export_feeds(synthetic_market, folder, cutoff=config.IS_END)
    return folder


@pytest.mark.parametrize("version", ["A", "B"])
def test_backtrader_matches_pandas_engine(version, synthetic_market, feed_dir):
    pandas_daily = backtest.run(synthetic_market, backtest.Spec(name=version, version=version))
    bt_daily, _, meta = main_futures.run_backtest(version, feed_dir)
    assert meta["source"] == "SYNTHETIC"

    result = compare(pandas_daily, bt_daily)
    assert result["checks"]["correlation"], result["correlation"]
    assert result["checks"]["sharpe"]
    assert result["checks"]["final_equity"]
    assert result["checks"]["max_drawdown"]

    # The comparison only means something if the rules were actually used.
    assert pandas_daily["trades"].sum() > 50
    assert pandas_daily["braking"].any()
    assert pandas_daily["cap_binding"].any()
