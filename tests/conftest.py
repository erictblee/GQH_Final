"""Shared set-up for the tests.

The tests NEVER read data/processed/ (which may hold real data). They make
their own synthetic data in a temporary folder and load only the in-sample
part of it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Let the tests import config.py and the src/ package.
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "tests"))

import config
import make_synthetic_data
from src import market


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A temporary folder of synthetic data, made once for all tests."""
    folder = tmp_path_factory.mktemp("synthetic")
    make_synthetic_data.make(folder)
    return folder


@pytest.fixture(scope="session")
def synthetic_market(data_dir):
    """The in-sample synthetic market."""
    return market.load_market(cutoff=config.IS_END, data_dir=data_dir)


def tiny_market(p_new, p_old=None, roll=None, is_open=None, start=1, first_day="2020-01-01"):
    """Build a small hand-made Market from price lists, so a test can check
    the engine's arithmetic against numbers worked out by hand.

    p_new: dict {asset: list of prices}. All three assets must be present.
    """
    n = len(next(iter(p_new.values())))
    dates = pd.bdate_range(first_day, periods=n)
    p_new = pd.DataFrame(p_new, index=dates, dtype=float)[config.ASSETS]
    p_old = p_new.copy() if p_old is None else pd.DataFrame(p_old, index=dates, dtype=float)[config.ASSETS]
    false = pd.DataFrame(False, index=dates, columns=config.ASSETS)
    true = ~false
    roll = false if roll is None else pd.DataFrame(roll, index=dates)[config.ASSETS]
    is_open = true if is_open is None else pd.DataFrame(is_open, index=dates)[config.ASSETS]
    returns = ((p_old - p_new.shift(1)) / p_new.shift(1)).fillna(0.0)
    return market.Market(
        dates=dates, assets=list(config.ASSETS), p_new=p_new, p_old=p_old,
        returns=returns, roll=roll, is_open=is_open,
        volume=pd.DataFrame(0.0, index=dates, columns=config.ASSETS),
        held=pd.DataFrame("X", index=dates, columns=config.ASSETS),
        start=start, source="SYNTHETIC",
    )
