"""Export our futures panel as CSV files that backtrader can read.

    python tools/export_backtrader_feeds.py            in-sample only (default)
    python tools/export_backtrader_feeds.py --final    full sample (final run only)
    python tools/export_backtrader_feeds.py --ag       the grains of the ENSO strategy
    python tools/export_backtrader_feeds.py --xa       the 11 futures of the cross-asset
                                                       strategy, up to the end of validation

It writes one file per asset into data/processed/bt_feeds/ plus meta.json
(with --ag: into data/processed_ag/bt_feeds/).

The important column is `adj`, a DIFFERENCE-ADJUSTED continuous price:

    adj[0] = p_new[0]
    adj[t] = adj[t-1] + (p_old[t] - p_new[t-1])

Each day `adj` moves by exactly the dollar move of the contract we held, so
holding n contracts on `adj` earns exactly the same dollars as our engine.
On a roll day the jump from the old contract's price to the new contract's
price is left out, so roll gaps never appear as profit or loss.

`adj` is only good for P&L. For anything that needs the REAL price level
(position sizes, costs) the file also carries the real settlements.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let this script find config.py and src/ when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import market


def export_feeds(m, out_dir, cutoff):
    """Write {ASSET}.csv for every asset, and meta.json, into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset in m.assets:
        p_new = m.p_new[asset].to_numpy()
        p_old = m.p_old[asset].to_numpy()

        # Daily dollar move of the held contract (0 on the first day).
        daily_move = np.zeros(len(p_new))
        daily_move[1:] = p_old[1:] - p_new[:-1]
        # Start at the first real price and add up the daily moves.
        adj = p_new[0] + np.cumsum(daily_move)

        # CHECK: adj must reproduce our engine's returns to within 1e-12.
        rebuilt_returns = np.zeros(len(adj))
        rebuilt_returns[1:] = (adj[1:] - adj[:-1]) / p_new[:-1]
        worst = np.max(np.abs(rebuilt_returns - m.returns[asset].to_numpy()))
        if worst > 1e-12:
            raise ValueError(f"{asset}: adj does not reproduce returns (error {worst})")

        # Will this market be open on the NEXT calendar day? Exchange holidays
        # are published in advance, so knowing this is not look-ahead. The
        # strategy uses it to avoid sending an order into a closed market.
        is_open = m.is_open[asset].to_numpy()
        next_open = np.ones(len(is_open), dtype=bool)
        next_open[:-1] = is_open[1:]

        feed = pd.DataFrame({
            "datetime": m.dates.strftime("%Y-%m-%d"),
            "open": adj, "high": adj, "low": adj, "close": adj,   # backtrader needs OHLC
            "adj": adj,
            "settle_held": p_new,          # real settlement, contract held after the day
            "settle_old": p_old,           # real settlement, contract held coming in
            "roll": m.roll[asset].to_numpy().astype(int),
            "is_open": is_open.astype(int),
            "next_open": next_open.astype(int),
            "volume": m.volume[asset].to_numpy(),
        })
        feed.to_csv(out_dir / f"{asset}.csv", index=False)

    meta = {
        "source": m.source,                               # SYNTHETIC or real
        "cutoff": cutoff,                                 # null means full sample
        "start_date": m.dates[m.start].strftime("%Y-%m-%d"),
        "assets": m.assets,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="include the out-of-sample period (final run only)")
    parser.add_argument("--ag", action="store_true",
                        help="export corn, soybeans and wheat (ENSO strategy) instead of ES, ZN and GC")
    parser.add_argument("--xa", action="store_true",
                        help="export the eleven futures of the cross-asset strategy")
    args = parser.parse_args()

    cutoff = None if args.final else config.IS_END
    if args.xa:
        # The test period (2020 onward) stays locked: stop at the end of validation.
        cutoff = None if args.final else config.XA_VALIDATION_END
        m = market.load_market(cutoff, data_dir=config.XA_PROCESSED_DIR,
                               assets=config.XA_ASSETS, warmup=config.XA_WARMUP_DAYS)
        out_dir = config.XA_PROCESSED_DIR / "bt_feeds"
    elif args.ag:
        m = market.load_market(cutoff, data_dir=config.AG_PROCESSED_DIR,
                               assets=config.AG_ASSETS, warmup=config.AG_WARMUP_DAYS)
        out_dir = config.AG_PROCESSED_DIR / "bt_feeds"
    else:
        m = market.load_market(cutoff)
        out_dir = config.PROCESSED_DIR / "bt_feeds"
    meta = export_feeds(m, out_dir, cutoff)
    print(f"DATA SOURCE: {meta['source']}. Feeds written to {out_dir}")
    print(json.dumps(meta, indent=2))
