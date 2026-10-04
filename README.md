# Risk parity on ES, ZN and GC futures

A pre-registered test of an equal-risk-contribution (ERC) portfolio of three
CME futures: E-mini S&P 500 (ES), 10-Year T-Note (ZN) and Gold (GC).

**Read [HYPOTHESIS.md](HYPOTHESIS.md) first.** It is the specification, and it
was committed before any backtest code existed (see `git log`).

> **Status.** The in-sample period (through 2024-09-30) has been run on real
> Databento data. The out-of-sample period (from 2024-10-01) has not been
> downloaded, loaded or run. It will be run exactly once.

## What is in this repo

| File | What it does |
|---|---|
| `HYPOTHESIS.md` | The pre-registered strategy, predictions and failure criteria. |
| `config.py` | Every fixed setting, in one place. No other file hardcodes a strategy number. |
| `data/download.py` | Downloads raw settlements and contract definitions from Databento. |
| `data/build_processed.py` | Turns the raw files into the tables the strategy reads. |
| `data/download_external.py` | Downloads public FRED and Kenneth French data for the diagnostics. |
| `data/gap_fills.csv` | The days with no settlement record (see the Deviations log). |
| `data/DATA_FINGERPRINT.txt`, `data/DOWNLOAD_DATE.txt` | A checksum of the data, and what was downloaded when. |
| `src/market.py` | Builds the daily panel from individual contracts: which contract is held, rolls, returns. |
| `src/signals.py` | The ERC solver and the target weights. |
| `src/backtest.py` | The backtest engine (pandas / numpy). This is the reference implementation. |
| `src/metrics.py` | Sharpe ratio, drawdown and the other summary numbers. |
| `src/analysis.py` | Checks of the predictions: regimes, regressions, bootstrap. |
| `src/diagnostics.py` | The Amendment 1 diagnostics. |
| `run_all.py` | Runs the six variants, the cost stress tests, the benchmarks and every diagnostic. |
| `tools/` | Export to backtrader, engine comparison, roll-date check, report, the one-time out-of-sample script. |
| `tests/` | Automated tests, and `make_synthetic_data.py` which makes fake data for them. |

## Setup

Python 3.11 or newer.

```bash
pip install -r requirements.txt
python -m pytest -q
```

The Databento key goes in a file called `.env` in this folder (never committed):

```
DATABENTO_API_KEY=db-...
```

## Data

```bash
python data/download.py            # raw Databento files -> data/raw/        (costs money)
python data/download.py --fills    # daily bars for the gap days             (under one cent)
python data/build_processed.py     # raw files -> data/processed/
python data/download_external.py   # public FRED and Kenneth French files    (free)
```

All four default to the in-sample period. `data/raw/`, `data/processed/` and
`data/external/` are never committed. The strategy reads three files from
`data/processed/`:

| File | Columns |
|---|---|
| `settlements.csv` | `date, asset, contract, settle, volume, price_type`: one row per contract per day |
| `contracts.csv` | `asset, contract, year, month, expiration`: one row per contract |
| `source.txt` | one word, `DATABENTO` or `SYNTHETIC`; it is printed on every output |

The data was downloaded on 2026-10-03 (see `data/DOWNLOAD_DATE.txt`).

To make fake data for development instead (it refuses to overwrite real data):

```bash
python tests/make_synthetic_data.py
```

## Running everything (in-sample)

```bash
python run_all.py                          # every variant, benchmark and diagnostic
python tools/export_backtrader_feeds.py    # then the two backtrader runs, see below
python tools/compare_engines.py
python tools/check_roll_dates.py
python tools/make_report.py                # writes results/REPORT.md
```

Results go to `results/` (not committed).

## The out-of-sample rule

Everything from `config.OOS_START` (2024-10-01) is the holdout. Every script
loads in-sample data by default and only loads later data when `--final` is
passed. Do not pass `--final` during development.

The single out-of-sample run is one command, which refuses to run twice:

```bash
bash tools/run_out_of_sample.sh --i-have-the-go-ahead
```

**It has not been run.** The checklist is at the end of `results/REPORT.md`.

## Second implementation (organizers' backtrader kit)

The strategy is implemented a second time, independently, as a backtrader
strategy inside the organizers' starter kit. If two engines written in
different ways give the same daily returns, the mechanics are very unlikely
to contain a bug.

### How to run it

The commands assume the two folders sit side by side:

```
gqh-erc-strategy/                  (this repo)
gqh-webull-backtrader-starter/     (the organizers' kit)
```

and one virtual environment that has both sets of dependencies (the kit's,
plus this repo's `requirements.txt`). No Webull credentials are needed.

```bash
# 1. In this repo: export the in-sample data as backtrader feeds.
cd gqh-erc-strategy
python tools/export_backtrader_feeds.py

# 2. In the starter kit: run versions A and B in backtrader.
cd ../gqh-webull-backtrader-starter
STRATEGY_REPO=../gqh-erc-strategy BT_VERSION=A python examples/backtest/main_futures.py
STRATEGY_REPO=../gqh-erc-strategy BT_VERSION=B python examples/backtest/main_futures.py

# 3. Back in this repo: compare the two engines.
cd ../gqh-erc-strategy
python tools/compare_engines.py
```

Step 3 writes `results/engine_comparison.md` and `results/engine_comparison.png`,
and stops with an error if a tolerance fails. `python -m pytest -q` runs the
same comparison on synthetic data (`tests/test_backtrader_match.py`).

The files added to the starter kit (nothing in the kit was modified):

| File | What it does |
|---|---|
| `examples/backtest/futures_feed.py` | A data feed with the real settlement prices and roll flags. |
| `examples/strategies/erc_risk_parity.py` | The strategy (versions A and B) and its cost model. |
| `examples/backtest/main_futures.py` | The entry point. |
| `examples/backtest/.env.futures.example` | Documents `STRATEGY_REPO` and `BT_VERSION`. |

### What it checks

Both engines use the same settings (`config.py`) and the same ERC solver
(`src/signals.py`, which has its own tests). Everything else is separate
code: in the backtrader version, backtrader's own broker does the fills, the
daily profit and loss and the trade costs. So the comparison verifies the
engine mechanics: fills at the next settlement, P&L, rolls, costs, the
rebalance band, the drawdown brake and the leverage cap.

It does **not** check the ERC solver (shared) or `src/market.py` (both engines
read the panel it builds). Those are covered by `tests/test_strategy.py`.

### Tolerances

| Check | Tolerance |
|---|---|
| Daily-return correlation | at least 0.999 |
| Sharpe ratio difference | at most 0.02 |
| Final equity difference | at most 0.5% |
| Max drawdown difference | at most 0.5 percentage points |

On synthetic data both versions pass with daily returns equal to about 1e-15
and identical positions on every day.

### Known timing differences, and how they are handled

1. **Roll costs.** backtrader applies `broker.add_cash()` on the next bar, so
   its own account value shows a roll cost one day late. The strategy corrects
   for this when it records equity (it subtracts today's roll cost by hand),
   so the recorded results match the pandas engine on the same day. The kit's
   HTML report uses backtrader's own value and therefore shows roll costs one
   day late.
2. **Orders on market-closed days.** If a market is closed on the day an order
   would fill, the pandas engine drops the order and holds the position. The
   backtrader strategy gets the same result by not sending the order: the feed
   has a `next_open` column, taken from the exchange calendar. Without it,
   backtrader would fill at the previous day's stale price.
3. **The HTML report** plots the difference-adjusted price `adj`, not real
   prices, and shows each date shifted to the previous evening (the kit
   converts timestamps to US Eastern time).

## Deviations log

Every data or implementation choice that HYPOTHESIS.md does not spell out, or
that departs from it. No trading rule, setting, prediction, pass/fail
criterion or variant count has been changed.

### Data

| Date | What | Why |
|---|---|---|
| 2026-10-03 | **Which settlement record is used.** CME publishes several settlement records per contract per day. We keep, in order of preference: a record flagged final; otherwise a preliminary record flagged rounded; otherwise the latest record. | The hypothesis says "official daily settlement prices" but not which record. Where a final and a rounded preliminary record both exist they agree on 98-100% of contract-days. Final records are sparse before 2016. |
| 2026-10-03 | **Two records dated on a weekend are dropped** (one GC record dated 2011-08-07, one ES record dated 2023-07-15). | CME does not settle on weekends. Each would have added a day on which every market looked closed. |
| 2026-10-03 | **Calendar spreads and other non-outright symbols are dropped** (731 ZN rows with symbols such as `ZN:CF ...`). | Only outright futures are traded. |
| 2026-10-03 | **Missing settlements: 11 asset-days.** ES on 2014-06-12; ES and GC on 2014-09-23, -24, -25; ZN and GC on 2020-02-27 and 2020-06-30. Checked against the raw data: this is not a parsing problem on our side. On the 2014 days Databento has no ES or GC statistics of any type; on the 2020 days it has other statistics but no settlement record for those dates. | Vendor gaps. |
| 2026-10-03 | **Four of them filled with Databento daily closes** (schema `ohlcv-1d`, same held contract): ZN and GC on 2020-02-27 and 2020-06-30. Rows are marked `price_type = daily_close`. | HYPOTHESIS.md names Databento daily closes as the fallback for missing settlements. |
| 2026-10-03 | **Seven could not be filled** (all the 2014 days): Databento has no daily bar either. They stay missing and are treated as "market closed": position held, zero return, and the move appears on the next day with a price. | HYPOTHESIS.md, "Data handling". No data is made up. |
| 2026-10-03 | **Caveats on the four fills.** (a) An `ohlcv-1d` bar covers a UTC calendar day, so its close is a few hours after the settlement time. On the neighbouring days, where both exist, close and settlement differ by up to 0.2% for ZN and up to 1.3% for GC (in the week of 2020-02-24). (b) The two bars for 2020-06-30 look incomplete: open equals close and volume is about half of normal. The two-day return across each gap is unaffected; only its split between the two days is approximate. | So the team can decide whether to keep these fills. Removing a line from `data/gap_fills.csv` and rebuilding undoes one. |
| 2026-10-03 | **Two roll dates are earlier than the calendar would give**, because the gaps remove "business days" (a business day is defined as a day with a settlement record). GC rolled out of the October 2014 contract on 2014-09-18 (it would be 2014-09-23 with complete data). ZN rolled out of the March 2020 contract on 2020-02-20 (it would be 2020-02-21). The fills do not change this, since they add a price only for the held contract. | Follows the hypothesis's own definition of a business day. |
| 2026-10-03 | **Cleared volume exists only from late November 2015** in Databento's statistics, and the last day of the sample has none (it is published the next day). The capacity diagnostic therefore covers 2015-11-27 onward. | Vendor coverage. |
| 2026-10-03 | **FRED high-yield spread (BAMLH0A0HYM2) is available only from 2023-10-03.** | FRED now publishes about three years of history for ICE BofA series. |
| 2026-10-03 | **Spend.** Databento quoted $3.32 for the in-sample statistics and definitions. `data/DOWNLOAD_DATE.txt` itemises $1.65 of it; the rest was downloaded before per-file logging was added and is not itemised. The gap-fill bars cost a further $0.0004. | Record keeping. |
| 2026-10-03 | **`data/external/` is not committed**, in addition to `data/raw/` and `data/processed/`. | It is third-party data that anyone can re-download with `data/download_external.py`. |

### Implementation

| Date | What | Why |
|---|---|---|
| 2026-10-03 | **Rebalance on a roll day:** close the old contract count at the old contract's settlement, open the new target count at the new contract's settlement, cost on both legs. | Amendment 1's clarification, read literally. |
| 2026-10-03 | **Order into a market that is closed the next day:** the order is dropped, not delayed. | "If a market is closed, we hold the prior position." |
| 2026-10-03 | **Annualized return** is the mean daily return times 252 (not compounded). | So that Sharpe = annualized return / annualized volatility. |
| 2026-10-03 | **Benchmarks** take their first position on the common start date, then rebalance on the last trading day of each month. | The hypothesis says "at every month-end" and does not say how they start. |
| 2026-10-03 | **Roll dates at the very end of the data.** A roll date is "5 business days before the anchor", and business days are only known once they have passed. If data ends inside that final window the roll cannot be seen; `load_market` warns. It does not happen at 2024-09-30. | The hypothesis's "no separate holiday calendar is needed" holds only in hindsight. |
| 2026-10-03 | **Constant-leverage portfolio (prediction 2):** the multiplier is found by rescaling until realized volatility matches B-no-brake's over the run (2.07x in-sample), with the 4x cap. | The hypothesis fixes the target but not the method. |
| 2026-10-03 | **Demeaned gold:** the average is taken over the evaluated days of each sample period separately (in-sample, and out-of-sample in the final run), and subtracted from gold's P&L only. | "The sample mean for the period being evaluated." |
| 2026-10-03 | **Newey-West lags** follow floor(4 x (n/100)^(2/9)). | The hypothesis asks for Newey-West errors but does not fix the lag. |
| 2026-10-03 | **Bootstrap:** seed 12345; paired resampling for Sharpe differences; 95% percentile intervals. | Amendment 1 says "fixed seed" without a value. |
| 2026-10-03 | **Prediction 6 "similar"** is read as: no window's Sharpe differs from the 60-day window's by more than its bootstrap 95% interval. | The hypothesis gives no number. This reading is the report's, not the hypothesis's. |
| 2026-10-03 | **Capacity model:** roll legs count as trades (old count in the old contract, new count in the new one); average volume is the traded contract's own 20-day trailing volume. | Amendment 1 does not say whether rolls are included. |
| 2026-10-03 | **Total return:** one day's T-bill interest is the DTB3 annual rate divided by 252. | "An approximate total-return figure." |
| 2026-10-03 | **Roll-date holiday check** uses the standard US market-holiday rules, not CME's published calendar. | The published calendar has not been compared yet. |
| 2026-10-03 | **Brake-noise diagnostics:** every run with a brake has a `diag_*_no_brake` twin. Reporting only. | To see which conclusions depend on the day the brake fires. |
| 2026-10-03 | **Margin diagnostic skipped.** | No CME initial margins have been entered. |
| 2026-10-03 | **`IS_END = 2024-09-30`** and the "Sample periods" section were added to HYPOTHESIS.md from the team's text, before any real data was processed. An earlier draft of this code used 2024-12-31 on synthetic data only. | The first committed copy of the hypothesis lacked that section. |

## Roll-date check

```bash
python tools/check_roll_dates.py
```

Lists every roll date from 2015 to 2023 that falls on a weekend or a CME
holiday. It only reports; it changes no rule. On the real data all 126 roll
dates fall on trading days.

## Not built

- Margin-to-equity (waiting for CME initial margins to be entered in `config.py`).
- The optional check of every settlement against a daily close.

## Second strategy: ENSO on corn, soybeans and wheat

Added 2026-10-03. It is separate from the ES/ZN/GC risk parity above, whose
settings, data and results are untouched. The specification is
`ENSO_SPEC.md`: Part 1 is the team's text, Part 2 lists every choice that
text leaves open, written before the first backtest.

In one sentence: each week, score corn, soybeans and wheat by price momentum
plus the NOAA Niño-3.4 (ENSO) state, hold only the ones with a positive
score, give the better-scored ones a bigger share of the portfolio's risk,
and scale the whole portfolio to 10% volatility.

| File | What it does |
|---|---|
| `ENSO_SPEC.md` | The specification and the choices fixed in advance. |
| `config.py` (last section) | Every setting of this strategy. |
| `data/download_ag.py` | Buys the grain settlements from Databento (one file per period). |
| `data/build_ag.py` | Turns that file into `data/processed_ag/`. |
| `data/download_enso.py` | Downloads NOAA's free Niño-3.4 table. |
| `src/enso.py` | Z-score, ENSO state, and the date each value became usable. |
| `src/alpha.py` | Alpha score, risk budgets, weights, whole contracts. |
| `src/signals.py` (`risk_budget_weights`) | The risk-budget solver. |
| `src/ag_backtest.py` | The weekly-rebalance engine. |
| `src/ag_metrics.py` | The performance numbers. |
| `run_enso.py` | Runs every ablation, benchmark and robustness test; writes `results/ENSO_REPORT.md`. |
| `tests/test_enso.py` | The invariants of ENSO_SPEC.md section 33, on fake data. |

### Running it (in-sample)

```bash
python data/download_enso.py      # free
python data/download_ag.py        # charged by Databento (about $0.21); skipped if already there
python data/build_ag.py
python run_enso.py                # writes results/ENSO_REPORT.md
python -m pytest -q
```

Second implementation in the organizers' kit (`examples/strategies/enso_risk_budget.py`):

```bash
python tools/export_backtrader_feeds.py --ag
cd ../gqh-webull-backtrader-starter
STRATEGY_REPO=../gqh-erc-strategy BT_VERSION=ENSO python examples/backtest/main_futures.py
```

On the real in-sample data the two engines agree on every position and on
every daily return to 16 decimal places (checked by hand on 2026-10-03;
there is no automated test of this yet).

### Out-of-sample rule

The same as for the main strategy. Out-of-sample is 2024-10-01 to
2026-09-30 and **has not been downloaded**. The one final run is the four
commands above with `--final` added to each of the three data scripts and to
`run_enso.py`. Do it once, after the team has agreed that nothing in
ENSO_SPEC.md or `config.py` will change.

### Data notes and deviations (ENSO strategy)

| Date | What | Why |
|---|---|---|
| 2026-10-03 | **Five years of futures data at most** (2021-10-01 to 2026-09-30); only the in-sample three years are on disk. | Team instruction. |
| 2026-10-03 | **One download, 66 MB, $0.21** (the main strategy needed 37 files, 277 MB). Contracts are requested by name, outright only, and no `definition` files are downloaded: contract months and last trading days come from CME's rule. | Team instruction to keep downloads smaller. |
| 2026-10-03 | Databento flagged three days of the download as "degraded": 2021-12-05 and 2022-01-02 (both Sundays, no settlement) and 2024-09-18. All three markets have a settlement on 2024-09-18. | Vendor notice; recorded, nothing changed. |
| 2026-10-03 | **No missing settlements:** on all 754 in-sample days all three held contracts have a settlement record. | Check of the built data. |
| 2026-10-03 | **`expiration` in `data/processed_ag/contracts.csv` is the weekday before the 15th of the delivery month**, ignoring exchange holidays. It only puts contracts in order; rolls are counted back from first notice day using days that have a settlement record. | No `definition` download. |
| 2026-10-03 | **NOAA values are today's version, not the version published at the time.** Timing is point-in-time (usable from the 15th of the next month); revisions cannot be undone. | NOAA publishes no vintages of `sstoi.indices`. |
| 2026-10-03 | **Margin limit skipped.** | No CME initial margins entered (`config.AG_INITIAL_MARGIN`). |
| 2026-10-03 | Existing files changed, behaviour of the main strategy unchanged (its 24 tests still pass): `config.py` (grain entries added to three dictionaries, new section at the end), `src/market.py` (`load_market` takes optional `assets` and `warmup`), `data/build_processed.py` (reader takes an optional folder), `src/signals.py` (new function), `tools/export_backtrader_feeds.py` (`--ag`). In the kit: `examples/backtest/main_futures.py` accepts `BT_VERSION=ENSO`. | To reuse the tested roll and feed code instead of copying it. |

## Third strategy: cross-asset signal-weighted risk budgeting

Added 2026-10-03, after the ENSO grain strategy lost money in-sample and the
team revised the design. The specification is `XA_SPEC.md`. The two earlier
strategies and their results are untouched.

In one sentence: each week, score eleven futures (stocks, bonds, energy,
metals, grains) by their 12-month trend, plus the ENSO effect on the grains;
go long the positive scores and short the negative ones; give each a share
of the portfolio's risk equal to its share of total signal strength; scale
the whole portfolio to 10% volatility.

| File | What it does |
|---|---|
| `XA_SPEC.md` | The specification, written before the first run. |
| `config.py` ("Cross-asset" section) | Every setting. |
| `data/download_xa.py` | Buys the five new markets and the older grain history (one file per year). |
| `data/build_xa.py` | Joins them with the data already on disk into `data/processed_xa/`. |
| `src/xa_alpha.py` | Trend, ENSO adjustment, signed risk budgets, weights. |
| `src/ag_backtest.py` | The same engine as the ENSO strategy; it now also writes a transaction ledger. |
| `run_xa.py` | Runs the four models and the robustness tests; writes `results/XA_REPORT.md`. |
| `tests/test_xa.py` | Tests on fake data. |

### Running it

```bash
python data/download_enso.py      # free
python data/download_xa.py        # charged by Databento; files already there are skipped
python data/build_xa.py
python run_xa.py                  # development + validation only
python -m pytest -q
```

In the organizers' kit (`examples/strategies/cross_asset_risk_budget.py`):

```bash
python tools/export_backtrader_feeds.py --xa
cd ../gqh-webull-backtrader-starter
STRATEGY_REPO=../gqh-erc-strategy BT_VERSION=XA4 python examples/backtest/main_futures.py
```

`BT_VERSION` can be `XA1`, `XA2`, `XA3` or `XA4` (the four models).

### Locked periods

| Period | How to run it | Status |
|---|---|---|
| Test, 2020-01-01 to 2024-09-30 | `python data/download_xa.py --test`, `python data/build_xa.py`, `python run_xa.py --test` | Locked. Not run. Needs the team's go-ahead. |
| Final, 2024-10-01 onward | the same with `--final` (and `data/download_ag.py --final`, `data/download_enso.py --final`, `data/download.py --final`) | Locked. Not downloaded. |

The test period is not pristine: see the disclosure in `XA_SPEC.md`.

### Data notes and deviations (cross-asset strategy)

| Date | What | Why |
|---|---|---|
| 2026-10-03 | **Bought:** ten yearly files, 314 MB, $1.36 (NQ, ZB, CL, NG, HG for 2010-06 to 2019-12; grains for 2010-06 to 2021-09). ES, ZN, GC and the later grain data were reused from disk. | The revision asks for a long history; this lifts the earlier "five years, small download" limit for this strategy. |
| 2026-10-03 | **History starts 2010-06, not 2000.** | Databento's CME data starts there. |
| 2026-10-03 | **Nothing after 2019-12-31 was downloaded for the five new markets**, and `run_xa.py` cuts every asset at that date by default. | Keeps the test period locked. |
| 2026-10-03 | **Missing settlement days** (treated as "market closed", order waits): ES 4, NQ 4, CL 3, NG 3, GC 3, HG 3, in 2011-2019. | The same Databento gaps as in the main strategy's log (2014). |
| 2026-10-03 | **Contract expiry dates by rule, not downloaded:** third Friday for NQ; for the others the date only orders the contracts. | No `definition` files. |
| 2026-10-03 | **Cross-asset mean not subtracted from the signal**, and **no turnover limit**: both explained in XA_SPEC.md. | Departures from the revision's text. |
| 2026-10-03 | **Engines agree:** on development + validation, models 1 and 4 in backtrader match the pandas engine on every position and on daily returns to 15 decimal places (checked by hand). | Second implementation. |
| 2026-10-03 | **Kit HTML report:** for the ENSO and cross-asset strategies its "Trades", "Win Rate" and "Trades Net P&L" cards now show position episodes (unbroken long or short stretches), not fills. The risk parity report is unchanged. | The cards were empty (W0/L0, $0.00). |
