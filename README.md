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

## Fourth iteration: the A/B/C/D attribution experiment

Added 2026-10-03. The plan is `ABCD_PLAN.md` (the team's text, unchanged).
This section records how the plan was turned into code. **Everything down to
"Results" was written and committed before the first run.** The third
strategy (`XA_SPEC.md`, `run_xa.py`, models M1 to M4) is kept as it was, as
the primary replication; nothing in it was changed.

Universe, data, periods, roll rule, costs (2 bps per side, rolls on both
legs), weekly rebalance, $10M, whole contracts, 12-month (252-day) trend,
EWMA covariance (60-day half-life, 252-day window), 10% target, 2x-per-asset
and 4x-gross limits: all taken over from `XA_SPEC.md` unchanged.

### The four models

| Model | Signal | Sizing |
|---|---|---|
| A Trend | `sign` of the 252-day compounded return: +1, -1, or 0 (no position) | Equal notional: `weight_i = signal_i / N`, longs + shorts = 1x equity. No volatility, no covariance, no leverage. |
| B Trend + InvVol | as A | `q_i = signal_i / σ_i`, then scaled to 10% **as if every correlation were zero**: `q × 0.10 / sqrt(Σ q_i² σ_i²)`. |
| C Trend + Covariance | as A | The same `q`, scaled with the full covariance: `q × 0.10 / sqrt(q'Σq)`. |
| D Trend + ENSO + Covariance | as A, plus the ENSO overlay on ZC, ZS, ZW | as C |

So A → B adds volatility information, B → C adds correlations (the two
portfolios have the same shape and differ by one common factor each week),
and C → D adds ENSO. The covariance is used only to scale. There is no
optimizer and no risk-budget solver in these models, and no asset is forced
to carry an equal or a signal-sized share of risk.

### Choices the plan left open

| Choice | What was done | Why |
|---|---|---|
| The old signal was a volatility-standardized score with a 0.25 dead zone and signal-sized risk budgets | Replaced, in A to D only, by the plain `sign` the plan writes (section 3). | The plan's formula. M1 to M4 keep the old one. |
| How B is "normalized" | To the same 10% target as C, with correlations set to zero. | Then B and C differ in exactly one thing: whether correlations are used. |
| A's size | 1x equity gross. | "Simple equal notional." Its volatility is whatever it turns out to be; Sharpe ratios are still comparable. |
| ENSO in Model D | The existing additive form, on the existing table (`config.ENSO_EFFECT`): `signal = trend + effect`. On a +1/-1 trend this means: double the grain position when ENSO agrees with the trend, close it when ENSO disagrees, unchanged when ENSO has no view. | Plan section 4: "preserve [the current formulation] for the primary replication". |
| The overlay form, tested separately | `D_mult_overlay`: `signal = trend × 1.5` when ENSO agrees, `× 0.5` when it disagrees. It never flips or closes a position. | Plan section 4's preferred form. The two numbers are round, symmetric and chosen before any run; they are not fitted. |
| "The historical relationship must be measured" | The report shows each grain's own return in each ENSO state, and the grains' P&L in every separate ENSO stretch. The table of effects itself is **not** re-estimated from prices. | Fitting it on this sample would be the full-sample estimation section 15 forbids. |
| No-trade band | Not implemented. | Optional in the plan; it would be a new free parameter. |
| Maximum risk contribution per asset | Not a separate limit. The 2x position limit and the 4x gross limit apply after scaling. | With sign signals every asset's standalone risk is equal by construction; risk shares are reported. |
| "OOS Sharpe" column | The validation period (2017 to 2019). | The test period (2020 to 2024-09) is still locked and not downloaded; unlocking it is the team's decision. |
| "Trades" | Position episodes (one unbroken long or short stretch in one market), as in the 277 of the plan's section 19. Fills are reported as well. | A weekly resize is not a trade one can win or lose. |

### Runs fixed in advance

| Group | Runs |
|---|---|
| The four models | `A_trend`, `B_trend_invvol`, `C_trend_cov`, `D_trend_enso_cov` |
| Overlay form | `D_mult_overlay` |
| Diversification (section 13) | `A_grains`: Model A on ZC, ZS, ZW only, against `A_trend` |
| Long/short (section 13) | `A_long_only`, `A_grains_long_only`, `C_long_only`: a negative signal means no position |
| ENSO test (section 14) | Grains only, Model C's sizing: `grains_trend`, `grains_enso_only`, `grains_trend_enso` |
| Robustness of D (section 16) | trend 126 and 189 days; covariance half-life 40 and 90 days; volatility target 8% and 12%; rebalance every 2 weeks; cost 4 bps |

Robustness runs are reported, never selected from.

### Known limits

- **ENSO values are today's version, not the version published at the
  time** (NOAA publishes no vintages of this file). Timing is point-in-time
  (a month's value is used from the 15th of the next month); later revisions
  to a value cannot be undone. The 30-year z-score is rolling, so it uses no
  future months. Plan section 4 asks for unrevised data; this is the one
  place it is not met.
- The old agriculture-only strategy ran on 2021-12 to 2024-09. Here the
  grains-only runs use 2011 to 2019, so they are a like-for-like comparison
  with Model A, not a re-run of the old result.
- The backtrader kit has not been given these models; only the pandas
  engine runs them. The engine itself is the one already checked against
  backtrader, with one addition (an optional "every k-th week" rebalance,
  off by default).

### Files and how to run

| File | What it does |
|---|---|
| `ABCD_PLAN.md` | The plan. |
| `config.py` ("A/B/C/D" section) | The settings that are new. |
| `src/abcd_alpha.py` | Signal and sizing of the models. |
| `src/ag_metrics.py` (end) | Annual P&L, monthly distribution, market-by-market table. |
| `run_abcd.py` | Runs everything above; writes `results/ABCD_REPORT.md`. |
| `tests/test_abcd.py` | Tests on fake data. |

```bash
python run_abcd.py                # development + validation only
python run_abcd.py --test         # locked: needs the test data and the team's go-ahead
```

### Results (first and only run, 2026-10-03)

Development + validation, 2011-06 to 2019-12. The full tables are in
`results/ABCD_REPORT.md` (not committed; `python run_abcd.py` rebuilds it).
The first six columns are the whole run. "OOS Sharpe" is the validation
period, 2017 to 2019: **the test and final periods are still locked.**

| Model | CAGR | Vol | Sharpe | Max DD | Trades | Costs | Development Sharpe | OOS Sharpe (validation) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Trend | -0.4% | 7.9% | -0.01 | 23.1% | 261 | $334,225 | 0.25 | -0.58 |
| B Trend + InvVol | 2.1% | 12.7% | 0.23 | 28.4% | 261 | $831,788 | 0.44 | -0.14 |
| C Trend + Covariance | 3.1% | 10.5% | 0.34 | 23.9% | 261 | $821,545 | 0.56 | -0.06 |
| D Trend + ENSO + Covariance | 4.2% | 10.6% | 0.44 | 20.2% | 264 | $863,054 | 0.60 | 0.15 |

Every model's 95% interval for the Sharpe ratio includes zero (D, whole
run: -0.16 to 1.04).

What each layer adds, as a difference in Sharpe ratio with its 95% interval:

| Step | Whole run | Development | Validation |
|---|---|---|---|
| Inverse volatility (B - A) | +0.24 (-0.04 to 0.51) | +0.19 (-0.15 to 0.53) | +0.44 (0.00 to 0.88) |
| Covariance (C - B) | +0.11 (0.01 to 0.22) | +0.12 (-0.01 to 0.26) | +0.08 (-0.05 to 0.24) |
| ENSO (D - C) | +0.10 (-0.05 to 0.25) | +0.04 (-0.13 to 0.22) | +0.22 (-0.02 to 0.47) |
| ENSO as a multiplier (D_mult - C) | +0.04 (-0.03 to 0.11) | +0.01 (-0.08 to 0.10) | +0.11 (-0.01 to 0.23) |
| Diversification (A on 11 markets - A on grains) | +0.52 (-0.11 to 1.11) | +0.63 (-0.12 to 1.41) | +0.29 (-0.57 to 1.09) |
| Shorting, Model C (long/short - long-only) | -0.10 (-0.61 to 0.39) | +0.18 (-0.47 to 0.83) | -0.62 (-1.40 to 0.09) |
| Grains alone: trend + ENSO - trend | +0.23 (-0.05 to 0.51) | +0.05 (-0.28 to 0.36) | +0.57 (0.14 to 1.09) |

### Conclusion

1. **Does cross-asset trend work?** Not shown. Plain equal-notional trend
   (A) broke even over the whole run and lost in validation. With
   volatility sizing the models made money in development (Sharpe 0.44 to
   0.60) and roughly nothing in validation (-0.14 to 0.15). In every model
   the profit came from being long stocks, short natural gas and long
   bonds; all three grains lost money in all four models.
2. **Does volatility scaling help?** Probably. B beat A in both periods
   (+0.19, +0.44), and the interval only just includes zero. But B ignores
   correlations, so it ran at 12.7% volatility against a 10% target and
   had the deepest drawdown (28.4%).
3. **Does covariance help?** Yes, modestly and consistently. C holds B's
   positions and only rescales them, and that brought realized volatility
   to 10.5%, cut the drawdown from 28.4% to 23.9% and raised the Sharpe
   ratio by about 0.1 in both periods, at the same cost. Changing the
   covariance half-life to 40 or 90 days changes nothing (0.45, 0.43).
4. **Does ENSO add incremental information?** Suggestive, not established.
   D beat C by +0.10 over the whole run with an interval that includes
   zero; almost all of it came in validation. In the grains D beat C in
   all five La Nina stretches (+$1.18M, of which $0.99M in two of them:
   early 2012 and 2017-12 to 2018-05), and in three of six El Nino
   stretches (-$0.13M in total), so the soybean El Nino rule added
   nothing. There were only 227 La Nina days. Two further cautions: the
   grain trend lost money throughout, so any rule that closes grain
   positions would have helped; and the multiplier form, which never
   closes a position, gained less than half as much (+0.04).
5. **Which model should be carried forward?** C, as the simplest model
   whose every layer earned its place in both periods, with D alongside it
   as the pre-specified challenger. This is not a choice by Sharpe ratio:
   D's is higher, but its edge over C cannot be told apart from chance.
   Nothing is changed in response to these results: the grains stay in,
   the 252-day lookback stays, shorting stays (long-only C did as well or
   better here, 0.45 against 0.34, but 2011 to 2019 was a rising market
   for stocks and bonds, and the difference is well inside its interval).
6. **What should be tested next on untouched data?** Run A to D exactly as
   they are on the locked test period (2020-01 to 2024-09,
   `python run_abcd.py --test`). It contains the three-year La Nina of
   2020 to 2023 and the El Nino of 2023 to 2024, which is the independent
   ENSO variation this sample lacks. The questions, fixed now: is C's
   Sharpe ratio positive; is C - B still positive; is D - C positive, and
   does it come from the grains in La Nina; does the 4 bps run stay
   positive. Then one run of the final period.

Other observations:

- **Robustness of D** (whole-run Sharpe 0.44): 126-day trend 0.35; 189-day
  0.27; half-life 40 and 90 days 0.45 and 0.43; 8% and 12% targets 0.44;
  every two weeks 0.36; 4 bps 0.37. All positive and in the same direction,
  none distinguishable from zero. The result is most sensitive to the trend
  lookback.
- **Compared with the earlier M4** (`run_xa.py`: +63.2%, Sharpe 0.575 over
  the same days): D made +42.8% at 0.44. The plain `sign` signal with equal
  standalone risk did worse here than M4's signal-sized risk budgets. Both
  are reported; neither is selected on that basis.
- **Costs** took about 9 points of D's 52.3% gross return ($863k), a little
  over half of it rolls.
- **Win rate**: about one position in three made money in every model
  (31.8% to 34.1%), as expected of trend following.
