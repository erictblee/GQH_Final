# Cross-asset signal-weighted risk budgeting: specification

Written 2026-10-03, **before the first backtest of this strategy**. It
implements the team's revision of the same date, made after the ENSO grain
strategy (ENSO_SPEC.md) lost money in-sample. All numbers are in the
"Cross-asset" section of `config.py`. Later changes go in the README.

This is a third strategy. HYPOTHESIS.md (ES/ZN/GC risk parity) and
ENSO_SPEC.md (long-or-cash grains) and their results are unchanged.

## Research question

> Can observable, economically motivated signals generate directional
> forecasts across liquid futures, and can a simple covariance-aware
> risk-budgeting framework improve the portfolio built from those forecasts?

ENSO is now one incremental-information test (on the grains), not the
foundation.

## What changed from ENSO_SPEC.md, and why (the team's revision)

| Change | Reason given in the revision |
|---|---|
| Universe: 11 futures in 5 asset groups instead of 3 grains | Three related grains give a risk allocator almost nothing to diversify. |
| Long **and short** instead of long-or-cash | A trend signal that cannot short throws away half its information. |
| Signal: 12-month trend for every asset; ENSO only added to the grains | ENSO should not be the universal signal. |
| Covariance: EWMA with a 60-day half-life, no shrinkage parameter | One arbitrary parameter fewer; test covariance against plain inverse volatility. |
| Four models instead of seven ablations | Each layer gets one clean on/off comparison. |
| History from 2010 instead of 2021 | One ENSO cycle cannot test a climate hypothesis. |
| A transaction ledger | The kit's HTML report shows "W0/L0" and "$0.00 trade P&L": its trade fields are not filled in. |

Things the revision says **not** to do, and which are therefore not done:
no switch to a shorter momentum because it looked better; no tuning of the
ENSO threshold; no extra covariance parameters; no optimizer; no choosing
assets by past performance.

## Universe

Chosen for liquidity and for being the standard contract of its kind, not
from any backtest:

| Group | Futures |
|---|---|
| Equity indices | ES (S&P 500), NQ (Nasdaq 100) |
| Government bonds | ZN (10-year Treasury), ZB (30-year Treasury) |
| Energy | CL (crude oil), NG (natural gas) |
| Metals | GC (gold), HG (copper) |
| Agriculture | ZC (corn), ZS (soybeans), ZW (wheat) |

Contract months held: ES, NQ, ZN, ZB quarterly; GC even months; HG
H K N U Z; CL and NG every month; grains every listed month.

**Roll rule** (fixed, same family as before): hold the nearest eligible
contract and roll 5 business days before the anchor. Anchor = expiration
(third Friday) for ES and NQ; first notice day (last business day of the
month before delivery) for ZN, ZB, GC, HG and the grains; for CL and NG the
last business day **two** months before delivery, because they stop trading
before the month before delivery ends.

## Data and periods

Databento `GLBX.MDP3` settlements, from its first day (2010-06-06). The
revision asked for 2000 onward "if the data is available"; it is not, in
this source. NOAA Niño-3.4 as in ENSO_SPEC.md (usable from the 15th of the
following month; today's values, not vintages).

| Period | Dates | Status |
|---|---|---|
| Warm-up | first 253 trading days | no trading |
| Development | mid 2011 to 2016-12-31 | run now |
| Validation | 2017-01-01 to 2019-12-31 | run now |
| Test | 2020-01-01 to 2024-09-30 | **locked**: `--test`, on the team's go-ahead |
| Final | 2024-10-01 to 2026-09-30 | **locked**: `--final`, not downloaded |

For the five new markets and for grains before 2021-10, data after
2019-12-31 is **not downloaded** until the test run.

**Disclosure.** The test period is not pristine. We have already seen (a)
the grain strategy's results for 2021-12 to 2024-09, which motivated this
revision, and (b) ES/ZN/GC risk-parity results for 2010 to 2024-09. The
revision's design choices (go long/short, widen the universe) were made
knowing that grains fell in 2022-2024. Only the final period is untouched.

## Signal

For every asset, each week:

1. **Trend.** `M_i` = compounded return of the held contracts over the last
   252 trading days (roll gaps excluded).
2. **Standardize.** `A_i = M_i / (σ_i × sqrt(252/252))`, where `σ_i` is the
   asset's annualized EWMA volatility. `A_i = 1` is a one-standard-deviation
   rise over the year.
3. **ENSO (grains only, models 2 and 4).** `A_i = A_i + β_i`, with `β_i`
   from the same table as ENSO_SPEC.md:

   | ENSO effect | El Niño | Neutral | La Niña |
   |---|---|---|---|
   | Corn | 0 | 0 | +1 |
   | Soybeans | −1 | 0 | +1 |
   | Wheat | 0 | 0 | +1 |

   ENSO state: z-score over 30 years, threshold ±0.5, unchanged.
4. **Direction.** `A_i > 0` long, `A_i < 0` short. `|A_i| < 0.25`: no position.

`A_i` is a standardized directional attractiveness score, not an expected
return.

**One departure from the revision's formula.** The revision's Step 1 writes
`z_i = (A_i − mean) / std` across assets. Subtracting the cross-asset mean
would short an asset with a positive trend whenever the others trend more,
which contradicts the revision's own rule "positive → long, negative →
short" (its sections 6 and 7). We follow the rule: **the mean is not
subtracted.** Dividing by the cross-asset standard deviation would cancel
out in the next step, so it is omitted too.

## Portfolio

1. **Signed risk budget:** `b_i = A_i / Σ|A_j|` over the active assets. Its
   sign is the direction, its size the share of portfolio risk the asset
   should carry. Shares above 70% are cut to 70% and the excess is spread
   over the others.
2. **Risk:** EWMA covariance of daily returns, half-life 60 trading days,
   over the last 252 days, annualized by 252. No shrinkage.
3. **Weights:** find weights with those signs whose risk contributions are
   `|b_i|` (the Maillard-Roncalli-Teiletche log-barrier method already used,
   applied after flipping the sign of each short asset's returns).
4. **Volatility target:** scale all weights so predicted volatility is 10%.
5. **Limits:** no asset above 2× equity; longs plus shorts at most 4×
   equity; whole contracts; each limit is recorded when active. Margin
   limit as before (skipped until CME margins are entered).
6. **"Without covariance"** (models 1 and 2) uses the same steps with every
   correlation set to zero, which gives `w_i ∝ sign × sqrt(|b_i|) / σ_i`:
   inverse volatility, weighted by signal strength. This way models 1 → 3
   and 2 → 4 differ in exactly one thing.
7. **Turnover limit: not implemented.** The revision lists one; no number
   was given, and adding one would be a new free parameter. Turnover is
   held down by weekly rebalancing and is reported. If the team wants a
   hard limit it must name the number before the test run.

Trading, costs and accounting are those of ENSO_SPEC.md Part 2, items 11 to
13: $10M, weekly decision at the first trading day's settlement, fill at
the next settlement, 2 bps per side, rolls charged on both legs. No
drawdown brake, no rebalance band.

## Models

| Model | ENSO on grains | Covariance | Question |
|---|---|---|---|
| 1 Trend | no | no | Does the basic alpha work? |
| 2 Trend + ENSO | yes | no | Does ENSO add information? |
| 3 Trend + covariance | no | yes | Does covariance improve construction? |
| 4 Trend + ENSO + covariance | yes | yes | Does the complete architecture add value? |

Reported with 95% stationary-bootstrap intervals for each Sharpe ratio and
for the differences M2−M1, M4−M3, M3−M1, M4−M2 and M4−M1.

**Robustness of model 4** (reported, never selected from): 126-day trend;
4 bps cost; fractional contracts.

## Reporting

`run_xa.py` writes, per run: the daily table (alpha, budgets, target and
actual risk shares, target and actual contracts, leverage, predicted and
actual volatility, costs, P&L per asset); a **transaction ledger** (one row
per fill: date, asset, contract, kind = rebalance / roll_close / roll_open,
contracts, price, notional, cost); and for model 4 a table of **position
episodes** (each unbroken long or short stretch in one asset, with its net
P&L), from which wins and losses are counted.

## What would count as failure

As in ENSO_SPEC.md section 30, applied to the four models: trend has no
edge (model 1); ENSO adds nothing (M2−M1 and M4−M3 indistinguishable from
zero or negative); covariance adds nothing after costs (M3−M1, M4−M2);
results hinge on the lookback, vanish at 4 bps, or come from a handful of
days; or the edge disappears in the test or final period.
