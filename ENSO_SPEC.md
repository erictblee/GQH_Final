# ENSO strategy on corn, soybeans and wheat: specification

This is a second strategy, separate from the ES/ZN/GC risk parity of
HYPOTHESIS.md, which is unchanged.

- **Part 1** is the team's specification, as given on 2026-10-03.
- **Part 2** lists every choice Part 1 leaves open. Part 2 was written on
  2026-10-03 **before the first backtest on real data**, and all its numbers
  are in `config.py`. Later changes go in the README's Deviations log.

---

# Part 1: Strategy Specification

## 1. Objective

Test whether a climate-informed momentum signal based on the NOAA Niño-3.4 ENSO index adds predictive value to agricultural futures momentum, and whether that information can be converted into improved portfolio construction through an alpha-tilted, covariance-aware risk-budgeting framework.

The strategy trades three CME agricultural futures:

* Corn
* Soybeans
* Wheat

The strategy is long-or-cash unless explicitly extended later to permit short positions.

The research must separate three questions:

1. Does momentum predict future returns?
2. Does ENSO information improve the momentum signal?
3. Does allocating portfolio risk according to the resulting signal improve performance relative to equal-risk or volatility-only allocation?

## 2. Core Architecture

The strategy has five layers:

### Layer 1 — Alpha Signal

For each commodity:

`Momentum + ENSO -> expected attractiveness score`

The score represents the estimated relative attractiveness of holding the commodity.

### Layer 2 — Observable Risk

Estimate:

* Individual commodity volatility
* Pairwise correlations
* Covariance matrix

using only historical futures returns available at the decision time.

### Layer 3 — Alpha-Tilted Risk Budget

Convert alpha scores into desired risk-contribution shares.

The strategy does NOT directly run an unconstrained mean-variance optimizer.

Instead:

`Alpha scores -> risk-budget shares -> covariance-aware portfolio weights`

### Layer 4 — Portfolio Volatility Target

Scale the resulting portfolio to a fixed annualized volatility target.

Initial target:

`10% annualized volatility`

The 10% target applies to the entire portfolio, not individually to each commodity.

### Layer 5 — Futures Execution

Convert portfolio weights into actual futures contracts while accounting for:

* Contract multipliers
* Futures prices
* Contract rolls
* Transaction costs
* Available capital
* Margin requirements

## 3. Trading Universe

Initial universe:

* CME Corn futures (ZC)
* CME Soybean futures (ZS)
* CME Wheat futures (ZW)

Use a continuous-contract representation for signal/risk calculations, but execute using the currently active tradeable contract.

The continuous series must be constructed without look-ahead.

Contract rolls must be determined using a predefined rule, such as:

* liquidity-based roll, or
* fixed days before expiration.

The roll rule must be fixed before the backtest.

## 4. Market Data

Required market data:

1. Daily futures settlement prices
2. Futures contract specifications
3. Contract expiration/roll information
4. NOAA Niño-3.4 ENSO observations
5. NOAA publication/release timestamps or vintages where available

All data must be point-in-time.

A data value may only influence a trading decision if it was publicly available before that decision.

## 5. Return Construction

Calculate daily futures returns from settlement prices.

Preferred return:

`r_i,t = P_i,t / P_i,t-1 - 1`

For futures research, also maintain a return series that correctly accounts for contract rolls.

Do not treat a mechanical price jump caused solely by rolling contracts as an economic return.

If using back-adjusted continuous prices, document the adjustment methodology and ensure it is consistent with the execution/roll return calculation.

## 6. Momentum Signal

For each commodity, calculate a predetermined momentum measure.

Initial specification:

`M_i,t = P_i,t / P_i,t-L - 1`

where:

* `L` = momentum lookback
* initial value: 60 trading days

Alternative lookbacks may be tested only as predefined robustness tests.

The momentum signal must use information available at the previous trading day's close.

Do not optimize the lookback on the complete sample.

## 7. ENSO Signal

Use the NOAA Niño-3.4 index.

The strategy must use the ENSO observation that was actually available to an investor at the decision date.

No future ENSO observations may be substituted into historical decisions.

Initial ENSO representation:

`E_t = standardized Niño-3.4 value`

Use a predefined transformation rather than allowing the optimizer to discover arbitrary thresholds.

Initial approach:

* Calculate a rolling historical z-score of Niño-3.4.
* Use only observations available at time `t`.
* Define a fixed threshold for identifying materially positive/negative ENSO states.

Example:

`ENSO_state = +1 if z >= threshold`
`ENSO_state = -1 if z <= -threshold`
`ENSO_state = 0 otherwise`

The exact threshold must be selected before the final out-of-sample test.

If the friend's original ENSO strategy already has a specific climate filter, preserve that specification rather than replacing it merely to improve performance.

## 8. Combined Alpha Score

For each commodity:

`A_i,t = f(M_i,t, ENSO_t)`

The preferred initial implementation should be simple and interpretable.

Rather than immediately attempting to estimate a continuous expected return, construct a normalized attractiveness score.

Example structure:

`A_i,t = w_M * normalized(M_i,t) + w_E * ENSO_effect_i,t`

where:

* `w_M` and `w_E` are fixed ex-ante coefficients
* momentum is normalized cross-sectionally or historically
* ENSO effect is determined from the predefined climate rule

The ENSO component should be commodity-specific if the research hypothesis predicts different effects for corn, soybeans, and wheat.

Do NOT initially treat the raw score as a literal expected return in dollars or percentage points.

It is an alpha score.

If later evidence supports calibration, estimate:

`E[r_i,t+H | information_t]`

using a walk-forward training process.

## 9. Long-or-Cash Rule

The initial strategy should avoid shorting.

For each commodity:

`positive_alpha_i = max(A_i,t, 0)`

If all alpha scores are non-positive:

`portfolio exposure = 0`

Otherwise, allocate risk only among commodities with positive alpha scores.

This prevents the portfolio construction layer from turning a weak predictive signal into a forced position.

## 10. Volatility Estimation

Estimate each commodity's realized volatility from historical daily returns.

Initial specification:

`σ_i,t = annualized standard deviation of last 60 trading-day returns`

Annualization:

`σ_annual = σ_daily * sqrt(252)`

Only information available before the portfolio decision may be used.

Do not use implied volatility or other unobservable forward-looking volatility in the initial version.

## 11. Covariance Estimation

Estimate covariance using the same rolling 60-day window.

For commodities `i` and `j`:

`Cov_i,j,t = Cov(r_i, r_j)`

Annualize consistently:

`Σ_t = 252 * Cov_daily`

The covariance matrix therefore contains:

* Corn variance
* Soybean variance
* Wheat variance
* Corn/Soy correlation
* Corn/Wheat correlation
* Soy/Wheat correlation

No forecasted future covariance is required.

The strategy simply asks:

"Given the recent observed behavior of these assets, how much portfolio risk would these positions create together?"

## 12. Covariance Robustness

Do not rely blindly on the raw 60-day covariance matrix.

Because correlations can be noisy, use a simple shrinkage/blending approach.

Initial specification:

`Σ_final = λ * Σ_rolling + (1-λ) * Σ_diagonal`

where:

* `Σ_rolling` = observed rolling covariance matrix
* `Σ_diagonal` = matrix containing individual variances but zero correlations
* `λ` = predetermined shrinkage coefficient

Initial:

`λ = 0.5`

This means the strategy uses 50% observed covariance information and 50% conservative zero-correlation covariance.

This is deliberately simple.

Robustness tests may evaluate:

* λ = 0
* λ = 0.5
* λ = 1

The main specification should be fixed before the final out-of-sample period.

## 13. Alpha-to-Risk-Budget Conversion

Do not use a full mean-variance optimizer.

Instead, convert positive alpha scores into desired risk-contribution budgets.

For each commodity:

`b_i = g(A_i)`

Initial transformation:

`b_i = exp(k * Z_i) / sum(exp(k * Z_j))`

where:

* `Z_i` = standardized positive alpha score
* `k` = alpha-tilt strength

Initial:

`k = 1`

If a commodity has non-positive alpha, its risk budget is zero.

Special case:

If all alpha scores are equal and positive, the risk budgets become approximately equal.

Thus the framework naturally connects to risk parity.

## 14. Risk-Budget Portfolio Construction

Find portfolio weights `w` such that each asset's contribution to total portfolio volatility approximately matches its desired risk budget.

Portfolio variance:

`σ_p² = w' Σ_final w`

Portfolio volatility:

`σ_p = sqrt(w' Σ_final w)`

Marginal contribution to risk:

`MRC_i = (Σ_final w)_i / σ_p`

Risk contribution:

`RC_i = w_i * MRC_i`

Risk contribution percentage:

`RC_i_pct = RC_i / σ_p`

The desired condition is:

`RC_i_pct ≈ b_i`

for all active assets.

Weights must satisfy:

`w_i >= 0`

and:

`sum(w_i) = 1`

during the risk-budget construction stage.

This is a risk-budgeting problem, NOT a return-maximization problem.

## 15. Portfolio Volatility Target

Once the relative portfolio weights have been determined, calculate their estimated volatility:

`σ_est = sqrt(w' Σ_final w)`

Scale all weights by:

`scale = target_vol / σ_est`

where:

`target_vol = 0.10`

Final portfolio weights:

`w_final = scale * w`

This permits leverage through futures.

The resulting estimated portfolio volatility should therefore be approximately 10%.

Important:

The strategy does NOT force every commodity to have 10% volatility.

It targets approximately 10% volatility for the combined portfolio.

## 16. Maximum Exposure Constraints

To prevent the covariance/risk-budget system from producing extreme futures exposure, impose simple limits.

Initial constraints:

* Maximum portfolio risk contribution per asset: 70%
* Maximum gross notional exposure: predefined multiple of capital
* Minimum tradable contract size
* Maximum contracts determined by available capital/margin

If constraints prevent the desired risk-budget solution, use the closest feasible solution and record that the constraint was active.

Do not silently alter the portfolio.

## 17. Rebalancing Frequency

Do not rebalance every day initially.

Initial specification:

`Rebalance weekly`

At each weekly rebalance:

1. Update momentum
2. Update ENSO information if a new observation is available
3. Update volatility
4. Update covariance
5. Calculate alpha scores
6. Calculate risk budgets
7. Calculate covariance-aware risk-budget weights
8. Apply 10% portfolio volatility scaling
9. Convert target weights into futures contracts
10. Execute trades

Between rebalances, hold positions constant except for contract-roll operations.

## 18. ENSO Update Frequency

NOAA ENSO information should only change when a new valid observation becomes available.

Do not interpolate future ENSO information backward.

If the monthly ENSO observation is unchanged between releases, the previous available value remains active.

## 19. Contract Execution

At each rebalance:

`target_dollar_exposure_i = w_final_i * portfolio_equity`

For futures:

`contract_notional_i = futures_price_i * contract_multiplier_i`

Target contracts:

`N_i = target_dollar_exposure_i / contract_notional_i`

Because futures contracts are indivisible:

`N_i = round_to_feasible_integer(N_i)`

Recalculate actual portfolio volatility after integer rounding.

Record both:

* theoretical continuous weights
* actual executed futures positions

## 20. Transaction Costs

Apply transaction costs to:

* Opening trades
* Closing trades
* Rebalancing trades
* Contract rolls

Initial assumption:

`2 basis points per side`

The cost must be applied to the appropriate notional traded.

Do not apply costs merely to portfolio returns.

Transaction cost:

`cost_t = traded_notional_t * 0.0002`

Run sensitivity tests using alternative realistic costs.

## 21. Roll Methodology

The roll rule must be deterministic and known before the backtest.

For example:

* roll from front contract to next contract when next contract becomes more liquid, or
* roll a fixed number of trading days before expiration.

The exact method should be selected before examining final performance.

Rolling should:

1. Close old contract
2. Open new contract
3. Apply transaction costs to both sides
4. Preserve intended economic exposure

Roll trades should not accidentally appear as alpha-driven rebalances.

## 22. Portfolio Accounting

Maintain:

* Cash
* Futures margin
* Unrealized P&L
* Realized P&L
* Transaction costs
* Roll costs
* Portfolio equity

Futures P&L:

`PnL_i,t = contracts_i * contract_multiplier_i * (settlement_t - settlement_t-1)`

Adjust correctly for contract changes during rolls.

Do not treat futures notional exposure as money invested.

## 23. Benchmark Strategies

At minimum, compare the proposed strategy against:

### Benchmark A — Momentum only

Same futures universe.

Momentum signal determines exposure.

No ENSO information.

### Benchmark B — Momentum + ENSO, simple sizing

Same alpha signal.

Use inverse-volatility sizing.

No covariance.

### Benchmark C — Momentum + ENSO, equal-risk

Same alpha signal.

Use covariance-aware equal-risk contribution.

No alpha tilt in risk budgets.

### Benchmark D — Proposed strategy

Momentum + ENSO.

Alpha-tilted risk budgets.

Rolling covariance.

10% portfolio volatility target.

This decomposition is critical.

It tells us whether performance comes from:

* momentum,
* ENSO,
* volatility scaling,
* covariance,
* or alpha-aware risk allocation.

## 24. Ablation Tests

Run controlled experiments where only one component changes.

Required:

1. Momentum only
2. ENSO only
3. Momentum + ENSO
4. Momentum + ENSO + volatility scaling
5. Momentum + ENSO + inverse volatility
6. Momentum + ENSO + covariance
7. Momentum + ENSO + alpha-tilted covariance risk budgeting

The goal is incremental attribution.

Do not judge the final strategy solely from its absolute Sharpe ratio.

## 25. Out-of-Sample Design

The most important requirement is strict temporal separation.

Recommended structure:

### Development period

Used to:

* establish signal definitions
* choose reasonable lookbacks
* estimate fixed coefficients
* develop the implementation

### Validation period

Used to test design choices without changing the core hypothesis.

### Final untouched OOS period

Completely locked before results are viewed.

No:

* parameter changes
* threshold changes
* lookback changes
* covariance changes
* signal changes
* benchmark changes

may be made after seeing final OOS performance.

## 26. Walk-Forward Calibration

If the alpha score is eventually converted into an estimated expected return, calibration must be walk-forward.

At time `t`:

`mu_hat_i,t = model(training data <= t-1)`

Never:

`mu_hat_i,t = model(full historical dataset)`

The model must not see future returns.

Initially, avoid continuous expected-return calibration unless necessary. The simpler alpha-score framework is preferred.

## 27. Parameter Philosophy

The strategy should minimize free parameters.

Parameters should be limited to:

* Momentum lookback
* ENSO threshold
* Volatility lookback
* Covariance lookback
* Covariance shrinkage coefficient
* Alpha-tilt strength
* Portfolio volatility target
* Rebalance frequency
* Position/exposure constraints
* Transaction costs

Do not optimize all parameters simultaneously.

Prefer economically motivated defaults.

## 28. Primary Research Hypothesis

The main hypothesis is:

> Climate information contained in the NOAA Niño-3.4 ENSO index provides incremental predictive information beyond price momentum in agricultural futures, and an alpha-tilted covariance-aware risk-budgeting framework can convert that information into superior portfolio-level risk-adjusted returns without relying on a highly parameterized optimizer.

## 29. Secondary Hypothesis

A secondary hypothesis is:

> Incorporating recent observed covariance improves portfolio construction relative to volatility-only sizing because the three agricultural futures are not independent and their correlations vary over time.

This hypothesis must be tested rather than assumed.

## 30. Main Failure Conditions

The strategy should be considered unsuccessful if:

1. ENSO provides no statistically or economically meaningful improvement over momentum alone.
2. Alpha tilting consistently worsens out-of-sample performance.
3. Covariance-aware allocation does not improve risk-adjusted results after transaction costs.
4. Performance depends heavily on one arbitrary parameter choice.
5. Performance disappears under reasonable transaction-cost assumptions.
6. The strategy generates excessive turnover.
7. The strategy's apparent edge disappears in the untouched OOS period.
8. Results are driven by a small number of extreme observations.
9. ENSO timing contains look-ahead bias.
10. The strategy requires continuously changing parameters to remain profitable.

## 31. Required Performance Metrics

Report at minimum:

* CAGR
* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Calmar ratio
* Worst month
* Worst year
* Average annual turnover
* Transaction costs
* Gross return
* Net return
* Percentage of time invested
* Average leverage
* Maximum leverage
* Number of trades
* Number of contract rolls
* Percentage of positive months
* Performance by commodity
* Performance by ENSO regime

Also report confidence intervals or statistical significance where appropriate.

## 32. Required Diagnostic Outputs

The backtest should produce time series for:

* Alpha score for each commodity
* ENSO state
* Momentum signal
* Estimated volatility
* Estimated correlations
* Covariance matrix
* Desired risk budget
* Actual risk contribution
* Target futures contracts
* Actual futures contracts
* Portfolio leverage
* Portfolio volatility
* Portfolio equity
* Drawdown
* Transaction costs

This makes it possible to determine whether the strategy is behaving as intended.

## 33. Required Invariants / Automated Tests

The implementation should contain tests ensuring:

### No look-ahead

Every signal at date `t` uses only data available before the trade.

### ENSO availability

A newly released ENSO observation cannot influence trades before its publication date.

### Risk budget

Active assets' actual risk contributions should approximately match their desired risk budgets before constraints.

### Volatility target

After continuous scaling:

`estimated portfolio volatility ≈ 10%`

### Non-negative exposure

No negative position unless shorting is explicitly enabled.

### Roll integrity

Rolling contracts must not create artificial P&L.

### Transaction costs

Every trade generates the correct cost.

### Integer contracts

Actual executed positions contain only feasible whole contracts.

### No future parameter fitting

Final OOS results cannot influence parameters used during the OOS period.

## 34. Conceptual Interpretation

The strategy intentionally separates three different questions:

### Alpha

"What appears likely to outperform?"

Answered by:

`Momentum + ENSO`

### Risk

"How risky is each asset, and how do these assets interact?"

Answered by:

`Rolling volatility + rolling covariance`

### Portfolio construction

"How much of the portfolio's risk should each asset receive?"

Answered by:

`Alpha-tilted risk budgets`

### Overall risk

"How much total risk should the portfolio take?"

Answered by:

`10% portfolio volatility target`

This separation is fundamental.

The system is therefore NOT saying:

"Every asset should receive equal risk."

Instead it says:

> "Give more of the portfolio's risk budget to assets with stronger predicted attractiveness, but account for their volatility and correlation with the other positions, then scale the entire portfolio to a controlled level of total risk."

## 35. Preferred Initial Configuration

Unless there is a strong reason to change them, the first implementation should use:

| Parameter                   |                         Initial value |
| --------------------------- | ------------------------------------: |
| Assets                      |                 Corn, Soybeans, Wheat |
| Momentum lookback           |                       60 trading days |
| Volatility lookback         |                       60 trading days |
| Covariance lookback         |                       60 trading days |
| Covariance shrinkage        |                                   50% |
| Alpha tilt                  |                    Moderate (`k = 1`) |
| Portfolio volatility target |                                   10% |
| Rebalance                   |                                Weekly |
| Position direction          |                          Long-or-cash |
| Transaction cost            |                        2 bps per side |
| ENSO frequency              |                Monthly, point-in-time |
| Leverage                    | Futures-based, subject to constraints |

These are starting specifications, not parameters to be optimized automatically.

## 36. Final Algorithm

At each weekly decision date `t`:

1. Load all futures prices available through `t-1`.
2. Load the latest NOAA Niño-3.4 observation actually available by `t-1`.
3. Calculate 60-day momentum for each commodity.
4. Calculate the ENSO component of each commodity's alpha signal.
5. Combine them into `A_i,t`.
6. Set non-positive alpha scores to zero.
7. Calculate 60-day annualized volatility.
8. Calculate 60-day covariance matrix.
9. Shrink covariance toward the diagonal matrix.
10. Convert positive alpha scores into risk-budget shares.
11. Solve for non-negative portfolio weights whose risk contributions approximate those shares.
12. Calculate estimated portfolio volatility.
13. Scale weights to a 10% annualized volatility target.
14. Apply exposure/leverage constraints.
15. Convert target dollar exposures into integer futures contracts.
16. Compare target positions with current positions.
17. Execute required trades.
18. Apply transaction costs.
19. Apply roll trades according to the predetermined roll rule.
20. Mark the portfolio to settlement each trading day.
21. Repeat at the next rebalance date.

The backtest must produce both gross and net performance.

The final untouched OOS period must be reported exactly as generated, without subsequent parameter adjustment.

---

# Part 2: Choices fixed before the first backtest (2026-10-03)

Part 1 leaves these open. Each was decided without looking at any backtest
result. **The team should read items 4 and 5 in particular: they are research
judgments, made here from published work because no "friend's original ENSO
filter" was available to copy.**

## Data and periods

1. **Five years of futures data in total, never more** (team instruction):
   2021-10-01 to 2026-09-30.

   | Period | Dates | Use |
   |---|---|---|
   | Development | 2021-10-01 to 2023-09-30 | build and check the code |
   | Validation | 2023-10-01 to 2024-09-30 | look at design choices |
   | Out-of-sample | 2024-10-01 to 2026-09-30 | locked; one final run |

   The in-sample end (2024-09-30) is the same date as in HYPOTHESIS.md. The
   out-of-sample data is not downloaded until the final run. The first 61
   trading days are a warm-up (no trading), so the first decision is in
   late December 2021.
2. **Futures data:** Databento `GLBX.MDP3`, `statistics` schema (official
   settlements and cleared volume), requested by contract name for outright
   contracts only. One file per period. The settlement record used per
   contract and day is chosen as in the main strategy (final, else rounded
   preliminary, else latest). Missing days are left missing ("market closed").
3. **Contracts and roll rule:** every listed delivery month (corn H K N U Z;
   soybeans F H K N Q U X; wheat H K N U Z). We hold the nearest contract and
   roll 5 business days before first notice day (the last business day of the
   month before the delivery month): the same rule the main strategy uses
   for ZN and GC. Contract months and last trading days follow CME's rule
   (business day before the 15th of the delivery month) instead of being
   downloaded. Prices are in cents per bushel; one contract is 5,000 bushels,
   so the multiplier is $50 per cent.

## ENSO

4. **Source and timing.** NOAA Climate Prediction Center monthly file
   `sstoi.indices`, Niño-3.4 anomaly (against the 1991-2020 average, which was
   already fixed before our first data day). The value for month M may be
   used **from the 15th of month M+1** (NOAA publishes in the first week or
   so; its ENSO bulletin appears on the second Thursday, the 14th at the
   latest). **Limitation:** NOAA does not publish old vintages of this file,
   so small later revisions to past values cannot be undone. Timing is
   point-in-time; the values are today's.
5. **Z-score, threshold and effect.**
   `z = (anomaly - mean) / standard deviation`, over the 360 monthly values
   (30 years) ending at that month. `z >= 0.5` is El Niño, `z <= -0.5` is
   La Niña, otherwise neutral. (0.5 mirrors NOAA's own ±0.5 °C convention;
   one standard deviation of the anomaly is a little under 1 °C, so our
   threshold is slightly easier to reach than NOAA's.)

   The effect on each price follows the yield results of Iizumi et al.
   (2014), "Impacts of El Niño Southern Oscillation on the global yields of
   major crops", *Nature Communications* 5:3712: La Niña years lower the
   world-average yield of maize, soybeans and wheat; El Niño years raise
   soybean yields and have a mixed effect on maize and wheat. Lower yield
   means less supply, which is bullish for the price.

   | ENSO effect | El Niño | Neutral | La Niña |
   |---|---|---|---|
   | Corn | 0 | 0 | +1 |
   | Soybeans | −1 | 0 | +1 |
   | Wheat | 0 | 0 | +1 |

## Alpha

6. **Momentum** is the compounded return of the held contract over the last
   60 trading days (built from roll-adjusted returns, so a roll gap is never
   counted). It is **normalized historically**: divided by
   `σ_i × sqrt(60/252)`, the size of a typical 60-day move for that
   commodity. A value of 1 is a one-standard-deviation rise.
7. **Alpha:** `A_i = 1.0 × normalized momentum_i + 1.0 × ENSO effect_i`.
   Both parts are in the same unit (about one standard deviation), so equal
   coefficients of 1 are used. `Z_i` in section 13 is the positive alpha
   score itself, since it is already standardized by construction.
   (Standardizing again across the two or three active assets would throw
   away the size of the scores.)

## Portfolio

8. **Risk-budget solver:** minimize `½ w'Σw − Σ b_i ln(w_i)`, the
   Maillard-Roncalli-Teiletche method the main strategy already uses for
   ERC, with budgets `b_i` in place of `1/N`. Solved on the active assets
   only, then scaled to `sum(w) = 1`, then to the 10% target.
9. **70% risk limit:** a budget above 70% is cut to 70% and the excess is
   given to the other active assets in proportion. With a single active
   asset the limit cannot be met (that asset carries all the risk and
   holding anything else would break long-or-cash); the position is kept
   and the limit is recorded as active.
10. **Gross notional limit:** 4× equity, as in the main strategy. **Margin
    limit:** margin posted ≤ 50% of equity, applied only once the team
    enters CME initial margins in `config.AG_INITIAL_MARGIN`; until then it
    is reported as skipped.
11. **Capital:** $10,000,000, as in the main strategy. No drawdown brake and
    no rebalance band (Part 1 has neither).

## Trading

12. **Weekly rebalance:** the decision is made at the settlement of the
    first trading day of each calendar week, using data through that
    settlement, and filled at the **next** trading day's settlement. (This
    is Part 1's "data through t−1, trade at t".) If a market is closed on
    the fill day the order waits for its next open day.
13. **Costs:** 2 bps of traded notional per side. A roll trades the old
    contract count out and the new count in, cost on both legs, and is
    reported separately from rebalancing.

## Runs (all reported; none selected from)

14. **Ablations** (section 24), each long-or-cash on positive alpha:

    | Run | Signal | Sizing |
    |---|---|---|
    | 1 | momentum | equal weights adding to 1, no volatility scaling |
    | 2 | ENSO | same |
    | 3 | momentum + ENSO | same |
    | 4 | momentum + ENSO | equal weights scaled to 10% volatility |
    | 5 = benchmark B | momentum + ENSO | inverse volatility, scaled to 10% ignoring correlations |
    | 6 = benchmark C | momentum + ENSO | equal risk contribution, scaled to 10% |
    | 7 = benchmark D | momentum + ENSO | alpha-tilted risk budgets, scaled to 10% |

    **Benchmark A** is run 7 with the ENSO part removed (momentum only, same
    tilted sizing), so that D − A isolates ENSO.
15. **Robustness** of run 7, one setting changed at a time: λ = 0 and 1;
    k = 2; momentum lookback 20 and 40 days (shorter, so that every run
    shares the same start date); cost 4 bps; fractional contracts.
16. **Statistics:** 95% stationary-bootstrap intervals (the main strategy's
    settings) for each Sharpe ratio and for the Sharpe differences D − A,
    3 − 1, 4 − 3, C − B and D − C.

## A warning written in advance

Three years in-sample and two out-of-sample is a short history, and it
contains roughly one ENSO cycle (a La Niña, an El Niño and neutral spells).
Whatever the numbers turn out to be, they cannot establish or reject an
ENSO effect with statistical confidence. The decomposition can show whether
the machinery works and which layer moves the result; it cannot prove the
hypothesis.
