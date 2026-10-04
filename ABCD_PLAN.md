# Research Hypothesis: Cross-Asset Trend + Risk Budgeting

## 1. Research Objective

Test whether a diversified, long/short cross-asset futures strategy can generate robust risk-adjusted returns by combining:

1. A common directional trend/momentum signal.
2. Inverse-volatility position sizing.
3. Covariance-aware portfolio volatility scaling.
4. An optional, asset-specific ENSO overlay for agricultural futures.
5. A fixed approximately 10% annualized portfolio volatility target.
6. Realistic transaction costs and deterministic futures rolls.

The primary question is:

> Does covariance-aware risk management and economically motivated signal information improve a simple cross-asset trend strategy out of sample?

Do NOT optimize parameters to maximize the existing backtest. The purpose of this iteration is attribution and robustness, not finding the best-performing specification.

---

# 2. Universe

Use the currently tested 11-market universe:

* ES — S&P 500 futures
* NQ — Nasdaq futures
* ZN — 10-Year Treasury futures
* ZB — 30-Year Treasury futures
* CL — Crude Oil futures
* NG — Natural Gas futures
* GC — Gold futures
* HG — Copper futures
* ZC — Corn futures
* ZS — Soybean futures
* ZW — Wheat futures

Use the same historical data source, continuous-contract methodology, settlement-price convention, and roll methodology as the existing backtest unless there is a documented data-quality problem.

Do not expand the universe further in this iteration.

---

# 3. Core Signal

The baseline signal should be a simple directional trend/momentum signal.

For each asset i:

```
trend_i,t = sign(price_t / price_{t-L} - 1)
```

where L is the already-specified momentum lookback from the existing strategy.

Long/short is allowed:

* positive trend -> long
* negative trend -> short
* zero/insufficient signal -> no position

Do not change the momentum lookback merely because another lookback performs better in hindsight.

If multiple lookbacks are already implemented, report them separately rather than selecting the winner after seeing results.

---

# 4. ENSO Overlay

ENSO is NOT a universal signal.

Apply ENSO only to agricultural futures:

* ZC
* ZS
* ZW

For non-agricultural markets:

```
final_signal = trend_signal
```

For agricultural markets, test whether ENSO contains incremental information beyond trend.

The ENSO implementation must use only information that was actually available before the trading decision.

Do not use revised/future ENSO information.

Do not simply assume that La Niña means long agriculture. The historical relationship must be measured.

The preferred implementation is an overlay/tilt on the directional trend signal rather than a hard universal gate.

For example:

```
agriculture_score =
    trend_score * ENSO_multiplier
```

where the multiplier is determined ex ante and is not optimized on the full sample.

If the current code uses a different ENSO formulation, preserve it for the primary replication and then test the overlay formulation separately.

---

# 5. Portfolio Construction

The portfolio construction should deliberately progress from simple to complex.

## Model A — Baseline Cross-Asset Trend

Directional trend only.

Use simple equal notional or equal signal exposure.

No covariance optimization.

Purpose:

Determine whether the diversified universe itself fixes the weaknesses of the original agriculture-only strategy.

---

## Model B — Trend + Inverse Volatility

For each active asset:

```
raw_weight_i = signal_i / volatility_i
```

where volatility is estimated only from information available at time t.

Normalize the resulting portfolio.

Purpose:

Determine whether volatility normalization improves the baseline.

---

## Model C — Trend + Inverse Volatility + Covariance Scaling

Use the same directional signal and inverse-volatility positions as Model B.

Estimate the covariance matrix from historical returns.

Use covariance ONLY to determine aggregate portfolio risk and scale the portfolio toward the target volatility.

Do NOT initially use a complicated full mean-variance optimizer.

For raw position vector q:

```
portfolio_vol = sqrt(q' Sigma q)
```

Then scale:

```
scale = target_vol / portfolio_vol

final_position = q * scale
```

with:

```
target_vol = 10% annualized
```

This gives us covariance-aware portfolio risk control without introducing an unstable optimizer.

---

## Model D — Trend + ENSO + Covariance

This is the current primary hypothesis.

For non-agricultural markets:

```
signal_i = trend_i
```

For agricultural markets:

```
signal_i = trend_i adjusted by the pre-specified ENSO overlay
```

Then:

```
q_i = signal_i / volatility_i
```

Calculate portfolio volatility using the covariance matrix:

```
sigma_p = sqrt(q' Sigma q)
```

Scale all positions:

```
q_final = q * (0.10 / sigma_p)
```

Apply position and leverage constraints after scaling.

This should be called:

**Cross-Asset Trend + ENSO + Covariance**

Do not call the signal an "expected return forecast" unless it has been explicitly calibrated into expected returns using a proper walk-forward procedure.

---

# 6. Covariance Estimation

Use a rolling historical covariance estimate.

The covariance estimate must be strictly point-in-time.

Start with a straightforward rolling/EWMA estimator.

Do NOT introduce multiple covariance models simultaneously.

The first comparison should simply be:

1. No covariance
2. Rolling/EWMA covariance

If covariance produces little improvement, that is an important result.

With only 11 markets, covariance estimation is manageable, but estimation error still matters.

---

# 7. Risk Target

Target:

```
10% annualized portfolio volatility
```

The target is a portfolio-level risk constraint.

It does NOT imply that every asset must contribute equal risk.

Do not force equal risk contribution unless explicitly testing risk parity as a separate model.

The central question is whether directional information should determine how the portfolio's risk budget is distributed.

---

# 8. Position Constraints

Use conservative, transparent constraints.

At minimum:

* Maximum position/risk contribution per asset.
* Maximum gross portfolio exposure.
* No unlimited leverage.
* No single market allowed to dominate the portfolio.
* Optional no-trade band to reduce unnecessary turnover.

All constraints must be fixed before evaluating the final OOS period.

Do not optimize the constraints to maximize Sharpe.

---

# 9. Rebalancing

Use the existing rebalance frequency unless there is a compelling implementation reason to change it.

If comparing frequencies, treat each frequency as a separate pre-specified model.

Do not select the frequency based on full-sample performance.

---

# 10. Transaction Costs

Retain realistic transaction costs.

Include:

* Entry trades
* Exit trades
* Position changes
* Contract rolls

Use the same cost assumptions as the existing backtest for comparability.

Report:

1. Gross performance
2. Net performance
3. Total trading costs
4. Turnover/trade count

The previous agriculture-only strategy showed that transaction costs were not the primary explanation for failure, so do not spend excessive effort optimizing around tiny cost differences.

---

# 11. Required Experiments

Run the following four models on exactly the same universe, dates, data, execution rules, and cost assumptions:

### A. Cross-Asset Trend

Trend only.

### B. Cross-Asset Trend + Inverse Vol

Trend with volatility-normalized sizing.

### C. Cross-Asset Trend + Covariance

Trend + inverse volatility + covariance-based portfolio scaling.

### D. Cross-Asset Trend + ENSO + Covariance

Full proposed model.

The comparison is the core experiment.

---

# 12. Required Performance Statistics

For every model report:

* CAGR
* Total return
* Annualized volatility
* Sharpe ratio
* Sortino ratio if available
* Maximum drawdown
* Calmar ratio if available
* Total P&L
* Annual P&L
* Monthly return distribution
* Number of trades
* Turnover
* Trading costs
* Gross vs net return
* Percentage of profitable months
* Percentage of profitable trades

Also report performance by individual futures contract:

* Total P&L
* Contribution to total P&L
* Annualized return
* Annualized volatility
* Sharpe
* Maximum drawdown
* Number of trades
* Long vs short contribution

---

# 13. Attribution Tests

The most important question is not simply whether Model D makes money.

Determine WHY.

Answer:

### Does diversification help?

Compare Model A on:

* agriculture only
* full 11-market universe

### Does inverse volatility help?

Compare A vs B.

### Does covariance help?

Compare B vs C.

### Does ENSO help?

Compare C vs D.

### Does long/short help?

If the old agriculture strategy was long-only, explicitly compare long-only vs long/short where feasible.

---

# 14. ENSO-Specific Test

For agriculture only, separately evaluate:

1. Trend without ENSO.
2. ENSO overlay without trend.
3. Trend + ENSO.

Do not claim ENSO works simply because the full strategy performs better.

The incremental contribution of ENSO must be measured.

Report:

```
ΔSharpe
ΔCAGR
ΔMaxDD
ΔP&L
Δturnover
```

Most importantly, evaluate ENSO across multiple ENSO regimes/cycles.

The previous sample contained too little independent ENSO variation to make a strong causal claim.

---

# 15. Time-Series Validation

The backtest must be split into:

1. Development/training period
2. Validation period
3. Final untouched out-of-sample period

Any parameter that is estimated from historical data must be estimated using only information available at that point in time.

Use walk-forward estimation where appropriate.

Do not use the full sample to estimate:

* momentum parameters
* volatility parameters
* covariance parameters
* ENSO coefficients
* signal scaling
* position limits
* optimizer parameters

---

# 16. Robustness Requirements

After the primary comparison, perform modest robustness tests.

Do NOT conduct an enormous parameter sweep.

Test only economically reasonable perturbations such as:

* nearby momentum lookback
* nearby covariance lookback
* slightly different volatility target
* modest transaction-cost increase
* slightly different rebalance frequency

The purpose is to determine whether the strategy is a narrow parameter accident.

A good result should remain directionally similar across reasonable perturbations.

---

# 17. Important Anti-Overfitting Rule

Do not modify the strategy because an individual asset or parameter performs poorly in the current sample.

The following are NOT valid reasons to change the model:

* "Wheat loses money."
* "NG is volatile."
* "ES contributes too much."
* "This lookback has the highest Sharpe."
* "Removing this asset improves the backtest."

Any modification must be justified economically and tested prospectively.

---

# 18. Current Hypothesis

The working hypothesis is:

> Diversification across sufficiently different futures markets allows a simple trend signal to produce more robust opportunities than the original agriculture-only strategy. Volatility normalization should improve comparability across markets, while covariance-aware scaling should control portfolio-level risk when markets become correlated. ENSO may provide incremental predictive information for agricultural markets, but it should be treated as an optional asset-specific overlay rather than a universal regime filter. The expected improvement should therefore come primarily from the broader cross-asset opportunity set and robust portfolio construction, with ENSO required to demonstrate incremental out-of-sample value rather than being assumed useful.

---

# 19. Current Evidence

The previous agriculture-only strategy was unsuccessful:

* approximately -16% cumulative return
* negative Sharpe
* approximately 29% maximum drawdown
* momentum itself was weak
* wheat was a major detractor
* ENSO's apparent benefit was heavily concentrated in one La Niña/Russia-Ukraine episode
* covariance had only a small effect

The updated 11-market model is materially better:

* approximately +63.2% cumulative return
* approximately 0.575 Sharpe
* approximately 20.7% maximum drawdown
* 277 trades
* approximately 33.6% winning trades

Treat this improvement as evidence supporting the new hypothesis, NOT as confirmation.

The next task is attribution.

---

# 20. Success Criterion

Do not define success as "highest Sharpe."

The strongest result would be:

1. Cross-asset trend is profitable.
2. Inverse-volatility sizing improves robustness.
3. Covariance scaling reduces drawdowns or stabilizes volatility without destroying returns.
4. ENSO provides measurable incremental value specifically in agriculture.
5. Results survive reasonable parameter perturbations.
6. Results survive realistic transaction costs.
7. Results remain positive or economically meaningful in untouched OOS data.

If covariance or ENSO does not add value, remove it.

The research goal is to find the simplest specification that survives out-of-sample testing—not the most complicated specification that performs well in-sample.

---

# 21. Output Format

Produce a final comparison table:

| Model                       | CAGR | Vol | Sharpe | Max DD | Trades | Costs | OOS Sharpe |
| --------------------------- | ---: | --: | -----: | -----: | -----: | ----: | ---------: |
| A Trend                     |      |     |        |        |        |       |            |
| B Trend + InvVol            |      |     |        |        |        |       |            |
| C Trend + Covariance        |      |     |        |        |        |       |            |
| D Trend + ENSO + Covariance |      |     |        |        |        |       |            |

Then provide:

### Conclusion

Answer directly:

1. Does cross-asset trend work?
2. Does volatility scaling help?
3. Does covariance help?
4. Does ENSO add incremental information?
5. Which model should be carried forward?
6. What should be tested next on untouched OOS data?

Do not select the final model based solely on the highest in-sample Sharpe.
