# Hypothesis (committed before any backtest code exists)

## Edge source
Risk premium earned because of investors' leverage constraints (structural constraint), with position sizing based on volatility clustering.

## Universe
Three CME futures, daily data from Databento (dataset GLBX.MDP3):
- ES: E-mini S&P 500 (stocks). Contract months Mar, Jun, Sep, Dec.
- ZN: 10-Year T-Note (bonds). Contract months Mar, Jun, Sep, Dec.
- GC: Gold. We trade only the liquid even months (Feb, Apr, Jun, Aug, Oct, Dec).

The universe is fixed in advance. No contracts will be added or removed after seeing results.

## Behavior
1. **Stocks and bonds (leverage aversion).** Many investors cannot or will not borrow, so they overweight high-risk assets such as stocks to reach their return targets. Low-risk assets such as bonds then offer more return per unit of risk (Black 1972; Frazzini & Pedersen 2014; Asness, Frazzini & Pedersen 2012). Futures provide cheap leverage, so a portfolio with equal risk from each asset can collect this premium.
2. **Volatility management.** Volatility is persistent (Bollerslev 1986), but expected returns do not rise in proportion to it. Scaling exposure inversely to recent volatility therefore improves risk-adjusted returns (Moreira & Muir 2017).
3. **Gold (diversifier, not a premium source).** We do not assume gold earns a risk premium. Its long-run real return is weak and unreliable (Erb & Harvey 2013), so we assume its expected excess return is about zero. Gold is included because its returns are only weakly related to stocks and bonds, and it has often held its value during stock market stress (Baur & Lucey 2010). Risk parity's main weakness is that stocks and bonds can fall together when inflation drives markets, as in 2022. A third asset whose diversification does not depend on the stock-bond relationship should reduce drawdowns in those periods, at the cost of some expected return.

## Prior work and what we add
- Risk parity: Qian (2005).
- Equal risk contribution (ERC) using the full covariance matrix: Maillard, Roncalli & Teïletche (2010). Version B below is a standard ERC portfolio.
- Volatility-managed portfolios: Moreira & Muir (2017).

The core strategy is an implementation of these established methods, not a new strategy. What we add:
(a) a pre-registered test of whether correlation-aware sizing (B) protects better than correlation-blind sizing (A) in periods of positive stock-bond correlation;
(b) an explicit test, with a defined procedure, of whether gold earns its place through diversification rather than through its price rise.

## Disclosures about prior knowledge
- We chose this design knowing that risk parity suffered in 2022, so 2022 is not an independent test.
- The team also broadly knows how stocks, bonds and gold moved from 2024 to 2026, including gold's large rally, which overlaps the out-of-sample period. The holdout protects against tuning parameters on the test period. It does not protect against general knowledge of market history.

## Capital, weights and equity
- Starting capital: $10,000,000.
- A weight w(i,t) is notional exposure to asset i as a fraction of current equity. For example, w = 1.5 means futures notional equal to 1.5x current equity in that contract.
- Equity grows with the strategy's daily excess return (futures returns net of costs). T-bill interest on cash is excluded from both the equity curve and the Sharpe ratio, so all reported figures are excess returns.

## Prices, contracts and rolls
- **Prices:** official CME daily settlement prices from Databento's statistics data (settlement price records). Every use of "settlement" in this document and in the code refers to these prices. If settlement data cannot be obtained, we will switch to Databento daily closes everywhere, state the change in the note and README, and log it as a deviation from this plan.
- **Asynchronous settlement times:** ES, ZN and GC settle at different times of the day. This makes measured correlations between them slightly less reliable. We note this as a limitation.
- **Contract data:** we download every individual contract in the date range, plus Databento's instrument definitions (which give each contract's expiration). First notice day for ZN and GC is computed from CME's rule: the last business day of the month before the contract month.
- **Business day:** for any contract, a business day is a day that has a CME settlement record for that contract. This is used for first notice day and roll timing, so no separate holiday calendar is needed.
- **Contract held:** for each asset, the nearest eligible contract that is not yet inside its roll window (quarterly months for ES and ZN; even months for GC).
- **Roll date:** ZN and GC roll 5 business days before first notice day; ES rolls 5 business days before expiration. On the roll date we close the old contract and open the new one at that day's settlement price, and both legs are charged transaction costs.
- **Returns:** daily returns r(i,t) are always computed from two settlement prices of the same contract, so roll gaps never appear as returns.

## Strategy definitions
Shared inputs, computed at the close of day t from data through day t only:
- Annualized volatility: sigma(i,t) = sqrt(252) x standard deviation of the last 60 daily returns.
- Annualized covariance matrix Sigma(t): sample covariance of the same 60 days x 252.
- Volatility target sigma* = 10% per year; N = 3 assets.

**A: inverse volatility (correlation-blind).** Target weight w(i,t) = sigma* / (sqrt(N) x sigma(i,t)). This hits the volatility target only if correlations are zero.

**B: equal risk contribution (correlation-aware; main strategy).** Relative weights w-hat solve

minimize 1/2 w'Sigma w - (1/N) sum ln(w_i), subject to w_i > 0,

which gives every asset an equal risk contribution (Maillard, Roncalli & Teïletche 2010). They are then scaled so that sqrt(w'Sigma w) = sigma*.

If all correlations are zero, A and B hold identical positions. Any difference between them comes only from correlation information.

## Rules applied identically to both versions
- **Execution:** target weights computed at day t's settlement are traded at day t+1's settlement.
- **Gross leverage cap:** if the sum of target weights exceeds 4, all target weights are scaled down proportionally until the sum equals 4.
- **Drawdown brake:** if equity falls more than 15% below its prior peak, sigma* drops to 5%. It returns to 10% once drawdown recovers to under 7.5%. The brake uses equity through day t and acts at day t+1.
- **Current weight:** between trades, each asset's current weight drifts with prices and equity: current w(i,t) = contracts held x price x multiplier / equity(t).
- **Rebalance band:** trade asset i only if |current w(i,t) - target w(i,t)| > 0.2 x target w(i,t). When we trade, we trade all the way to target.
- **Contracts:** target contracts = round(target w x equity / (price x multiplier)).
- **Costs:** 2 bps of traded notional per side on every rebalance and every roll leg. Stress test at 4 bps.

**On the volatility target level.** Time-varying volatility scaling is part of the strategy and is expected to improve the Sharpe ratio (Behavior 2). The choice of 10% rather than another level is a different matter: it does not affect the Sharpe ratio, except when the 4x leverage cap or the drawdown brake binds. We will report the share of days on which the cap binds and the share of days on which the brake is active.

## Common start date
All variants, benchmarks, the constant-leverage portfolio and all diagnostics start on the same date: 120 trading days after the first date on which all three markets have data. The longest window (120 days) needs this much history, so every comparison is measured over the same dates. Each strategy starts flat with $10,000,000 on that date, and the drawdown brake tracks its peak from that date. Shorter-window variants use only the most recent 40, 60 or 90 days of the available history.

## Benchmarks
- **Equal-weight:** 1/3 notional in each of ES, ZN and GC (gross 1x equity), rebalanced back to 1/3 each at every month-end and traded at the next day's settlement. Weights drift between rebalances.
- **60/40:** 60% notional ES and 40% notional ZN, rebalanced monthly in the same way.
- Benchmarks use the same contracts, the same roll rules and the same 2 bps per side costs on rebalances and rolls, so the comparison is fair.
- For drawdown comparisons only, each benchmark's returns are multiplied by a constant so that its realized volatility matches B's over the same period. This scaling does not change any Sharpe ratio.

## Positive-correlation periods
A day t is a positive-correlation day if the trailing 60-day correlation between ES and ZN daily returns, measured through day t-1, is above zero. This 60-day regime window is fixed for every variant, including the 40-, 90- and 120-day variants, so all variants are classified on the same days.

Because these days are not contiguous, "drawdown in positive-correlation periods" means: chain together a strategy's daily returns on positive-correlation days only, in date order, and compute the max drawdown of that chained series. Realized volatility in these periods is the standard deviation of the same chained returns x sqrt(252). Predictions 4 and 5 use this method.

## Testable predictions
1. **Versus benchmarks.** After costs, B has a higher Sharpe ratio than the equal-weight and 60/40 benchmarks. On a volatility-matched basis, B also has a smaller max drawdown. We report Calmar ratios (annualized return / max drawdown) alongside.
2. **Volatility management adds value.** We build a constant-leverage comparison portfolio: B's normalized ERC weights (w-hat scaled to sum to 1) times a fixed multiplier, with the same costs, rolls and rebalance band. To isolate the effect of volatility targeting, this test compares B without the drawdown brake against the constant-leverage portfolio without the drawdown brake. The 4x gross leverage cap applies to both. Regressing B-no-brake's returns on the constant-leverage portfolio's returns gives a positive intercept, as in Moreira & Muir (2017). The intercept does not depend on the multiplier chosen; we set it so the portfolio's volatility matches B-no-brake's (capped at 4x if needed).
3. **Low-risk premium.** Within B, the bond leg earns a Sharpe ratio at least as high as the stock leg. A leg's Sharpe ratio is the annualized Sharpe ratio (mean / standard deviation x sqrt(252)) of that leg's daily P&L contribution within B: w(i,t-1) x r(i,t), minus that leg's transaction costs, as a fraction of equity.
4. **Gold as diversifier.** B with gold has a higher Sharpe ratio and a smaller volatility-matched drawdown than B without gold (ES and ZN only), with the improvement concentrated in periods when the trailing 60-day ES-ZN correlation is positive. This must also hold in the demeaned-gold diagnostic below.
5. **Correlation awareness.** In positive-correlation periods, B has a smaller drawdown than A. Because A ignores correlation, it is expected to take more risk than its target in these periods, so a smaller drawdown for B could simply reflect lower risk. We therefore also test whether B's risk estimate was more accurate: in positive-correlation periods, B's realized volatility is closer to its target than A's, and A's realized volatility exceeds its target. To keep the comparison fair when the drawdown brake has lowered the target, each day's return is first rescaled to a 10% target (multiplied by 10% / sigma* in force at day t-1) before computing realized volatility.
6. **Plateau.** Results for B are similar with 40-, 90- and 120-day windows.

## Diagnostics (not strategy variants)
- **Demeaned-gold run.** Rerun B with gold's daily returns demeaned (the sample mean for the period being evaluated is subtracted from each day's return). Rolling volatilities and correlations are unchanged, so target weights are identical and only gold's average return is removed. Contract counts and the timing of drawdown-brake triggers can still differ, because equity follows a different path. If B with demeaned gold still beats B without gold, gold earned its place through diversification, not through its price rise.
- **Factor regressions,** run on weekly returns for (i) and monthly returns for (ii), with Newey-West standard errors. Daily data is not used because Kenneth French factors are measured at the 4 PM NYSE close while futures settle at different times.
  - (i) On ES, ZN and GC returns. The intercept measures the value of our dynamic sizing beyond static exposure to the three assets. Because this regression counts the leveraged bond position as beta, any alpha here can only come from timing and gold.
  - (ii) On the Kenneth French market, size, value and momentum factors. This checks whether the strategy is secretly an equity-factor strategy.
- **Rounding effect.** We also run B with fractional contracts to show how much integer rounding changes results at $10M.
- **B without the drawdown brake,** used only for prediction 2 (see above).
- **Usage statistics.** Share of days the leverage cap binds and the drawdown brake is active.

Benchmarks, the constant-leverage portfolio and all diagnostics above are not counted as strategy variants.

## What would make it fail
- OOS Sharpe at or below zero, or no better than 60/40 after costs.
- Profits come almost entirely from one regime, for example the bond leg during the falling-rate years 2010-2020, with losses in rising-rate periods.
- Regression (ii) explains the returns fully (alpha near zero), or regression (i) shows no value from dynamic sizing.
- B without gold matches or beats B with demeaned gold, meaning gold's benefit came from its price rise rather than diversification.
- Only the 60-day window works.

## Data handling
- **Missing data / holidays:** if a market is closed, we hold the prior position with zero return for that market. We never fill with future data. Zero-return days slightly lower volatility estimates, but because all three contracts trade on CME and share most holidays, the effect is negligible.
- **Survivorship:** three contracts chosen in advance that traded over the whole sample. We acknowledge mild selection bias, since we picked well-known, liquid markets.
- **Warm-up:** the first 120 trading days after all three markets have data are used only for estimation and are excluded from reported metrics for every variant and benchmark (see Common start date).

## Planned variants (all will be reported)
1. A: inverse volatility
2. B: ERC (main strategy)
3. B without gold (ES and ZN only)
4. B with a 40-day window
5. B with a 90-day window
6. B with a 120-day window

Plus cost stress tests at 4 bps for A and B. Any additional variant tried will be added to this count in the note.

## Sample periods
- **In-sample:** from the common start date (120 trading days after all three markets have data) through 2024-09-30.
- **Out-of-sample:** 2024-10-01 through the last bar available on our download date. The download date will be recorded in the README. The OOS period is untouched during development and run exactly once at the end.

## Amendment 1 (added before any backtest on real data; see the commit timestamp)

This amendment adds diagnostics and clarifies implementation details. It does not change any trading rule, fixed setting, prediction, pass/fail criterion or the variant count.

**Added diagnostics (reported for both sample periods):**
1. **Cost justification.** Estimated cost per side = half a tick (crossing half the bid-ask spread) plus an assumed $2.50 per contract in exchange, clearing and broker fees, compared with the 2 bps we charge.
2. **Capacity.** Square-root market-impact model: impact as a fraction of traded notional = 1.0 x daily volatility x sqrt(contracts traded / average daily volume). Daily volatility uses the 60 days before the trade; average daily volume is the 20-day trailing CME cleared volume of the held contract (from Databento statistics). Trades are scaled linearly to fund sizes of $10M, $100M, $500M, $1B, $5B, $10B and $25B. Capacity is defined as the largest size tested that keeps at least 75% of B's Sharpe ratio. We also report participation rates and the largest position as a share of average daily volume.
3. **Stress windows,** fixed now from general market history: 2011 US downgrade (2011-07-22 to 2011-10-03), 2013 taper tantrum (2013-05-22 to 2013-09-05), 2015 China devaluation (2015-08-10 to 2015-08-25), 2018 Volmageddon (2018-01-26 to 2018-02-08), 2018 Q4 selloff (2018-09-20 to 2018-12-24), 2020 COVID crash (2020-02-19 to 2020-03-23), 2022 inflation shock (2022-01-03 to 2022-10-12), 2023 regional banks (2023-03-08 to 2023-03-24), 2024 August volatility spike (2024-07-16 to 2024-08-05), and, out of sample, the 2025 tariff selloff (2025-02-19 to 2025-04-08).
4. **Position risk.** Worst one-day and five-day loss of each leg, worst portfolio day, longest continuous spell with the drawdown brake on, and maximum actual gross leverage.
5. **Statistical uncertainty.** Stationary block bootstrap (Politis & Romano 1994; 2,000 samples, mean block length 20 days, fixed seed) for B's Sharpe ratio and for the Sharpe differences between B and A, B without gold, equal-weight and 60/40.
6. **Regime diagnostics (FRED).** 10-year breakeven inflation (T10YIE), VIX (VIXCLS), 10-year real yield (DFII10) and high-yield credit spread (BAMLH0A0HYM2), each lagged one day. We report the correlation between the 60-day ES-ZN correlation and breakeven inflation, and average breakevens and VIX on positive-correlation days versus other days. These are explanatory only and never used as trading signals.
7. **Asynchronous settlement check.** Daily versus weekly ES-ZN and ES-GC correlations.
8. **Total return.** Excess returns plus T-bill interest (FRED DTB3, lagged one day), as an approximate total-return figure.
9. **Margin.** Margin-to-equity using current CME initial margins (entered with their date), scaled back in time by price. Reported only if margins are entered.
10. **Data integrity.** Parsed contract months are checked against Databento definitions, settlements can optionally be checked against daily closes, and a fingerprint of the settlement data is committed so others can confirm they downloaded the same data.
11. **Automated tests** of roll handling, ERC risk contributions, look-ahead, cost charging and the leverage cap.

**Implementation clarifications:**
- On a roll date, the new contract is opened with the same number of contracts as the old one unless a rebalance is scheduled that day.
- Contract counts for a trade are computed from day t's equity and settlement price; the trade fills at day t+1's settlement.
- The 4x leverage cap applies to target weights. Actual leverage can drift slightly above 4x between rebalances; we report the maximum.

## Amendment 2: deviations (added 2026-10-03, after the first in-sample run on real data and before any out-of-sample data was downloaded)

This amendment changes no trading rule, fixed setting, prediction, pass/fail criterion or the variant count.

Every data-handling and implementation choice that this document does not spell out, and every departure from it, is recorded with its reason and date in the "Deviations log" section of README.md. The main ones are: which CME settlement record is used when several exist; two weekend-dated records dropped; eleven asset-days with no settlement record, of which four are filled with Databento daily closes and seven cannot be filled; and cleared volume being available only from late 2015.

## Amendment 3: a second, separate strategy (added 2026-10-03)

A second strategy (momentum plus ENSO risk budgeting on corn, soybean and wheat futures) has been added to this repo. Its specification is `ENSO_SPEC.md`. It does not change anything in this document: the ES/ZN/GC universe, settings, predictions, variants, sample periods and pass/fail criteria above are as they were. It shares the in-sample end date (2024-09-30) and the out-of-sample rule.
