"""Cross-asset strategy: from returns and the ENSO state to LONG/SHORT target
weights (XA_SPEC.md).

One function, decide(), does the whole calculation for one rebalance day.
Both engines call it (src/ag_backtest.py and the backtrader strategy).

    trend (12 months)            ->  alpha score for every asset
    + ENSO effect, grains only   ->  (only in the models that use ENSO)
    sign of alpha                ->  long or short
    size of alpha                ->  that asset's share of the risk budget
    EWMA volatility / covariance ->  weights that deliver those risk shares
    whole portfolio              ->  scaled to 10% predicted volatility

A "weight" is notional exposure as a fraction of equity; negative = short.
"""

from dataclasses import dataclass

import numpy as np

import config
from src import alpha as ag_alpha
from src import signals


@dataclass
class XaSpec:
    """One thing to backtest. The defaults are Model 4 (everything on)."""

    name: str = "M4_trend_enso_cov"
    use_enso: bool = True             # add the ENSO effect to the grains' alpha
    use_cov: bool = True              # True: full covariance. False: volatilities only
    mom_lookback: int = config.XA_MOM_LOOKBACK
    cost_bps: float = config.COST_BPS
    integer_contracts: bool = True


def ewma_cov(returns_so_far, half_life=config.XA_HALF_LIFE):
    """Annualized covariance in which recent days count more.

    A day that is `half_life` (60) days old counts half as much as today, a
    day 120 days old a quarter as much, and so on. Only the last
    config.XA_COV_WINDOW (252) days are used.
    """
    window = np.asarray(returns_so_far)[-config.XA_COV_WINDOW:]
    age = np.arange(len(window))[::-1]              # 0 for the newest day
    day_weights = 0.5 ** (age / half_life)
    # aweights tells numpy how much each day (row) counts.
    return np.cov(window, rowvar=False, aweights=day_weights) * config.TRADING_DAYS


def decide(spec, returns_so_far, enso_state, assets):
    """Everything for one rebalance day. Returns a dict; "target" holds the
    signed weights (0 for an asset we do not hold)."""
    n = len(assets)
    returns_so_far = np.asarray(returns_so_far)

    # --- Risk ---
    cov = ewma_cov(returns_so_far)
    sigma = np.sqrt(np.diag(cov))
    # Without covariance: keep each asset's own variance, set every
    # correlation to zero. This is "inverse volatility" sizing.
    cov_used = cov if spec.use_cov else np.diag(np.diag(cov))

    # --- Alpha ---
    # Trend: compounded return of the held contracts over the lookback,
    # divided by the size of a typical move over that many days. 1.0 = a
    # one-standard-deviation rise; -1.0 = a one-standard-deviation fall.
    momentum = np.prod(1.0 + returns_so_far[-spec.mom_lookback:], axis=0) - 1.0
    alpha = momentum / (sigma * np.sqrt(spec.mom_lookback / config.TRADING_DAYS))
    if spec.use_enso:
        # Grains only: +1 bullish, -1 bearish, 0 no view (config.ENSO_EFFECT).
        for i, asset in enumerate(assets):
            if asset in config.ENSO_EFFECT:
                alpha[i] += config.ENSO_EFFECT[asset][enso_state]

    # No position where the signal is tiny.
    active = np.abs(alpha) >= config.XA_MIN_SIGNAL

    target = np.zeros(n)
    budgets = np.zeros(n)
    risk_limit_active = False
    position_limit_active = False
    if active.any():
        direction = np.sign(alpha[active])                  # +1 long, -1 short
        # Risk budget: each asset's share of the total signal strength.
        share = np.abs(alpha[active]) / np.abs(alpha[active]).sum()
        share, risk_limit_active = ag_alpha.limit_budgets(share)

        # The risk-budget solver wants positive weights. Trick: flip the
        # sign of every SHORT asset's returns (which flips the sign of its
        # covariances with the others). In that flipped world all positions
        # are long, and the solver's answer has the right risk shares.
        part = cov_used[np.ix_(active, active)]
        flipped = part * np.outer(direction, direction)
        size = signals.risk_budget_weights(flipped, share)  # positive, adds up to 1
        weights = direction * size                          # put the signs back

        # Scale the whole portfolio to the volatility target.
        weights = weights * config.VOL_TARGET / np.sqrt(weights @ part @ weights)

        # Position limit: no single asset above 2x equity (either direction).
        too_big = np.abs(weights) > config.XA_MAX_WEIGHT
        if too_big.any():
            position_limit_active = True
            weights = np.clip(weights, -config.XA_MAX_WEIGHT, config.XA_MAX_WEIGHT)

        target[active] = weights
        budgets[active] = direction * share

    # Gross leverage cap: longs plus shorts may not exceed 4x equity.
    gross = np.abs(target).sum()
    cap_binding = bool(gross > config.GROSS_CAP)
    if cap_binding:
        target = target * config.GROSS_CAP / gross

    return {
        "target": target,
        "alpha": alpha,
        "momentum": momentum,
        "sigma": sigma,
        "cov": cov,
        "cov_final": cov_used,
        "budgets": budgets,                                  # signed risk shares
        "risk_shares": ag_alpha.risk_shares(target, cov_used),
        "predicted_vol": float(np.sqrt(target @ cov_used @ target)),
        "risk_limit_active": risk_limit_active,
        "position_limit_active": position_limit_active,
        "cap_binding": cap_binding,
    }
