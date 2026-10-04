"""The A/B/C/D attribution models (ABCD_PLAN.md): from returns and the ENSO
state to LONG/SHORT target weights.

One function, decide(), does the whole calculation for one rebalance day.
The engine (src/ag_backtest.py) calls it exactly as it calls xa_alpha.decide.

    sign of the 12-month trend       ->  +1 long, -1 short, 0 no position
    ENSO overlay, grains only        ->  (only in the models that use ENSO)
    sizing "equal"   (Model A)       ->  equal notional, 1x equity in total
    sizing "invvol"  (Model B)       ->  signal / volatility, scaled to 10%
                                         as if all correlations were zero
    sizing "cov"     (Models C, D)   ->  the SAME signal / volatility, scaled
                                         to 10% using the full covariance

The covariance is used only to measure the risk of the whole portfolio and
scale it. There is no optimizer and no risk-budget solver here.

A "weight" is notional exposure as a fraction of equity; negative = short.
"""

from dataclasses import dataclass

import numpy as np

import config
from src import alpha as ag_alpha
from src import xa_alpha


@dataclass
class AbcdSpec:
    """One thing to backtest. The defaults are Model D."""

    name: str = "D_trend_enso_cov"
    sizing: str = "cov"               # "equal", "invvol" or "cov"
    enso: str = "add"                 # "none", "add", "mult" or "only" (see signal_scores)
    universe: tuple = tuple(config.XA_ASSETS)   # the assets this run may hold
    long_only: bool = False           # True: a negative signal means no position
    mom_lookback: int = config.XA_MOM_LOOKBACK
    half_life: int = config.XA_HALF_LIFE
    vol_target: float = config.VOL_TARGET
    rebalance_weeks: int = 1          # decide every this many weeks
    cost_bps: float = config.COST_BPS
    integer_contracts: bool = True


def signal_scores(spec, momentum, enso_state, assets):
    """The signed signal of every asset. Its sign is the direction; its size
    multiplies the position (1 everywhere unless ENSO changes a grain).

        "none"  trend only: +1 or -1.
        "add"   trend + ENSO effect on the grains (the form used so far).
                On a +1/-1 trend this doubles the position when ENSO agrees
                with the trend and closes it when ENSO disagrees.
        "mult"  trend x 1.5 when ENSO agrees, x 0.5 when it disagrees.
        "only"  the ENSO effect alone, no trend (grains; zero elsewhere).
    """
    trend = np.sign(momentum)
    # +1 bullish, -1 bearish, 0 no view; always 0 for the non-grains.
    effect = np.array([config.ENSO_EFFECT.get(a, {}).get(enso_state, 0.0) for a in assets])

    if spec.enso == "none":
        signal = trend
    elif spec.enso == "add":
        signal = trend + effect
    elif spec.enso == "mult":
        agreement = trend * effect                  # +1 agree, -1 disagree, 0 no view
        signal = trend * np.where(agreement > 0, config.ABCD_ENSO_AGREE,
                                  np.where(agreement < 0, config.ABCD_ENSO_DISAGREE, 1.0))
    elif spec.enso == "only":
        signal = effect
    else:
        raise ValueError(f"unknown ENSO form {spec.enso!r}")

    if spec.long_only:
        signal = np.maximum(signal, 0.0)
    in_universe = np.array([a in spec.universe for a in assets])
    return np.where(in_universe, signal, 0.0)


def decide(spec, returns_so_far, enso_state, assets):
    """Everything for one rebalance day. Returns a dict; "target" holds the
    signed weights (0 for an asset we do not hold)."""
    n = len(assets)
    returns_so_far = np.asarray(returns_so_far)

    # --- Risk: the same EWMA estimate in every model ---
    cov = xa_alpha.ewma_cov(returns_so_far, spec.half_life)
    sigma = np.sqrt(np.diag(cov))
    # The covariance each model scales with. Model B sets every correlation
    # to zero. Model A scales with nothing; the full matrix is kept so its
    # predicted volatility can still be reported.
    cov_used = np.diag(np.diag(cov)) if spec.sizing == "invvol" else cov

    # --- Signal ---
    # Trend: compounded return of the held contracts over the lookback.
    momentum = np.prod(1.0 + returns_so_far[-spec.mom_lookback:], axis=0) - 1.0
    signal = signal_scores(spec, momentum, enso_state, assets)
    active = signal != 0

    target = np.zeros(n)
    position_limit_active = False
    if active.any():
        if spec.sizing == "equal":
            # Equal notional per unit of signal; longs + shorts add up to 1x.
            weights = config.ABCD_EQUAL_GROSS * signal[active] / np.abs(signal[active]).sum()
        elif spec.sizing in ("invvol", "cov"):
            q = signal[active] / sigma[active]
            part = cov_used[np.ix_(active, active)]
            weights = q * spec.vol_target / np.sqrt(q @ part @ q)
        else:
            raise ValueError(f"unknown sizing {spec.sizing!r}")

        # Position limit: no single asset above 2x equity (either direction).
        too_big = np.abs(weights) > config.XA_MAX_WEIGHT
        if too_big.any():
            position_limit_active = True
            weights = np.clip(weights, -config.XA_MAX_WEIGHT, config.XA_MAX_WEIGHT)
        target[active] = weights

    # Gross leverage cap: longs plus shorts may not exceed 4x equity.
    gross = np.abs(target).sum()
    cap_binding = bool(gross > config.GROSS_CAP)
    if cap_binding:
        target = target * config.GROSS_CAP / gross

    return {
        "target": target,
        "alpha": signal,
        "momentum": momentum,
        "sigma": sigma,
        "cov": cov,
        "cov_final": cov_used,
        "budgets": np.zeros(n),                              # these models set no risk budgets
        "risk_shares": ag_alpha.risk_shares(target, cov_used),
        "predicted_vol": float(np.sqrt(target @ cov_used @ target)),
        "risk_limit_active": False,
        "position_limit_active": position_limit_active,
        "cap_binding": cap_binding,
    }
