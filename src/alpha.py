"""ENSO strategy: from returns and the ENSO state to target weights.

One function, decide(), does the whole calculation for one rebalance day.
Both engines call it (the pandas engine in src/ag_backtest.py and the
backtrader strategy in the organizers' kit), so the two can only differ in
their mechanics: fills, P&L, rolls and costs.

The steps follow ENSO_SPEC.md sections 6 to 16:

    momentum + ENSO  ->  alpha score  ->  keep only positive scores
    returns          ->  volatility, covariance, shrunk covariance
    alpha            ->  risk budgets (who should carry how much of the risk)
    budgets + cov    ->  relative weights that add up to 1
    weights          ->  scaled so predicted portfolio volatility is 10%

A "weight" is notional exposure as a fraction of equity.
"""

from dataclasses import dataclass

import numpy as np

import config
from src import signals


@dataclass
class AgSpec:
    """One thing to backtest. The defaults are the proposed strategy
    (benchmark D in ENSO_SPEC.md)."""

    name: str = "D_tilt"
    signal: str = "MOM+ENSO"          # "MOM", "ENSO" or "MOM+ENSO"
    sizing: str = "TILT"              # "EW", "EW_VT", "IV", "ERC" or "TILT" (see size())
    mom_lookback: int = config.MOM_LOOKBACK
    shrinkage: float = config.SHRINKAGE
    tilt_k: float = config.TILT_K
    cost_bps: float = config.COST_BPS
    integer_contracts: bool = True


def alpha_scores(spec, window_returns, sigma, enso_state, assets):
    """The attractiveness score of each asset. Returns (alpha, momentum)."""
    # Momentum: the compounded return of the last `mom_lookback` days of the
    # held contract. Built from returns, not raw prices, so the price gap
    # between two contracts on a roll day never counts as momentum.
    recent = window_returns[-spec.mom_lookback:]
    momentum = np.prod(1.0 + recent, axis=0) - 1.0

    # Normalize: divide by the move that would be "typical" over that many
    # days, given the asset's own volatility. 1.0 = a one-standard-deviation
    # rise. This puts corn, soybeans and wheat on the same scale.
    typical_move = sigma * np.sqrt(spec.mom_lookback / config.TRADING_DAYS)
    momentum_z = momentum / typical_move

    # ENSO effect: +1 bullish, -1 bearish, 0 no view (config.ENSO_EFFECT).
    enso_effect = np.array([config.ENSO_EFFECT[a][enso_state] for a in assets])

    if spec.signal == "MOM":
        alpha = momentum_z
    elif spec.signal == "ENSO":
        alpha = enso_effect
    elif spec.signal == "MOM+ENSO":
        alpha = config.ALPHA_W_MOM * momentum_z + config.ALPHA_W_ENSO * enso_effect
    else:
        raise ValueError(f"unknown signal {spec.signal!r}")
    return alpha, momentum


def risk_budgets(positive_alpha, k):
    """Turn the positive alpha scores of the ACTIVE assets into risk-budget
    shares that add up to 1, then apply the 70% limit.

        b_i = exp(k * alpha_i) / sum of exp(k * alpha_j)

    A bigger score gets a bigger share. Equal scores (or k = 0) give equal
    shares, which is plain risk parity. Returns (budgets, limit_was_active).
    """
    # Subtracting the largest score first changes nothing in the result
    # (it cancels out in the division) but avoids huge numbers in exp().
    grown = np.exp(k * (positive_alpha - positive_alpha.max()))
    return limit_budgets(grown / grown.sum())


def limit_budgets(budgets):
    """Apply the 70% limit to risk-budget shares that add up to 1.
    Returns (budgets, limit_was_active)."""
    limit = config.MAX_RISK_SHARE
    if budgets.max() <= limit:
        return budgets, False
    if len(budgets) == 1:
        # A single asset always carries 100% of the risk; the limit cannot
        # be met without holding an asset the signal does not like. We keep
        # the position and report that the limit was active.
        return budgets, True
    # Cut the too-big shares to 70% and hand what was cut to the others, in
    # proportion to what they already have. Repeat in case that pushes
    # another asset over the limit.
    capped = np.zeros(len(budgets), dtype=bool)
    while (budgets > limit + 1e-12).any():
        capped |= budgets > limit
        budgets[capped] = limit
        left_over = 1.0 - limit * capped.sum()
        budgets[~capped] *= left_over / budgets[~capped].sum()
    return budgets, True


def size(spec, positive_alpha, cov, cov_final):
    """Target weights of the ACTIVE assets, for each way of sizing.

        "EW"     equal weights adding up to 1. No volatility scaling.
        "EW_VT"  equal weights, scaled to the 10% volatility target.
        "IV"     inverse volatility, scaled to 10% as if correlations were zero.
        "ERC"    equal risk contributions using the covariance, scaled to 10%.
        "TILT"   alpha-tilted risk budgets using the covariance, scaled to 10%.

    Returns (weights, budgets, limit_was_active). `budgets` is the risk share
    each asset is MEANT to carry (empty for the sizings that have no such aim).
    """
    n = len(positive_alpha)
    no_budgets = np.full(n, np.nan)

    if spec.sizing == "EW":
        return np.full(n, 1.0 / n), no_budgets, False

    if spec.sizing == "EW_VT":
        shape, budgets, limit_active = np.full(n, 1.0 / n), no_budgets, False
        predicted_vol = np.sqrt(shape @ cov_final @ shape)
    elif spec.sizing == "IV":
        sigma = np.sqrt(np.diag(cov))
        shape, budgets, limit_active = (1.0 / sigma) / (1.0 / sigma).sum(), no_budgets, False
        # "No covariance": the volatility is predicted from the variances only.
        predicted_vol = np.sqrt(np.sum(shape ** 2 * sigma ** 2))
    elif spec.sizing in ("ERC", "TILT"):
        if spec.sizing == "ERC":
            budgets, limit_active = np.full(n, 1.0 / n), False
        else:
            budgets, limit_active = risk_budgets(positive_alpha, spec.tilt_k)
        shape = signals.risk_budget_weights(cov_final, budgets)
        predicted_vol = np.sqrt(shape @ cov_final @ shape)
    else:
        raise ValueError(f"unknown sizing {spec.sizing!r}")

    # Stretch or shrink every weight by the same factor so that the predicted
    # volatility of the whole portfolio equals the target.
    return shape * config.VOL_TARGET / predicted_vol, budgets, limit_active


def decide(spec, returns_so_far, enso_state, assets):
    """Everything for one rebalance day.

    returns_so_far: daily returns up to and including the decision day, one
                    row per day and one column per asset. Only the last
                    config.AG_WINDOW rows are used.
    enso_state:     "el_nino", "neutral" or "la_nina", as known that day.

    Returns a dict. "target" holds the weights (0 for an asset we do not
    hold); the rest are diagnostics.
    """
    n = len(assets)
    window_returns = np.asarray(returns_so_far)[-config.AG_WINDOW:]

    # Risk: annualized covariance of the last 60 days, then blend it half and
    # half with the same matrix with its correlations set to zero.
    cov = signals.annualized_cov(window_returns)
    sigma = np.sqrt(np.diag(cov))
    diagonal_only = np.diag(np.diag(cov))
    cov_final = spec.shrinkage * cov + (1.0 - spec.shrinkage) * diagonal_only

    # Alpha, and the long-or-cash rule: only positive scores get a position.
    alpha, momentum = alpha_scores(spec, window_returns, sigma, enso_state, assets)
    active = alpha > 0

    target = np.zeros(n)
    budgets = np.zeros(n)
    limit_active = False
    if active.any():
        rows_and_columns = np.ix_(active, active)       # the active part of a matrix
        weights, active_budgets, limit_active = size(
            spec, alpha[active], cov[rows_and_columns], cov_final[rows_and_columns])
        target[active] = weights
        budgets[active] = active_budgets

    # Gross leverage cap: weights may not add up to more than 4.
    gross = target.sum()
    cap_binding = bool(gross > config.GROSS_CAP)
    if cap_binding:
        target = target * config.GROSS_CAP / gross

    return {
        "target": target,
        "alpha": alpha,
        "momentum": momentum,
        "sigma": sigma,
        "cov": cov,
        "cov_final": cov_final,
        "budgets": budgets,
        "risk_shares": risk_shares(target, cov_final),
        "predicted_vol": float(np.sqrt(target @ cov_final @ target)),
        "risk_limit_active": limit_active,
        "cap_binding": cap_binding,
    }


def risk_shares(weights, cov):
    """Each asset's share of portfolio risk (zeros when nothing is held)."""
    contributions = weights * (cov @ weights)
    total = contributions.sum()
    return contributions / total if total > 0 else np.zeros(len(weights))


def whole_contracts(spec, target, equity, prices, multipliers, assets):
    """Turn target weights into contract counts.

    contracts = target dollars / dollars per contract, rounded to a whole
    number. If the team has entered CME margins, the counts are cut so the
    margin posted stays under config.MARGIN_CAP of equity.
    Returns (contract counts, margin_limit_active).
    """
    wanted = target * equity / (prices * multipliers)
    if spec.integer_contracts:
        wanted = np.round(wanted)

    margin_active = False
    if config.AG_INITIAL_MARGIN:
        margin_each = np.array([config.AG_INITIAL_MARGIN[a] for a in assets])
        needed = float(np.abs(wanted) @ margin_each)    # shorts need margin too
        allowed = config.MARGIN_CAP * equity
        if needed > allowed:
            margin_active = True
            wanted = wanted * allowed / needed
            if spec.integer_contracts:
                wanted = np.trunc(wanted)       # round TOWARD ZERO, to stay under the limit
    return wanted, margin_active
