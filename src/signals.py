"""Signals: turn a window of daily returns into target weights.

A "weight" is notional exposure as a fraction of equity (1.5 = futures worth
1.5x our equity). See HYPOTHESIS.md, "Strategy definitions".

    Version A (inverse volatility): ignores correlations.
    Version B (equal risk contribution, ERC): uses the full covariance matrix.
"""

import numpy as np

import config


def annualized_cov(window_returns):
    """Annualized covariance matrix of a block of daily returns.

    window_returns: array with one row per day and one column per asset.
    """
    # np.cov with ddof=1 is the usual "sample" covariance (divide by n - 1).
    # rowvar=False tells numpy that each COLUMN is one asset.
    daily_cov = np.cov(window_returns, rowvar=False, ddof=1)
    # Variances add up over time, so a year is 252 times one day.
    return daily_cov * config.TRADING_DAYS


def erc_weights(cov, tolerance=1e-12, max_rounds=10_000):
    """Relative ERC weights (w-hat in HYPOTHESIS.md), before any scaling.

    They solve:   minimize  1/2 w'Cw - (1/N) * sum(ln(w_i)),   with w_i > 0.

    At the minimum every asset contributes the same amount of risk
    (Maillard, Roncalli & Teiletche 2010).

    Method: "one asset at a time". Hold every weight fixed except w_i. Setting
    the slope of the objective to zero for w_i gives

        C_ii * w_i + b - (1/N) / w_i = 0,   where b = sum over j != i of C_ij * w_j

    Multiply by w_i to get a quadratic equation:

        C_ii * w_i^2 + b * w_i - 1/N = 0

    whose positive root is the best w_i. Repeat over the assets, again and
    again, until the weights stop changing.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]

    # Starting guess: inverse volatility. (Any positive start works.)
    w = 1.0 / np.sqrt(np.diag(cov))

    for _ in range(max_rounds):
        w_before = w.copy()
        for i in range(n):
            # b = how much the OTHER assets co-move with asset i.
            b = cov[i] @ w - cov[i, i] * w[i]
            # Positive root of the quadratic above.
            w[i] = (-b + np.sqrt(b * b + 4.0 * cov[i, i] / n)) / (2.0 * cov[i, i])
        # Stop once a full round changes nothing (to 12 decimal places).
        if np.max(np.abs(w - w_before)) < tolerance * np.max(w):
            return w

    raise RuntimeError("ERC solver did not converge")


def risk_contributions(weights, cov):
    """Each asset's share of total portfolio variance (the shares sum to 1)."""
    weights = np.asarray(weights, dtype=float)
    # w_i * (C w)_i is asset i's contribution; they add up to w'Cw.
    contributions = weights * (cov @ weights)
    return contributions / contributions.sum()


def target_weights(cov, version, sigma_star):
    """Target weights for one day.

    cov:        annualized covariance matrix (from annualized_cov).
    version:    "A" or "B".
    sigma_star: the volatility target in force (10%, or 5% while braking).

    Returns (weights, cap_binding), where cap_binding is True when the 4x
    gross leverage cap forced the weights down.
    """
    n = cov.shape[0]

    if version == "A":
        # Inverse volatility. sigma_i is each asset's own annualized volatility.
        sigma = np.sqrt(np.diag(cov))
        weights = sigma_star / (np.sqrt(n) * sigma)
    elif version == "B":
        # ERC shape, then stretch or shrink so predicted volatility = sigma*.
        w_hat = erc_weights(cov)
        predicted_vol = np.sqrt(w_hat @ cov @ w_hat)
        weights = w_hat * sigma_star / predicted_vol
    else:
        raise ValueError(f"unknown version {version!r}: use 'A' or 'B'")

    # Gross leverage cap: if the weights add up to more than 4, shrink them
    # all by the same factor so they add up to exactly 4.
    gross = weights.sum()
    cap_binding = bool(gross > config.GROSS_CAP)
    if cap_binding:
        weights = weights * config.GROSS_CAP / gross

    return weights, cap_binding


def constant_leverage_weights(cov, leverage):
    """Weights of the constant-leverage comparison portfolio (prediction 2).

    B's ERC weights are scaled to add up to 1 and multiplied by a fixed
    number `leverage`. The mix of assets still follows ERC, but the overall
    size no longer reacts to volatility. The 4x gross cap applies.

    Returns (weights, cap_binding).
    """
    w_hat = erc_weights(cov)
    weights = w_hat / w_hat.sum() * leverage

    gross = weights.sum()                    # equals `leverage`
    cap_binding = bool(gross > config.GROSS_CAP)
    if cap_binding:
        weights = weights * config.GROSS_CAP / gross
    return weights, cap_binding


def risk_budget_weights(cov, budgets, tolerance=1e-12, max_rounds=10_000):
    """Relative weights whose risk contributions match the given budgets.

    cov:     covariance matrix of the assets we want to hold.
    budgets: the share of portfolio risk each asset should carry. All must
             be positive and they must add up to 1.

    This is erc_weights with one change: asset i's term in the objective is
    weighted by its own budget b_i instead of the equal share 1/N,

        minimize  1/2 w'Cw - sum(b_i * ln(w_i)),   with w_i > 0,

    so the quadratic for one weight becomes  C_ii * w_i^2 + b * w_i - b_i = 0.
    With equal budgets it gives exactly the ERC weights. The result is scaled
    so the weights add up to 1.
    """
    cov = np.asarray(cov, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    n = cov.shape[0]

    w = 1.0 / np.sqrt(np.diag(cov))          # starting guess: inverse volatility
    for _ in range(max_rounds):
        w_before = w.copy()
        for i in range(n):
            b = cov[i] @ w - cov[i, i] * w[i]
            w[i] = (-b + np.sqrt(b * b + 4.0 * cov[i, i] * budgets[i])) / (2.0 * cov[i, i])
        if np.max(np.abs(w - w_before)) < tolerance * np.max(w):
            return w / w.sum()

    raise RuntimeError("risk-budget solver did not converge")
