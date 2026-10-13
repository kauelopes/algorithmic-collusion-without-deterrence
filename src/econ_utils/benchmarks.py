"""
Equilibrium benchmarks for Cournot (linear demand) and Bertrand (logit demand).

Cournot benchmarks are closed-form. Bertrand/logit benchmarks are solved
numerically via fixed-point iteration — accurate to < 1e-10.

Returns are plain Python dicts so they can be serialized to YAML/JSON directly.
"""

import math


def nash_symmetric(a: float, b: float, c: float, n: int) -> dict:
    """
    Symmetric Nash equilibrium of the n-firm linear Cournot game.

    Each firm solves max_{q_i} (a - b*(q_i + (n-1)*q_j) - c) * q_i,
    yielding the symmetric fixed point q_i = q_NE for all i.

    q_NE  = (a - c) / (b * (n + 1))
    Q_NE  = n * q_NE
    p_NE  = (a + n*c) / (n + 1)
    pi_NE = b * q_NE^2   (per firm)
    """
    q_NE = (a - c) / (b * (n + 1))
    Q_NE = n * q_NE
    p_NE = a - b * Q_NE          # equivalent to (a + n*c) / (n+1)
    pi_NE = (p_NE - c) * q_NE
    return {
        "q_per_firm": q_NE,
        "Q_total": Q_NE,
        "price": p_NE,
        "profit_per_firm": pi_NE,
    }


def monopoly(a: float, b: float, c: float) -> dict:
    """
    Joint monopoly outcome (full collusion, single decision-maker).

    Q_M  = (a - c) / (2*b)
    p_M  = (a + c) / 2
    pi_M = (a - c)^2 / (4*b)   (total profit)
    """
    Q_M = (a - c) / (2 * b)
    p_M = a - b * Q_M            # equivalent to (a + c) / 2
    pi_M = (p_M - c) * Q_M
    return {
        "Q_total": Q_M,
        "price": p_M,
        "profit_total": pi_M,
    }


def competitive(a: float, b: float, c: float) -> dict:
    """
    Competitive equilibrium (price equals marginal cost).

    Q_C = (a - c) / b
    p_C = c
    pi_C = 0
    """
    Q_C = (a - c) / b
    p_C = c
    return {
        "Q_total": Q_C,
        "price": p_C,
        "profit_total": 0.0,
    }


# ---------------------------------------------------------------------------
# Bertrand / logit demand benchmarks (numerical)
# ---------------------------------------------------------------------------

def _logit_sym_share(p: float, a: float, mu: float, a0: float, n: int) -> float:
    """Symmetric logit demand share when all N firms charge price p."""
    outside = math.exp(a0 / mu)
    logit   = math.exp((a - p) / mu)
    return logit / (outside + n * logit)


def _logit_nash_fp(
    a: float, mu: float, a0: float, c: float, n: int,
    tol: float = 1e-10, max_iter: int = 2000,
) -> float:
    """
    Bertrand-Nash price for symmetric firms with logit demand.

    Fixed-point: p = c + mu / (1 - q(p))
    where q(p) is the symmetric logit share at price p.
    Uses damped iteration to ensure convergence.
    """
    p = c + mu  # initial guess
    for _ in range(max_iter):
        q     = _logit_sym_share(p, a, mu, a0, n)
        p_new = c + mu / (1.0 - q)
        p_new = 0.5 * p + 0.5 * p_new  # damping
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def _logit_monopoly_fp(
    a: float, mu: float, a0: float, c: float, n: int,
    tol: float = 1e-10, max_iter: int = 2000,
) -> float:
    """
    Joint-monopoly price for N symmetric firms with logit demand.

    Fixed-point: p = c + mu / (1 - n*q(p))
    Uses stronger damping because the contraction radius is smaller.
    """
    p_ne = _logit_nash_fp(a, mu, a0, c, n)
    p    = p_ne * 1.3  # start above Nash
    for _ in range(max_iter):
        q     = _logit_sym_share(p, a, mu, a0, n)
        denom = 1.0 - n * q
        if denom <= 1e-12:
            p = p * 0.99  # step back if denominator blows up
            continue
        p_new = c + mu / denom
        p_new = 0.7 * p + 0.3 * p_new  # strong damping
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def logit_benchmarks(
    a: float,
    mu: float,
    c: float,
    n: int,
    a0: float = 0.0,
) -> dict:
    """
    Equilibrium benchmarks for N-firm Bertrand with logit demand.

    Demand (per firm, symmetric prices p):
        q_i = exp((a - p_i)/mu) / (exp(a0/mu) + sum_j exp((a - p_j)/mu))
    Profit: pi_i = (p_i - c) * q_i

    Returns
    -------
    dict with keys:
        p_NE, q_NE, pi_NE          — Bertrand-Nash (symmetric)
        p_monopoly, q_monopoly_per_firm, pi_monopoly_per_firm  — joint monopoly
        p_competitive              — marginal cost = c
    """
    p_NE   = _logit_nash_fp(a, mu, a0, c, n)
    q_NE   = _logit_sym_share(p_NE, a, mu, a0, n)
    pi_NE  = (p_NE - c) * q_NE

    p_M    = _logit_monopoly_fp(a, mu, a0, c, n)
    q_M    = _logit_sym_share(p_M, a, mu, a0, n)
    pi_M   = (p_M - c) * q_M

    return {
        "p_NE":                 p_NE,
        "q_NE":                 q_NE,
        "pi_NE":                pi_NE,
        "p_monopoly":           p_M,
        "q_monopoly_per_firm":  q_M,
        "pi_monopoly_per_firm": pi_M,
        "p_competitive":        float(c),
    }
