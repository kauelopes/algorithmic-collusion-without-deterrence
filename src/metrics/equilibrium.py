"""
Equilibrium metrics for Cournot RL experiments.

All computations use jax.numpy. Functions that return Python scalars or
strings convert at the boundary via float() / int() — required outside JIT.
"""

import jax.numpy as jnp
import numpy as np


def nash_distance(q: float, q_NE: float) -> float:
    return float(jnp.abs(jnp.array(q) - jnp.array(q_NE)))


def collusion_index(Q_total: float, Q_NE_total: float, Q_monopoly: float) -> float:
    """
    CI = (Q_NE - Q_total) / (Q_NE - Q_monopoly)
    CI = 0 → Nash,  CI = 1 → Monopoly,  CI < 0 → more competitive than Nash.
    """
    Q_total    = jnp.array(Q_total)
    Q_NE_total = jnp.array(Q_NE_total)
    Q_monopoly = jnp.array(Q_monopoly)
    denom = Q_NE_total - Q_monopoly
    ci = jnp.where(jnp.abs(denom) < 1e-12, 0.0, (Q_NE_total - Q_total) / denom)
    return float(ci)


def dynamics_label(q_series: np.ndarray, q_NE: float) -> str:
    """
    Classify dynamics from a quantity time series (last 20% used).

    fixed_point : std(q) < 0.01 * q_NE
    unstable    : std(q) > 0.5  * q_NE
    cycling     : lag-1 autocorrelation > 0.3
    converging  : otherwise
    """
    arr = jnp.array(q_series)
    n   = arr.shape[0]
    if n < 10:
        return "insufficient_data"

    tail  = arr[int(0.8 * n):]
    std_q = float(jnp.std(tail))

    if std_q < 0.01 * q_NE:
        return "fixed_point"
    if std_q > 0.5 * q_NE:
        return "unstable"

    if tail.shape[0] > 2:
        mean    = jnp.mean(tail)
        centered = tail - mean
        var     = jnp.var(tail)
        autocorr = float(
            jnp.where(
                var > 1e-12,
                jnp.mean(centered[:-1] * centered[1:]) / var,
                0.0,
            )
        )
        if abs(autocorr) > 0.3:
            return "cycling"

    return "converging"


def convergence_step(
    nash_dist_series: np.ndarray,
    steps: np.ndarray,
    threshold: float,
) -> int:
    """First step at which nash_dist stays below threshold permanently. -1 if never."""
    arr = jnp.array(nash_dist_series)
    for i in range(arr.shape[0]):
        if bool(jnp.all(arr[i:] < threshold)):
            return int(steps[i])
    return -1
