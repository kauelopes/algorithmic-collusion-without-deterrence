"""
Inverse demand functions for Cournot markets.

All functions map total quantity Q to market price p(Q).
Pure functions — no side effects, JAX-compatible (jnp ops, no Python control
flow on traced values).
"""

import jax.numpy as jnp


def linear_demand(Q, a: float, b: float):
    """
    Linear inverse demand: p(Q) = max(0, a - b*Q).

    Parameters
    ----------
    Q : float or jax.Array
        Total market quantity (sum of all firms).
    a : float
        Demand intercept (choke price).
    b : float
        Demand slope (price sensitivity to quantity).

    Returns
    -------
    float or jax.Array
        Market price. Floored at zero (quantity beyond choke point yields p=0).
    """
    return jnp.maximum(0.0, a - b * Q)
