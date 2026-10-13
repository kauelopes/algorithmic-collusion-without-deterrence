"""
Cost functions for Cournot firms.

All functions map firm quantity q_i to total cost c_i(q_i).
Pure functions — no side effects, JAX-compatible.
"""


def linear_cost(q, c: float):
    """
    Linear cost: c(q) = c * q.

    Parameters
    ----------
    q : float or jax.Array
        Quantity produced by the firm.
    c : float
        Marginal cost (constant).

    Returns
    -------
    float or jax.Array
        Total cost.
    """
    return c * q


def profit(q, Q_total, a: float, b: float, c: float):
    """
    Firm profit given own quantity and total market quantity.

    pi(q, Q) = p(Q) * q - cost(q)
             = max(0, a - b*Q) * q - c * q

    Parameters
    ----------
    q : float or jax.Array
        Own quantity.
    Q_total : float or jax.Array
        Total market quantity (including own).
    a, b : float
        Linear demand parameters.
    c : float
        Marginal cost.

    Returns
    -------
    float or jax.Array
        Firm profit.
    """
    from src.econ_utils.demand_functions import linear_demand
    price = linear_demand(Q_total, a, b)
    return price * q - linear_cost(q, c)
