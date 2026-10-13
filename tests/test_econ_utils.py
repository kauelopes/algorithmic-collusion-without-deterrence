"""
Unit tests for src/econ_utils — demand, cost, profit, and analytical benchmarks.

Canonical parameters used throughout: a=1, b=1, c=0, n=2.
Expected values (closed-form):
  Nash:        q_NE = 1/3,  Q_NE = 2/3,  p_NE = 1/3,  pi_NE = 1/9
  Monopoly:    Q_M  = 1/2,  p_M  = 1/2,  pi_M  = 1/4
  Competitive: Q_C  = 1,    p_C  = 0,    pi_C  = 0
"""

import pytest
import jax.numpy as jnp

from src.econ_utils.demand_functions import linear_demand
from src.econ_utils.cost_functions import linear_cost, profit
from src.econ_utils.benchmarks import nash_symmetric, monopoly, competitive

TOL = 1e-6
A, B, C, N = 1.0, 1.0, 0.0, 2


# ---------------------------------------------------------------------------
# demand_functions
# ---------------------------------------------------------------------------

class TestLinearDemand:
    def test_interior(self):
        assert abs(linear_demand(0.5, A, B) - 0.5) < TOL

    def test_zero_quantity(self):
        assert abs(linear_demand(0.0, A, B) - 1.0) < TOL

    def test_choke_price(self):
        # Q = a/b → p = 0
        assert abs(linear_demand(1.0, A, B) - 0.0) < TOL

    def test_floored_at_zero(self):
        # Q beyond choke point must not go negative
        assert linear_demand(2.0, A, B) == 0.0

    def test_jax_array_input(self):
        Q = jnp.array(0.5)
        result = linear_demand(Q, A, B)
        assert abs(float(result) - 0.5) < TOL


# ---------------------------------------------------------------------------
# cost_functions
# ---------------------------------------------------------------------------

class TestLinearCost:
    def test_zero_mc(self):
        assert linear_cost(0.5, 0.0) == 0.0

    def test_positive_mc(self):
        assert abs(linear_cost(2.0, 0.3) - 0.6) < TOL

    def test_jax_array_input(self):
        q = jnp.array(2.0)
        assert abs(float(linear_cost(q, 0.3)) - 0.6) < TOL


class TestProfit:
    def test_nash_profit(self):
        # At Nash: q=1/3, Q=2/3, p=1/3, pi=1/9
        q_NE = 1 / 3
        Q_NE = 2 / 3
        pi = profit(q_NE, Q_NE, A, B, C)
        assert abs(float(pi) - 1 / 9) < TOL

    def test_zero_quantity(self):
        assert profit(0.0, 0.0, A, B, C) == 0.0

    def test_profit_consistency(self):
        # pi = p(Q)*q - c*q
        q, Q = 0.4, 0.7
        expected = linear_demand(Q, A, B) * q - linear_cost(q, C)
        assert abs(float(profit(q, Q, A, B, C)) - float(expected)) < TOL

    def test_positive_mc_reduces_profit(self):
        q, Q = 1 / 3, 2 / 3
        pi_zero_cost = profit(q, Q, A, B, 0.0)
        pi_positive_cost = profit(q, Q, A, B, 0.1)
        assert pi_zero_cost > pi_positive_cost


# ---------------------------------------------------------------------------
# benchmarks — Nash
# ---------------------------------------------------------------------------

class TestNashSymmetric:
    def setup_method(self):
        self.ne = nash_symmetric(A, B, C, N)

    def test_q_per_firm(self):
        assert abs(self.ne["q_per_firm"] - 1 / 3) < TOL

    def test_Q_total(self):
        assert abs(self.ne["Q_total"] - 2 / 3) < TOL

    def test_price(self):
        assert abs(self.ne["price"] - 1 / 3) < TOL

    def test_profit_per_firm(self):
        assert abs(self.ne["profit_per_firm"] - 1 / 9) < TOL

    def test_price_consistent_with_demand(self):
        p_check = linear_demand(self.ne["Q_total"], A, B)
        assert abs(float(p_check) - self.ne["price"]) < TOL

    def test_profit_consistent(self):
        q = self.ne["q_per_firm"]
        Q = self.ne["Q_total"]
        pi_check = profit(q, Q, A, B, C)
        assert abs(float(pi_check) - self.ne["profit_per_firm"]) < TOL

    def test_n3_values(self):
        # n=3: q_NE = 1/4, Q_NE = 3/4, p_NE = 1/4
        ne3 = nash_symmetric(A, B, C, 3)
        assert abs(ne3["q_per_firm"] - 0.25) < TOL
        assert abs(ne3["Q_total"] - 0.75) < TOL
        assert abs(ne3["price"] - 0.25) < TOL


# ---------------------------------------------------------------------------
# benchmarks — Monopoly
# ---------------------------------------------------------------------------

class TestMonopoly:
    def setup_method(self):
        self.m = monopoly(A, B, C)

    def test_Q_total(self):
        assert abs(self.m["Q_total"] - 0.5) < TOL

    def test_price(self):
        assert abs(self.m["price"] - 0.5) < TOL

    def test_profit_total(self):
        assert abs(self.m["profit_total"] - 0.25) < TOL

    def test_price_consistent_with_demand(self):
        p_check = linear_demand(self.m["Q_total"], A, B)
        assert abs(float(p_check) - self.m["price"]) < TOL

    def test_monopoly_output_less_than_nash(self):
        ne = nash_symmetric(A, B, C, N)
        assert self.m["Q_total"] < ne["Q_total"]

    def test_monopoly_price_greater_than_nash(self):
        ne = nash_symmetric(A, B, C, N)
        assert self.m["price"] > ne["price"]


# ---------------------------------------------------------------------------
# benchmarks — Competitive
# ---------------------------------------------------------------------------

class TestCompetitive:
    def setup_method(self):
        self.c = competitive(A, B, C)

    def test_Q_total(self):
        assert abs(self.c["Q_total"] - 1.0) < TOL

    def test_price_equals_marginal_cost(self):
        assert abs(self.c["price"] - C) < TOL

    def test_zero_profit(self):
        assert self.c["profit_total"] == 0.0

    def test_competitive_output_greater_than_nash(self):
        ne = nash_symmetric(A, B, C, N)
        assert self.c["Q_total"] > ne["Q_total"]


# ---------------------------------------------------------------------------
# ordering invariants across benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarkOrdering:
    """
    Economic theory predicts:
      Q_M < Q_NE < Q_C   (monopoly restricts output most)
      p_M > p_NE > p_C   (monopoly charges highest price)
    """

    def setup_method(self):
        self.ne = nash_symmetric(A, B, C, N)
        self.m = monopoly(A, B, C)
        self.comp = competitive(A, B, C)

    def test_quantity_ordering(self):
        assert self.m["Q_total"] < self.ne["Q_total"] < self.comp["Q_total"]

    def test_price_ordering(self):
        assert self.m["price"] > self.ne["price"] > self.comp["price"]

    def test_nash_profit_between_monopoly_and_competitive(self):
        pi_NE_total = self.ne["profit_per_firm"] * N
        assert self.comp["profit_total"] < pi_NE_total < self.m["profit_total"]
