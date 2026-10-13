"""
Tests for src/environment/multi_agent_cournot_env.py

Canonical params: a=1, b=1, c=0, n=2.
  q_NE = 1/3, Q_NE = 2/3, p_NE = 1/3, pi_NE = 1/9 ≈ 0.1111
  q_monopoly_per_firm = 1/4, Q_monopoly = 1/2, pi_monopoly_total = 1/4
"""

import numpy as np
import pytest

from src.environment.multi_agent_cournot_env import MultiAgentCournotEnv

TOL = 1e-6

Q_NE = 1.0 / 3.0
PI_NE = 1.0 / 9.0
Q_MONO = 1.0 / 2.0
PI_MONO_TOTAL = 1.0 / 4.0


def make_env(n=2, info="bandit") -> MultiAgentCournotEnv:
    return MultiAgentCournotEnv({
        "demand": {"a": 1.0, "b": 1.0},
        "cost": {"c": 0.0},
        "market": {"n_firms": n},
        "state": {"info_structure": info},
    })


# ---------------------------------------------------------------------------
# Init and reset
# ---------------------------------------------------------------------------

class TestInit:
    def test_invalid_info_structure_raises(self):
        with pytest.raises(ValueError, match="Unknown info_structure"):
            make_env(info="unknown")

    def test_n_less_than_2_raises(self):
        with pytest.raises(ValueError):
            MultiAgentCournotEnv({
                "demand": {"a": 1.0, "b": 1.0},
                "cost": {"c": 0.0},
                "market": {"n_firms": 1},
                "state": {"info_structure": "bandit"},
            })

    def test_reset_returns_n_observations(self):
        env = make_env(n=2)
        obs = env.reset()
        assert len(obs) == 2

    def test_reset_n3_returns_3_observations(self):
        env = make_env(n=3)
        obs = env.reset()
        assert len(obs) == 3

    def test_reset_obs_are_arrays(self):
        env = make_env()
        obs = env.reset()
        for o in obs:
            assert isinstance(o, np.ndarray)

    def test_reset_deterministic(self):
        env = make_env()
        obs1 = env.reset(seed=0)
        obs2 = env.reset(seed=0)
        for o1, o2 in zip(obs1, obs2):
            np.testing.assert_array_equal(o1, o2)


# ---------------------------------------------------------------------------
# Observation shapes
# ---------------------------------------------------------------------------

class TestObsShape:
    def test_bandit_obs_shape(self):
        env = make_env(n=2, info="bandit")
        obs = env.reset()
        assert obs[0].shape == (1,)
        assert obs[1].shape == (1,)

    def test_price_obs_shape(self):
        env = make_env(n=2, info="price")
        obs = env.reset()
        assert obs[0].shape == (1,)

    def test_price_obs_shape_n3(self):
        env = make_env(n=3, info="price")
        obs = env.reset()
        assert obs[0].shape == (1,)

    def test_full_obs_shape_n2(self):
        env = make_env(n=2, info="full")
        obs = env.reset()
        # Each agent observes (n-1)=1 rival quantity
        assert obs[0].shape == (1,)
        assert obs[1].shape == (1,)

    def test_full_obs_shape_n3(self):
        env = make_env(n=3, info="full")
        obs = env.reset()
        # Each agent observes (n-1)=2 rival quantities
        assert obs[0].shape == (2,)

    def test_obs_size_method(self):
        assert make_env(n=2, info="bandit").obs_size() == 1
        assert make_env(n=2, info="price").obs_size() == 1
        assert make_env(n=2, info="full").obs_size() == 1
        assert make_env(n=3, info="full").obs_size() == 2


# ---------------------------------------------------------------------------
# Step correctness
# ---------------------------------------------------------------------------

class TestStepCorrectness:
    def test_step_returns_correct_structure(self):
        env = make_env()
        env.reset()
        obs, rewards, dones, info = env.step([0.3, 0.3])
        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(dones) == 2
        assert isinstance(info, dict)

    def test_dones_always_false(self):
        env = make_env()
        env.reset()
        _, _, dones, _ = env.step([0.3, 0.3])
        assert all(d is False for d in dones)

    def test_nash_quantities_give_nash_profit(self):
        """Both agents at q_NE → each earns pi_NE."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, info = env.step([Q_NE, Q_NE])
        assert rewards[0] == pytest.approx(PI_NE, abs=TOL)
        assert rewards[1] == pytest.approx(PI_NE, abs=TOL)
        assert info["Q_total"] == pytest.approx(2 * Q_NE, abs=TOL)

    def test_nash_quantities_give_nash_profit_n3(self):
        """N=3: each at q_NE(n=3) → correct symmetric payoff."""
        env = make_env(n=3)
        bm = env.benchmarks()
        q = bm["q_NE"]
        pi = bm["pi_NE"]
        env.reset()
        _, rewards, _, _ = env.step([q, q, q])
        for r in rewards:
            assert r == pytest.approx(pi, abs=TOL)

    def test_symmetric_play_equal_rewards(self):
        """Symmetric quantities → equal profits."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([0.25, 0.25])
        assert rewards[0] == pytest.approx(rewards[1], abs=TOL)

    def test_higher_rival_quantity_reduces_reward(self):
        """If rival increases output, own profit decreases (Cournot externality)."""
        env = make_env(n=2)
        env.reset()
        _, r_low, _, _ = env.step([Q_NE, 0.1])
        env.reset()
        _, r_high, _, _ = env.step([Q_NE, 0.6])
        assert r_low[0] > r_high[0]

    def test_total_quantity_in_info(self):
        env = make_env(n=2)
        env.reset()
        _, _, _, info = env.step([0.2, 0.3])
        assert info["Q_total"] == pytest.approx(0.5, abs=TOL)

    def test_price_consistent_with_demand(self):
        """p = a - b*Q = 1 - 0.5 = 0.5 when Q=0.5."""
        env = make_env(n=2)
        env.reset()
        _, _, _, info = env.step([0.25, 0.25])
        assert info["price"] == pytest.approx(1.0 - 0.5, abs=TOL)

    def test_action_clipped_to_competitive_limit(self):
        """Actions above Q_competitive should be clipped without error."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([999.0, 999.0])
        assert all(np.isfinite(r) for r in rewards)


# ---------------------------------------------------------------------------
# Observations per info_structure
# ---------------------------------------------------------------------------

class TestObservations:
    def test_bandit_obs_is_own_profit(self):
        env = make_env(n=2, info="bandit")
        env.reset()
        _, _, _, info = env.step([Q_NE, Q_NE])
        obs2, _, _, _ = env.step([Q_NE, Q_NE])
        # After first step, obs = own profit from previous step
        assert obs2[0][0] == pytest.approx(PI_NE, abs=TOL)
        assert obs2[1][0] == pytest.approx(PI_NE, abs=TOL)

    def test_price_obs_is_market_price(self):
        env = make_env(n=2, info="price")
        env.reset()
        env.step([Q_NE, Q_NE])    # execute step to update state
        obs2, _, _, info = env.step([Q_NE, Q_NE])
        # obs should be the price from the previous step
        assert obs2[0][0] == pytest.approx(info["price"], abs=TOL)

    def test_full_obs_contains_rival_quantities(self):
        """With n=2, full obs for agent 0 = agent 1's quantity."""
        env = make_env(n=2, info="full")
        env.reset()
        obs2, _, _, _ = env.step([0.2, 0.4])
        # obs updated to reflect last step's quantities
        # Agent 0's obs = rival (agent 1) quantity = 0.4
        assert obs2[0][0] == pytest.approx(0.4, abs=TOL)
        # Agent 1's obs = rival (agent 0) quantity = 0.2
        assert obs2[1][0] == pytest.approx(0.2, abs=TOL)

    def test_full_obs_n3_excludes_own_quantity(self):
        """With n=3, agent 0's obs has 2 entries (agents 1 and 2 quantities)."""
        env = make_env(n=3, info="full")
        env.reset()
        obs2, _, _, _ = env.step([0.1, 0.2, 0.3])
        assert obs2[0].shape == (2,)
        # Agent 0 sees agents 1 and 2 (order determined by np.delete)
        np.testing.assert_allclose(sorted(obs2[0].tolist()), [0.2, 0.3], atol=TOL)

    def test_price_obs_same_for_all_agents(self):
        """Price is a public signal — all agents observe the same value."""
        env = make_env(n=3, info="price")
        env.reset()
        obs2, _, _, _ = env.step([0.1, 0.2, 0.15])
        assert obs2[0][0] == pytest.approx(obs2[1][0], abs=TOL)
        assert obs2[1][0] == pytest.approx(obs2[2][0], abs=TOL)


# ---------------------------------------------------------------------------
# Counterfactual profits (CE-gap metric)
# ---------------------------------------------------------------------------

class TestCounterfactualProfits:
    def setup_method(self):
        self.env = make_env(n=2)
        self.env.reset()
        self.grid = np.linspace(0.0, 1.0, 20)

    def test_returns_array_of_correct_shape(self):
        self.env.step([Q_NE, Q_NE])
        cf = self.env.counterfactual_profits(0, self.grid)
        assert cf.shape == (20,)

    def test_actual_action_achieves_actual_reward(self):
        """Counterfactual profit at the exact actual action equals the reward."""
        _, rewards, _, _ = self.env.step([Q_NE, Q_NE])
        # Use a grid containing Q_NE exactly to avoid discretization error
        exact_grid = np.array([Q_NE])
        cf = self.env.counterfactual_profits(0, exact_grid)
        assert cf[0] == pytest.approx(rewards[0], abs=TOL)

    def test_best_response_maximises_counterfactual(self):
        """Given rival at q_NE, agent 0's best response maximises counterfactual."""
        self.env.step([Q_NE, Q_NE])
        cf = self.env.counterfactual_profits(0, self.grid)
        best_q = self.grid[np.argmax(cf)]
        # Best response to q_NE (for n=2) is also q_NE analytically
        assert abs(best_q - Q_NE) < 0.1  # within one grid step

    def test_counterfactual_for_agent1(self):
        """Works for any agent index."""
        self.env.step([0.2, 0.4])
        cf0 = self.env.counterfactual_profits(0, self.grid)
        cf1 = self.env.counterfactual_profits(1, self.grid)
        # Different rivals → different counterfactual profiles
        assert not np.allclose(cf0, cf1)

    def test_zero_quantity_gives_zero_profit(self):
        """Producing nothing → profit = 0."""
        self.env.step([Q_NE, Q_NE])
        cf = self.env.counterfactual_profits(0, np.array([0.0]))
        assert cf[0] == pytest.approx(0.0, abs=TOL)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_benchmarks_returns_dict(self):
        env = make_env()
        bm = env.benchmarks()
        assert isinstance(bm, dict)

    def test_nash_quantity_correct(self):
        env = make_env(n=2)
        bm = env.benchmarks()
        assert bm["q_NE"] == pytest.approx(Q_NE, abs=TOL)

    def test_monopoly_quantity_correct(self):
        env = make_env(n=2)
        bm = env.benchmarks()
        assert bm["Q_monopoly"] == pytest.approx(Q_MONO, abs=TOL)

    def test_nash_quantity_n3(self):
        """For n=3, a=1, b=1, c=0: q_NE = 1/(3+1) = 0.25."""
        env = make_env(n=3)
        bm = env.benchmarks()
        assert bm["q_NE"] == pytest.approx(0.25, abs=TOL)

    def test_monopoly_per_firm_is_total_over_n(self):
        env = make_env(n=2)
        bm = env.benchmarks()
        assert bm["q_monopoly_per_firm"] == pytest.approx(
            bm["Q_monopoly"] / 2, abs=TOL
        )
