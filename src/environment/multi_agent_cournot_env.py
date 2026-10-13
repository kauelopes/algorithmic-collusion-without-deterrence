"""
Multi-agent Cournot environment — JAX-optimised.

N learning agents simultaneously choose quantities each period. The
environment is non-stationary from any individual agent's perspective
because every firm adapts.

Core market arithmetic (step and counterfactual profits) is JIT-compiled
via module-level function caches, following the same pattern as the agent
modules. One compiled kernel per (n_firms, a, b, c, q_max) tuple; cache
persists across instantiations of the same market configuration.

Observation space per agent depends on info_structure:

  bandit : own profit this period         → shape (1,)
  price  : market price this period       → shape (1,)
  full   : rivals' quantities this period → shape (n-1,)

The environment also exposes counterfactual_profits() for CE-gap computation.
Counterfactuals are vmapped over the action grid in a single JIT call.

Config dict expected keys:
  demand.a, demand.b
  cost.c
  market.n_firms
  state.info_structure  ∈ {"bandit", "price", "full"}
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.econ_utils.benchmarks import competitive, monopoly, nash_symmetric


# ---------------------------------------------------------------------------
# JIT-compiled kernel cache
# ---------------------------------------------------------------------------

_step_cache: dict = {}
_cf_cache: dict = {}


def _get_step_fn(n: int, a: float, b: float, c: float, q_max: float):
    """
    Return a JIT-compiled market-clearing function for N firms.

    Compiled once per (n, a, b, c, q_max) configuration.
    Takes a float32 actions array [n], returns:
      q          : clipped quantities          [n]
      Q_total    : scalar
      price      : scalar
      profits    : per-firm profits            [n]
      rivals_q   : each agent's rivals' qtys  [n, n-1]
    """
    key = (n, a, b, c, q_max)
    if key not in _step_cache:
        # Precompute rival index matrix — shape [n, n-1], static per n
        rival_idx = jnp.array(
            [[j for j in range(n) if j != i] for i in range(n)]
        )
        a_ = jnp.float32(a)
        b_ = jnp.float32(b)
        c_ = jnp.float32(c)
        qm = jnp.float32(q_max)

        @jax.jit
        def _step(actions_f32: jnp.ndarray):
            q = jnp.clip(actions_f32, 0.0, qm)
            Q = q.sum()
            price = jnp.maximum(jnp.float32(0.0), a_ - b_ * Q)
            profits = (price - c_) * q
            rivals_q = q[rival_idx]          # integer-array gather: [n, n-1]
            return q, Q, price, profits, rivals_q

        _step_cache[key] = _step
    return _step_cache[key]


def _get_cf_fn(a: float, b: float, c: float, q_max: float):
    """
    Return a JIT+vmap counterfactual profits function.

    Compiled once per (a, b, c, q_max).
    Signature: (action_grid [K], Q_opponents scalar) → profits [K]
    """
    key = (a, b, c, q_max)
    if key not in _cf_cache:
        a_ = jnp.float32(a)
        b_ = jnp.float32(b)
        c_ = jnp.float32(c)
        qm = jnp.float32(q_max)

        def _single(q_prime: jnp.ndarray, Q_opp: jnp.ndarray) -> jnp.ndarray:
            qp = jnp.clip(q_prime, 0.0, qm)
            price = jnp.maximum(jnp.float32(0.0), a_ - b_ * (qp + Q_opp))
            return (price - c_) * qp

        _cf_cache[key] = jax.jit(jax.vmap(_single, in_axes=(0, None)))
    return _cf_cache[key]


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class MultiAgentCournotEnv:
    def __init__(self, config: dict):
        d = config["demand"]
        self.a: float = float(d["a"])
        self.b: float = float(d["b"])
        self.c: float = float(config["cost"]["c"])
        self.n: int = int(config["market"]["n_firms"])
        self.info_structure: str = config["state"]["info_structure"]

        if self.info_structure not in ("bandit", "price", "full"):
            raise ValueError(f"Unknown info_structure: {self.info_structure!r}")
        if self.n < 2:
            raise ValueError(f"n_firms must be >= 2, got {self.n}")

        self._nash = nash_symmetric(self.a, self.b, self.c, self.n)
        self._monopoly = monopoly(self.a, self.b, self.c)
        self._competitive = competitive(self.a, self.b, self.c)
        self._q_max: float = self._competitive["Q_total"]

        # Warm up JIT compilation at construction time
        self._step_fn = _get_step_fn(self.n, self.a, self.b, self.c, self._q_max)
        self._cf_fn = _get_cf_fn(self.a, self.b, self.c, self._q_max)

        # Last-step state (updated by step, read by _obs_for)
        self._last_q: np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._last_price: float = float(self.a)
        self._last_profits: np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._last_rivals_q: np.ndarray = np.zeros((self.n, self.n - 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0) -> list[np.ndarray]:
        """
        Reset environment. Returns initial observations for all N agents.

        Returns
        -------
        obs_list : list[np.ndarray], len=N — obs_list[i] is agent i's obs (zeros).
        """
        self._last_q = np.zeros(self.n, dtype=np.float32)
        self._last_price = float(self.a)
        self._last_profits = np.zeros(self.n, dtype=np.float32)
        self._last_rivals_q = np.zeros((self.n, self.n - 1), dtype=np.float32)
        return [self._obs_for(i) for i in range(self.n)]

    def step(
        self, actions: list[float]
    ) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
        """
        Execute one market period with all N agents acting simultaneously.

        Parameters
        ----------
        actions : list[float], len=N
            Quantity chosen by each agent.

        Returns
        -------
        obs_list : list[np.ndarray]   next observations, one per agent
        rewards  : list[float]        profits, one per agent
        dones    : list[bool]         always [False]*N (infinite horizon)
        info     : dict               full market outcome
        """
        actions_f32 = jnp.array(actions, dtype=jnp.float32)
        q, Q_total, price, profits, rivals_q = self._step_fn(actions_f32)

        # Update state (numpy views — cheap copy from device)
        self._last_q = np.array(q)
        self._last_price = float(price)
        self._last_profits = np.array(profits)
        self._last_rivals_q = np.array(rivals_q)

        rewards = self._last_profits.tolist()
        obs_list = [self._obs_for(i) for i in range(self.n)]
        dones = [False] * self.n

        info = {
            "quantities": self._last_q.tolist(),
            "Q_total": float(Q_total),
            "price": float(price),
            "rewards": rewards,
            "q_NE": self._nash["q_per_firm"],
            "Q_NE": self._nash["Q_total"],
            "nash_dist_mean": float(
                np.mean(np.abs(self._last_q - self._nash["q_per_firm"]))
            ),
        }
        return obs_list, rewards, dones, info

    def counterfactual_profits(
        self, agent_idx: int, action_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute agent i's profit for each action in action_grid, holding
        all other agents' last quantities fixed.

        Used to compute the CE-gap metric:
            CE-gap = max_{i, a'} E_mu[π_i(a', a_{-i}) − π_i(a_i, a_{-i})]

        Parameters
        ----------
        agent_idx   : int            index of the agent to evaluate
        action_grid : np.ndarray     shape [K], candidate actions for agent i

        Returns
        -------
        cf_profits : np.ndarray      shape [K], π_i(a', a_{-i}) for each a'
        """
        Q_opponents = jnp.float32(
            float(self._last_q.sum()) - float(self._last_q[agent_idx])
        )
        grid_f32 = jnp.array(action_grid, dtype=jnp.float32)
        cf = self._cf_fn(grid_f32, Q_opponents)
        return np.array(cf)

    def expected_counterfactual_profits(
        self, agent_idx: int, cf_grid: np.ndarray, actions_history: np.ndarray
    ) -> np.ndarray:
        """
        Expected counterfactual profit over the empirical action distribution.

        For each candidate quantity q' in cf_grid, computes
        E_t[π_i(q', q_{-i,t})] averaged over all T steps in actions_history.

        Parameters
        ----------
        agent_idx       : int
        cf_grid         : np.ndarray  shape (K,)
        actions_history : np.ndarray  shape (T, N) — quantities at each step

        Returns
        -------
        np.ndarray  shape (K,)
        """
        grid_f32 = jnp.array(cf_grid, dtype=jnp.float32)
        total    = np.zeros(len(cf_grid))
        for q_t in actions_history:
            Q_opponents = jnp.float32(float(q_t.sum()) - float(q_t[agent_idx]))
            total += np.array(self._cf_fn(grid_f32, Q_opponents))
        return total / len(actions_history)

    def benchmarks(self) -> dict:
        """Return all analytical benchmarks as a flat dict."""
        return {
            "q_NE": self._nash["q_per_firm"],
            "Q_NE": self._nash["Q_total"],
            "p_NE": self._nash["price"],
            "pi_NE": self._nash["profit_per_firm"],
            "q_monopoly_per_firm": self._monopoly["Q_total"] / self.n,
            "Q_monopoly": self._monopoly["Q_total"],
            "p_monopoly": self._monopoly["price"],
            "pi_monopoly_total": self._monopoly["profit_total"],
            "Q_competitive": self._competitive["Q_total"],
            "p_competitive": self._competitive["price"],
        }

    def obs_size(self) -> int:
        """Dimension of the observation vector for each agent."""
        if self.info_structure in ("bandit", "price"):
            return 1
        return self.n - 1  # full

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obs_for(self, agent_idx: int) -> np.ndarray:
        """Build obs for agent i from cached last-step values (no recomputation)."""
        if self.info_structure == "bandit":
            return np.array([self._last_profits[agent_idx]], dtype=np.float32)
        elif self.info_structure == "price":
            return np.array([self._last_price], dtype=np.float32)
        else:  # full
            return self._last_rivals_q[agent_idx].astype(np.float32)
