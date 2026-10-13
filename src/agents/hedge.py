"""
Hedge (EXP3) agent for continuous Cournot action spaces.

Maintains a probability distribution over a discrete action grid using
multiplicative exponential weights (Hedge). In the bandit setting — where
only the reward for the played action is observed — uses importance-weighted
updates (EXP3 variant):

    r̂[i] = reward / p[i]   if action i was played, else 0
    log_w[i] += eta * r̂[i]

This guarantees sublinear external regret O(sqrt(T · K · log K)) where
K = n_actions, regardless of how rewards are generated (adversarial or
stochastic). In a stationary Cournot environment the policy converges to a
near-deterministic distribution over the Nash grid point.

Unlike Q-learning (value-based) and REINFORCE (policy-gradient), Hedge:
  — is observation-independent: maintains one weight vector, ignores obs
  — requires no reward baseline or entropy regularisation
  — has a single hyperparameter (eta) controlling the exploration rate

Interface is identical to q_learning.py and reinforce.py:
  init(rng, config)                               → HedgeState
  select_action(state, obs, rng, config, training) → (float, HedgeState)
  greedy_action(state, obs, config)               → float
  update(state, transition, config)               → (HedgeState, log_dict)

Config dict keys:
  q_max      : float — action upper bound (p_max for Bertrand)
  action_min : float — action lower bound (default 0.0; use p_min for Bertrand)
  n_actions  : int   — number of discrete grid points (default 20)
  eta        : float — learning rate for weight updates (default 0.01)
  obs_dim    : int   — ignored (Hedge is observation-independent)
"""

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

from src.agents.q_learning import Transition


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HedgeState(NamedTuple):
    log_weights: jnp.ndarray  # shape [n_actions], unnormalized log-weights
    step: int


# ---------------------------------------------------------------------------
# JIT-compiled kernel cache
# ---------------------------------------------------------------------------

_sample_cache: dict = {}
_step_cache: dict = {}


def _get_sample_fn(n_actions: int):
    """JIT-compiled action sampling; compiled once per n_actions."""
    if n_actions not in _sample_cache:
        @jax.jit
        def _sample(log_weights, rng, q_max, action_min):
            q_values = jnp.linspace(action_min, q_max, n_actions)
            idx = jax.random.categorical(rng, log_weights)
            return q_values[idx]

        _sample_cache[n_actions] = _sample
    return _sample_cache[n_actions]


def _get_step_fn(n_actions: int):
    """JIT-compiled EXP3 update; compiled once per n_actions."""
    if n_actions not in _step_cache:
        @jax.jit
        def _step(log_weights, action_float, reward, q_max, action_min, eta):
            q_values = jnp.linspace(action_min, q_max, n_actions)

            # Recover action index by nearest grid point
            action_idx = jnp.argmin(jnp.abs(q_values - action_float))

            # Current sampling distribution
            probs = jax.nn.softmax(log_weights)
            p_played = probs[action_idx]

            # EXP3 importance-weighted update (only played arm)
            r_hat = reward / (p_played + 1e-8)
            new_log_weights = log_weights.at[action_idx].add(eta * r_hat)

            # Diagnostics
            new_probs = jax.nn.softmax(new_log_weights)
            entropy = -jnp.sum(new_probs * jnp.log(new_probs + 1e-8))
            greedy_idx = jnp.argmax(new_log_weights)
            greedy_q = q_values[greedy_idx]

            return new_log_weights, entropy, greedy_q

        _step_cache[n_actions] = _step
    return _step_cache[n_actions]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def init(rng: jax.Array, config: dict) -> HedgeState:
    """
    Initialise Hedge with a uniform distribution over the action grid.

    Parameters
    ----------
    rng    : JAX PRNGKey (unused; kept for interface parity)
    config : dict with keys listed in module docstring
    """
    n_actions = int(config.get("n_actions", 20))

    # Warm up JIT compilation
    _get_sample_fn(n_actions)
    _get_step_fn(n_actions)

    return HedgeState(
        log_weights=jnp.zeros(n_actions, dtype=jnp.float32),
        step=0,
    )


def select_action(
    agent_state: HedgeState,
    obs,
    rng: jax.Array,
    config: dict,
    training: bool = True,
) -> tuple[float, HedgeState]:
    """
    Sample from the Hedge distribution (training) or return the argmax (eval).

    The obs argument is accepted for interface parity but ignored — Hedge is
    observation-independent in the bandit setting.

    Returns
    -------
    action : float   quantity in [0, q_max]
    state  : HedgeState   unchanged (weights updated only in `update`)
    """
    n_actions = int(config.get("n_actions", 20))
    q_max = float(config["q_max"])

    action_min = float(config.get("action_min", 0.0))

    if training:
        sample_fn = _get_sample_fn(n_actions)
        action = sample_fn(
            agent_state.log_weights, rng,
            jnp.float32(q_max), jnp.float32(action_min),
        )
    else:
        q_values = jnp.linspace(action_min, q_max, n_actions)
        action = q_values[jnp.argmax(agent_state.log_weights)]

    return float(action), agent_state


def greedy_action(
    agent_state: HedgeState,
    obs,
    config: dict,
) -> float:
    """
    Return the action with the highest weight (argmax policy).
    Deterministic; use this for convergence metrics.
    """
    n_actions  = int(config.get("n_actions", 20))
    q_max      = float(config["q_max"])
    action_min = float(config.get("action_min", 0.0))
    q_values   = jnp.linspace(action_min, q_max, n_actions)
    return float(q_values[jnp.argmax(agent_state.log_weights)])


def update(
    agent_state: HedgeState,
    transition: Transition,
    config: dict,
) -> tuple[HedgeState, dict]:
    """
    EXP3 importance-weighted weight update (JIT-compiled).

    r̂[played] = reward / p[played]
    log_w[played] += eta * r̂[played]
    """
    n_actions  = int(config.get("n_actions", 20))
    eta        = jnp.float32(config.get("eta", 0.01))
    q_max      = jnp.float32(config["q_max"])
    action_min = jnp.float32(config.get("action_min", 0.0))

    action = jnp.float32(transition.action)
    reward = jnp.float32(transition.reward)

    step_fn = _get_step_fn(n_actions)
    new_log_weights, entropy, greedy_q = step_fn(
        agent_state.log_weights, action, reward, q_max, action_min, eta
    )

    new_state = HedgeState(
        log_weights=new_log_weights,
        step=agent_state.step + 1,
    )
    log = {
        "policy_loss": float(-entropy),   # convention: negative entropy as "loss"
        "entropy":     float(entropy),
        "greedy_q":    float(greedy_q),
    }
    return new_state, log
