"""
REINFORCE agent with Gaussian MLP policy for continuous action spaces.

The policy is a feedforward MLP that outputs (mean_raw, log_std_raw) for a
Gaussian distribution over action a ∈ [action_min, q_max].

Action parameterisation:
  mean = action_min + (q_max - action_min) * sigmoid(mean_raw)  →  always in (action_min, q_max)
  std  = softplus(log_std_raw) + 0.01                           →  always positive

Update rule:
  advantage = reward - baseline          (baseline = 0 when disabled)
  loss = -log π(a|s) * advantage  −  entropy_coef * H(π(·|s))

Stability features (all optional, controlled via config):
  lr_schedule    : cosine / linear / warmup_cosine / constant  (optim_utils)
  grad_clip_norm : gradient clipping by global norm            (optim_utils)
  reward_baseline: EMA of past rewards subtracted from return  (this module)

Interface is identical to q_learning.py:
  init(rng, config)                              → ReinforcePolicyState
  select_action(state, obs, rng, config, training) → (float, ReinforcePolicyState)
  greedy_action(state, obs, config)              → float
  update(state, transition, config)              → (ReinforcePolicyState, log_dict)

Config dict keys:
  q_max          : float       — action upper bound (p_max for Bertrand)
  action_min     : float       — action lower bound (default 0.0; use p_min for Bertrand)
  hidden_dims    : list[int]   — MLP hidden layer sizes, e.g. [64, 64]
  learning_rate  : float       — base Adam learning rate
  gamma          : float       — discount factor (interface parity; not used here)
  entropy_coef   : float       — weight on entropy regularisation term
  obs_dim        : int         — observation dimensionality (default 1)
  lr_schedule    : dict        — {type, decay_steps, lr_min, ...}  (optional)
  grad_clip_norm : float       — global norm clip threshold          (optional)
  reward_baseline: dict        — {type: ema, beta: 0.99}            (optional)
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from src.agents.q_learning import Transition  # shared across all agents
from src.agents.optim_utils import build_optimizer, optimizer_config_key


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ReinforcePolicyState(NamedTuple):
    params: Any            # Flax FrozenDict
    opt_state: Any         # Optax optimizer state
    step: int
    reward_ema: jnp.ndarray  # running reward baseline (scalar float32)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class _PolicyNet(nn.Module):
    hidden_dims: tuple

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for d in self.hidden_dims:
            x = nn.Dense(d)(x)
            x = nn.relu(x)
        mean_raw = nn.Dense(1)(x).squeeze(-1)
        log_std_raw = nn.Dense(1)(x).squeeze(-1)
        return mean_raw, log_std_raw


# ---------------------------------------------------------------------------
# JIT-compiled kernel cache
# ---------------------------------------------------------------------------

_forward_cache: dict = {}
_step_cache: dict = {}


def _baseline_config_key(config: dict) -> tuple:
    cfg = config.get("reward_baseline") or {"type": "none"}
    return (cfg.get("type", "none"), float(cfg.get("beta", 0.99)))


def _get_forward_fn(hidden_dims: tuple):
    """Return a JIT-compiled forward pass, compiled once per architecture."""
    if hidden_dims not in _forward_cache:
        net = _PolicyNet(hidden_dims=hidden_dims)

        @jax.jit
        def _fwd(params, obs_input, q_max, action_min):
            mean_raw, log_std_raw = net.apply(params, obs_input)
            mean = action_min + (q_max - action_min) * jax.nn.sigmoid(mean_raw)
            std = jax.nn.softplus(log_std_raw) + 0.01
            return mean, std

        _forward_cache[hidden_dims] = _fwd
    return _forward_cache[hidden_dims]


def _get_step_fn(hidden_dims: tuple, config: dict):
    """Return a JIT-compiled (forward + backward + optimizer + baseline) function."""
    cache_key = (hidden_dims, optimizer_config_key(config), _baseline_config_key(config))
    if cache_key not in _step_cache:
        net = _PolicyNet(hidden_dims=hidden_dims)
        optimizer = build_optimizer(config)

        baseline_cfg = config.get("reward_baseline") or {"type": "none"}
        use_baseline = baseline_cfg.get("type", "none") == "ema"
        beta = jnp.float32(baseline_cfg.get("beta", 0.99))

        @jax.jit
        def _step(params, opt_state, obs_input, action, reward, q_max,
                  action_min, entropy_coef, reward_ema):
            advantage = reward - reward_ema if use_baseline else reward

            def loss_fn(p):
                mean_raw, log_std_raw = net.apply(p, obs_input)
                mean = action_min + (q_max - action_min) * jax.nn.sigmoid(mean_raw)
                std = jax.nn.softplus(log_std_raw) + 0.01
                log_prob = (
                    -0.5 * ((action - mean) / std) ** 2
                    - jnp.log(std)
                    - 0.5 * jnp.log(2.0 * jnp.pi)
                )
                entropy = 0.5 * (1.0 + jnp.log(2.0 * jnp.pi * std ** 2))
                return -log_prob * advantage - entropy_coef * entropy, (entropy, mean)

            (loss, (entropy, mean)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            new_ema = beta * reward_ema + (1.0 - beta) * reward if use_baseline \
                      else reward_ema

            return new_params, new_opt_state, new_ema, loss, entropy, mean

        _step_cache[cache_key] = _step
    return _step_cache[cache_key]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _bias_mean_toward(params, init_q: float, q_max: float, action_min: float, hidden_dims: tuple):
    """
    Set the bias of the mean output head so that
    action_min + (q_max - action_min) * sigmoid(bias) ≈ init_q.
    """
    from flax.core import freeze, unfreeze
    span = q_max - action_min
    p = float(jnp.clip(jnp.float32((init_q - action_min) / span), 0.01, 0.99))
    bias_val = float(jnp.log(jnp.float32(p / (1.0 - p))))  # logit
    mean_dense_key = f"Dense_{len(hidden_dims)}"
    mutable = unfreeze(params)
    old_bias = mutable["params"][mean_dense_key]["bias"]
    mutable["params"][mean_dense_key]["bias"] = jnp.full_like(old_bias, bias_val)
    return freeze(mutable)


def _prep_obs(obs) -> jnp.ndarray:
    """Return a non-empty 1-D float32 array; bandit (empty obs) gets [1.0]."""
    arr = jnp.array(obs, dtype=jnp.float32)
    if arr.shape == () or arr.size == 0:
        return jnp.array([1.0], dtype=jnp.float32)
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def init(rng: jax.Array, config: dict) -> ReinforcePolicyState:
    """
    Initialise the policy network, optimizer, and reward baseline.

    Parameters
    ----------
    rng    : JAX PRNGKey used to initialise network weights
    config : dict with keys listed in module docstring
    """
    hidden_dims = tuple(int(d) for d in config["hidden_dims"])
    obs_dim = int(config.get("obs_dim", 1))

    net = _PolicyNet(hidden_dims=hidden_dims)
    dummy_obs = jnp.ones((max(1, obs_dim),), dtype=jnp.float32)

    params = net.init(rng, dummy_obs)

    if "init_q" in config:
        params = _bias_mean_toward(
            params,
            float(config["init_q"]),
            float(config["q_max"]),
            float(config.get("action_min", 0.0)),
            hidden_dims,
        )

    optimizer = build_optimizer(config)
    opt_state = optimizer.init(params)

    # Warm up JIT compilation so the first training step doesn't pay compile cost
    _get_forward_fn(hidden_dims)
    _get_step_fn(hidden_dims, config)

    return ReinforcePolicyState(
        params=params,
        opt_state=opt_state,
        step=0,
        reward_ema=jnp.float32(0.0),
    )


def select_action(
    agent_state: ReinforcePolicyState,
    obs,
    rng: jax.Array,
    config: dict,
    training: bool = True,
) -> tuple[float, ReinforcePolicyState]:
    """
    Sample from the Gaussian policy (training) or return the mean (eval).

    Returns
    -------
    action : float   continuous quantity in [0, q_max]
    state  : ReinforcePolicyState   unchanged (updated only in `update`)
    """
    hidden_dims    = tuple(int(d) for d in config["hidden_dims"])
    q_max_val      = float(config["q_max"])
    action_min_val = float(config.get("action_min", 0.0))
    obs_input      = _prep_obs(obs)
    q_max_arr      = jnp.float32(q_max_val)
    action_min_arr = jnp.float32(action_min_val)

    fwd = _get_forward_fn(hidden_dims)
    mean, std = fwd(agent_state.params, obs_input, q_max_arr, action_min_arr)

    if training:
        noise = jax.random.normal(rng, shape=mean.shape)
        action = mean + std * noise
    else:
        action = mean

    return float(jnp.clip(action, action_min_val, q_max_val)), agent_state


def greedy_action(
    agent_state: ReinforcePolicyState,
    obs,
    config: dict,
) -> float:
    """
    Return the policy mean (deterministic) — no sampling.
    Use this for convergence metrics, not for training steps.
    """
    hidden_dims    = tuple(int(d) for d in config["hidden_dims"])
    q_max_val      = float(config["q_max"])
    action_min_val = float(config.get("action_min", 0.0))
    obs_input      = _prep_obs(obs)
    q_max_arr      = jnp.float32(q_max_val)
    action_min_arr = jnp.float32(action_min_val)

    fwd = _get_forward_fn(hidden_dims)
    mean, _ = fwd(agent_state.params, obs_input, q_max_arr, action_min_arr)
    return float(jnp.clip(mean, action_min_val, q_max_val))


def update(
    agent_state: ReinforcePolicyState,
    transition: Transition,
    config: dict,
) -> tuple[ReinforcePolicyState, dict]:
    """
    REINFORCE gradient step (JIT-compiled).

    advantage = reward - reward_ema   (when reward_baseline.type == "ema")
    loss = -log π(a|s) * advantage  −  entropy_coef * H(π(·|s))
    """
    hidden_dims    = tuple(int(d) for d in config["hidden_dims"])
    q_max_val      = float(config["q_max"])
    action_min_val = float(config.get("action_min", 0.0))

    obs_input      = _prep_obs(transition.obs)
    action         = jnp.float32(transition.action)
    reward         = jnp.float32(transition.reward)
    q_max_arr      = jnp.float32(q_max_val)
    action_min_arr = jnp.float32(action_min_val)
    entropy_coef_arr = jnp.float32(config["entropy_coef"])

    step_fn = _get_step_fn(hidden_dims, config)
    new_params, new_opt_state, new_ema, loss, entropy, mean = step_fn(
        agent_state.params, agent_state.opt_state,
        obs_input, action, reward, q_max_arr, action_min_arr,
        entropy_coef_arr, agent_state.reward_ema,
    )

    new_state = ReinforcePolicyState(
        params=new_params,
        opt_state=new_opt_state,
        step=agent_state.step + 1,
        reward_ema=new_ema,
    )
    log = {
        "policy_loss": float(loss),
        "entropy":     float(entropy),
        "greedy_q":    float(jnp.clip(mean, action_min_val, q_max_val)),
        "reward_ema":  float(new_ema),
    }
    return new_state, log


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_state(state: ReinforcePolicyState, path) -> None:
    """Save policy params, step, and reward_ema to .npz (optimizer state not saved)."""
    import flax.serialization as fser
    params_bytes = fser.to_bytes(state.params)
    np.savez(
        path,
        params_bytes=np.frombuffer(params_bytes, dtype=np.uint8),
        step=np.array(state.step),
        reward_ema=np.array(float(state.reward_ema), dtype=np.float32),
    )


def load_state(path, config: dict) -> ReinforcePolicyState:
    """
    Load ReinforcePolicyState from .npz.

    Optimizer state is re-initialised from config (sufficient for evaluation;
    if resuming training, the Adam moments will be reset to zero).
    """
    import flax.serialization as fser
    data = np.load(path, allow_pickle=False)

    hidden_dims = tuple(int(d) for d in config["hidden_dims"])
    obs_dim = int(config.get("obs_dim", 1))
    net = _PolicyNet(hidden_dims=hidden_dims)
    dummy_obs = jnp.ones((max(1, obs_dim),), dtype=jnp.float32)
    dummy_params = net.init(jax.random.PRNGKey(0), dummy_obs)

    params = fser.from_bytes(dummy_params, bytes(data["params_bytes"].tobytes()))

    optimizer = build_optimizer(config)
    opt_state = optimizer.init(params)

    return ReinforcePolicyState(
        params=params,
        opt_state=opt_state,
        step=int(data["step"]),
        reward_ema=jnp.float32(float(data["reward_ema"])),
    )
