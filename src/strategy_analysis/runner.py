"""
Episode runner for trained-agent vs. strategy-player evaluation.

Runs a fixed-length episode with no exploration (training=False for both
agents) and returns per-step price and profit series.
"""

from __future__ import annotations

import jax

from src.agents.q_learning import Transition


def run_episode(
    env,
    agent_a_module,
    agent_a_state,
    agent_a_config: dict,
    agent_b_module,
    agent_b_state,
    agent_b_config: dict,
    n_steps: int,
    rng,
    trained_agent_idx: int = 0,
) -> dict:
    """
    Run one episode with agent_a (trained) vs agent_b (strategy).

    trained_agent_idx controls which environment slot agent_a occupies.
    agent_b occupies the remaining slot (only 2-agent envs supported).

    Returns
    -------
    dict with:
        price_a       : list[float]
        price_b       : list[float]
        profit_a      : list[float]
        profit_b      : list[float]
        steps         : list[int]
        strategy_name : str
    """
    b_idx = 1 - trained_agent_idx

    obs_list = env.reset()
    obs_a = obs_list[trained_agent_idx]
    obs_b = obs_list[b_idx]

    state_a = agent_a_state
    state_b = agent_b_state

    price_a: list[float] = []
    price_b: list[float] = []
    profit_a: list[float] = []
    profit_b: list[float] = []
    steps: list[int] = []

    for t in range(n_steps):
        rng, rng_a, rng_b = jax.random.split(rng, 3)

        action_a, state_a = agent_a_module.select_action(
            state_a, obs_a, rng_a, agent_a_config, training=False
        )
        action_b, state_b = agent_b_module.select_action(
            state_b, obs_b, rng_b, agent_b_config, training=False
        )

        actions = [None, None]
        actions[trained_agent_idx] = action_a
        actions[b_idx] = action_b

        next_obs_list, rewards, _, _ = env.step(actions)

        reward_a = float(rewards[trained_agent_idx])
        reward_b = float(rewards[b_idx])

        tr_b = Transition(
            obs=obs_b,
            action=action_b,
            reward=reward_b,
            next_obs=next_obs_list[b_idx],
            done=False,
        )
        state_b, _ = agent_b_module.update(state_b, tr_b, agent_b_config)

        price_a.append(float(action_a))
        price_b.append(float(action_b))
        profit_a.append(reward_a)
        profit_b.append(reward_b)
        steps.append(t + 1)

        obs_a = next_obs_list[trained_agent_idx]
        obs_b = next_obs_list[b_idx]

    return {
        "price_a": price_a,
        "price_b": price_b,
        "profit_a": profit_a,
        "profit_b": profit_b,
        "steps": steps,
        "strategy_name": agent_b_config.get("type", "unknown"),
    }
