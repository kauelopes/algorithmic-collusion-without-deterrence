"""
Noise-Nash player — plays p_NE + N(0, noise_sigma), clipped to [p_min, p_max].

Models a firm that targets the Nash price but has imperfect price control.

Config keys: p_NE, p_min, p_max, noise_sigma (default 0.05)
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.strategy_players.base import StrategyState


def init(rng, config: dict) -> StrategyState:
    return StrategyState(own_last_price=float(config["p_NE"]))


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    p_NE = float(config["p_NE"])
    p_min = float(config["p_min"])
    p_max = float(config["p_max"])
    sigma = float(config.get("noise_sigma", 0.05))

    noise = float(jax.random.normal(rng)) * sigma
    price = float(np.clip(p_NE + noise, p_min, p_max))

    new_state = StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
    )
    return price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return float(config["p_NE"])


def update(state: StrategyState, transition, config: dict):
    return state, {}
