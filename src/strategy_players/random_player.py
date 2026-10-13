"""
Random player — draws a price uniformly from [p_min, p_max] each period.

Config keys: p_min, p_max
"""

import jax
import jax.numpy as jnp
from src.strategy_players.base import StrategyState


def init(rng, config: dict) -> StrategyState:
    return StrategyState()


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    p_min = float(config["p_min"])
    p_max = float(config["p_max"])
    price = float(jax.random.uniform(rng, minval=p_min, maxval=p_max))
    new_state = StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
    )
    return price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    # Deterministic fallback: midpoint of price range
    return (float(config["p_min"]) + float(config["p_max"])) / 2.0


def update(state: StrategyState, transition, config: dict):
    return state, {}
