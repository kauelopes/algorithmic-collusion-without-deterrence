"""
Nash player — always prices at the Bertrand-Nash equilibrium price p_NE.

Config keys: p_NE
"""

import numpy as np
from src.strategy_players.base import StrategyState


def init(rng, config: dict) -> StrategyState:
    return StrategyState()


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    price = float(config["p_NE"])
    return price, state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return float(config["p_NE"])


def update(state: StrategyState, transition, config: dict):
    return state, {}
