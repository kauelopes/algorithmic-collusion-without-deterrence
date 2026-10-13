"""
Monopoly player — always prices at the joint-monopoly price p_monopoly.

Config keys: p_monopoly
"""

from src.strategy_players.base import StrategyState


def init(rng, config: dict) -> StrategyState:
    return StrategyState()


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    price = float(config["p_monopoly"])
    return price, state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return float(config["p_monopoly"])


def update(state: StrategyState, transition, config: dict):
    return state, {}
