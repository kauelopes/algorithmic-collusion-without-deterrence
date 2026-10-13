"""
Tit-for-tat player — copies the rival's price from the previous period.

Starts at p_monopoly. If rival prices cannot be observed (obs too small),
falls back to p_NE. The player directly mirrors the rival's last price;
it does not use defect_threshold for action selection (only the grim player
does). defect_threshold is accepted in config for interface consistency but
is not used here.

Config keys: p_NE, p_monopoly, own_idx (default 0), n_agents (default 2)
"""

from src.strategy_players.base import StrategyState, extract_rival_price


def init(rng, config: dict) -> StrategyState:
    return StrategyState(own_last_price=float(config["p_monopoly"]))


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    own_idx = int(config.get("own_idx", 0))
    n_agents = int(config.get("n_agents", 2))
    fallback = float(config["p_NE"])

    rival_price = extract_rival_price(obs, own_idx, n_agents, fallback)
    new_state = StrategyState(
        own_last_price=rival_price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
    )
    return rival_price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    own_idx = int(config.get("own_idx", 0))
    n_agents = int(config.get("n_agents", 2))
    fallback = float(config["p_NE"])
    return extract_rival_price(obs, own_idx, n_agents, fallback)


def update(state: StrategyState, transition, config: dict):
    new_state = StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
    )
    return new_state, {}
