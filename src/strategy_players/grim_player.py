"""
Grim trigger player — cooperates at p_monopoly until any defection, then
punishes forever at p_NE.

Defection is detected when the rival's observed price falls below
p_NE + defect_threshold. Once triggered, the player never reverts.

Config keys: p_NE, p_monopoly, defect_threshold (default 0.05),
             own_idx (default 0), n_agents (default 2)
"""

from src.strategy_players.base import StrategyState, extract_rival_price


def init(rng, config: dict) -> StrategyState:
    return StrategyState(
        own_last_price=float(config["p_monopoly"]),
        triggered=False,
    )


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    p_NE = float(config["p_NE"])
    p_monopoly = float(config["p_monopoly"])
    threshold = float(config.get("defect_threshold", 0.05))
    own_idx = int(config.get("own_idx", 0))
    n_agents = int(config.get("n_agents", 2))

    triggered = state.triggered
    if not triggered:
        rival_price = extract_rival_price(obs, own_idx, n_agents, p_NE + threshold)
        if rival_price < p_NE + threshold:
            triggered = True

    price = p_NE if triggered else p_monopoly
    new_state = StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=triggered,
        step=state.step + 1,
    )
    return price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    p_NE = float(config["p_NE"])
    p_monopoly = float(config["p_monopoly"])
    threshold = float(config.get("defect_threshold", 0.05))
    own_idx = int(config.get("own_idx", 0))
    n_agents = int(config.get("n_agents", 2))

    triggered = state.triggered
    if not triggered:
        rival_price = extract_rival_price(obs, own_idx, n_agents, p_NE + threshold)
        if rival_price < p_NE + threshold:
            triggered = True

    return p_NE if triggered else p_monopoly


def update(state: StrategyState, transition, config: dict):
    new_state = StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
    )
    return new_state, {}
