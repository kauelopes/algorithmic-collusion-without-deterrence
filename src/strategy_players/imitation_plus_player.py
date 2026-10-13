"""
Imitation-plus player — imita o rival e ACRESCENTA: joga o preço do rival no período
anterior escalado por (1 + rate), limitado a p_max.

Modela uma firma que segue o rival mas sempre tenta puxar um pouco para cima. Diferente
do `imitation_player` puro, ela não apenas espelha: propõe uma escalada, e recua junto se
o rival cortar.

⚠️ CORRIGIDO em 2026-08-12. A versão anterior escalava o PRÓPRIO preço
(`own_last_price * (1+rate)`), o que a fazia subir sozinha até o teto e ficar lá,
ignorando o rival por completo — não era imitação nenhuma. O nome dizia "imitation" e o
comportamento era uma rampa monotônica.

Config keys: p_NE (fallback), p_max, rate (default 0.10), own_idx, n_agents
"""

from src.strategy_players.base import StrategyState, extract_rival_price


def init(rng, config: dict) -> StrategyState:
    return StrategyState(own_last_price=float(config["p_NE"]))


def _rival_plus(obs, config: dict) -> float:
    rival = extract_rival_price(
        obs,
        int(config.get("own_idx", 1)),
        int(config.get("n_agents", 2)),
        float(config["p_NE"]),
    )
    rate  = float(config.get("rate", 0.10))
    p_max = float(config["p_max"])
    return min(rival * (1.0 + rate), p_max)


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    price = _rival_plus(obs, config)
    return price, StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
        prev_profit=state.prev_profit,
    )


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return _rival_plus(obs, config)


def update(state: StrategyState, transition, config: dict):
    return StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
        prev_profit=state.last_profit,
    ), {}
