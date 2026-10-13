"""
Win-stay-lose-shift (WSLS) player.

Starts at p_NE. If last profit >= previous profit (or first step): keep same
price. If last profit < previous profit: shift price up by shift_delta.
If already at p_max when shifting, wrap around to p_NE.

⚠️ CORRIGIDO em 2026-08-12. O código comparava `last_profit >= 0.0` em vez de
`>= prev_profit`. Como o lucro é SEMPRE positivo neste ambiente (p > c em toda a
grade), a condição nunca falhava e o player ficava travado em p_NE para sempre —
ou seja, era um `nash_player` com outro nome. Isso ilumina retroativamente o achado
do p1 "nash ≡ grim ≡ wsls ≡ imitation em 24/24": para o `wsls` a identidade era
**por construção**, não confound do grim disparar em t=0.

Config keys: p_NE, p_max, shift_delta (default 0.05)
"""

from src.strategy_players.base import StrategyState


def init(rng, config: dict) -> StrategyState:
    return StrategyState(
        own_last_price=float(config["p_NE"]),
        last_profit=0.0,
    )


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    p_NE = float(config["p_NE"])
    p_max = float(config["p_max"])
    delta = float(config.get("shift_delta", 0.05))

    if state.step == 0:
        price = p_NE
    elif state.last_profit >= state.prev_profit:
        price = state.own_last_price
    else:
        if state.own_last_price >= p_max:
            price = p_NE
        else:
            price = min(state.own_last_price + delta, p_max)

    new_state = StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
        prev_profit=state.prev_profit,
    )
    return price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    p_NE = float(config["p_NE"])
    p_max = float(config["p_max"])
    delta = float(config.get("shift_delta", 0.05))

    if state.step == 0:
        return p_NE
    if state.last_profit >= state.prev_profit:
        return state.own_last_price
    if state.own_last_price >= p_max:
        return p_NE
    return min(state.own_last_price + delta, p_max)


def update(state: StrategyState, transition, config: dict):
    new_state = StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
        prev_profit=state.last_profit,   # o de agora vira o "anterior"
    )
    return new_state, {}
