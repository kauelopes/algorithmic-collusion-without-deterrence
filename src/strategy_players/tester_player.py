"""
Tester player (c5_tesoura) — roteiro determinístico de desvios periódicos.

Coopera em p_monopoly e corta para p_low por EXATAMENTE 1 período a cada
tester_k períodos, a partir do período tester_phase (0-based, contado pelo
próprio state.step — não condiciona no rival nem no relógio do ambiente):

    corta  ⟺  t >= tester_phase  e  (t - tester_phase) % tester_k == 0

É o instrumento do braço T1/T5: contra os cortes REAIS do roteiro, executar a
punição é melhor resposta — a lâmina "auto-refutável" da tesoura vira testável.
`tester_k > total_steps` (ou tester_phase > total_steps) dá o braço "kinf"
que nunca corta (atrofia por ausência de desvio).

Config keys: tester_k (default 4), tester_phase (default 3),
             p_low (default p_NE), p_monopoly, p_NE
"""

from src.strategy_players.base import StrategyState


def _price(step: int, config: dict) -> float:
    k = int(config.get("tester_k", 4))
    phase = int(config.get("tester_phase", 3))
    p_low = float(config.get("p_low", config["p_NE"]))
    p_monopoly = float(config["p_monopoly"])
    cut = step >= phase and (step - phase) % k == 0
    return p_low if cut else p_monopoly


def init(rng, config: dict) -> StrategyState:
    return StrategyState(own_last_price=float(config["p_monopoly"]))


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    price = _price(state.step, config)
    new_state = StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
    )
    return price, new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return _price(state.step, config)


def update(state: StrategyState, transition, config: dict):
    new_state = StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
    )
    return new_state, {}
