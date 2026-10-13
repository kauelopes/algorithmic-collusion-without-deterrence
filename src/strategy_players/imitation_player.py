"""
Imitation player — AJUSTE PARCIAL na direção do preço do rival.

`p_t = p_own_{t-1} + adjust · (p_rival_{t-1} − p_own_{t-1})`

Segue o rival, mas devagar: cobre uma fração `adjust` da distância a cada período. É
rigidez de preço / ajuste parcial, a forma mais comum de imitação na literatura de
oligopólio — e, ao contrário da imitação pura, **preserva memória do próprio nível**,
então dois desses convergem gradualmente em vez de trocarem de preço num passo.

Com `adjust = 1.0` degenera em imitação pura; o default 0.5 mantém o comportamento
distinto dos vizinhos:

| player | regra |
|---|---|
| `tit_for_tat_player` | `p_rival_{t-1}` — cópia exata |
| `imitation_plus_player` | `p_rival_{t-1} + shift` — cópia com acréscimo |
| **`imitation_player`** | **ajuste parcial na direção do rival** |

⚠️ **CORRIGIDO DUAS VEZES, e o segundo bug nasceu do primeiro conserto.**

A versão original repetia o PRÓPRIO preço, nunca lia o `obs`, e ficava travada em p_NE —
era um `nash_player` com outro nome. O conserto de 2026-08-12 a fez copiar o preço do
rival, o que a tornou **bit-idêntica ao `tit_for_tat_player`**: medido no A5 da c4, o gap
de desvio contra os dois era igual até a sexta casa decimal, nos 8 runs. O pool `full`
declarava 10 oponentes e tinha 8 comportamentos, sobre-representando "espelhar o rival"
em 2×.

Daí o ajuste parcial: cumpre o nome (imita) sem duplicar um vizinho. E daí também o
teste de unicidade em `tests/test_strategy_players.py`, que barra qualquer par de players
com trajetória idêntica contra o mesmo rival.

Config keys: p_NE (fallback do primeiro passo), adjust (default 0.5), own_idx, n_agents
"""

from src.strategy_players.base import StrategyState, extract_rival_price


def init(rng, config: dict) -> StrategyState:
    return StrategyState(own_last_price=float(config["p_NE"]))


def _rival(obs, config: dict) -> float:
    return extract_rival_price(
        obs,
        int(config.get("own_idx", 1)),
        int(config.get("n_agents", 2)),
        float(config["p_NE"]),          # fallback: obs ausente ou pequeno demais
    )


def _target(state: StrategyState, obs, config: dict) -> float:
    """Ajuste parcial: cobre `adjust` da distância até o preço do rival."""
    rival = _rival(obs, config)
    adjust = float(config.get("adjust", 0.5))
    own = float(state.own_last_price)
    return own + adjust * (rival - own)


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    price = _target(state, obs, config)
    return price, StrategyState(
        own_last_price=price,
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
        prev_profit=state.prev_profit,
    )


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return _target(state, obs, config)


def update(state: StrategyState, transition, config: dict):
    # `own_last_price` recebe o preço EFETIVAMENTE jogado (o env pode ter discretizado),
    # que é o ponto de partida correto do próximo ajuste.
    return StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
        prev_profit=state.last_profit,
    ), {}
