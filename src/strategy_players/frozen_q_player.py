"""
Frozen Q-table player — joga a política gulosa de uma Q-table congelada.

É como um snapshot histórico do agente aprendiz entra na população pela mesma porta
que as estratégias fixas: expõe a interface de strategy player
(`init / select_action / greedy_action / update`) e nunca aprende.

Config keys (montadas por `src.training.population`, não escritas à mão):
  _q_state      : QLearningState congelado (a Q-table NÃO é mutada aqui)
  n_actions, q_max, action_min, memory_length  : para o encoding de estado
  trained_slot  : índice de firma em que a Q-table foi TREINADA (default 0)
  own_idx       : índice de firma que este player ocupa AGORA  (default 1)
  label         : nome legível do snapshot (ex.: "snapshot@0500000")

⚠️ **Permutação de observação — a armadilha que custou um achado no p1.**
O estado é `s = idx(p_0)·n + idx(p_1)`, indexado por FIRMA, e os dois agentes recebem
o mesmo vetor de observação. Logo o dígito mais significativo é o preço próprio só para
a firma 0. Uma Q-table treinada no slot 0 e jogada no slot 1 sem permutar o obs seria
lida `(rival, own)` — transposta em relação ao que aprendeu. Quando
`trained_slot != own_idx`, este módulo troca as duas posições antes do lookup, de modo
que a tabela sempre veja a ordem em que foi treinada.

⚠️ Nunca chamar `q_learning.update` sobre `_q_state`: o update é **in-place**
(`q_learning.py:214`) e mutaria a tabela congelada — e, como snapshots compartilham
array com quem os criou se o `.copy()` for esquecido, a corrupção seria silenciosa.
Por isso `update` aqui é no-op.
"""

from __future__ import annotations

import numpy as np

from src.agents import q_learning
from src.strategy_players.base import StrategyState

# Não pode ser declarado como entrada de pool num YAML: exige `_q_state`, que só existe
# em tempo de execução. Entra na população pelo bloco `self_play`, não por `entries`.
DECLARABLE = False


def _present_obs(obs, config: dict):
    """Reordena o obs para a convenção de firma em que a Q-table foi treinada."""
    trained_slot = int(config.get("trained_slot", 0))
    own_idx      = int(config.get("own_idx", 1))
    if config.get("obs_egocentric"):
        # obs egocêntrico ([p_rival, features...], p3 c1b): já vem na perspectiva de
        # quem observa — permutar aqui trocaria o preço do rival por uma feature
        return obs
    if trained_slot == own_idx or obs is None:
        return obs
    arr = np.asarray(obs)
    if arr.ndim != 1 or arr.shape[0] <= max(trained_slot, own_idx):
        # bandit / own_price (obs escalar) — não há o que permutar
        return obs
    swapped = arr.copy()
    swapped[trained_slot], swapped[own_idx] = arr[own_idx], arr[trained_slot]
    return swapped


def init(rng, config: dict) -> StrategyState:
    # reinicia o autômato do snapshot a cada episódio
    config["_q_runtime"] = config["_q_state"]._replace(m=0, m_prev=0)
    return StrategyState()


def select_action(state: StrategyState, obs, rng, config: dict, training: bool = False):
    """Sempre guloso — um snapshot congelado não explora, independente de `training`.

    ⚠️ Com autômato (M>1) o snapshot precisa CARREGAR seu próprio estado interno ao longo
    do episódio; senão jogaria sempre como se estivesse em m=0 e o mecanismo que ele
    aprendeu (ex.: punição absorvente) nunca se manifestaria. Guardamos `m` em
    `StrategyState.step` não — usamos o campo dedicado do `_q_state` via `greedy_step`.
    """
    q_state = config.get("_q_runtime", config["_q_state"])
    price, q_state = q_learning.greedy_step(q_state, _present_obs(obs, config), config)
    config["_q_runtime"] = q_state          # o autômato avança dentro do episódio
    new_state = StrategyState(
        own_last_price=float(price),
        last_profit=state.last_profit,
        triggered=state.triggered,
        step=state.step + 1,
    )
    return float(price), new_state


def greedy_action(state: StrategyState, obs, config: dict) -> float:
    return float(q_learning.greedy_action(config["_q_state"], _present_obs(obs, config), config))


def update(state: StrategyState, transition, config: dict):
    """No-op: a Q-table é congelada. Só o bookkeeping do StrategyState avança."""
    new_state = StrategyState(
        own_last_price=float(transition.action),
        last_profit=float(transition.reward),
        triggered=state.triggered,
        step=state.step,
    )
    return new_state, {}
