"""
Tabular Q-Learning agent for discrete Cournot action spaces.

Action space: q_i ∈ {0, Δ, 2Δ, ..., q_max} with n_actions grid points.
State space:  last memory_length own quantities, each discretized to the
              same grid. Total states = n_actions^memory_length (or 1 if
              memory_length=0).

Interface is uniform across all agents in this project:
  init(rng, config)           → AgentState
  select_action(state, obs, rng, training) → (action_float, AgentState)
  update(state, transition)   → (AgentState, log_dict)

JAX PRNGs are used for all randomness so the interface is compatible with
neural-network agents. The Q-table itself is a numpy array — tabular lookup
does not benefit from JIT compilation.

Config dict keys:
  n_actions          : int   — number of discrete action levels
  q_max              : float — maximum action value (mapped to n_actions-1)
  action_min         : float — minimum action value (default 0.0); use p_min for Bertrand
  alpha              : float — learning rate
  gamma              : float — discount factor
  epsilon_start      : float — initial exploration rate
  epsilon_end        : float — final exploration rate
  epsilon_decay_steps: int   — steps over which epsilon decays
  epsilon_decay_type : str   — "linear" (default) or "geometric" (Calvano et al.)
  memory_length      : int   — history length (must match env config)
  q_init             : str   — "zeros" (default), "uniform" or "opponent_optimized"
  q_init_value       : float — valor do `uniform`. Sem ele usa π_NE/(1-γ)=4.459, que fica
                               ABAIXO do teto realista π_M/(1-γ)=6.75 e portanto NÃO é
                               otimista. Para otimismo de verdade passe > 6.75.
  n_internal         : int   — M, estados internos do AUTÔMATO (default 1 = sem autômato)
  alpha_mode         : str   — "global" (default) ou "count"
  alpha_omega        : float — expoente do modo count (default 0.7)
  alpha_start/end    : float — no modo count viram TETO e PISO de alpha
  alpha_decay_steps  : int   — só no modo global

Q-table initialisation modes
  zeros              : Q[:, :] = 0 — argmax always returns index 0 (p_min) before learning
  uniform            : Q[:, :] = q_init_value (ou π_Nash/(1-γ)) — remove o viés do p_min
  opponent_optimized : Q[:, p] = mean_{rivals} π(p, rivals) / (1-γ) — Calvano "type O"
                       Requires caller to pass q_init_table via config["_q_init_table"].
                       ⚠️ Replica UMA linha em todas as linhas ⇒ política inicial
                       constante ⇒ atrator global POR CONSTRUÇÃO (CI de nulo +0.2857).
                       É o nulo relevante, não o aleatório uniforme.

Estado interno (`n_internal: M > 1`) — autômato finito
  s = (estado público)·M + m        a = idx(preço)·M + m'
  O agente escolhe o preço E para onde sua memória vai, o que torna representáveis grim
  (M=2, punição absorvente) e punição por T períodos (M=T+1) — que a memória-1 sobre o par
  de preços NÃO representa: ali o gatilho é o estado, não a história.
  ⚠️ M=1 é bit-a-bit idêntico ao comportamento histórico. É o portão da refatoração,
     verificado por teste contra uma Q-table real do p1.
  Análise de agentes-autômato: `src/qtable_analysis/automaton.py`.

Taxa de aprendizado por CONTAGEM (`alpha_mode: count`)
  alpha(s,a) = clip(1/(1+n(s,a))^omega, alpha_end, alpha_start)   — Robbins-Monro
  O decaimento GLOBAL trata todas as células igual: uma visitada pela primeira vez no
  passo 12M recebia alpha=0.01 e nunca separava as ações. O teto em `alpha_start` é o que
  preserva a inicialização otimista — sem ele a 1ª visita teria alpha=1.0.

`visits` (contagem por (s,a)) é SEMPRE alocado e salvo no .npz — o mapa de visitação vem
de graça em todo run. `load_state` aceita arquivos antigos sem essa chave.

⚠️ Armadilhas deste módulo em `docs/dev-notes/armadilhas_p2.md` (§2 NamedTuple
reconstruído perde campos, §8 decaimento global de alpha, §11 mutação in-place).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class QLearningState(NamedTuple):
    q_table: np.ndarray   # shape: (n_states·M, n_actions·M)
    epsilon: float
    step: int
    m: int = 0            # estado interno ATUAL do autômato (após a última ação)
    m_prev: int = 0       # estado interno ANTES da última ação — `update` precisa dos dois
    visits: np.ndarray | None = None   # contagem por (s,a); ver `alpha_mode: count`
    # --- variantes do p3 (etapa 7) — todos None no baseline, que fica bit-exato ---
    q_table2: np.ndarray | None = None    # Double Q: a segunda tabela (B)
    trace: np.ndarray | None = None       # Q(λ) de Watkins: eligibility trace
    model_next: np.ndarray | None = None  # Dyna-Q: s' determinístico por (s,a); -1 = não visto
    model_rew: np.ndarray | None = None   # Dyna-Q: recompensa por (s,a)
    model_seen: list | None = None        # Dyna-Q: (s,a) na ordem da 1ª visita
    model_pri: np.ndarray | None = None   # Dyna-Q prioritized: |TD| corrente por (s,a)


# ---------------------------------------------------------------------------
# Estado interno (autômato) — M = 1 é exatamente o comportamento histórico
# ---------------------------------------------------------------------------
#
# Com M > 1 o agente deixa de ser uma política Markov sobre o estado público e vira um
# AUTÔMATO FINITO (arcabouço de Abreu-Rubinstein). Ele escolhe, a cada passo, o preço
# E para onde sua memória vai:
#
#     estado :  s = (idx(p_0)·n + idx(p_1))·M + m
#     ação   :  a = idx(preço)·M + m'
#
# Isso torna representáveis estratégias que a memória-1 sobre o par de preços NÃO
# representa — grim (M=2, punição absorvente) e punição por T períodos (M=T+1). Ver
# docs/theory/representacao_de_estado_e_estrategias.md §5-6.
#
# `m` é INTERNO, não privado: as transições dependem só do estado público, então o
# oponente poderia inferi-lo. Isso é deliberado — dispositivo de coordenação observável
# (Abreu-Pearce-Stacchetti). Aleatorização privada cairia em monitoramento privado, onde
# a teoria diz que coordenação fica MAIS difícil.
#
# ⚠️ M = 1 tem de ser bit-a-bit idêntico ao comportamento anterior: s·1+0 = s e
#    idx(p)·1+0 = idx(p). É o portão que torna a refatoração segura.


def n_internal(config: dict) -> int:
    """Número de estados internos do autômato. 1 = agente sem memória interna."""
    return max(1, int(config.get("n_internal", 1)))


# ---------------------------------------------------------------------------
# Variantes do p3 (etapa 7): Dyna-Q, Q(λ), Double Q — config-ativadas, default OFF
# ---------------------------------------------------------------------------
#
#   planning:  {enabled: true, k_updates: 10, model_scope: global|per_episode}
#   lambda:    0.8                (Q(λ) de Watkins — corta o trace no passo não-guloso)
#   double_q:  true               (duas tabelas, seleção/avaliação cruzadas)
#
# São MUTUAMENTE EXCLUSIVAS (validado em `init`) — combiná-las multiplicaria os modos
# de falha sem braço de controle correspondente na triagem. Com todas OFF, `update` e
# as seleções seguem exatamente o caminho histórico (pino:
# `tests/test_q_learning.py::test_variantes_off_e_bit_exato`).
#
# ⚠️ Dyna-Q: o replay usa uma sequência DETERMINÍSTICA (hash do step) sobre as células
# vistas, não um RNG — mantém o run reproduzível e o resume bit-exato sem enfiar um
# gerador dentro do estado. Contra um pool, o modelo mistura oponentes;
# `model_scope: per_episode` zera o modelo na rotação (via `reset_episode_state`).


def _variant_of(config: dict) -> str | None:
    on = [k for k, flag in (
        ("planning", bool((config.get("planning") or {}).get("enabled", False))),
        ("lambda", config.get("lambda") is not None),
        ("double_q", bool(config.get("double_q", False))),
    ) if flag]
    if len(on) > 1:
        raise ValueError(f"variantes de q_learning são mutuamente exclusivas: {on}")
    return on[0] if on else None


def _q_row(state: QLearningState, s: int) -> np.ndarray:
    """Linha de Q usada para AGIR. Com Double Q, a soma A+B (mesmo argmax da média)."""
    if state.q_table2 is None:
        return state.q_table[s]
    return state.q_table[s] + state.q_table2[s]


def _ucb_action(state: QLearningState, s: int, c: float) -> int:
    """Exploração dirigida por contagem (p3, c1b): argmax de Q + c·√(ln t / (1+n)).

    Substitui o ε-greedy quando `agent.exploration: {type: ucb, c: ...}` — em vez de
    explorar uniformemente, o agente é puxado para as células SUB-VISITADAS do estado
    corrente, que é onde o SCE do p2 mora (off-path sub-explorado). Determinística:
    nenhum sorteio. `epsilon` continua decaindo no estado, mas não é consultado.
    """
    row = _q_row(state, s)
    if state.visits is None:
        return int(np.argmax(row))
    bonus = c * np.sqrt(np.log(state.step + 2.0) / (1.0 + state.visits[s]))
    return int(np.argmax(row + bonus))


def reset_episode_state(state: QLearningState, config: dict) -> QLearningState:
    """Hook de fronteira de episódio para os trainers: zera o trace do Q(λ) (crédito
    não atravessa um reset de mercado) e, com `model_scope: per_episode`, o modelo do
    Dyna (o episódio seguinte é contra OUTRO oponente)."""
    if state.trace is not None:
        state.trace[:] = 0.0
    if (state.model_next is not None
            and str((config.get("planning") or {}).get("model_scope", "global"))
            == "per_episode"):
        state.model_next[:] = -1
        state.model_rew[:] = 0.0
        state.model_seen.clear()
        if state.model_pri is not None:
            state.model_pri[:] = 0.0
    return state


class Transition(NamedTuple):
    obs: np.ndarray
    action: float
    reward: float
    next_obs: np.ndarray
    done: bool


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def init(rng: jax.Array, config: dict) -> QLearningState:
    """
    Initialise Q-table and epsilon.

    Parameters
    ----------
    rng    : JAX PRNGKey (not used here, kept for interface uniformity)
    config : dict with keys listed in module docstring

    Q-table initialisation is controlled by config["q_init"]:
      "zeros"              — all zeros (default; biased toward p_min at argmax)
      "uniform"            — uniform value π_Nash/(1-γ); requires config["_q_init_value"]
      "opponent_optimized" — per-action mean payoff/(1-γ); requires config["_q_init_table"]
                             with shape (n_actions,) — broadcast across all states
    """
    n_actions     = config["n_actions"]
    memory_length = config["memory_length"]
    M             = n_internal(config)
    radices       = config.get("obs_radices")
    if radices is not None:
        # estado expandido (p3, etapa 8): o espaço público é o produto dos radices
        n_pub = 1
        for r in radices:
            if int(r) < 2:
                raise ValueError(f"obs_radices com radix inválido: {r}")
            n_pub *= int(r)
    else:
        n_pub = n_actions ** memory_length if memory_length > 0 else 1
    n_states      = n_pub * M
    n_cols        = n_actions * M
    q_init_mode   = config.get("q_init", "zeros")

    if q_init_mode == "crafted":
        # c4_estabilidade: tabela COMPLETA fabricada (Bellman-consistente), injetada
        # pelo trainer via `_q_init_full` (e opcionalmente `_q_init_visits`). O
        # conteúdo entra na impressão digital da campanha via hash do .npz.
        q_table = np.array(config["_q_init_full"], dtype=np.float64, copy=True)
        if q_table.shape != (n_states, n_cols):
            raise ValueError(
                f"q_init crafted: tabela {q_table.shape} não bate com "
                f"(n_states={n_states}, n_cols={n_cols})"
            )
    elif q_init_mode == "opponent_optimized":
        row = np.asarray(config["_q_init_table"], dtype=np.float64)  # shape (n_actions,)
        # replicado em M para não favorecer nenhuma transição de m — se o q_init
        # preferisse um m', o autômato seria plantado em vez de aprendido
        q_table = np.tile(np.repeat(row, M), (n_states, 1))
    elif q_init_mode == "uniform":
        fill = float(config["_q_init_value"])
        q_table = np.full((n_states, n_cols), fill, dtype=np.float64)
    else:
        q_table = np.zeros((n_states, n_cols), dtype=np.float64)

    variant = _variant_of(config)
    return QLearningState(
        q_table=q_table,
        epsilon=float(config["epsilon_start"]),
        step=0,
        m=0,
        m_prev=0,
        # contagem sempre alocada (int32, ~54 KB para 13.500 células). Serve ao
        # `alpha_mode: count` e, de graça, dá o mapa de visitação que o p1 precisou
        # de um experimento dedicado para obter (FIG-01). `_q_init_visits` (c4) é o
        # botão de "convicção": visitas plantadas ⇒ α menor por célula desde o início.
        visits=(np.array(config["_q_init_visits"], dtype=np.int32, copy=True)
                if config.get("_q_init_visits") is not None
                else np.zeros_like(q_table, dtype=np.int32)),
        q_table2=(np.array(q_table, copy=True) if variant == "double_q" else None),
        trace=(np.zeros_like(q_table) if variant == "lambda" else None),
        model_next=(np.full(q_table.shape, -1, dtype=np.int64)
                    if variant == "planning" else None),
        model_rew=(np.zeros_like(q_table) if variant == "planning" else None),
        model_seen=([] if variant == "planning" else None),
        model_pri=(np.zeros_like(q_table)
                   if variant == "planning"
                   and str((config.get("planning") or {}).get("mode", "uniform"))
                   == "prioritized" else None),
    )


def _exploration_banned_set(config: dict) -> frozenset | None:
    """Máscara de exploração (c5, braço T4): `exploration_mask_states: [[i, j], ...]`
    em coordenadas (own, rival) da grade — nesses estados o ε-greedy age GULOSO
    ("não audite suas ameaças"). A expansão para linhas planas depende do slot
    (`own_slot`: agente 0 lê (own, rival); agente 1, (rival, own)) e cobre os M
    estados internos. Cacheada no próprio config (cópia por slot no trainer).
    Ausente ⇒ None, caminho bit-exato. Só suporta o layout plano (sem obs_radices)."""
    mask = config.get("exploration_mask_states")
    if not mask:
        return None
    cached = config.get("_exploration_banned")
    if cached is None:
        if config.get("obs_radices"):
            raise ValueError("exploration_mask_states não suporta obs_radices")
        n = int(config["n_actions"])
        slot = int(config.get("own_slot", 0))
        M = n_internal(config)
        rows: set[int] = set()
        for i, j in mask:
            base = (int(i) * n + int(j)) if slot == 0 else (int(j) * n + int(i))
            for m in range(M):
                rows.add(base * M + m)
        cached = frozenset(rows)
        config["_exploration_banned"] = cached
    return cached


def _oracle_tables(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Tabelas do Dyna estratégico (c5, T3): a partir de `planning.oracle_file`
    (.npz com `pi` — a política do RIVAL nas coordenadas (own, rival) DELE — e
    `P[a, b]` — lucro de estágio de quem joga `a` contra `b`), pré-computa
    `next[s, a] = s'` e `rew[s, a] = P[a, B[s]]`, na convenção de slot
    (`own_slot`: o estado é sempre `s = i_f0·n + i_f1`, indexado por FIRMA).
    Cacheadas no config (cópia por slot no trainer). Só layout plano (M=1)."""
    cached = config.get("_oracle_tables")
    if cached is not None:
        return cached
    if n_internal(config) > 1 or config.get("obs_radices"):
        raise ValueError("planning.opponent_model oracle só suporta o layout plano "
                         "(M=1, sem obs_radices)")
    pcfg = config.get("planning") or {}
    data = np.load(pcfg["oracle_file"])
    pi, P = data["pi"].astype(int), data["P"]
    n = pi.shape[0]
    slot = int(config.get("own_slot", 0))
    nxt = np.empty((n * n, n), dtype=np.int64)
    rew = np.empty((n * n, n))
    for i0 in range(n):
        for i1 in range(n):
            s = i0 * n + i1
            for a in range(n):
                if slot == 0:
                    b = int(pi[i1, i0])          # rival = firma 1: own=i1, rival=i0
                    nxt[s, a] = a * n + b
                else:
                    b = int(pi[i0, i1])          # rival = firma 0: own=i0, rival=i1
                    nxt[s, a] = b * n + a
                rew[s, a] = float(P[a, b])
    cached = (nxt, rew)
    config["_oracle_tables"] = cached
    return cached


def select_action(
    agent_state: QLearningState,
    obs: np.ndarray,
    rng: jax.Array,
    config: dict,
    training: bool = True,
) -> tuple[float, QLearningState]:
    """
    Epsilon-greedy action selection.

    Returns
    -------
    action : float   continuous value in [action_min, q_max]
    state  : QLearningState   unchanged (epsilon updated only in `update`)
    """
    n_actions  = config["n_actions"]
    q_max      = float(config["q_max"])
    action_min = float(config.get("action_min", 0.0))
    M          = n_internal(config)

    state_idx = _state_idx(obs, agent_state.m, n_actions, q_max, action_min,
                           config["memory_length"], M, config.get("obs_radices"))

    rng_eps, rng_act = jax.random.split(rng)

    _expl = config.get("exploration") or {}
    if training and str(_expl.get("type", "")) == "ucb":
        a_comp = _ucb_action(agent_state, state_idx, float(_expl.get("c", 1.0)))
    elif training and float(jax.random.uniform(rng_eps)) < agent_state.epsilon:
        _banned = _exploration_banned_set(config)
        if _banned is not None and state_idx in _banned:
            a_comp = int(np.argmax(_q_row(agent_state, state_idx)))
        else:
            a_comp = int(jax.random.randint(rng_act, shape=(), minval=0, maxval=n_actions * M))
    else:
        a_comp = int(np.argmax(_q_row(agent_state, state_idx)))

    price_idx, m_next = divmod(a_comp, M)
    action = _idx_to_action(price_idx, n_actions, q_max, action_min)
    return action, agent_state._replace(m_prev=agent_state.m, m=m_next)


def select_action_numpy(
    agent_state: QLearningState,
    obs: np.ndarray,
    eps_roll: float,
    rand_act: int,
    config: dict,
    training: bool = True,
) -> tuple[float, QLearningState]:
    """Action selection using pre-generated random values (no JAX dispatch).

    Equivalent to select_action but accepts pre-generated floats instead of a
    JAX PRNGKey. Use when the caller pre-batches all random numbers to avoid
    per-step JAX dispatch overhead in tight training loops.
    """
    n_actions  = config["n_actions"]
    q_max      = float(config["q_max"])
    action_min = float(config.get("action_min", 0.0))
    M          = n_internal(config)
    state_idx  = _state_idx(obs, agent_state.m, n_actions, q_max, action_min,
                            config["memory_length"], M, config.get("obs_radices"))

    _expl = config.get("exploration") or {}
    if training and str(_expl.get("type", "")) == "ucb":
        a_comp = _ucb_action(agent_state, state_idx, float(_expl.get("c", 1.0)))
    elif training and eps_roll < agent_state.epsilon:
        # `rand_act` vem pré-gerado em [0, n_actions); com M>1 o espaço composto é
        # M vezes maior, então o caller tem de pré-gerar em [0, n_actions·M).
        _banned = _exploration_banned_set(config)
        if _banned is not None and state_idx in _banned:
            a_comp = int(np.argmax(_q_row(agent_state, state_idx)))
        else:
            a_comp = int(rand_act)
    else:
        a_comp = int(np.argmax(_q_row(agent_state, state_idx)))

    price_idx, m_next = divmod(a_comp, M)
    return (_idx_to_action(price_idx, n_actions, q_max, action_min),
            agent_state._replace(m_prev=agent_state.m, m=m_next))


def greedy_action(
    agent_state: QLearningState,
    obs: np.ndarray,
    config: dict,
) -> float:
    """
    Return the greedy (argmax Q) action without any exploration.
    Use this for convergence metrics — not for training steps.
    """
    n_actions  = config["n_actions"]
    q_max      = float(config["q_max"])
    action_min = float(config.get("action_min", 0.0))
    M          = n_internal(config)
    state_idx  = _state_idx(obs, agent_state.m, n_actions, q_max, action_min,
                            config["memory_length"], M, config.get("obs_radices"))
    price_idx  = int(np.argmax(_q_row(agent_state, state_idx))) // M
    return _idx_to_action(price_idx, n_actions, q_max, action_min)


def greedy_step(
    agent_state: QLearningState,
    obs: np.ndarray,
    config: dict,
) -> tuple[float, QLearningState]:
    """Como `greedy_action`, mas AVANÇA o estado interno.

    `greedy_action` é uma consulta (não move `m`); para simular uma trajetória gulosa
    com M > 1 é preciso carregar o autômato, e é isso que esta função faz.
    """
    n_actions  = config["n_actions"]
    q_max      = float(config["q_max"])
    action_min = float(config.get("action_min", 0.0))
    M          = n_internal(config)
    state_idx  = _state_idx(obs, agent_state.m, n_actions, q_max, action_min,
                            config["memory_length"], M, config.get("obs_radices"))
    price_idx, m_next = divmod(int(np.argmax(_q_row(agent_state, state_idx))), M)
    return (_idx_to_action(price_idx, n_actions, q_max, action_min),
            agent_state._replace(m_prev=agent_state.m, m=m_next))


def update(
    agent_state: QLearningState,
    transition: Transition,
    config: dict,
) -> tuple[QLearningState, dict]:
    """
    Apply Q-learning update rule, with optional linear alpha and epsilon decay.

    Q(s,a) ← Q(s,a) + α_t * [r + γ * max_a' Q(s',a') − Q(s,a)]

    Alpha schedule (optional): if alpha_start / alpha_end / alpha_decay_steps
    are present in config, alpha decays linearly from alpha_start to alpha_end.
    Falls back to fixed config["alpha"] when those keys are absent.
    """
    n_actions     = config["n_actions"]
    q_max         = float(config["q_max"])
    action_min    = float(config.get("action_min", 0.0))
    gamma         = float(config["gamma"])
    memory_length = config["memory_length"]

    alpha_start = float(config.get("alpha_start", config["alpha"]))
    alpha_end   = float(config.get("alpha_end",   config["alpha"]))

    M  = n_internal(config)
    # s usa o m ANTES da ação; a ação composta e s' usam o m DEPOIS. Com M=1 isto
    # colapsa exatamente no índice histórico.
    s  = _state_idx(transition.obs,      agent_state.m_prev, n_actions, q_max, action_min,
                    memory_length, M, config.get("obs_radices"))
    s_ = _state_idx(transition.next_obs, agent_state.m,      n_actions, q_max, action_min,
                    memory_length, M, config.get("obs_radices"))
    a  = _action_to_idx(float(transition.action), n_actions, q_max, action_min) * M + agent_state.m

    visits = agent_state.visits
    if visits is None:                       # estado carregado de um .npz antigo
        visits = np.zeros_like(agent_state.q_table, dtype=np.int32)

    if str(config.get("alpha_mode", "global")) == "count":
        alpha = _alpha_by_count(
            int(visits[s, a]), alpha_start, alpha_end,
            float(config.get("alpha_omega", 0.7)),
        )
    else:
        alpha = _decay_alpha(
            step=agent_state.step, alpha_start=alpha_start, alpha_end=alpha_end,
            decay_steps=int(config.get("alpha_decay_steps", 0)),
        )
    visits[s, a] += 1

    if agent_state.q_table2 is not None:
        # Double Q — alternância determinística pelo passo (par → atualiza A, ímpar → B),
        # seleção pela soma A+B (ver _q_row). O td_error logado é o da tabela atualizada.
        if agent_state.step % 2 == 0:
            q_upd, q_est = agent_state.q_table, agent_state.q_table2
        else:
            q_upd, q_est = agent_state.q_table2, agent_state.q_table
        a_star    = int(np.argmax(q_upd[s_]))
        td_target = transition.reward + gamma * float(q_est[s_, a_star])
        td_error  = td_target - q_upd[s, a]
        q_upd[s, a] += alpha * td_error
    elif agent_state.trace is not None:
        # Q(λ) de Watkins — trace substitutivo; corta no passo não-guloso (a política
        # alvo é a gulosa; crédito de exploração não retropropaga).
        q, e = agent_state.q_table, agent_state.trace   # locais: NamedTuple é imutável,
        was_greedy = a == int(np.argmax(q[s]))          # mas os arrays mutam in-place
        td_target  = transition.reward + gamma * float(np.max(q[s_]))
        td_error   = td_target - q[s, a]
        lam = float(config["lambda"])
        e[s, a] = 1.0
        q += alpha * td_error * e
        if was_greedy:
            e *= gamma * lam
        else:
            e[:] = 0.0
    else:
        td_target = transition.reward + gamma * float(np.max(agent_state.q_table[s_]))
        td_error  = td_target - agent_state.q_table[s, a]

        # In-place update — training is single-threaded; copy is unnecessary
        agent_state.q_table[s, a] += alpha * td_error

    _pcfg = config.get("planning") or {}
    if agent_state.model_next is not None and str(_pcfg.get("opponent_model", "none")) == "oracle":
        # Dyna ESTRATÉGICO (c5, braço T3) — o replay NÃO usa o modelo de mundo
        # aprendido: usa a função de reação do RIVAL (oráculo da fabricação) e a
        # recompensa ANALÍTICA de estágio. `s' = (a, B[s])` e `r = P[a, B[s]]` —
        # nunca `model_rew` (off-path não tem recompensa realizada, e a realizada
        # carrega exatamente a história que o braço quer neutralizar).
        # Replay uniforme (hash do step) sobre TODAS as (s, a) — sem isso as células
        # de punição jamais seriam re-derivadas e o braço não testaria nada.
        # `visits` intocado pelo replay (mesma regra do Dyna de mundo).
        _o_next, _o_rew = _oracle_tables(config)
        k = int(_pcfg.get("k_updates", 10))
        q = agent_state.q_table
        alpha_count = str(config.get("alpha_mode", "global")) == "count"

        def _alpha_of(ps, pa):
            return (_alpha_by_count(int(visits[ps, pa]), alpha_start, alpha_end,
                                    float(config.get("alpha_omega", 0.7)))
                    if alpha_count else alpha)

        _total = _o_next.size
        _n_cols = q.shape[1]
        for i in range(k):
            flat = (agent_state.step * 2654435761 + i * 97) % _total
            ps, pa = divmod(flat, _n_cols)
            ns = int(_o_next[ps, pa])
            q[ps, pa] += _alpha_of(ps, pa) * (
                float(_o_rew[ps, pa]) + gamma * float(np.max(q[ns])) - q[ps, pa])
    elif agent_state.model_next is not None:
        # Dyna-Q — modelo tabular determinístico + k updates de replay por passo real.
        # `visits` NÃO é incrementado por replay: alpha_mode=count conta amostras
        # REAIS, e o replay usa o α corrente da célula sem consumi-lo.
        #
        # mode: uniform      — sequência hash do step sobre as células vistas
        #       prioritized  — prioritized sweeping: replay da célula de MAIOR |TD|
        #                      corrente (argmax sobre `model_pri`); depois do replay a
        #                      prioridade da célula vira o |TD| novo, então células
        #                      resolvidas saem sozinhas da fila. Determinístico
        #                      (argmax do numpy é o primeiro índice em empate).
        if agent_state.model_next[s, a] < 0:
            agent_state.model_seen.append((s, a))
        agent_state.model_next[s, a] = s_
        agent_state.model_rew[s, a]  = transition.reward
        seen = agent_state.model_seen
        k = int((config.get("planning") or {}).get("k_updates", 10))
        q = agent_state.q_table
        n_seen = len(seen)
        alpha_count = str(config.get("alpha_mode", "global")) == "count"

        def _alpha_of(ps, pa):
            return (_alpha_by_count(int(visits[ps, pa]), alpha_start, alpha_end,
                                    float(config.get("alpha_omega", 0.7)))
                    if alpha_count else alpha)

        pri = agent_state.model_pri
        if pri is not None:
            pri[s, a] = abs(td_error)
            n_cols_ = q.shape[1]
            for _ in range(k):
                flat = int(np.argmax(pri))
                if pri.flat[flat] <= 0.0:
                    break
                ps, pa = divmod(flat, n_cols_)
                ns = int(agent_state.model_next[ps, pa])
                td_p = (agent_state.model_rew[ps, pa]
                        + gamma * float(np.max(q[ns])) - q[ps, pa])
                q[ps, pa] += _alpha_of(ps, pa) * td_p
                pri[ps, pa] = abs(td_p) * (1.0 - _alpha_of(ps, pa))
        else:
            for i in range(min(k, n_seen)):
                ps, pa = seen[(agent_state.step * 2654435761 + i * 97) % n_seen]
                ns = int(agent_state.model_next[ps, pa])
                q[ps, pa] += _alpha_of(ps, pa) * (
                    agent_state.model_rew[ps, pa]
                    + gamma * float(np.max(q[ns])) - q[ps, pa])

    new_q_table = agent_state.q_table

    new_epsilon = _decay_epsilon(
        step=agent_state.step,
        epsilon_start=float(config["epsilon_start"]),
        epsilon_end=float(config["epsilon_end"]),
        decay_steps=int(config["epsilon_decay_steps"]),
        decay_type=str(config.get("epsilon_decay_type", "linear")),
    )

    # ⚠️ PRESERVAR m/m_prev. Reconstruir o NamedTuple sem eles usa os defaults (0,0) e
    # ZERA o autômato a cada passo — o agente escolheria m'=1 e nunca chegaria a m=1.
    # Foi exatamente o bug que invalidou a primeira c3: 100% das linhas m=1 intocadas.
    # (E o mesmo vale para os campos de variante — por isso _replace, não reconstrução.)
    new_state = agent_state._replace(
        q_table=new_q_table,
        epsilon=new_epsilon,
        step=agent_state.step + 1,
        visits=visits,
    )
    log = {"td_error": td_error, "epsilon": new_epsilon, "alpha": alpha}
    return new_state, log


# ---------------------------------------------------------------------------
# Q-table initialisation helpers
# ---------------------------------------------------------------------------

def opponent_optimized_qtable(
    n_actions: int,
    action_min: float,
    q_max: float,
    n_rivals: int,
    gamma: float,
    a: float,
    mu: float,
    c: float,
    a0: float = 0.0,
) -> np.ndarray:
    """Compute Calvano "type O" initialisation vector (shape: n_actions,).

    Q[p_i] = mean_{all rival combos} π_i(p_i, rivals) / (1 - γ)

    Rivals are assumed to draw uniformly from the same price grid.
    For N=2 this is mean over 15 rival prices; for N=3 over 15^2 combos.
    The result is broadcast across all Q-table rows by init().
    """
    grid = np.array([
        action_min + i / (n_actions - 1) * (q_max - action_min)
        for i in range(n_actions)
    ])

    q_row = np.zeros(n_actions, dtype=np.float64)
    rival_combos = list(_cartesian_product([grid] * n_rivals))

    for p_i_idx, p_i in enumerate(grid):
        total = 0.0
        for rivals in rival_combos:
            logit_i = np.exp((a - p_i) / mu)
            denom   = np.exp(a0 / mu) + logit_i + sum(np.exp((a - pj) / mu) for pj in rivals)
            share_i = logit_i / denom
            total  += (p_i - c) * share_i
        q_row[p_i_idx] = (total / len(rival_combos)) / (1.0 - gamma)

    return q_row


def _cartesian_product(arrays):
    """Yield all tuples from the cartesian product of a list of arrays."""
    if not arrays:
        yield ()
        return
    for item in arrays[0]:
        for rest in _cartesian_product(arrays[1:]):
            yield (item,) + rest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _state_idx(
    obs: np.ndarray,
    m: int,
    n_actions: int,
    q_max: float,
    action_min: float,
    memory_length: int,
    M: int = 1,
    radices: list[int] | None = None,
) -> int:
    """Índice de linha da Q-table: `(estado público)·M + m`.

    Com M = 1 devolve exatamente `_obs_to_state_idx` — o portão de compatibilidade.
    """
    s_pub = _obs_to_state_idx(obs, n_actions, q_max, action_min, memory_length, radices)
    return s_pub * M + (int(m) % M if M > 1 else 0)


def _obs_to_state_idx(
    obs: np.ndarray,
    n_actions: int,
    q_max: float,
    action_min: float,
    memory_length: int,
    radices: list[int] | None = None,
) -> int:
    """Map continuous obs array to flat Q-table row index.

    Sem `radices` (o caminho histórico, BIT-EXATO): todo componente do obs é um preço,
    discretizado pela grade, radix uniforme `n_actions`.

    Com `radices` (p3, etapa 8 — `agent.obs_radices`): mixed-radix heterogêneo. A
    convenção que separa preço de feature é o próprio radix: componente com
    `radices[k] == n_actions` é PREÇO (passa pela grade); qualquer outro valor é uma
    FEATURE CATEGÓRICA que o env já emite como inteiro (só clampada ao suporte).
    `len(radices)` tem de ser `len(obs)` — divergir é erro, não truncamento.
    """
    if radices is not None:
        if len(radices) != len(obs):
            raise ValueError(
                f"obs_radices tem {len(radices)} dígitos para um obs de {len(obs)} "
                f"componentes — o config não bate com o env (state.features?)"
            )
        idx = 0
        for v, r in zip(obs, radices):
            r = int(r)
            if r == n_actions:
                i = _action_to_idx(float(v), n_actions, q_max, action_min)
            else:
                i = int(round(float(v)))
                i = 0 if i < 0 else (r - 1 if i > r - 1 else i)
            idx = idx * r + i
        return int(idx)

    if memory_length == 0:
        return 0
    indices = [_action_to_idx(float(v), n_actions, q_max, action_min) for v in obs]
    # Mixed-radix encoding: most-recent value is least significant digit
    idx = 0
    for i in indices:
        idx = idx * n_actions + i
    return int(idx)


def _action_to_idx(v: float, n_actions: int, q_max: float, action_min: float) -> int:
    """Discretise a continuous action value to the nearest grid index.

    Clamp em Python puro em vez de `np.clip`: 0.10 µs contra 3.00 µs (30×), e a função é
    chamada ~14 vezes por passo. BIT-EXATO — `round` e comparações não envolvem
    aritmética nova. Ver docs/dev-notes/todo_optimizations.md OPT-02.
    """
    span = q_max - action_min
    raw  = (v - action_min) / span * (n_actions - 1) if span > 0 else 0.0
    i    = round(raw)
    return 0 if i < 0 else (n_actions - 1 if i > n_actions - 1 else int(i))


def _idx_to_action(idx: int, n_actions: int, q_max: float, action_min: float) -> float:
    """Convert a grid index back to its continuous action value."""
    return action_min + float(idx) / (n_actions - 1) * (q_max - action_min)


# Backward-compatible aliases (used by existing Cournot code via config action_min=0)
def _quantity_to_idx(q: float, n_actions: int, q_max: float) -> int:
    return _action_to_idx(q, n_actions, q_max, 0.0)


def _idx_to_quantity(idx: int, n_actions: int, q_max: float) -> float:
    return _idx_to_action(idx, n_actions, q_max, 0.0)


def _decay_epsilon(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    decay_steps: int,
    decay_type: str = "linear",
) -> float:
    """Epsilon decay — linear (default) or geometric (Calvano et al.).

    Geometric: ε_t = ε_start * (1 − δ)^t, where δ is chosen so that
    ε reaches ε_end exactly at decay_steps.
    """
    if decay_type == "geometric":
        if epsilon_start <= 0 or epsilon_end <= 0 or decay_steps <= 0:
            return epsilon_end
        delta = 1.0 - (epsilon_end / epsilon_start) ** (1.0 / decay_steps)
        return max(epsilon_end, epsilon_start * (1.0 - delta) ** step)
    # linear (default)
    frac = min(1.0, step / decay_steps)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


def _alpha_by_count(
    n: int,
    alpha_start: float,
    alpha_end: float,
    omega: float,
) -> float:
    """`alpha(s,a) = clip(1/(1+n)^omega, alpha_end, alpha_start)` — Robbins-Monro.

    **Por que isto e não o decaimento global.** O decaimento por PASSO trata todas as
    células igual: uma visitada pela primeira vez no passo 12M recebe o mesmo
    `alpha=0.01` de uma visitada um milhão de vezes, e nunca separa as ações. Foi o que
    a margem mediana de 0.03 nos braços populacionais mediu (contra 0.39 no co-treino).
    Com contagem, célula rara continua aprendendo e célula saturada estabiliza — que é
    a condição clássica de convergência do Q-learning, não um truque.

    ⚠️ **O teto em `alpha_start` não é cosmético.** Sem ele a primeira visita teria
    `alpha = 1.0` e o Q saltaria direto para o alvo TD, ANULANDO a inicialização
    otimista — cujo mecanismo é justamente decair devagar do valor alto, forçando
    revisita. Com o teto em 0.15 e omega=0.7, as **15 primeiras visitas** mantêm 0.15 e
    só depois o decaimento por contagem assume.

    Sequência com omega=0.7 (teto 0.15, piso 0.01):
        n=0..14 → 0.150   n=15 → 0.144   n=100 → 0.040   n=1000 → 0.010
    """
    if n < 0:
        n = 0
    a = 1.0 / (1.0 + n) ** omega
    return float(min(alpha_start, max(alpha_end, a)))


def _decay_alpha(
    step: int,
    alpha_start: float,
    alpha_end: float,
    decay_steps: int,
) -> float:
    """Linear alpha decay. Returns alpha_start unchanged when decay_steps == 0."""
    if decay_steps <= 0:
        return alpha_start
    frac = min(1.0, step / decay_steps)
    return alpha_start + frac * (alpha_end - alpha_start)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_state(state: QLearningState, path) -> None:
    """Save Q-table and metadata to .npz file.

    Convenção para Double Q (p3): `q_table` no arquivo é a MÉDIA (A+B)/2 — o
    estimador combinado padrão, na escala de um Q normal, com o MESMO argmax da soma
    usada para agir. Toda a camada de análise lê `q_table` e continua correta; as
    tabelas cruas vão em `q_table_a`/`q_table_b` para o resume. Trace do Q(λ) e o
    modelo do Dyna são TRANSIENTES de treino — não entram no npz (o resume via
    checkpoint pickle os preserva; um load do npz recomeça com eles zerados).
    """
    extra = {}
    q_main = state.q_table
    if state.q_table2 is not None:
        q_main = (state.q_table + state.q_table2) / 2.0
        extra["q_table_a"] = state.q_table
        extra["q_table_b"] = state.q_table2
    np.savez(
        path,
        q_table=q_main,
        epsilon=np.array(state.epsilon),
        step=np.array(state.step),
        m=np.array(state.m),
        m_prev=np.array(state.m_prev),
        **({} if state.visits is None else {"visits": state.visits}),
        **extra,
    )


def load_state(path) -> QLearningState:
    """Load QLearningState from .npz file.

    ⚠️ Retrocompatível por obrigação: há 650+ Q-tables em disco do p1, da c1 e da c2
    salvas ANTES do estado interno existir, com apenas as chaves
    {q_table, epsilon, step}. Toda a camada de análise as carrega. Arquivos sem `m`
    assumem M = 1 (m = 0), que é o comportamento histórico.
    """
    data = np.load(path)
    keys = set(data.files)
    return QLearningState(
        q_table=(data["q_table_a"] if "q_table_a" in keys else data["q_table"]),
        epsilon=float(data["epsilon"]),
        step=int(data["step"]),
        m=int(data["m"]) if "m" in keys else 0,
        m_prev=int(data["m_prev"]) if "m_prev" in keys else 0,
        visits=data["visits"] if "visits" in keys else None,
        q_table2=(data["q_table_b"] if "q_table_b" in keys else None),
    )
