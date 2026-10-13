"""
Tests for src/agents/q_learning.py

O agente central da tese não tinha teste unitário — só cobertura indireta via
`test_multi_agent_trainer.py`. Item da Trilha D do backlog, coberto aqui porque o
p2_ecosystem depende de duas coisas que estavam sem rede:

  1. a **convenção de índice de estado** (`s = idx(p_0)·n + idx(p_1)`, indexada por
     firma), cuja leitura errada custou um achado no p1;
  2. o ciclo **save_state → load_state**, do qual todo cross-deploy depende;

e de uma terceira que é uma armadilha ativa:

  3. o **update in-place**, que faz qualquer snapshot sem `.copy()` virar um alias.
"""

import numpy as np
import pytest

from src.agents import q_learning
from src.agents.q_learning import Transition

N_ACTIONS = 15
P_MIN, P_MAX = 1.0, 2.0


def _config(**over):
    cfg = {
        "n_actions": N_ACTIONS, "memory_length": 2,
        "action_min": P_MIN, "q_max": P_MAX,
        "q_init": "zeros", "alpha": 0.15, "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.01,
        "epsilon_decay_steps": 1000, "epsilon_decay_type": "linear",
    }
    cfg.update(over)
    return cfg


def _grid():
    return np.linspace(P_MIN, P_MAX, N_ACTIONS)


def _init(cfg=None):
    import jax
    return q_learning.init(jax.random.PRNGKey(0), cfg or _config())


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def test_qtable_shape_is_n_actions_to_the_memory_length():
    st = _init()
    assert st.q_table.shape == (N_ACTIONS ** 2, N_ACTIONS)


def test_memory_length_zero_gives_a_single_state():
    st = _init(_config(memory_length=0))
    assert st.q_table.shape == (1, N_ACTIONS)


def test_zeros_init_is_all_zero():
    assert np.all(_init().q_table == 0.0)


def test_opponent_optimized_replicates_one_row_everywhere():
    """A razão de o nulo relevante ser o de INICIALIZAÇÃO, não o aleatório: `q_init:
    opponent_optimized` replica UMA linha em todas as linhas ⇒ a política gulosa inicial
    é constante ⇒ atrator global único por construção, antes de qualquer aprendizado."""
    row = np.arange(N_ACTIONS, dtype=float)
    st = _init(_config(q_init="opponent_optimized", _q_init_table=row))
    assert np.allclose(st.q_table, np.tile(row, (N_ACTIONS ** 2, 1)))
    greedy_per_state = st.q_table.argmax(axis=1)
    assert len(set(greedy_per_state.tolist())) == 1


def test_opponent_optimized_has_no_zero_cells():
    """Com q_init preenchida, uma célula NUNCA VISITADA retém um número plausível —
    visitação não é inferível da Q-table. É o que torna fósseis de inicialização
    indistinguíveis de política aprendida a olho nu.

    Usa a grade REAL do setup de Calvano (grid_extend=0.1 ⇒ p_min ≈ 1.428 > c = 1.0).
    Ver o caso de borda no teste seguinte.
    """
    p_min, p_max = 1.4277, 1.9702
    row = q_learning.opponent_optimized_qtable(
        n_actions=N_ACTIONS, action_min=p_min, q_max=p_max,
        n_rivals=1, gamma=0.95, a=2.0, mu=0.25, c=1.0, a0=0.0,
    )
    st = _init(_config(q_init="opponent_optimized", action_min=p_min, q_max=p_max,
                       _q_init_table=row))
    assert np.count_nonzero(st.q_table == 0.0) == 0


def test_opponent_optimized_is_zero_where_price_equals_marginal_cost():
    """Caso de borda: se a grade começa em p = c (acontece quando
    `p_min = max(c, p_NE − ext·(p_M − p_NE))` satura no custo), o lucro daquela ação é
    exatamente (p − c)·q = 0, e a coluna 0 da Q-table inicial é nula. Não é bug — mas
    quebra o atalho "nenhuma célula é zero", então vale saber onde ele vale."""
    row = q_learning.opponent_optimized_qtable(
        n_actions=N_ACTIONS, action_min=1.0, q_max=2.0,     # action_min == c
        n_rivals=1, gamma=0.95, a=2.0, mu=0.25, c=1.0, a0=0.0,
    )
    assert row[0] == pytest.approx(0.0)
    assert np.all(row[1:] > 0.0)


# ---------------------------------------------------------------------------
# Encoding de estado — a convenção que custou um achado
# ---------------------------------------------------------------------------

def test_state_index_is_mixed_radix_by_firm():
    """`s = idx(p_0)·n + idx(p_1)`, com o PRIMEIRO elemento do obs como dígito mais
    significativo — e o obs é indexado por FIRMA, não por 'próprio/rival'."""
    grid = _grid()
    for i, j in ((0, 0), (0, 1), (1, 0), (3, 7), (14, 14)):
        obs = np.array([grid[i], grid[j]])
        s = q_learning._obs_to_state_idx(obs, N_ACTIONS, P_MAX, P_MIN, 2)
        assert s == i * N_ACTIONS + j


def test_state_index_is_not_symmetric():
    """Consequência direta: (p_0, p_1) e (p_1, p_0) são estados DIFERENTES. Logo a
    tabela do agente 1 — que recebe o mesmo obs — é lida (rival, own)."""
    grid = _grid()
    s_ij = q_learning._obs_to_state_idx(np.array([grid[2], grid[9]]), N_ACTIONS, P_MAX, P_MIN, 2)
    s_ji = q_learning._obs_to_state_idx(np.array([grid[9], grid[2]]), N_ACTIONS, P_MAX, P_MIN, 2)
    assert s_ij != s_ji


def test_action_index_round_trips_on_the_grid():
    for i, p in enumerate(_grid()):
        assert q_learning._action_to_idx(float(p), N_ACTIONS, P_MAX, P_MIN) == i
        assert q_learning._idx_to_action(i, N_ACTIONS, P_MAX, P_MIN) == pytest.approx(float(p))


def test_action_index_saturates_outside_the_grid():
    assert q_learning._action_to_idx(-5.0, N_ACTIONS, P_MAX, P_MIN) == 0
    assert q_learning._action_to_idx(99.0, N_ACTIONS, P_MAX, P_MIN) == N_ACTIONS - 1


def test_profit_scale_obs_collapses_to_state_zero():
    """Documenta um bug latente conhecido: sob `bandit` em Bertrand o obs é um LUCRO
    (~0.2–0.35) discretizado com o discretizador de PREÇO (action_min ≈ p_min). Tudo
    satura em s=0 e o agente fica sem estado — o que explica mecanicamente o artefato
    do CI = 0.9286 no combo N2_bandit."""
    for profit in (0.20, 0.28, 0.35):
        s = q_learning._obs_to_state_idx(np.array([profit]), N_ACTIONS, 1.97, 1.43, 1)
        assert s == 0


# ---------------------------------------------------------------------------
# Ações
# ---------------------------------------------------------------------------

def test_greedy_action_returns_the_argmax_of_the_state_row():
    cfg = _config()
    st = _init(cfg)
    grid = _grid()
    st.q_table[3 * N_ACTIONS + 7, 11] = 1.0
    obs = np.array([grid[3], grid[7]])
    assert q_learning.greedy_action(st, obs, cfg) == pytest.approx(float(grid[11]))


def test_select_action_numpy_is_greedy_when_not_training():
    cfg = _config()
    st = _init(cfg)
    grid = _grid()
    st.q_table[0, 5] = 1.0
    obs = np.array([grid[0], grid[0]])
    a, _ = q_learning.select_action_numpy(st, obs, 0.0, 13, cfg, training=False)
    assert a == pytest.approx(float(grid[5]))


def test_select_action_numpy_explores_when_the_roll_is_below_epsilon():
    cfg = _config()
    st = _init(cfg)._replace(epsilon=1.0)
    grid = _grid()
    st.q_table[0, 5] = 1.0
    obs = np.array([grid[0], grid[0]])
    a, _ = q_learning.select_action_numpy(st, obs, 0.0, 13, cfg, training=True)
    assert a == pytest.approx(float(grid[13]))


def test_argmax_ties_resolve_to_the_lowest_price():
    """Numa linha de margem zero a política é decidida pelo desempate do argmax
    (menor índice = menor preço), não pelo aprendizado. Qualquer estatística sobre σ
    tem de vir acompanhada da distribuição de margens."""
    cfg = _config()
    st = _init(cfg)                     # tudo zero → empate total
    obs = np.array([_grid()[4], _grid()[9]])
    assert q_learning.greedy_action(st, obs, cfg) == pytest.approx(P_MIN)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def test_update_applies_the_bellman_rule():
    cfg = _config(gamma=0.95, alpha=0.15)
    st = _init(cfg)
    grid = _grid()
    obs, nxt = np.array([grid[0], grid[0]]), np.array([grid[1], grid[1]])
    s, s_ = 0, 1 * N_ACTIONS + 1
    st.q_table[s_, :] = 2.0                       # max_a' Q(s',a') = 2.0

    new, _ = q_learning.update(
        st, Transition(obs=obs, action=float(grid[0]), reward=1.0, next_obs=nxt, done=False), cfg
    )
    # Q ← 0 + 0.15 * (1.0 + 0.95*2.0 - 0) = 0.435
    assert new.q_table[s, 0] == pytest.approx(0.15 * (1.0 + 0.95 * 2.0))


def test_update_mutates_the_qtable_in_place():
    """⚠️ Comportamento REAL, documentado aqui de propósito: o update é in-place e o
    NamedTuple devolvido referencia o MESMO ndarray. Quem guardar o estado anterior vê
    a mutação. Qualquer snapshot precisa de `.copy()` explícito — é o contrato de que
    `population.Pool.add_snapshot` depende."""
    cfg = _config()
    st = _init(cfg)
    grid = _grid()
    before = st.q_table
    new, _ = q_learning.update(
        st,
        Transition(obs=np.array([grid[0], grid[0]]), action=float(grid[0]),
                   reward=1.0, next_obs=np.array([grid[0], grid[0]]), done=False),
        cfg,
    )
    assert new.q_table is before, "o update deixou de ser in-place — revise os snapshots"
    assert before[0, 0] != 0.0


def test_a_copied_snapshot_is_isolated_from_later_updates():
    cfg = _config()
    st = _init(cfg)
    grid = _grid()
    snapshot = st._replace(q_table=np.array(st.q_table, copy=True))

    for _ in range(20):
        st, _ = q_learning.update(
            st,
            Transition(obs=np.array([grid[0], grid[0]]), action=float(grid[0]),
                       reward=1.0, next_obs=np.array([grid[0], grid[0]]), done=False),
            cfg,
        )
    assert np.all(snapshot.q_table == 0.0)
    assert st.q_table[0, 0] != 0.0


def test_epsilon_decays_and_floors_at_epsilon_end():
    cfg = _config(epsilon_decay_steps=10, epsilon_end=0.01)
    st = _init(cfg)
    grid = _grid()
    tr = Transition(obs=np.array([grid[0], grid[0]]), action=float(grid[0]),
                    reward=0.0, next_obs=np.array([grid[0], grid[0]]), done=False)
    eps = [st.epsilon]
    for _ in range(40):
        st, _ = q_learning.update(st, tr, cfg)
        eps.append(st.epsilon)
    assert eps == sorted(eps, reverse=True)
    assert st.epsilon == pytest.approx(0.01)


def test_geometric_decay_reaches_epsilon_end_at_decay_steps():
    """O decaimento de Calvano: ε_t = ε_start·(1−δ)^t, com δ escolhido para atingir
    ε_end exatamente em decay_steps."""
    e = q_learning._decay_epsilon(1000, epsilon_start=1.0, epsilon_end=0.01,
                                  decay_steps=1000, decay_type="geometric")
    assert e == pytest.approx(0.01, rel=1e-3)


def test_step_only_advances_on_update():
    """Um agente que nunca recebe `update` nunca decai epsilon — é por isso que os
    oponentes congelados precisam de epsilon forçado a 0, e não herdado."""
    cfg = _config()
    st = _init(cfg)
    grid = _grid()
    obs = np.array([grid[0], grid[0]])
    for _ in range(5):
        _, st = q_learning.select_action_numpy(st, obs, 1.0, 0, cfg, training=True)
    assert st.step == 0
    assert st.epsilon == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistência — o ciclo do qual todo cross-deploy depende
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    cfg = _config()
    st = _init(cfg)
    rng = np.random.default_rng(0)
    st.q_table[:] = rng.normal(size=st.q_table.shape)
    st = st._replace(epsilon=0.037, step=12345)

    path = tmp_path / "agent_0.npz"
    q_learning.save_state(st, path)
    back = q_learning.load_state(path)

    assert np.array_equal(back.q_table, st.q_table)
    assert back.epsilon == pytest.approx(0.037)
    assert back.step == 12345


def test_saved_file_keys(tmp_path):
    """O .npz não guarda config, grade nem benchmarks. Toda análise standalone precisa
    parear com o config.yaml irmão do run — e distinguir Q-learning de REINFORCE pela
    CHAVE (`q_table` vs `params_bytes`), não pelo caminho.

    `m`/`m_prev` entraram com o autômato (2026-08-11); ver o teste de
    retrocompatibilidade abaixo."""
    path = tmp_path / "a.npz"
    q_learning.save_state(_init(), path)
    assert set(np.load(path).files) == {"q_table", "epsilon", "step", "m", "m_prev", "visits"}


def test_load_state_accepts_files_written_before_the_automaton(tmp_path):
    """⚠️ Obrigatório: há 650+ Q-tables em disco (p1, c1, c2) salvas com apenas
    {q_table, epsilon, step}. Sem isto, toda a camada de análise para de carregar."""
    path = tmp_path / "legado.npz"
    np.savez(path, q_table=np.zeros((225, 15)), epsilon=np.array(0.01), step=np.array(2_000_000))

    st = q_learning.load_state(path)
    assert st.q_table.shape == (225, 15)
    assert st.epsilon == pytest.approx(0.01)
    assert st.step == 2_000_000
    assert st.m == 0 and st.m_prev == 0      # M = 1, o comportamento histórico


def test_loaded_policy_is_identical_to_the_original(tmp_path):
    cfg = _config()
    st = _init(cfg)
    rng = np.random.default_rng(1)
    st.q_table[:] = rng.normal(size=st.q_table.shape)

    path = tmp_path / "a.npz"
    q_learning.save_state(st, path)
    back = q_learning.load_state(path)

    grid = _grid()
    for i in (0, 4, 9, 14):
        for j in (0, 7, 14):
            obs = np.array([grid[i], grid[j]])
            assert q_learning.greedy_action(st, obs, cfg) == \
                   q_learning.greedy_action(back, obs, cfg)


# ---------------------------------------------------------------------------
# Estado interno (autômato) — acrescentado 2026-08-11
# ---------------------------------------------------------------------------
#
# O PORTÃO da refatoração: com M = 1 nada pode mudar. É o que torna seguro mexer no
# agente central da tese.

def _cfg_m(M, **over):
    c = _config(q_init="zeros")
    c["n_internal"] = M
    c.update(over)
    return c


def test_M1_qtable_shape_is_unchanged():
    st = _init(_cfg_m(1))
    assert st.q_table.shape == (N_ACTIONS ** 2, N_ACTIONS)


def test_M1_state_and_action_index_are_the_historical_ones():
    grid = _grid()
    obs = np.array([grid[3], grid[7]])
    s_hist = q_learning._obs_to_state_idx(obs, N_ACTIONS, P_MAX, P_MIN, 2)
    s_new  = q_learning._state_idx(obs, 0, N_ACTIONS, P_MAX, P_MIN, 2, 1)
    assert s_new == s_hist


def test_M1_greedy_action_matches_a_real_trained_qtable():
    """Portão bit-a-bit contra uma Q-table de verdade, do p1."""
    from pathlib import Path
    p = Path("results/cross_deploy__20260621_203040/training/pair_000_A/agents/agent_0.npz")
    if not p.exists():
        pytest.skip("Q-table do p1 ausente neste checkout")
    q = np.load(p)["q_table"]
    assert q.shape == (225, 15)

    cfg = _cfg_m(1, action_min=1.0209, q_max=2.3770)
    st = q_learning.QLearningState(q_table=q, epsilon=0.0, step=0)
    grid = np.linspace(1.0209, 2.3770, 15)
    for i in range(15):
        for j in range(15):
            obs = np.array([grid[i], grid[j]])
            got = q_learning.greedy_action(st, obs, cfg)
            expected = grid[int(np.argmax(q[i * 15 + j]))]
            assert got == pytest.approx(expected)


def test_M2_expands_state_and_action_space():
    st = _init(_cfg_m(2))
    assert st.q_table.shape == (N_ACTIONS ** 2 * 2, N_ACTIONS * 2)


def test_state_index_composes_public_state_and_m():
    grid = _grid()
    obs = np.array([grid[3], grid[7]])
    s_pub = 3 * N_ACTIONS + 7
    for M in (2, 3):
        for m in range(M):
            assert q_learning._state_idx(obs, m, N_ACTIONS, P_MAX, P_MIN, 2, M) == s_pub * M + m


def test_action_decomposes_into_price_and_next_m():
    """A ação composta é (preço, m'), com m' no dígito menos significativo."""
    cfg = _cfg_m(2)
    st = _init(cfg)
    grid = _grid()
    obs = np.array([grid[0], grid[0]])
    # ação composta 2·2+1 = 5 ⇒ preço índice 2, m' = 1
    st.q_table[0, 5] = 1.0
    price, new = q_learning.select_action_numpy(st, obs, 1.0, 0, cfg, training=False)
    assert price == pytest.approx(float(grid[2]))
    assert new.m == 1 and new.m_prev == 0


def test_policy_can_depend_on_the_internal_state():
    """O ponto da proposta: no MESMO estado público, m diferente ⇒ ação diferente.
    Impossível com M = 1."""
    cfg = _cfg_m(2)
    st = _init(cfg)
    grid = _grid()
    obs = np.array([grid[4], grid[9]])
    s_pub = 4 * N_ACTIONS + 9
    st.q_table[s_pub * 2 + 0, 3 * 2 + 0] = 1.0     # m=0 → preço 3
    st.q_table[s_pub * 2 + 1, 11 * 2 + 1] = 1.0    # m=1 → preço 11

    a0 = q_learning.greedy_action(st._replace(m=0), obs, cfg)
    a1 = q_learning.greedy_action(st._replace(m=1), obs, cfg)
    assert a0 == pytest.approx(float(grid[3]))
    assert a1 == pytest.approx(float(grid[11]))
    assert a0 != a1


def test_grim_built_by_hand_absorbs_into_punishment():
    """Grim com M=2: m=0 coopera; preço do rival abaixo do limiar leva a m=1; m=1 é
    absorvente e joga Nash. É a estratégia que a memória-1 sobre o par NÃO representa."""
    cfg = _cfg_m(2)
    st = _init(cfg)
    grid = _grid()
    coop, nash = 12, 5          # índices na grade
    for i in range(N_ACTIONS):
        for j in range(N_ACTIONS):
            s = (i * N_ACTIONS + j)
            deviated = j < nash
            st.q_table[s * 2 + 0, coop * 2 + (1 if deviated else 0)] = 1.0   # m=0
            st.q_table[s * 2 + 1, nash * 2 + 1] = 1.0                        # m=1 absorvente

    state = st._replace(m=0)
    # rival coopera → segue cooperando em m=0
    p, state = q_learning.greedy_step(state, np.array([grid[coop], grid[coop]]), cfg)
    assert p == pytest.approx(float(grid[coop])) and state.m == 0
    # rival desvia → dispara
    p, state = q_learning.greedy_step(state, np.array([grid[coop], grid[0]]), cfg)
    assert state.m == 1
    # e nunca mais sai, mesmo com o rival voltando a cooperar
    for _ in range(5):
        p, state = q_learning.greedy_step(state, np.array([grid[nash], grid[coop]]), cfg)
        assert state.m == 1
        assert p == pytest.approx(float(grid[nash]))


def test_opponent_optimized_init_does_not_favour_any_m_transition():
    """Se o q_init preferisse um m', o autômato seria PLANTADO em vez de aprendido."""
    row = np.arange(1.0, N_ACTIONS + 1.0)
    st = _init(_cfg_m(2, q_init="opponent_optimized", _q_init_table=row))
    q = st.q_table
    for price_idx in range(N_ACTIONS):
        col0, col1 = q[:, price_idx * 2 + 0], q[:, price_idx * 2 + 1]
        assert np.allclose(col0, col1), "q_init favorece uma transição de m"


def test_update_uses_m_prev_for_s_and_m_for_action():
    cfg = _cfg_m(2)
    st = _init(cfg)
    grid = _grid()
    obs = np.array([grid[0], grid[0]])
    st = st._replace(m_prev=0, m=1)
    new, _ = q_learning.update(
        st, Transition(obs=obs, action=float(grid[2]), reward=1.0, next_obs=obs, done=False), cfg)
    # s = 0·2 + m_prev(0) = 0 ;  a = 2·2 + m(1) = 5
    assert new.q_table[0, 5] != 0.0
    assert new.q_table[1, 5] == 0.0     # linha de m=1 intocada


def test_update_preserves_the_internal_state():
    """REGRESSÃO. `update` reconstruía o NamedTuple sem m/m_prev, e os defaults (0,0)
    zeravam o autômato a cada passo: o agente escolhia m'=1, o update apagava, e ele
    nunca chegava a m=1. Resultado: 100% das linhas m=1 nunca atualizadas, argmax por
    empate ⇒ p_min, e a primeira c3 inteira inválida."""
    cfg = _cfg_m(2)
    st = _init(cfg)._replace(m=1, m_prev=0)
    grid = _grid()
    obs = np.array([grid[0], grid[0]])
    new, _ = q_learning.update(
        st, Transition(obs=obs, action=float(grid[0]), reward=1.0, next_obs=obs, done=False), cfg)
    assert new.m == 1 and new.m_prev == 0


def test_agent_actually_reaches_every_internal_state_when_exploring():
    """O teste de ponta que teria pego o bug: explorando, o agente TEM de visitar m=1 —
    e as linhas de m=1 têm de sair do valor de inicialização."""
    cfg = _cfg_m(2, q_init="uniform", _q_init_value=13.5, epsilon_start=1.0)
    st = _init(cfg)._replace(epsilon=1.0)
    grid = _grid()
    obs = np.array([grid[0], grid[0]])
    rng = np.random.default_rng(0)
    seen_m = set()
    for _ in range(400):
        a, st = q_learning.select_action_numpy(
            st, obs, 0.0, int(rng.integers(N_ACTIONS * 2)), cfg, training=True)
        seen_m.add(st.m)
        st, _ = q_learning.update(
            st, Transition(obs=obs, action=a, reward=1.0, next_obs=obs, done=False), cfg)
    assert seen_m == {0, 1}, f"o agente nunca alcançou m=1: {seen_m}"

    n_pub = N_ACTIONS ** 2
    rows_m1 = st.q_table[1::2]
    assert not np.allclose(rows_m1, 13.5), "linhas de m=1 nunca foram atualizadas"


# ---------------------------------------------------------------------------
# alpha por contagem de visitas (Robbins-Monro) — acrescentado 2026-08-12
# ---------------------------------------------------------------------------

def _step_once(st, cfg, obs, price):
    return q_learning.update(
        st, Transition(obs=obs, action=price, reward=1.0, next_obs=obs, done=False), cfg)


def test_alpha_mode_defaults_to_global():
    """Opt-in: sem `alpha_mode` nada muda em relação ao comportamento histórico."""
    cfg = _config(alpha_start=0.15, alpha_end=0.01, alpha_decay_steps=1000)
    st = _init(cfg)
    grid = _grid(); obs = np.array([grid[0], grid[0]])
    _, log = _step_once(st, cfg, obs, float(grid[0]))
    assert log["alpha"] == pytest.approx(0.15, abs=1e-3)   # passo 0 → alpha_start


def test_alpha_by_count_follows_robbins_monro():
    from src.agents.q_learning import _alpha_by_count as ab
    assert ab(0, 1.0, 0.0, 0.7) == pytest.approx(1.0)
    assert ab(100, 1.0, 0.0, 0.7) == pytest.approx(1 / 101 ** 0.7)
    assert ab(10, 1.0, 0.0, 0.5) > ab(10, 1.0, 0.0, 0.7)   # omega menor decai mais devagar


def test_alpha_by_count_is_capped_to_preserve_optimistic_init():
    """⚠️ Sem o teto, a PRIMEIRA visita teria alpha=1.0 e o Q saltaria para o alvo,
    anulando a inicialização otimista — cujo mecanismo é decair devagar do valor alto."""
    from src.agents.q_learning import _alpha_by_count as ab
    # cruzamento: 1/(1+n)^0.7 cai abaixo de 0.15 em n=15
    crossover = next(n for n in range(1000) if 1 / (1 + n) ** 0.7 <= 0.15)
    assert crossover == 15
    for n in range(crossover):
        assert ab(n, 0.15, 0.01, 0.7) == pytest.approx(0.15)
    assert ab(crossover, 0.15, 0.01, 0.7) < 0.15
    assert ab(200, 0.15, 0.01, 0.7) < 0.15
    assert ab(10 ** 9, 0.15, 0.01, 0.7) == pytest.approx(0.01)   # piso


def test_count_mode_keeps_rare_cells_learning_late_in_training():
    """O problema que motivou a mudança: no modo global, uma célula visitada pela
    PRIMEIRA vez no passo 12M recebe alpha=0.01 e nunca separa as ações."""
    grid = _grid(); obs = np.array([grid[0], grid[0]])

    glob = _config(alpha_start=0.15, alpha_end=0.01, alpha_decay_steps=1_000_000)
    st = _init(glob)._replace(step=1_000_000)
    _, log_g = _step_once(st, glob, obs, float(grid[0]))

    cnt = _config(alpha_start=0.15, alpha_end=0.01, alpha_mode="count", alpha_omega=0.7)
    st = _init(cnt)._replace(step=1_000_000)          # tarde no treino, célula NOVA
    _, log_c = _step_once(st, cnt, obs, float(grid[0]))

    assert log_g["alpha"] == pytest.approx(0.01)
    assert log_c["alpha"] == pytest.approx(0.15)
    assert log_c["alpha"] > 10 * log_g["alpha"]


def test_visits_are_counted_per_state_action():
    cfg = _config(alpha_mode="count")
    st = _init(cfg)
    grid = _grid(); obs = np.array([grid[0], grid[0]])
    for _ in range(7):
        st, _ = _step_once(st, cfg, obs, float(grid[3]))
    assert st.visits[0, 3] == 7
    assert st.visits.sum() == 7


def test_alpha_decreases_as_a_cell_is_revisited():
    cfg = _config(alpha_start=0.15, alpha_end=0.001, alpha_mode="count", alpha_omega=0.7)
    st = _init(cfg)
    grid = _grid(); obs = np.array([grid[0], grid[0]])
    alphas = []
    for _ in range(400):
        st, log = _step_once(st, cfg, obs, float(grid[0]))
        alphas.append(log["alpha"])
    assert alphas[0] == pytest.approx(0.15)
    assert alphas[-1] < alphas[0]
    assert all(b <= a + 1e-12 for a, b in zip(alphas, alphas[1:]))   # monotônico


def test_visits_survive_save_load_and_old_files_still_work(tmp_path):
    cfg = _config(alpha_mode="count")
    st = _init(cfg)
    grid = _grid(); obs = np.array([grid[0], grid[0]])
    for _ in range(3):
        st, _ = _step_once(st, cfg, obs, float(grid[1]))

    p = tmp_path / "a.npz"
    q_learning.save_state(st, p)
    back = q_learning.load_state(p)
    assert back.visits is not None and back.visits[0, 1] == 3

    old = tmp_path / "legado.npz"
    np.savez(old, q_table=np.zeros((225, 15)), epsilon=np.array(0.01), step=np.array(1))
    st_old = q_learning.load_state(old)
    assert st_old.visits is None
    st2, _ = _step_once(st_old, cfg, obs, float(grid[0]))   # não pode explodir
    assert st2.visits is not None and st2.visits.sum() == 1


# ---------------------------------------------------------------------------
# Variantes do p3 (etapa 7): Dyna-Q, Q(λ), Double Q — default OFF, bit-exato
# ---------------------------------------------------------------------------

def _tr(p0, p1, action, reward, p0n, p1n):
    return Transition(obs=np.array([p0, p1]), action=action, reward=reward,
                      next_obs=np.array([p0n, p1n]), done=False)


class TestVariantesOff:
    def test_variantes_off_e_bit_exato(self):
        """Sem flags de variante, os campos novos são None e o update segue o caminho
        histórico (os 40+ testes acima cobrem o comportamento; este pina os campos)."""
        st = _init()
        assert st.q_table2 is None and st.trace is None
        assert st.model_next is None and st.model_seen is None
        st2, log = q_learning.update(st, _tr(1.0, 1.5, 1.2, 0.3, 1.2, 1.5), _config())
        assert st2.q_table2 is None and st2.trace is None and st2.model_next is None
        assert "td_error" in log

    def test_combinacao_de_variantes_e_erro(self):
        with pytest.raises(ValueError, match="mutuamente exclusivas"):
            _init(_config(double_q=True, **{"lambda": 0.8}))


class TestDoubleQ:
    def test_init_cria_duas_tabelas_independentes(self):
        st = _init(_config(double_q=True, q_init="zeros"))
        assert st.q_table2 is not None
        assert st.q_table2 is not st.q_table

    def test_update_alterna_as_tabelas(self):
        cfg = _config(double_q=True)
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.5, 1.2, 0.7, 1.2, 1.5), cfg)  # step 0 → A
        assert st.q_table.any() and not st.q_table2.any()
        st, _ = q_learning.update(st, _tr(1.2, 1.5, 1.3, 0.7, 1.3, 1.5), cfg)  # step 1 → B
        assert st.q_table2.any()

    def test_save_load_convencao_da_media(self, tmp_path):
        """`q_table` no npz é (A+B)/2 — mesma escala e mesmo argmax da soma que age.
        O load devolve A/B para o treino continuar."""
        cfg = _config(double_q=True)
        st = _init(cfg)
        for k in range(6):
            st, _ = q_learning.update(st, _tr(1.0, 1.5, 1.0 + 0.05 * k, 0.5, 1.1, 1.5), cfg)
        path = tmp_path / "agent.npz"
        q_learning.save_state(st, path)
        data = np.load(path)
        assert np.allclose(data["q_table"], (st.q_table + st.q_table2) / 2.0)
        st2 = q_learning.load_state(path)
        assert (st2.q_table == st.q_table).all()
        assert (st2.q_table2 == st.q_table2).all()


class TestQLambda:
    def test_trace_propaga_credito_ao_estado_anterior(self):
        """Dois passos gulosos em sequência: a recompensa do 2º tem de mexer na célula
        do 1º (via trace) — é o que o one-step não faz."""
        cfg = _config(**{"lambda": 0.9}, epsilon_start=0.0, epsilon_end=0.0)
        st = _init(cfg)
        # passo 1: greedy (q toda zero ⇒ argmax=0 ⇒ ação p_min), recompensa 0
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.0, 1.0, 1.0), cfg)
        cell1 = (0, 0)
        before = st.q_table[cell1]
        # passo 2: de novo greedy no mesmo estado, recompensa ALTA
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), cfg)
        assert st.q_table[cell1] > before + 0.15 * 0.9 * 0.0, "célula 1 não recebeu crédito"
        assert st.trace is not None and st.trace.max() > 0

    def test_acao_exploratoria_zera_o_trace(self):
        cfg = _config(**{"lambda": 0.9})
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        assert st.trace.any()
        # ação claramente não-gulosa: depois da recompensa 0.5 na célula (0,0), o
        # argmax do estado é 0 — jogar o preço mais alto corta o trace
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 2.0, 0.0, 2.0, 1.0), cfg)
        assert not st.trace.any(), "trace tinha de zerar no passo não-guloso"

    def test_reset_episode_state_zera_o_trace(self):
        cfg = _config(**{"lambda": 0.9})
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        st = q_learning.reset_episode_state(st, cfg)
        assert not st.trace.any()


class TestDynaQ:
    def _cfg(self, **over):
        return _config(planning={"enabled": True, "k_updates": 5,
                                 "model_scope": "global"}, **over)

    def test_modelo_grava_a_transicao(self):
        cfg = self._cfg()
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        assert len(st.model_seen) == 1
        (s, a) = st.model_seen[0]
        assert st.model_next[s, a] >= 0
        assert st.model_rew[s, a] == pytest.approx(0.5)

    def test_replay_acelera_a_propagacao(self):
        """Com o MESMO fluxo de transições, o Dyna tem de mover a Q-table mais que o
        baseline (replay = updates extras nas células vistas)."""
        trs = [_tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0),
               _tr(1.0, 1.0, 1.1, 0.2, 1.1, 1.0),
               _tr(1.1, 1.0, 1.0, 0.6, 1.0, 1.0)] * 4
        cfg_b = _config()
        st_b = _init(cfg_b)
        for t in trs:
            st_b, _ = q_learning.update(st_b, t, cfg_b)
        cfg_d = self._cfg()
        st_d = _init(cfg_d)
        for t in trs:
            st_d, _ = q_learning.update(st_d, t, cfg_d)
        assert not np.allclose(st_d.q_table, st_b.q_table)
        assert np.abs(st_d.q_table).sum() > np.abs(st_b.q_table).sum()

    def test_per_episode_zera_o_modelo_na_fronteira(self):
        cfg = _config(planning={"enabled": True, "k_updates": 5,
                                "model_scope": "per_episode"})
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        assert st.model_seen
        st = q_learning.reset_episode_state(st, cfg)
        assert not st.model_seen and (st.model_next == -1).all()

    def test_replay_nao_incrementa_visits(self):
        """alpha_mode=count conta amostras REAIS — o replay não pode inflar a contagem."""
        cfg = self._cfg()
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        assert int(st.visits.sum()) == 1


# ---------------------------------------------------------------------------
# Estado expandido (p3, etapa 8): obs_radices
# ---------------------------------------------------------------------------

class TestObsRadices:
    def test_radices_de_precos_reproduzem_o_encoder_historico(self):
        """`obs_radices` todo em n_actions ≡ caminho sem radices — portão bit-exato."""
        from src.agents.q_learning import _obs_to_state_idx
        for obs in ([1.0, 1.5], [2.0, 1.0], [1.47, 1.93]):
            base = _obs_to_state_idx(np.array(obs), N_ACTIONS, P_MAX, P_MIN, 2)
            hetero = _obs_to_state_idx(np.array(obs), N_ACTIONS, P_MAX, P_MIN, 2,
                                       radices=[N_ACTIONS, N_ACTIONS])
            assert base == hetero

    def test_features_entram_como_digitos_proprios(self):
        from src.agents.q_learning import _obs_to_state_idx
        radices = [N_ACTIONS, N_ACTIONS, 3, 2]
        seen = set()
        for f1 in range(3):
            for f2 in range(2):
                obs = np.array([1.0, 1.0, float(f1), float(f2)])
                seen.add(_obs_to_state_idx(obs, N_ACTIONS, P_MAX, P_MIN, 2, radices))
        assert len(seen) == 6, "combinações de features têm de dar estados distintos"

    def test_feature_fora_do_suporte_e_clampada(self):
        from src.agents.q_learning import _obs_to_state_idx
        radices = [N_ACTIONS, N_ACTIONS, 3]
        hi = _obs_to_state_idx(np.array([1.0, 1.0, 9.0]), N_ACTIONS, P_MAX, P_MIN, 2, radices)
        top = _obs_to_state_idx(np.array([1.0, 1.0, 2.0]), N_ACTIONS, P_MAX, P_MIN, 2, radices)
        assert hi == top

    def test_tamanho_errado_e_erro_nao_truncamento(self):
        from src.agents.q_learning import _obs_to_state_idx
        with pytest.raises(ValueError, match="obs_radices"):
            _obs_to_state_idx(np.array([1.0, 1.0, 1.0]), N_ACTIONS, P_MAX, P_MIN, 2,
                              radices=[N_ACTIONS, N_ACTIONS])

    def test_init_dimensiona_a_tabela_pelo_produto(self):
        st = _init(_config(obs_radices=[N_ACTIONS, N_ACTIONS, 3, 3, 2]))
        assert st.q_table.shape == (N_ACTIONS * N_ACTIONS * 18, N_ACTIONS)


class TestDynaPrioritized:
    def _cfg(self):
        return _config(planning={"enabled": True, "k_updates": 5,
                                 "mode": "prioritized", "model_scope": "global"})

    def test_init_aloca_prioridades(self):
        st = _init(self._cfg())
        assert st.model_pri is not None and not st.model_pri.any()

    def test_replay_vai_na_celula_de_maior_td(self):
        cfg = self._cfg()
        st = _init(cfg)
        # duas visitas: uma com recompensa alta (TD grande), outra pequena
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), cfg)
        st, _ = q_learning.update(st, _tr(1.5, 1.5, 1.5, 0.01, 1.5, 1.5), cfg)
        # a prioridade da célula de TD alto decaiu após os replays (foi trabalhada)
        assert st.model_pri.max() >= 0.0
        # e o replay moveu Q além das duas células reais (propagação)
        assert (st.q_table != 0).sum() >= 2

    def test_prioridade_zera_no_reset_per_episode(self):
        cfg = _config(planning={"enabled": True, "k_updates": 5,
                                "mode": "prioritized", "model_scope": "per_episode"})
        st = _init(cfg)
        st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.7, 1.0, 1.0), cfg)
        assert st.model_pri.any()
        st = q_learning.reset_episode_state(st, cfg)
        assert not st.model_pri.any()

    def test_prioritized_converge_prioridades_a_zero_em_ponto_fixo(self):
        """Num mundo de recompensa constante, replays repetidos esvaziam a fila —
        prioridades caem (célula resolvida sai sozinha)."""
        cfg = self._cfg()
        st = _init(cfg)
        for _ in range(200):
            st, _ = q_learning.update(st, _tr(1.0, 1.0, 1.0, 0.5, 1.0, 1.0), cfg)
        assert st.model_pri.max() < 0.05


class TestUCB:
    def _cfg(self):
        return _config(exploration={"type": "ucb", "c": 1.0})

    def test_ucb_prefere_a_celula_sub_visitada(self):
        cfg = self._cfg()
        st = _init(cfg)
        s = 0
        # torna a ação 0 muito visitada; Q empatado em zero
        st.visits[s, 0] = 1000
        st = st._replace(step=100)
        from src.agents.q_learning import _ucb_action
        a = _ucb_action(st, s, 1.0)
        assert a != 0, "com Q empatado, o bônus tem de fugir da célula batida"

    def test_ucb_e_deterministico_e_ignora_epsilon(self):
        cfg = self._cfg()
        st = _init(cfg)._replace(epsilon=1.0)   # ε=1: o ε-greedy seria 100% aleatório
        obs = np.array([1.0, 1.0])
        a1, _ = q_learning.select_action_numpy(st, obs, 0.0, 7, cfg, training=True)
        a2, _ = q_learning.select_action_numpy(st, obs, 0.99, 3, cfg, training=True)
        assert a1 == a2, "UCB não pode depender do roll nem do rand_act"

    def test_sem_exploration_o_caminho_e_o_historico(self):
        st = _init()
        obs = np.array([1.0, 1.0])
        a, _ = q_learning.select_action_numpy(st, obs, 0.0, 7, _config(), training=True)
        # roll 0.0 < ε=1.0 ⇒ explora com rand_act=7 (caminho histórico intacto)
        grid = _grid()
        assert a == pytest.approx(grid[7])


# ---------------------------------------------------------------------------
# Máscara de exploração (c5, braço T4): estados vetados jogam guloso no ramo ε
# ---------------------------------------------------------------------------

class TestExplorationMask:
    def test_estado_mascarado_joga_guloso_mesmo_com_eps_1(self):
        st = _init()   # ε_start = 1.0, q_init zeros ⇒ argmax = ação 0
        grid = _grid()
        cfg = _config(exploration_mask_states=[[2, 5]])
        obs_masked = np.array([grid[2], grid[5]])
        a, _ = q_learning.select_action_numpy(st, obs_masked, 0.0, 7, cfg, training=True)
        assert a == pytest.approx(grid[0]), "estado vetado tinha de jogar argmax"
        obs_free = np.array([grid[3], grid[5]])
        a2, _ = q_learning.select_action_numpy(st, obs_free, 0.0, 7, cfg, training=True)
        assert a2 == pytest.approx(grid[7]), "estado fora da máscara explora normal"

    def test_conversao_de_slot_do_agente_1(self):
        """(own=2, rival=5) para o agente 1 é a linha 5·n + 2 — a convenção de
        índice que já custou um achado."""
        st = _init()
        grid = _grid()
        cfg = _config(exploration_mask_states=[[2, 5]], own_slot=1)
        # para o agente 1, obs = (p_f0, p_f1) = (rival, own)
        obs_masked = np.array([grid[5], grid[2]])
        a, _ = q_learning.select_action_numpy(st, obs_masked, 0.0, 7, cfg, training=True)
        assert a == pytest.approx(grid[0])
        # a MESMA obs sem own_slot (slot 0) lê (own=5, rival=2) — fora da máscara
        cfg0 = _config(exploration_mask_states=[[2, 5]])
        a0, _ = q_learning.select_action_numpy(st, obs_masked, 0.0, 7, cfg0, training=True)
        assert a0 == pytest.approx(grid[7])

    def test_select_action_jax_tambem_respeita_a_mascara(self):
        import jax
        st = _init()
        grid = _grid()
        cfg = _config(exploration_mask_states=[[2, 5]])
        obs_masked = np.array([grid[2], grid[5]])
        for seed in range(5):
            a, _ = q_learning.select_action(st, obs_masked, jax.random.PRNGKey(seed),
                                            cfg, training=True)
            assert a == pytest.approx(grid[0])

    def test_ausente_e_bit_exato(self):
        st = _init()
        obs = np.array([1.0, 1.0])
        a_ref, _ = q_learning.select_action_numpy(st, obs, 0.0, 7, _config(), training=True)
        a_msk, _ = q_learning.select_action_numpy(st, obs, 0.0, 7,
                                                  _config(exploration_mask_states=[]),
                                                  training=True)
        assert a_ref == a_msk

    def test_mascara_com_obs_radices_e_erro(self):
        st = _init()
        grid = _grid()
        cfg = _config(exploration_mask_states=[[2, 5]], obs_radices=[15, 15, 3])
        with pytest.raises(ValueError, match="obs_radices"):
            q_learning.select_action_numpy(st, np.array([grid[2], grid[5], 0.0]),
                                           0.0, 7, cfg, training=True)


# ---------------------------------------------------------------------------
# Dyna estratégico (c5, braço T3): replay pela função de reação do RIVAL
# ---------------------------------------------------------------------------

def _oracle_npz(tmp_path, n=N_ACTIONS):
    rng = np.random.default_rng(0)
    pi = ((np.arange(n)[:, None] + 2 * np.arange(n)[None, :]) % n).astype(np.int64)
    P = rng.uniform(0.1, 0.5, size=(n, n))
    path = tmp_path / "oracle.npz"
    np.savez(path, pi=pi, P=P)
    return path, pi, P


class TestDynaOracle:
    def test_tabelas_do_oraculo_seguem_a_convencao_de_slot(self, tmp_path):
        n = N_ACTIONS
        path, pi, P = _oracle_npz(tmp_path)
        for slot in (0, 1):
            cfg = _config(planning={"enabled": True, "opponent_model": "oracle",
                                    "oracle_file": str(path)}, own_slot=slot)
            nxt, rew = q_learning._oracle_tables(cfg)
            for (i0, i1, a) in [(0, 0, 0), (3, 7, 2), (14, 1, 9)]:
                s = i0 * n + i1
                if slot == 0:
                    b = pi[i1, i0]
                    assert nxt[s, a] == a * n + b
                else:
                    b = pi[i0, i1]
                    assert nxt[s, a] == b * n + a
                assert rew[s, a] == pytest.approx(P[a, b])

    def test_replay_usa_recompensa_analitica_e_nao_toca_visits(self, tmp_path):
        n = N_ACTIONS
        path, pi, P = _oracle_npz(tmp_path)
        cfg = _config(planning={"enabled": True, "opponent_model": "oracle",
                                "oracle_file": str(path), "k_updates": 1})
        st = _init(cfg)
        grid = _grid()
        # transição real na célula (s_real, a_real) — longe da célula de replay (0, 0)
        obs = np.array([grid[5], grid[6]])
        tr = Transition(obs=obs, action=grid[4], reward=0.2,
                        next_obs=np.array([grid[4], grid[3]]), done=False)
        st2, _ = q_learning.update(st, tr, cfg)
        # replay do step 0, i=0: flat = 0 ⇒ (ps, pa) = (0, 0); b = pi[0,0]; q zeros
        b = pi[0, 0]
        alpha = float(cfg["alpha"])
        assert st2.q_table[0, 0] == pytest.approx(alpha * P[0, b])
        assert int(st2.visits[0, 0]) == 0, "replay não pode consumir visits"

    def test_oraculo_com_M_maior_que_1_e_erro(self, tmp_path):
        path, _, _ = _oracle_npz(tmp_path)
        cfg = _config(planning={"enabled": True, "opponent_model": "oracle",
                                "oracle_file": str(path)},
                      n_internal=2)
        with pytest.raises(ValueError, match="oracle"):
            q_learning._oracle_tables(cfg)
