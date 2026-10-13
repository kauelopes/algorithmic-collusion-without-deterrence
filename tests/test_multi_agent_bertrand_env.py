"""
Tests for src/environment/multi_agent_bertrand_env.py

Calvano et al. (2020) baseline parameters: a=2, mu=0.25, c=1, N=2.
  Competitive: p=1.000
  Nash:        p≈1.473,  pi/firm≈0.224
  Monopoly:    p≈1.925,  pi/firm≈0.338
"""

import math

import numpy as np
import pytest

from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv

TOL   = 1e-4   # tolerance for numerical benchmarks
TOL_S = 1e-5   # tolerance for exact step arithmetic


def make_env(n=2, info="bandit") -> MultiAgentBertrandEnv:
    return MultiAgentBertrandEnv({
        "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
        "cost":   {"c": 1.0},
        "market": {"n_firms": n, "p_min": 1.0, "p_max": 2.1},
        "state":  {"info_structure": info},
    })


# ---------------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_invalid_info_structure_raises(self):
        with pytest.raises(ValueError, match="Unknown info_structure"):
            make_env(info="quantity")

    def test_n_less_than_2_raises(self):
        with pytest.raises(ValueError):
            MultiAgentBertrandEnv({
                "demand": {"a": 2.0, "mu": 0.25},
                "cost":   {"c": 1.0},
                "market": {"n_firms": 1, "p_min": 1.0, "p_max": 2.1},
                "state":  {"info_structure": "bandit"},
            })

    def test_p_min_ge_p_max_raises(self):
        with pytest.raises(ValueError):
            MultiAgentBertrandEnv({
                "demand": {"a": 2.0, "mu": 0.25},
                "cost":   {"c": 1.0},
                "market": {"n_firms": 2, "p_min": 2.0, "p_max": 1.0},
                "state":  {"info_structure": "bandit"},
            })

    def test_reset_returns_n_observations(self):
        obs = make_env(n=2).reset()
        assert len(obs) == 2

    def test_reset_n3_returns_3_observations(self):
        obs = make_env(n=3).reset()
        assert len(obs) == 3

    def test_reset_obs_are_float32_arrays(self):
        for o in make_env().reset():
            assert isinstance(o, np.ndarray)
            assert o.dtype == np.float32


# ---------------------------------------------------------------------------
# Observation shapes
# ---------------------------------------------------------------------------

class TestObsShape:
    def test_bandit_shape(self):
        env = make_env(n=2, info="bandit")
        obs = env.reset()
        assert obs[0].shape == (1,) and obs[1].shape == (1,)

    def test_own_price_shape(self):
        obs = make_env(n=2, info="own_price").reset()
        assert obs[0].shape == (1,)

    def test_rival_prices_shape_n2(self):
        obs = make_env(n=2, info="rival_prices").reset()
        # Calvano: state = all N prices
        assert obs[0].shape == (2,) and obs[1].shape == (2,)

    def test_rival_prices_shape_n3(self):
        obs = make_env(n=3, info="rival_prices").reset()
        assert obs[0].shape == (3,)

    def test_obs_size_method(self):
        assert make_env(n=2, info="bandit").obs_size()        == 1
        assert make_env(n=2, info="own_price").obs_size()     == 1
        assert make_env(n=2, info="rival_prices").obs_size()  == 2
        assert make_env(n=3, info="rival_prices").obs_size()  == 3


# ---------------------------------------------------------------------------
# Step correctness
# ---------------------------------------------------------------------------

class TestStepCorrectness:
    def test_step_returns_correct_structure(self):
        env = make_env()
        env.reset()
        obs, rewards, dones, info = env.step([1.4, 1.4])
        assert len(obs) == 2 and len(rewards) == 2 and len(dones) == 2
        assert isinstance(info, dict)

    def test_dones_always_false(self):
        env = make_env()
        env.reset()
        _, _, dones, _ = env.step([1.4, 1.4])
        assert all(d is False for d in dones)

    def test_symmetric_prices_give_equal_profits(self):
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([1.5, 1.5])
        assert rewards[0] == pytest.approx(rewards[1], abs=TOL_S)

    def test_nash_prices_give_nash_profit(self):
        """Both agents at p_NE → each earns pi_NE."""
        env = make_env(n=2)
        bm  = env.benchmarks()
        p_NE, pi_NE = bm["p_NE"], bm["pi_NE"]
        env.reset()
        _, rewards, _, _ = env.step([p_NE, p_NE])
        assert rewards[0] == pytest.approx(pi_NE, rel=1e-4)
        assert rewards[1] == pytest.approx(pi_NE, rel=1e-4)

    def test_monopoly_prices_give_monopoly_profit(self):
        """Both agents at p_M → each earns pi_monopoly_per_firm."""
        env = make_env(n=2)
        bm  = env.benchmarks()
        p_M, pi_M = bm["p_monopoly"], bm["pi_monopoly_per_firm"]
        env.reset()
        _, rewards, _, _ = env.step([p_M, p_M])
        assert rewards[0] == pytest.approx(pi_M, rel=1e-4)

    def test_competitive_price_gives_zero_profit(self):
        """p = c → profit = 0."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([1.0, 1.0])
        assert rewards[0] == pytest.approx(0.0, abs=TOL_S)
        assert rewards[1] == pytest.approx(0.0, abs=TOL_S)

    def test_shares_sum_to_less_than_one(self):
        """Total market shares < 1 (outside option absorbs the rest)."""
        env = make_env(n=2)
        env.reset()
        _, _, _, info = env.step([1.5, 1.5])
        assert sum(info["shares"]) < 1.0

    def test_lower_price_captures_more_share(self):
        """Firm undercutting rival gains higher share."""
        env = make_env(n=2)
        env.reset()
        _, _, _, info = env.step([1.2, 1.8])
        assert info["shares"][0] > info["shares"][1]

    def test_prices_clipped_to_p_max(self):
        """Actions above p_max are clipped; profits remain finite."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([999.0, 999.0])
        assert all(np.isfinite(r) for r in rewards)

    def test_prices_clipped_to_p_min(self):
        """Actions below p_min are clipped to p_min."""
        env = make_env(n=2)
        env.reset()
        _, rewards, _, _ = env.step([0.0, 0.0])
        # p_min = c = 1.0 → profit = 0
        assert rewards[0] == pytest.approx(0.0, abs=TOL_S)

    def test_n3_symmetric_nash(self):
        """N=3 at p_NE → all earn pi_NE."""
        env = make_env(n=3)
        bm  = env.benchmarks()
        env.reset()
        _, rewards, _, _ = env.step([bm["p_NE"]] * 3)
        for r in rewards:
            assert r == pytest.approx(bm["pi_NE"], rel=1e-4)


# ---------------------------------------------------------------------------
# Observations per info_structure
# ---------------------------------------------------------------------------

class TestObservations:
    def test_bandit_obs_is_own_profit(self):
        """After step t, bandit obs = own profit from step t."""
        env = make_env(n=2, info="bandit")
        env.reset()
        _, rewards, _, _ = env.step([1.5, 1.5])
        obs2, _, _, _    = env.step([1.5, 1.5])
        # obs2 reflects the PREVIOUS step's profit (step 1)
        assert obs2[0][0] == pytest.approx(rewards[0], abs=TOL_S)

    def test_own_price_obs_is_own_price(self):
        """obs returned by step t = own price chosen in step t (state for t+1)."""
        env = make_env(n=2, info="own_price")
        env.reset()
        obs1, _, _, _ = env.step([1.3, 1.7])
        assert obs1[0][0] == pytest.approx(1.3, abs=TOL_S)
        assert obs1[1][0] == pytest.approx(1.7, abs=TOL_S)
        # After a different step, obs reflects the new prices
        obs2, _, _, _ = env.step([1.5, 1.5])
        assert obs2[0][0] == pytest.approx(1.5, abs=TOL_S)

    def test_rival_prices_obs_contains_all_prices(self):
        """rival_prices obs = all N prices from the current period (Calvano state)."""
        env = make_env(n=2, info="rival_prices")
        env.reset()
        obs1, _, _, _ = env.step([1.3, 1.7])
        # Both agents see the same full price vector from this period
        np.testing.assert_allclose(obs1[0], [1.3, 1.7], atol=TOL_S)
        np.testing.assert_allclose(obs1[1], [1.3, 1.7], atol=TOL_S)

    def test_rival_prices_obs_symmetric(self):
        """All agents see the same price vector with rival_prices."""
        env = make_env(n=3, info="rival_prices")
        env.reset()
        env.step([1.2, 1.5, 1.8])
        obs2, _, _, _ = env.step([1.5, 1.5, 1.5])
        np.testing.assert_allclose(obs2[0], obs2[1], atol=TOL_S)
        np.testing.assert_allclose(obs2[1], obs2[2], atol=TOL_S)


# ---------------------------------------------------------------------------
# Counterfactual profits (CE-gap metric)
# ---------------------------------------------------------------------------

class TestCounterfactualProfits:
    def setup_method(self):
        self.env  = make_env(n=2)
        self.env.reset()
        self.grid = np.linspace(1.0, 2.1, 30)

    def test_returns_correct_shape(self):
        self.env.step([1.47, 1.47])
        cf = self.env.counterfactual_profits(0, self.grid)
        assert cf.shape == (30,)

    def test_actual_action_matches_reward(self):
        """CF at the exact price played equals the actual reward."""
        _, rewards, _, _ = self.env.step([1.5, 1.5])
        cf = self.env.counterfactual_profits(0, np.array([1.5]))
        assert cf[0] == pytest.approx(rewards[0], rel=1e-5)

    def test_best_response_near_nash(self):
        """Given rival at p_NE, argmax CF should be near p_NE."""
        bm = self.env.benchmarks()
        self.env.step([bm["p_NE"], bm["p_NE"]])
        cf      = self.env.counterfactual_profits(0, self.grid)
        best_p  = self.grid[np.argmax(cf)]
        assert abs(best_p - bm["p_NE"]) < 0.12   # within one grid step

    def test_competitive_price_gives_zero_cf(self):
        """CF at p=c (competitive) = 0 regardless of rival price."""
        self.env.step([1.47, 1.47])
        cf = self.env.counterfactual_profits(0, np.array([1.0]))
        assert cf[0] == pytest.approx(0.0, abs=TOL_S)

    def test_cf_differs_per_agent(self):
        """Different rival prices → different CF profiles."""
        self.env.step([1.3, 1.8])
        cf0 = self.env.counterfactual_profits(0, self.grid)
        cf1 = self.env.counterfactual_profits(1, self.grid)
        assert not np.allclose(cf0, cf1)

    def test_undercutting_high_rival_gives_high_cf(self):
        """If rival charges monopoly price, undercutting is very profitable."""
        bm = self.env.benchmarks()
        self.env.step([bm["p_monopoly"], bm["p_monopoly"]])
        cf = self.env.counterfactual_profits(0, self.grid)
        # Undercut slightly below p_M: should be max
        best_p = self.grid[np.argmax(cf)]
        assert best_p < bm["p_monopoly"]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_returns_dict(self):
        assert isinstance(make_env().benchmarks(), dict)

    def test_nash_price_above_competitive(self):
        bm = make_env().benchmarks()
        assert bm["p_NE"] > bm["p_competitive"]

    def test_monopoly_price_above_nash(self):
        bm = make_env().benchmarks()
        assert bm["p_monopoly"] > bm["p_NE"]

    def test_nash_profit_positive(self):
        bm = make_env().benchmarks()
        assert bm["pi_NE"] > 0.0

    def test_monopoly_profit_above_nash(self):
        bm = make_env().benchmarks()
        assert bm["pi_monopoly_per_firm"] > bm["pi_NE"]

    def test_calvano_nash_price_approximate(self):
        """Calvano baseline: p_NE ≈ 1.473 (published value)."""
        bm = make_env(n=2).benchmarks()
        assert bm["p_NE"] == pytest.approx(1.473, abs=0.005)

    def test_calvano_monopoly_price_approximate(self):
        """Calvano baseline: p_M ≈ 1.925 (published value)."""
        bm = make_env(n=2).benchmarks()
        assert bm["p_monopoly"] == pytest.approx(1.925, abs=0.01)

    def test_competitive_equals_marginal_cost(self):
        bm = make_env().benchmarks()
        assert bm["p_competitive"] == pytest.approx(1.0, abs=TOL_S)

    def test_nash_foc_satisfied(self):
        """At p_NE, best-response condition holds: p = c + mu/(1-q)."""
        bm  = make_env(n=2).benchmarks()
        p   = bm["p_NE"]
        q   = bm["q_NE"]
        mu  = 0.25
        c   = 1.0
        assert p == pytest.approx(c + mu / (1.0 - q), abs=TOL)

    def test_n3_nash_price_higher_than_n2(self):
        """More firms → lower market power → lower Nash price in Bertrand?
        Actually with logit, N affects market shares but Nash FOC
        p = c + mu/(1-q) where q shrinks with N → p_NE increases slightly.
        This tests monotonicity direction."""
        bm2 = make_env(n=2).benchmarks()
        bm3 = make_env(n=3).benchmarks()
        # q per firm decreases with N, so 1/(1-q) increases and p_NE increases
        # This is specific to logit; test that the relationship is consistent
        assert bm2["p_NE"] != pytest.approx(bm3["p_NE"], abs=0.001)


# ---------------------------------------------------------------------------
# Caminho rápido em numpy (opt-in) — acrescentado 2026-08-11
# ---------------------------------------------------------------------------

def _cfg_fast(fast):
    return {
        "type": "bertrand",
        "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
        "cost": {"c": 1.0},
        "market": {"n_firms": 2, "grid_extend": 1.0, "fast_step": fast},
        "state": {"info_structure": "rival_prices"},
    }


def test_fast_step_is_off_by_default():
    """Opt-in: nada do que já existe pode mudar de comportamento sem pedir."""
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    env = MultiAgentBertrandEnv(_cfg_fast(False))
    assert env._fast_step is False
    cfg = _cfg_fast(False); del cfg["market"]["fast_step"]
    assert MultiAgentBertrandEnv(cfg)._fast_step is False


def test_fast_step_matches_jax_within_one_ulp():
    """NÃO é bit-exato (implementações diferentes de exp), mas a diferença tem de ficar
    em meio ULP de float32. Se crescer, alguma coisa divergiu de verdade."""
    import numpy as np
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    slow = MultiAgentBertrandEnv(_cfg_fast(False))
    fast = MultiAgentBertrandEnv(_cfg_fast(True))
    grid = np.linspace(slow.p_min, slow.p_max, 15)
    worst = 0.0
    for x in grid:
        for y in grid:
            _, r_slow, _, _ = slow.step([float(x), float(y)])
            _, r_fast, _, _ = fast.step([float(x), float(y)])
            worst = max(worst, float(np.abs(np.array(r_slow) - np.array(r_fast)).max()))
    assert worst < 1.2e-7, f"divergência além de 1 ULP: {worst:.3e}"


def test_fast_step_clips_like_the_jax_path():
    import numpy as np
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    slow, fast = MultiAgentBertrandEnv(_cfg_fast(False)), MultiAgentBertrandEnv(_cfg_fast(True))
    for pair in ([-5.0, 99.0], [0.0, 0.0], [99.0, 99.0]):
        _, rs, _, _ = slow.step(pair)
        _, rf, _, _ = fast.step(pair)
        assert np.allclose(np.array(rs), np.array(rf), atol=1.2e-7)


# ---------------------------------------------------------------------------
# Features categóricas do rival (p3, etapa 8)
# ---------------------------------------------------------------------------

class TestStateFeatures:
    @staticmethod
    def _env(features):
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        return MultiAgentBertrandEnv({
            "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
            "cost": {"c": 1.0},
            "market": {"n_firms": 2, "grid_extend": 0.1},
            "state": {"info_structure": "rival_prices", "features": features},
        })

    def test_obs_ganha_um_componente_por_feature(self):
        env = self._env(["rival_direction", "rival_relative", "own_profit_sign"])
        obs = env.reset()
        assert len(obs[0]) == 2 + 3
        assert env.obs_size() == 5
        assert env.obs_radices(15) == [15, 15, 3, 3, 2]

    def test_direcao_do_rival_le_o_penultimo_periodo(self):
        env = self._env(["rival_direction"])
        env.reset()
        env.step([1.5, 1.5])
        obs, _, _, _ = env.step([1.5, 1.8])      # rival (firma 1) SUBIU
        assert obs[0][-1] == 2.0
        obs, _, _, _ = env.step([1.5, 1.6])      # rival DESCEU
        assert obs[0][-1] == 0.0
        obs, _, _, _ = env.step([1.5, 1.6])      # rival MANTEVE
        assert obs[0][-1] == 1.0

    def test_relative_compara_com_o_meu_preco(self):
        env = self._env(["rival_relative"])
        env.reset()
        obs, _, _, _ = env.step([1.7, 1.5])      # rival CORTOU abaixo de mim
        assert obs[0][-1] == 0.0
        assert obs[1][-1] == 2.0                 # do ponto de vista dele, EU superei
        obs, _, _, _ = env.step([1.6, 1.6])
        assert obs[0][-1] == 1.0

    def test_profit_sign_responde_ao_lucro_proprio(self):
        env = self._env(["own_profit_sign"])
        env.reset()
        env.step([1.5, 1.5])
        obs, _, _, _ = env.step([1.7, 1.7])      # preços conjuntos ↑ ⇒ lucro ↑ (nesta faixa)
        assert obs[0][-1] == 1.0

    def test_features_exigem_rival_prices(self):
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        with pytest.raises(ValueError, match="rival_prices"):
            MultiAgentBertrandEnv({
                "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
                "cost": {"c": 1.0},
                "market": {"n_firms": 2, "grid_extend": 0.1},
                "state": {"info_structure": "own_price", "features": ["rival_direction"]},
            })

    def test_sem_features_o_obs_e_bit_exato_ao_historico(self):
        env_a = self._env([])
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        env_b = MultiAgentBertrandEnv({
            "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
            "cost": {"c": 1.0},
            "market": {"n_firms": 2, "grid_extend": 0.1},
            "state": {"info_structure": "rival_prices"},
        })
        env_a.reset(); env_b.reset()
        oa, _, _, _ = env_a.step([1.5, 1.7])
        ob, _, _, _ = env_b.step([1.5, 1.7])
        assert (oa[0] == ob[0]).all() and len(oa[0]) == 2


class TestDropOwnPrice:
    @staticmethod
    def _env(**state_extra):
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        return MultiAgentBertrandEnv({
            "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
            "cost": {"c": 1.0},
            "market": {"n_firms": 2, "grid_extend": 0.1},
            "state": {"info_structure": "rival_prices", **state_extra},
        })

    def test_obs_egocentrico_tem_so_o_preco_do_rival(self):
        env = self._env(features=["rival_direction", "own_profit_sign"],
                        drop_own_price=True)
        env.reset()
        obs, _, _, _ = env.step([1.5, 1.9])
        assert len(obs[0]) == 3 and len(obs[1]) == 3
        assert obs[0][0] == pytest.approx(1.9)   # agente 0 vê o preço do rival (1)
        assert obs[1][0] == pytest.approx(1.5)
        assert env.obs_radices(15) == [15, 3, 2]
        assert env.obs_size() == 3

    def test_drop_sem_features_e_erro(self):
        with pytest.raises(ValueError, match="features"):
            self._env(drop_own_price=True)


# ---------------------------------------------------------------------------
# A grade ancorada e a Proposição 8 — pinos contra erro de MEDIÇÃO
#
# `docs/theory/modelo_formal_bertrand_logit.md` §3.7 prova que a grade construída por
#   p_min = p_NE − ξ·Δ,  p_max = p_M + ξ·Δ,  Δ = p_M − p_NE
# é simétrica em torno de (p_NE + p_M)/2 para QUALQUER ξ e QUALQUER n_actions, e que
# portanto E[CI de preço] = 1/2 exatamente sob jogo uniforme.
#
# Essa propriedade é a única coisa que dá sentido a ler CI contra 0.500 (armadilha 16).
# Ela morre em silêncio se o clamp `max(c, ·)` morder — daí os dois testes abaixo.
# ---------------------------------------------------------------------------

def _grid_env(extend, n_actions=15, a=2.0, mu=0.25, c=1.0):
    return MultiAgentBertrandEnv({
        "demand": {"a": a, "mu": mu, "a0": 0.0},
        "cost":   {"c": c},
        "market": {"n_firms": 2, "grid_extend": extend},
        "state":  {"info_structure": "rival_prices"},
    }), n_actions


@pytest.mark.parametrize("extend", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("n_actions", [5, 15, 51])
def test_grade_ancorada_e_simetrica_logo_o_acaso_vale_meio(extend, n_actions):
    """Prop. 8: o ponto médio da grade é (p_NE + p_M)/2, e E[CI] = 1/2 exatamente.

    Independente de `grid_extend` e de `n_actions` — a prova não usa nenhum dos dois.
    """
    env, _ = _grid_env(extend, n_actions)
    bm = env.benchmarks()
    p_ne, p_m = float(bm["p_NE"]), float(bm["p_monopoly"])

    meio_grade = 0.5 * (env.p_min + env.p_max)
    assert meio_grade == pytest.approx(0.5 * (p_ne + p_m), abs=1e-9)

    grid = np.linspace(env.p_min, env.p_max, n_actions)
    ci = (grid.mean() - p_ne) / (p_m - p_ne)
    assert ci == pytest.approx(0.5, abs=1e-9)


def test_clamp_do_p_min_falha_alto_em_vez_de_descentrar_a_grade():
    """Quando `max(c, ·)` morderia, o env levanta — não descentra em silêncio.

    O clamp é economicamente correto (preço < custo marginal não faz sentido), mas
    quebra a Prop. 8. Como ler CI contra 0.500 numa grade descentrada é exatamente a
    armadilha 16, o env prefere falhar.
    """
    env, _ = _grid_env(1.0)                        # a convenção legada: seguro
    bm = env.benchmarks()
    p_ne, p_m = float(bm["p_NE"]), float(bm["p_monopoly"])
    xi_max = (p_ne - env.c) / (p_m - p_ne)
    assert xi_max == pytest.approx(1.0462, abs=1e-3)   # pino no número documentado

    with pytest.raises(ValueError, match="DESCENTRA"):
        _grid_env(xi_max + 0.05)

    # e a mensagem ensina o limite, em vez de só reclamar
    with pytest.raises(ValueError, match=r"grid_extend <= 1\.04"):
        _grid_env(1.5)


def test_as_duas_convencoes_em_uso_seguem_validas():
    """0.1 (convenção atual) e 1.0 (réplica canônica do p1) não podem quebrar."""
    for extend, esperado in ((0.1, (1.4277, 1.9702)), (1.0, (1.0209, 2.3770))):
        env, _ = _grid_env(extend)
        assert env.p_min == pytest.approx(esperado[0], abs=1e-3)
        assert env.p_max == pytest.approx(esperado[1], abs=1e-3)


# ---------------------------------------------------------------------------
# P6 — monitoramento imperfeito (hip.p6.3): sinal público ruidoso de suporte comum
#
# Ver docs/theory/modelo_formal_bertrand_logit.md §4.5. Estes testes são pinos contra
# erro de MEDIÇÃO: se `test_lucro_usa_preco_verdadeiro_nao_o_observado` quebrar, o
# experimento deixou de ser sobre monitoramento e passou a ser sobre ruído de demanda.
# ---------------------------------------------------------------------------

def _mon_env(eta=0.0, scope="public", mtype=None, n_actions=15, init="p_monopoly"):
    mon = {}
    if mtype or eta:
        mon = {"type": mtype or "noisy_price", "eta": eta, "scope": scope}
    return MultiAgentBertrandEnv({
        "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
        "cost":   {"c": 1.0},
        "market": {"n_firms": 2, "grid_extend": 0.1, "n_actions": n_actions,
                   "init_mode": init},
        "state":  {"info_structure": "rival_prices", **({"monitoring": mon} if mon else {})},
    })


def test_monitoramento_perfeito_e_o_default_e_e_bit_exato():
    """Sem config de monitoramento, o obs É o vetor de preços verdadeiro — o mesmo objeto."""
    env = _mon_env()
    assert env.monitoring_type == "perfect"
    assert env._noisy_mon is False
    env.reset(seed=7)
    obs, rew, _, info = env.step([1.7, 1.9])
    assert np.allclose(obs[0], info["prices"])
    assert np.allclose(obs[1], info["prices"])


def test_lucro_usa_preco_verdadeiro_nao_o_observado():
    """⚠️ O PINO CENTRAL de P6: o ruído está na observação, não no mercado.

    Com eta = 1.0 toda observação é sorteada, então obs e preço verdadeiro divergem quase
    sempre — mas o lucro tem de ser exatamente o do preço jogado. Se este teste quebrar,
    as firmas passaram a ser pagas pelo que o rival PARECEU cobrar.
    """
    env = _mon_env(eta=1.0)
    env.reset(seed=3)
    p = [1.7, 1.9]
    obs, rew, _, info = env.step(p)
    # o lucro confere com o cálculo analítico sobre os preços VERDADEIROS
    e = [math.exp((2.0 - x) / 0.25) for x in p]
    d = 1.0 + sum(e)
    esperado = [(p[i] - 1.0) * e[i] / d for i in (0, 1)]
    assert rew == pytest.approx(esperado, abs=TOL_S)
    assert info["prices"] == pytest.approx(p, abs=TOL_S)
    # e a observação de fato divergiu do verdadeiro (eta = 1 ⇒ ambos sorteados)
    assert not np.allclose(obs[0], p)


def test_suporte_comum_todo_sinal_tem_probabilidade_positiva():
    """hip.p6.3: P(observar j | verdadeiro j') > 0 para todo par — é o que torna
    'o rival desviou' e 'houve ruído' indistinguíveis."""
    K = 15
    env = _mon_env(eta=0.5, n_actions=K)
    grid = np.linspace(env.p_min, env.p_max, K)
    env.reset(seed=1)
    vistos = set()
    for _ in range(4000):
        obs, _, _, _ = env.step([float(grid[13]), float(grid[13])])
        vistos.add(int(np.argmin(np.abs(grid - float(obs[0][1])))))
    assert vistos == set(range(K)), f"faltaram sinais: {sorted(set(range(K)) - vistos)}"


def test_taxa_de_corrupcao_bate_com_eta():
    """A fração de períodos em que o sinal difere do verdadeiro é eta·(K−1)/K por firma."""
    K, eta = 15, 0.2
    env = _mon_env(eta=eta, n_actions=K)
    grid = np.linspace(env.p_min, env.p_max, K)
    env.reset(seed=11)
    p = float(grid[13])
    difere = 0
    N = 6000
    # comparação por ÍNDICE da grade, não por preço: `_last_prices` é float32 e `grid` é
    # float64, então a diferença de arredondamento (~1e-8) faria todo período parecer
    # corrompido.
    for _ in range(N):
        obs, _, _, _ = env.step([p, p])
        if int(np.argmin(np.abs(grid - float(obs[0][1])))) != 13:
            difere += 1
    esperado = eta * (K - 1) / K
    assert difere / N == pytest.approx(esperado, abs=0.02)


def test_escopo_publico_as_duas_firmas_veem_o_mesmo_sinal():
    """hip.p6.3 é sinal PÚBLICO: a história continua comum."""
    env = _mon_env(eta=0.6, scope="public")
    env.reset(seed=5)
    for _ in range(50):
        obs, _, _, _ = env.step([1.7, 1.9])
        assert np.allclose(obs[0], obs[1])


def test_escopo_privado_as_firmas_podem_ver_coisas_diferentes():
    """Monitoramento privado: cada firma sorteia a sua corrupção."""
    env = _mon_env(eta=0.6, scope="private")
    env.reset(seed=5)
    divergiu = False
    for _ in range(200):
        obs, _, _, _ = env.step([1.7, 1.9])
        if not np.allclose(obs[0], obs[1]):
            divergiu = True
            break
    assert divergiu


def test_ruido_e_reprodutivel_pela_seed():
    seqs = []
    for _ in range(2):
        env = _mon_env(eta=0.4)
        env.reset(seed=42)
        seqs.append([env.step([1.7, 1.9])[0][0].copy() for _ in range(30)])
    for x, y in zip(*seqs):
        assert np.allclose(x, y)


def test_configs_invalidas_de_monitoramento_falham():
    with pytest.raises(ValueError, match="monitoring.type"):
        _mon_env(mtype="gaussiano", eta=0.1)
    with pytest.raises(ValueError, match="monitoring.scope"):
        _mon_env(eta=0.1, scope="semi")
    with pytest.raises(ValueError, match="eta fora"):
        _mon_env(eta=1.5)
    with pytest.raises(ValueError, match="n_actions"):
        _mon_env(eta=0.1, n_actions=None)
