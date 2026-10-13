"""
Unit tests for src/metrics — equilibrium and welfare.

Canonical params: a=1, b=1, c=0, n=2  →  q_NE=1/3, Q_NE=2/3, Q_M=1/2, Q_C=1
"""

import numpy as np
import pytest

from src.metrics.equilibrium import (
    nash_distance, collusion_index, dynamics_label, convergence_step
)
from src.metrics.welfare import (
    consumer_surplus, producer_surplus, total_welfare,
    deadweight_loss, welfare_table
)

TOL = 1e-5
A, B, C, N = 1.0, 1.0, 0.0, 2
Q_NE, Q_M, Q_C = 2/3, 0.5, 1.0


# ---------------------------------------------------------------------------
# equilibrium.py
# ---------------------------------------------------------------------------

class TestNashDistance:
    def test_at_nash(self):
        assert nash_distance(1/3, 1/3) < TOL

    def test_above_nash(self):
        assert abs(nash_distance(0.4, 1/3) - (0.4 - 1/3)) < TOL

    def test_below_nash(self):
        assert abs(nash_distance(0.2, 1/3) - (1/3 - 0.2)) < TOL


class TestCollusionIndex:
    def test_at_nash(self):
        assert abs(collusion_index(Q_NE, Q_NE, Q_M)) < TOL

    def test_at_monopoly(self):
        assert abs(collusion_index(Q_M, Q_NE, Q_M) - 1.0) < TOL

    def test_more_competitive_than_nash(self):
        assert collusion_index(Q_C, Q_NE, Q_M) < 0

    def test_zero_denominator(self):
        assert collusion_index(0.5, 0.5, 0.5) == 0.0

    def test_intermediate(self):
        # Midpoint between Nash and Monopoly → CI = 0.5
        Q_mid = (Q_NE + Q_M) / 2
        assert abs(collusion_index(Q_mid, Q_NE, Q_M) - 0.5) < TOL


class TestDynamicsLabel:
    def test_fixed_point(self):
        q_NE = 1/3
        series = np.full(100, q_NE)
        assert dynamics_label(series, q_NE) == "fixed_point"

    def test_unstable(self):
        q_NE = 1/3
        rng = np.random.default_rng(0)
        series = rng.uniform(0, 1, 100)
        assert dynamics_label(series, q_NE) == "unstable"

    def test_insufficient_data(self):
        assert dynamics_label(np.array([0.3, 0.35]), 1/3) == "insufficient_data"

    def test_cycling(self):
        # Perfect alternating series → high autocorrelation
        q_NE = 1/3
        vals = np.tile([0.32, 0.35], 50)
        label = dynamics_label(vals, q_NE)
        assert label in ("cycling", "converging")


class TestConvergenceStep:
    def test_converged_from_start(self):
        nd   = np.full(10, 0.01)
        steps = np.arange(1000, 11000, 1000)
        assert convergence_step(nd, steps, threshold=0.05) == 1000

    def test_never_converged(self):
        nd   = np.full(10, 0.1)
        steps = np.arange(1000, 11000, 1000)
        assert convergence_step(nd, steps, threshold=0.05) == -1

    def test_converges_midway(self):
        nd   = np.array([0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        steps = np.arange(1000, 11000, 1000)
        assert convergence_step(nd, steps, threshold=0.05) == 5000


# ---------------------------------------------------------------------------
# welfare.py
# ---------------------------------------------------------------------------

class TestConsumerSurplus:
    def test_zero_quantity(self):
        assert abs(consumer_surplus(0.0, A, B)) < TOL

    def test_at_competitive(self):
        # Q = Q_C = 1, p = 0 → CS = 0.5 * 1 * 1 = 0.5
        assert abs(consumer_surplus(Q_C, A, B) - 0.5) < TOL

    def test_at_nash(self):
        # Q_NE = 2/3, p_NE = 1/3 → CS = 0.5 * (1 - 1/3) * 2/3 = 2/9
        assert abs(consumer_surplus(Q_NE, A, B) - 2/9) < TOL

    def test_at_monopoly(self):
        # Q_M = 0.5, p_M = 0.5 → CS = 0.5 * 0.5 * 0.5 = 0.125
        assert abs(consumer_surplus(Q_M, A, B) - 0.125) < TOL

    def test_ordering(self):
        # Higher output → higher CS
        assert consumer_surplus(Q_C, A, B) > consumer_surplus(Q_NE, A, B) > consumer_surplus(Q_M, A, B)


class TestProducerSurplus:
    def test_zero_cost(self):
        # PS = price * Q  (c=0)
        price = 1/3
        qs = [1/3, 1/3]
        assert abs(producer_surplus(qs, price, C) - price * 2/3) < TOL

    def test_price_equals_cost(self):
        assert abs(producer_surplus([0.5], 0.0, 0.0)) < TOL

    def test_positive_cost_reduces_ps(self):
        ps0 = producer_surplus([0.3], 0.5, 0.0)
        ps1 = producer_surplus([0.3], 0.5, 0.1)
        assert ps0 > ps1


class TestTotalWelfare:
    def test_competitive_maximises_welfare(self):
        # W at Q_C should be ≥ W at Q_NE
        w_c  = total_welfare(Q_C,  [Q_C],  A, B, C)
        w_ne = total_welfare(Q_NE, [Q_NE], A, B, C)
        assert w_c >= w_ne

    def test_welfare_positive(self):
        assert total_welfare(Q_NE, [1/3, 1/3], A, B, C) > 0


class TestDeadweightLoss:
    def test_at_competitive_no_dwl(self):
        assert abs(deadweight_loss(Q_C, Q_C, A, B, C)) < TOL

    def test_monopoly_has_dwl(self):
        assert deadweight_loss(Q_M, Q_C, A, B, C) > 0

    def test_nash_has_dwl(self):
        assert deadweight_loss(Q_NE, Q_C, A, B, C) > 0

    def test_monopoly_dwl_greater_than_nash(self):
        dwl_m  = deadweight_loss(Q_M,  Q_C, A, B, C)
        dwl_ne = deadweight_loss(Q_NE, Q_C, A, B, C)
        assert dwl_m > dwl_ne


class TestWelfareTable:
    def setup_method(self):
        from src.econ_utils.benchmarks import nash_symmetric, monopoly, competitive
        ne   = nash_symmetric(A, B, C, N)
        mono = monopoly(A, B, C)
        comp = competitive(A, B, C)
        self.bm = {
            "q_NE": ne["q_per_firm"], "Q_NE": ne["Q_total"],
            "p_NE": ne["price"],      "pi_NE": ne["profit_per_firm"],
            "Q_monopoly": mono["Q_total"], "p_monopoly": mono["price"],
            "pi_monopoly": mono["profit_total"],
            "Q_competitive": comp["Q_total"], "p_competitive": comp["price"],
        }

    def test_all_outcomes_present(self):
        wt = welfare_table(Q_NE, N, self.bm, A, B, C)
        assert set(wt.keys()) == {"RL", "Nash", "Monopoly", "Competitive"}

    def test_rl_equals_nash_when_q_is_nash(self):
        wt = welfare_table(Q_NE, N, self.bm, A, B, C)
        assert abs(wt["RL"]["W"] - wt["Nash"]["W"]) < TOL

    def test_competitive_has_highest_welfare(self):
        wt = welfare_table(Q_NE, N, self.bm, A, B, C)
        assert wt["Competitive"]["W"] >= wt["Nash"]["W"] >= wt["Monopoly"]["W"]


# ---------------------------------------------------------------------------
# Métricas de lucro (acrescentadas 2026-08-11)
# ---------------------------------------------------------------------------

def test_collusion_index_profit_endpoints():
    from src.metrics.multi_agent import collusion_index_profit as cip
    assert cip(0.2229, 0.2229, 0.3375) == pytest.approx(0.0)     # Nash
    assert cip(0.3375, 0.2229, 0.3375) == pytest.approx(1.0)     # monopólio
    assert cip(0.1500, 0.2229, 0.3375) < 0                       # abaixo do competitivo


def test_collusion_index_profit_degenerate_benchmarks():
    from src.metrics.multi_agent import collusion_index_profit as cip
    assert cip(0.3, 0.25, 0.25) == 0.0


def test_discounted_value_matches_closed_form():
    """Trajetória constante ⇒ Σ γ^t π = π·(1−γ^T)/(1−γ)."""
    from src.metrics.multi_agent import discounted_value
    pi, g, T = 0.25, 0.95, 100
    got = discounted_value([pi] * T, g)
    assert got == pytest.approx(pi * (1 - g ** T) / (1 - g))


def test_discounted_value_weights_early_periods_more():
    from src.metrics.multi_agent import discounted_value
    assert discounted_value([1.0, 0.0], 0.9) > discounted_value([0.0, 1.0], 0.9)


def test_price_and_profit_indices_diverge_when_asymmetric():
    """O ponto que motivou a métrica: com preços iguais os dois índices concordam de
    perto; com preços descasados o de preço superestima a colusão."""
    import numpy as np
    from src.metrics.multi_agent import collusion_index_price as cipr
    from src.metrics.multi_agent import collusion_index_profit as cip

    a, mu, c = 2.0, 0.25, 1.0
    def profit(p_own, p_riv):
        e_i, e_j = np.exp((a - p_own) / mu), np.exp((a - p_riv) / mu)
        return (p_own - c) * e_i / (1.0 + e_i + e_j)

    p_ne, p_m = 1.4729, 1.9250
    pi_ne, pi_m = profit(p_ne, p_ne), profit(p_m, p_m)

    # simétrico: preço 70% do caminho
    p = p_ne + 0.7 * (p_m - p_ne)
    sym_price  = cipr(p, p_ne, p_m)
    sym_profit = cip(profit(p, p), pi_ne, pi_m)
    assert sym_profit > sym_price          # concavidade do lucro

    # assimétrico com a MESMA média de preço
    d = 0.25
    pa, pb = p + d, p - d
    asym_price  = cipr((pa + pb) / 2, p_ne, p_m)
    asym_profit = cip((profit(pa, pb) + profit(pb, pa)) / 2, pi_ne, pi_m)
    assert asym_price == pytest.approx(sym_price)   # mesma média de preço
    assert asym_profit < sym_profit                 # mas lucro conjunto pior
    assert asym_price > asym_profit                 # o de preço SUPERESTIMA


def test_mean_gap_understates_dispersion_in_cyclical_regimes():
    """O defeito que a c3 expôs: |Δ das MÉDIAS| ≈ 0 numa trajetória que ALTERNA, mesmo
    com um abismo em cada instante. É o gap INSTANTÂNEO que explica a perda de lucro."""
    import numpy as np
    ha = np.array([1.9, 1.5, 1.9, 1.5, 1.9, 1.5])
    hb = np.array([1.5, 1.9, 1.5, 1.9, 1.5, 1.9])
    gap_medias = abs(ha.mean() - hb.mean())
    gap_inst   = np.mean(np.abs(ha - hb))
    assert gap_medias == pytest.approx(0.0)      # some por completo
    assert gap_inst == pytest.approx(0.4)
    assert gap_inst > 10 * max(gap_medias, 1e-9)


def test_asymmetry_destroys_joint_profit_at_fixed_mean_price():
    """Com mu=0.25 a demanda logit é brutalmente sensível: mesmo preço MÉDIO, e o lucro
    conjunto despenca com o descasamento. É por isso que CI de preço e de lucro divergem."""
    import numpy as np
    from src.metrics.multi_agent import collusion_index_profit as cip
    a, mu, c = 2.0, 0.25, 1.0
    def profit(p_own, p_riv):
        e_i, e_j = np.exp((a - p_own) / mu), np.exp((a - p_riv) / mu)
        return (p_own - c) * e_i / (1.0 + e_i + e_j)
    pi_ne, pi_m = profit(1.4729, 1.4729), profit(1.9250, 1.9250)
    pbar = 1.7215
    cis = []
    for gap in (0.0, 0.2, 0.45):
        pa, pb = pbar + gap / 2, pbar - gap / 2
        cis.append(cip((profit(pa, pb) + profit(pb, pa)) / 2, pi_ne, pi_m))
    assert cis[0] > cis[1] > cis[2]              # monotonicamente pior
    assert cis[0] - cis[2] > 0.4                 # e a queda é grande


class TestLogitWelfare:
    """CS logit (p3/c4) — pinos nos limites conhecidos."""

    A, MU, C = 2.0, 0.25, 1.0

    def test_cs_decresce_no_preco(self):
        from src.metrics.welfare import logit_consumer_surplus
        cs_lo = logit_consumer_surplus([1.4729, 1.4729], self.A, self.MU)
        cs_hi = logit_consumer_surplus([1.9250, 1.9250], self.A, self.MU)
        assert cs_lo > cs_hi

    def test_precos_infinitos_colapsam_na_opcao_externa(self):
        from src.metrics.welfare import logit_consumer_surplus
        cs = logit_consumer_surplus([1e6, 1e6], self.A, self.MU, a0=0.0)
        assert cs == pytest.approx(0.0, abs=1e-9)   # μ·ln(exp(0/μ)) = 0

    def test_welfare_row_consistente_com_lucro_do_env(self):
        """O lucro total da linha de welfare tem de bater com o lucro do próprio
        ambiente — mesma fórmula logit, mesmo resultado."""
        from src.metrics.welfare import logit_welfare_row
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        env = MultiAgentBertrandEnv({
            "demand": {"a": self.A, "mu": self.MU, "a0": 0.0},
            "cost": {"c": self.C},
            "market": {"n_firms": 2, "grid_extend": 0.1},
            "state": {"info_structure": "rival_prices"},
        })
        env.reset()
        _, rewards, _, _ = env.step([1.7, 1.8])
        row = logit_welfare_row([1.7, 1.8], self.A, self.MU, self.C)
        assert row["ps"] == pytest.approx(sum(rewards), rel=1e-5)
        assert row["welfare"] == pytest.approx(row["cs"] + row["ps"])
