"""
Tests for src/metrics/multi_agent.py

Canonical params: a=1, b=1, c=0, n=2.
  q_NE = 1/3,  Q_NE = 2/3,  Q_M = 1/2
"""

import numpy as np
import pytest

from src.metrics.multi_agent import (
    collusion_index,
    dynamics_label,
    final_metrics,
    inter_agent_symmetry,
    nash_distance_joint,
    _lag1_autocorr,
)

TOL = 1e-6

Q_NE = 2.0 / 3.0
Q_M  = 0.5
q_NE = 1.0 / 3.0

BM = {
    "q_NE": q_NE,
    "Q_NE": Q_NE,
    "Q_monopoly": Q_M,
    "p_NE": 1.0 / 3.0,
    "pi_NE": 1.0 / 9.0,
    "p_monopoly": 0.5,
    "Q_competitive": 1.0,
    "p_competitive": 0.0,
    "a": 1.0, "b": 1.0, "c": 0.0,
}


# ---------------------------------------------------------------------------
# collusion_index
# ---------------------------------------------------------------------------

class TestCollusionIndex:
    def test_nash_gives_zero(self):
        assert collusion_index(Q_NE, Q_NE, Q_M) == pytest.approx(0.0, abs=TOL)

    def test_monopoly_gives_one(self):
        assert collusion_index(Q_M, Q_NE, Q_M) == pytest.approx(1.0, abs=TOL)

    def test_over_production_is_negative(self):
        assert collusion_index(Q_NE + 0.1, Q_NE, Q_M) < 0.0

    def test_collusive_midpoint(self):
        Q_mid = (Q_NE + Q_M) / 2.0
        ci = collusion_index(Q_mid, Q_NE, Q_M)
        assert ci == pytest.approx(0.5, abs=TOL)

    def test_degenerate_denom_returns_zero(self):
        # Q_NE == Q_M would be degenerate — just check no exception
        result = collusion_index(0.5, 0.5, 0.5)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# nash_distance_joint
# ---------------------------------------------------------------------------

class TestNashDistanceJoint:
    def test_at_nash_is_zero(self):
        assert nash_distance_joint([q_NE, q_NE], q_NE) == pytest.approx(0.0, abs=TOL)

    def test_symmetric_deviation(self):
        d = 0.05
        assert nash_distance_joint([q_NE + d, q_NE + d], q_NE) == pytest.approx(d, abs=TOL)

    def test_asymmetric_is_mean(self):
        assert nash_distance_joint([q_NE + 0.1, q_NE - 0.1], q_NE) == pytest.approx(0.1, abs=TOL)

    def test_n3(self):
        result = nash_distance_joint([0.2, 0.3, 0.4], 0.3)
        # mean(|0.2-0.3|, |0.3-0.3|, |0.4-0.3|) = mean(0.1, 0, 0.1) = 1/15
        assert result == pytest.approx(1.0 / 15.0, abs=TOL)

    def test_non_negative(self):
        assert nash_distance_joint([0.0, 1.0], q_NE) >= 0.0


# ---------------------------------------------------------------------------
# inter_agent_symmetry
# ---------------------------------------------------------------------------

class TestInterAgentSymmetry:
    def test_identical_quantities_is_zero(self):
        assert inter_agent_symmetry([0.3, 0.3]) == pytest.approx(0.0, abs=TOL)

    def test_different_quantities_nonzero(self):
        assert inter_agent_symmetry([0.2, 0.4]) > 0.0

    def test_n3_symmetric(self):
        assert inter_agent_symmetry([0.3, 0.3, 0.3]) == pytest.approx(0.0, abs=TOL)

    def test_known_value(self):
        # std([0, 1]) = 0.5
        assert inter_agent_symmetry([0.0, 1.0]) == pytest.approx(0.5, abs=TOL)

    def test_non_negative(self):
        assert inter_agent_symmetry([0.1, 0.5, 0.3]) >= 0.0


# ---------------------------------------------------------------------------
# _lag1_autocorr (internal helper)
# ---------------------------------------------------------------------------

class TestLag1Autocorr:
    def test_constant_series_returns_one(self):
        arr = np.full(20, 0.5)
        assert _lag1_autocorr(arr) == pytest.approx(1.0, abs=TOL)

    def test_alternating_series_is_negative(self):
        arr = np.array([0.0, 1.0] * 20, dtype=float)
        assert _lag1_autocorr(arr) < 0.0

    def test_iid_noise_near_zero(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(10000)
        assert abs(_lag1_autocorr(arr)) < 0.05

    def test_ar1_high_correlation(self):
        arr = np.zeros(1000)
        arr[0] = 1.0
        for t in range(1, 1000):
            arr[t] = 0.95 * arr[t - 1] + np.random.default_rng(t).standard_normal() * 0.01
        assert _lag1_autocorr(arr) > 0.80

    def test_single_element_returns_zero(self):
        assert _lag1_autocorr(np.array([1.0])) == pytest.approx(0.0, abs=TOL)


# ---------------------------------------------------------------------------
# dynamics_label
# ---------------------------------------------------------------------------

class TestDynamicsLabel:
    def test_insufficient_data(self):
        assert dynamics_label(np.array([0.6, 0.6, 0.6]), Q_NE, ci_final=0.0) == "insufficient_data"

    def test_unstable_high_std(self):
        # std > 0.20 * Q_NE
        arr = np.array([0.0, 1.5] * 30, dtype=float)
        label = dynamics_label(arr, Q_NE, ci_final=0.0)
        assert label == "unstable"

    def test_nash_fixed_point(self):
        # Constant series near Nash total Q, CI < 0.1
        arr = np.full(50, Q_NE)
        label = dynamics_label(arr, Q_NE, ci_final=0.02)
        assert label == "nash_fixed_point"

    def test_collusive_fixed_point(self):
        # Constant series near monopoly, CI > 0.2
        arr = np.full(50, Q_M)
        label = dynamics_label(arr, Q_NE, ci_final=0.8)
        assert label == "collusive_fixed_point"

    def test_generic_fixed_point_ambiguous_ci(self):
        # Constant series, 0.10 <= CI <= 0.20
        arr = np.full(50, Q_NE * 0.9)
        label = dynamics_label(arr, Q_NE, ci_final=0.15)
        assert label == "fixed_point"

    def test_cycling_label(self):
        # Sinusoidal series with moderate autocorrelation
        t = np.linspace(0, 4 * np.pi, 100)
        arr = Q_NE + 0.05 * np.sin(t)
        label = dynamics_label(arr, Q_NE, ci_final=0.0)
        assert label in ("cycling", "nash_fixed_point", "fixed_point", "converging")
        # At minimum, does not crash

    def test_returns_string(self):
        arr = np.linspace(0.3, 0.7, 20)
        result = dynamics_label(arr, Q_NE, ci_final=0.3)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# final_metrics
# ---------------------------------------------------------------------------

class TestFinalMetrics:
    def _make_history(self, n_records=10, N=2):
        """Synthetic history where both agents play q_NE."""
        records = []
        for step in range(1, n_records + 1):
            r = {
                "step": step * 10000,
                "collusion_index": 0.0,
                "nash_dist": 0.001,
                "ce_gap": 0.002,
                "sym_q": 0.001,
                "Q_rl_total": Q_NE,
            }
            for i in range(N):
                r[f"mean_q_{i}"]      = q_NE
                r[f"mean_reward_{i}"] = 1.0 / 9.0
                r[f"greedy_q_{i}"]    = q_NE
            records.append(r)
        return records

    def test_returns_dict(self):
        h = self._make_history()
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        assert isinstance(fm, dict)

    def test_required_keys_present(self):
        h = self._make_history()
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        required = {"ci", "nash_dist", "ce_gap", "sym_q", "dynamics_label",
                    "mean_q_0", "mean_q_1", "greedy_q_0", "greedy_q_1"}
        assert required <= set(fm.keys())

    def test_per_agent_keys_n3(self):
        h = self._make_history(N=3)
        fm = final_metrics(h, 50000, 10000, BM, N=3)
        assert "mean_q_2" in fm
        assert "greedy_q_2" in fm

    def test_ci_correct_at_nash(self):
        h = self._make_history()
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        assert fm["ci"] == pytest.approx(0.0, abs=TOL)

    def test_nash_dist_near_zero_at_nash(self):
        h = self._make_history()
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        assert fm["nash_dist"] < 0.01

    def test_dynamics_label_is_string(self):
        h = self._make_history()
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        assert isinstance(fm["dynamics_label"], str)

    def test_single_record_does_not_crash(self):
        h = self._make_history(n_records=1)
        fm = final_metrics(h, 50000, 10000, BM, N=2)
        assert isinstance(fm, dict)
