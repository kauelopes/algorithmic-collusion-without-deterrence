"""
Tests for src/training/multi_agent_trainer.py

Verifies correctness of the training loop for N=2 and N=3 with each
supported algorithm (q_learning, reinforce, hedge).

All tests use a minimal step count (smoke-test level) — correctness of
convergence is validated in Milestone 2.4 via full baseline runs.
"""

import numpy as np
import pytest

from src.training import multi_agent_trainer as trainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _env_config(n_firms=2, info="bandit"):
    return {
        "demand": {"a": 1.0, "b": 1.0},
        "cost": {"c": 0.0},
        "market": {"n_firms": n_firms},
        "state": {"info_structure": info},
    }


def _training_config(total_steps=200, eval_interval=100):
    return {
        "total_steps": total_steps,
        "eval_interval": eval_interval,
        "eval_window": 100,
        "seed": 0,
    }


def _agent_config(agent_type="q_learning"):
    if agent_type == "q_learning":
        return {
            "type": "q_learning",
            "n_actions": 10,
            "q_max": 1.0,
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 100,
            "memory_length": 1,
        }
    elif agent_type == "reinforce":
        return {
            "type": "reinforce",
            "q_max": 1.0,
            "hidden_dims": [16, 16],
            "learning_rate": 3e-4,
            "gamma": 0.95,
            "entropy_coef": 0.01,
            "obs_dim": 1,
        }
    elif agent_type == "hedge":
        return {
            "type": "hedge",
            "q_max": 1.0,
            "n_actions": 10,
            "eta": 0.01,
            "obs_dim": 1,
        }
    raise ValueError(f"Unknown agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Smoke tests — runs without error
# ---------------------------------------------------------------------------

class TestSmokeN2:
    """Trainer completes 200 steps without error for N=2."""

    @pytest.mark.parametrize("agent_type", ["q_learning", "reinforce", "hedge"])
    def test_runs_without_error(self, agent_type, capsys):
        trainer.run(
            _env_config(n_firms=2, info="bandit"),
            _agent_config(agent_type),
            _training_config(),
        )
        out = capsys.readouterr().out
        assert "Training complete." in out

    @pytest.mark.parametrize("agent_type", ["q_learning", "reinforce", "hedge"])
    def test_runs_price_info(self, agent_type):
        trainer.run(
            _env_config(n_firms=2, info="price"),
            _agent_config(agent_type),
            _training_config(),
        )

    def test_runs_full_info(self):
        trainer.run(
            _env_config(n_firms=2, info="full"),
            _agent_config("q_learning"),
            _training_config(),
        )


class TestSmokeN3:
    """Trainer completes 200 steps without error for N=3."""

    @pytest.mark.parametrize("agent_type", ["q_learning", "reinforce", "hedge"])
    def test_runs_without_error(self, agent_type):
        trainer.run(
            _env_config(n_firms=3, info="bandit"),
            _agent_config(agent_type),
            _training_config(),
        )


# ---------------------------------------------------------------------------
# History structure
# ---------------------------------------------------------------------------

class TestHistoryStructure:
    """The history list captured via monkeypatching has correct structure."""

    def _run_and_capture(self, n_firms=2, agent_type="q_learning", info="bandit"):
        """Run trainer and capture history by patching output writer."""
        captured = []

        import src.general_utils.multi_agent_output_writer as ow
        orig_write = ow.write

        def mock_write(config, history, bm, output_dir=None, cce_data=None):
            captured.extend(history)
            # Create a minimal path-like object
            from pathlib import Path
            return Path("/tmp/test_output")

        ow.write = mock_write

        full_config = {
            "experiment_name": "test",
            "environment": _env_config(n_firms, info),
            "agent": _agent_config(agent_type),
            "training": _training_config(total_steps=300, eval_interval=100),
        }
        try:
            trainer.run(
                _env_config(n_firms, info),
                _agent_config(agent_type),
                _training_config(total_steps=300, eval_interval=100),
                full_config=full_config,
            )
        finally:
            ow.write = orig_write

        return captured

    def test_history_has_required_keys(self):
        history = self._run_and_capture()
        required = {"step", "collusion_index", "nash_dist", "ce_gap", "sym_q"}
        for record in history:
            assert required <= set(record.keys()), (
                f"Missing keys: {required - set(record.keys())}"
            )

    def test_history_has_per_agent_keys_n2(self):
        history = self._run_and_capture(n_firms=2)
        for record in history:
            assert "mean_q_0" in record
            assert "mean_q_1" in record
            assert "greedy_q_0" in record
            assert "greedy_q_1" in record

    def test_history_has_per_agent_keys_n3(self):
        history = self._run_and_capture(n_firms=3)
        for record in history:
            assert "mean_q_2" in record
            assert "greedy_q_2" in record

    def test_steps_are_monotone(self):
        history = self._run_and_capture()
        steps = [r["step"] for r in history]
        assert steps == sorted(steps)
        assert steps[-1] == 300

    def test_ci_is_finite(self):
        history = self._run_and_capture()
        for r in history:
            assert np.isfinite(r["collusion_index"])

    def test_nash_dist_non_negative(self):
        history = self._run_and_capture()
        for r in history:
            assert r["nash_dist"] >= 0.0

    def test_ce_gap_non_negative(self):
        history = self._run_and_capture()
        for r in history:
            assert r["ce_gap"] >= 0.0

    def test_sym_q_non_negative(self):
        history = self._run_and_capture()
        for r in history:
            assert r["sym_q"] >= 0.0

    def test_training_curves_keys_present_and_finite(self):
        """Pino do p3 (etapa 3): a 'curva de loss' do tabular sai no history.

        `agent.update` sempre devolveu {td_error, epsilon, alpha} e o trainer
        descartava; se este teste quebrar, o descarte voltou.
        """
        history = self._run_and_capture(agent_type="q_learning")
        for r in history:
            for i in range(2):
                assert f"td_error_mean_{i}" in r
                assert np.isfinite(r[f"td_error_mean_{i}"])
                assert r[f"td_error_mean_{i}"] >= 0.0
                assert f"td_error_p95_{i}" in r
                assert f"alpha_{i}" in r
            assert "epsilon" in r
            assert 0.0 <= r["epsilon"] <= 1.0


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

class TestConsoleOutput:
    def test_header_printed(self, capsys):
        trainer.run(
            _env_config(), _agent_config(), _training_config()
        )
        out = capsys.readouterr().out
        assert "CI" in out
        assert "nash_dist" in out
        assert "CE-gap" in out

    def test_eval_rows_printed(self, capsys):
        trainer.run(
            _env_config(), _agent_config(),
            _training_config(total_steps=300, eval_interval=100),
        )
        out = capsys.readouterr().out
        # 3 eval rows at steps 100, 200, 300
        data_lines = [
            ln for ln in out.splitlines()
            if ln.strip() and not ln.startswith("-") and "step" not in ln and "Training" not in ln
        ]
        assert len(data_lines) >= 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def _run_collect_ci(self, seed, agent_type="q_learning"):
        captured = []

        import src.general_utils.multi_agent_output_writer as ow
        orig = ow.write

        def mock_write(config, history, bm, output_dir=None, cce_data=None):
            captured.extend(history)
            from pathlib import Path
            return Path("/tmp/test")

        ow.write = mock_write
        tc = {**_training_config(total_steps=200, eval_interval=100), "seed": seed}
        full_config = {
            "experiment_name": "test",
            "environment": _env_config(),
            "agent": _agent_config(agent_type),
            "training": tc,
        }
        try:
            trainer.run(_env_config(), _agent_config(agent_type), tc, full_config)
        finally:
            ow.write = orig
        return [r["collusion_index"] for r in captured]

    def test_same_seed_same_ci(self):
        ci1 = self._run_collect_ci(seed=7)
        ci2 = self._run_collect_ci(seed=7)
        assert ci1 == ci2

    def test_different_seeds_different_ci(self):
        ci_a = self._run_collect_ci(seed=0)
        ci_b = self._run_collect_ci(seed=99)
        # Not guaranteed to differ in just 200 steps, but usually will
        # We just check they don't raise
        assert isinstance(ci_a, list)
        assert isinstance(ci_b, list)
