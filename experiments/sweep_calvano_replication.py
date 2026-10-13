"""
Calvano et al. (2020) replication sweep — Q-learning only.

Runs N=2, rival_prices, q_learning with the corrected config:
  - Geometric epsilon decay (Calvano schedule)
  - Opponent-optimized Q-table initialisation (Calvano "type O")
  - Dynamic price grid centred on benchmarks (grid_extend=1.0)
  - Early stopping disabled
  - 2M training steps

Usage
-----
  python experiments/sweep_calvano_replication.py [options]

Options
-------
  --n_seeds N       seeds per combination (default 20)
  --seed_start S    first seed index (default 0)
  --n_workers N     parallel workers (default 4)
  --resume DIR      resume an interrupted run from results/DIR
  --compare DIR     after sweep, compare CI against baseline DIR
                    (default: auto-detect latest bertrand_sweep_*)

Output
------
  results/calvano_replication__YYYYMMDD_HHMMSS/
    progress.json
    seed_000/ … seed_019/
      training.log
      metrics_history.csv
      config.yaml
      tables/
      plots/
    comparison.txt     ← CI summary: old vs new
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

_RESULTS_DIR = _PROJ_ROOT / "results"

# ---------------------------------------------------------------------------
# Corrected Calvano config (all 5 fixes applied)
# ---------------------------------------------------------------------------

_ENV_CONFIG = {
    "type":   "bertrand",
    "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
    "cost":   {"c": 1.0},
    "market": {
        "n_firms":     2,
        "grid_extend": 1.0,          # fix 5: dynamic grid centred on benchmarks
    },
    "state": {"info_structure": "rival_prices"},
}

_AGENT_CONFIG = {
    "type":                "q_learning",
    "n_actions":           15,
    "memory_length":       2,        # N=2 rival_prices → 15^2 = 225 states
    "q_init":              "opponent_optimized",  # fix 4
    "alpha":               0.15,
    "gamma":               0.95,
    "epsilon_start":       1.0,
    "epsilon_end":         0.01,
    "epsilon_decay_steps": 1_150_000,  # delta≈4e-6 matches Calvano; ε=0.01 at step 1.15M, greedy for remaining 850k
    "epsilon_decay_type":  "geometric",           # fix 1
}

_TRAINING_CONFIG = {
    "total_steps":   2_000_000,      # fix 3
    "eval_interval":    10_000,
    "eval_window":     200_000,
    "early_stop": {"enabled": False},            # fix 2
}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_seed(seed: int, out_dir: Path) -> dict:
    """Train one seed. Runs in a subprocess (spawn)."""
    import jax
    import yaml
    from src.training.multi_agent_trainer import run as train

    seed_dir = out_dir / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    log_path = seed_dir / "training.log"
    training_cfg = {**_TRAINING_CONFIG, "seed": seed}

    with open(log_path, "w") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log_file
        try:
            result = train(
                env_config=_ENV_CONFIG,
                agent_config=_AGENT_CONFIG,
                training_config=training_cfg,
                full_config={
                    "environment": _ENV_CONFIG,
                    "agent":       _AGENT_CONFIG,
                    "training":    training_cfg,
                },
                output_dir=str(seed_dir),
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    return {
        "seed":     seed,
        "ci":       result["ci"],
        "ce_gap":   result["ce_gap"],
        "nash_dist":result["nash_dist"],
        "dynamics": result["dynamics_label"],
        "steps":    result["final_step"],
    }


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress(sweep_dir: Path) -> dict:
    p = sweep_dir / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_progress(sweep_dir: Path, progress: dict) -> None:
    (sweep_dir / "progress.json").write_text(
        json.dumps(progress, indent=2)
    )


# ---------------------------------------------------------------------------
# Comparison against baseline
# ---------------------------------------------------------------------------

def _find_baseline() -> Path | None:
    candidates = sorted(_RESULTS_DIR.glob("bertrand_sweep__*"), reverse=True)
    for c in candidates:
        seed_dir = c / "N2_rival_prices_q_learning" / "seed_000"
        if seed_dir.exists():
            return c
    return None


def _read_ci_from_sweep(sweep_dir: Path, combo: str, n_seeds: int) -> list[float]:
    import csv as _csv
    values = []
    for seed in range(n_seeds):
        hist = sweep_dir / combo / f"seed_{seed:03d}" / "metrics_history.csv"
        if not hist.exists():
            continue
        rows = list(_csv.DictReader(hist.open()))
        if rows:
            values.append(float(rows[-1]["collusion_index"]))
    return values


def _read_ci_from_replication(sweep_dir: Path, n_seeds: int) -> list[float]:
    import csv as _csv
    values = []
    for seed in range(n_seeds):
        hist = sweep_dir / f"seed_{seed:03d}" / "metrics_history.csv"
        if not hist.exists():
            continue
        rows = list(_csv.DictReader(hist.open()))
        if rows:
            values.append(float(rows[-1]["collusion_index"]))
    return values


def _write_comparison(sweep_dir: Path, new_ci: list[float],
                       baseline_dir: Path | None, n_seeds: int) -> None:
    import numpy as np

    lines = ["=" * 60]
    lines.append("Calvano Replication — CI Comparison")
    lines.append("=" * 60)
    lines.append(f"Target (Calvano paper): CI ∈ [0.60, 0.85]")
    lines.append("")

    if baseline_dir:
        old_ci = _read_ci_from_sweep(
            baseline_dir, "N2_rival_prices_q_learning", n_seeds
        )
        if old_ci:
            lines.append(f"Baseline sweep ({baseline_dir.name}):")
            lines.append(f"  seeds: {len(old_ci)}")
            lines.append(f"  CI mean:   {np.mean(old_ci):.4f}")
            lines.append(f"  CI median: {np.median(old_ci):.4f}")
            lines.append(f"  CI std:    {np.std(old_ci):.4f}")
            lines.append(f"  CI > 0.3:  {sum(v > 0.3 for v in old_ci)}/{len(old_ci)}")
            lines.append(f"  CI > 0.6:  {sum(v > 0.6 for v in old_ci)}/{len(old_ci)}")
            lines.append("")

    lines.append(f"Replication ({sweep_dir.name}):")
    lines.append(f"  seeds: {len(new_ci)}")
    if new_ci:
        lines.append(f"  CI mean:   {np.mean(new_ci):.4f}")
        lines.append(f"  CI median: {np.median(new_ci):.4f}")
        lines.append(f"  CI std:    {np.std(new_ci):.4f}")
        lines.append(f"  CI > 0.3:  {sum(v > 0.3 for v in new_ci)}/{len(new_ci)}")
        lines.append(f"  CI > 0.6:  {sum(v > 0.6 for v in new_ci)}/{len(new_ci)}")
        mean = np.mean(new_ci)
        ok = 0.60 <= mean <= 0.85
        lines.append(f"  Milestone 2B.2: {'OK ✓' if ok else 'FORA ✗'} (CI mean ∈ [0.60, 0.85])")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)
    (sweep_dir / "comparison.txt").write_text(text + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Calvano replication sweep")
    parser.add_argument("--n_seeds",   type=int, default=20)
    parser.add_argument("--seed_start",type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--resume",    type=str, default=None)
    parser.add_argument("--compare",   type=str, default=None)
    args = parser.parse_args()

    # Output directory
    if args.resume:
        sweep_dir = Path(args.resume) if Path(args.resume).is_absolute() \
                    else _RESULTS_DIR / args.resume
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _RESULTS_DIR / f"calvano_replication__{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(sweep_dir)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    pending = [s for s in seeds if str(s) not in progress or progress[str(s)] != "completed"]

    print(f"Sweep dir : {sweep_dir}")
    print(f"Seeds     : {args.n_seeds}  (pending: {len(pending)})")
    print(f"Workers   : {args.n_workers}")
    print(f"Steps     : {_TRAINING_CONFIG['total_steps']:,}")
    print(f"Epsilon   : geometric decay")
    print(f"Q-init    : opponent_optimized")
    print(f"Grid      : dynamic (grid_extend=1.0)")
    print(f"Early stop: disabled")
    print()

    if not pending:
        print("All seeds already completed — skipping training.")
    else:
        ctx = multiprocessing.get_context("spawn")
        t0  = time.time()

        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_run_seed, seed, sweep_dir): seed
                for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    row = future.result()
                    progress[str(seed)] = "completed"
                    _save_progress(sweep_dir, progress)
                    print(
                        f"  seed {seed:03d} | CI={row['ci']:.4f}  CE-gap={row['ce_gap']:.4f}"
                        f"  {row['dynamics']:25s}  steps={row['steps']:,}"
                    )
                except Exception as exc:
                    progress[str(seed)] = f"failed: {exc}"
                    _save_progress(sweep_dir, progress)
                    print(f"  seed {seed:03d} FAILED: {exc}")

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed/60:.1f} min")

    # Comparison
    print()
    if args.compare:
        baseline = Path(args.compare) if Path(args.compare).is_absolute() \
                   else _RESULTS_DIR / args.compare
    else:
        baseline = _find_baseline()

    new_ci = _read_ci_from_replication(sweep_dir, args.n_seeds)
    _write_comparison(sweep_dir, new_ci, baseline, args.n_seeds)


if __name__ == "__main__":
    main()
