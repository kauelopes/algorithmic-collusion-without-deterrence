"""
EXP-02 (vectorized): Grid granularity sensitivity via vectorized_q_trainer.

Same experiment as sweep_grid_sensitivity.py (same _BASE_ENV/_agent_config),
but each n_prices level runs its K seeds simultaneously on GPU/CPU with
jax.lax.scan + vmap, instead of K sequential trainer processes.

Differences vs the sequential sweep (documented for the paper):
  - CE-gap is not computed (vectorized trainer limitation) → NaN in outputs.
  - Seeding: one PRNGKey drives K independent streams, instead of one
    training.seed per process. Monte Carlo semantics are equivalent.
  - No per-step metrics history; only the final eval_window aggregate.

Output layout mirrors the sequential sweep so analysis scripts work:
  results/grid_sensitivity_vectorized__TS/
    n_prices_<k>/vectorized_results.csv + q_tables.npy
    aggregate/summary_table.csv + all_seeds.csv

Usage:
  python experiments/sweep_grid_sensitivity_vectorized.py \
      [--levels 5,15,50,100,200] [--k 20] [--seed 0] [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from experiments.sweep_grid_sensitivity import (  # noqa: E402
    _BASE_ENV,
    _BASE_TRAINING,
    _agent_config,
    _combo_key,
)
from src.training.vectorized_q_trainer import run_vectorized  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EXP-02 vectorized grid sensitivity")
    p.add_argument("--levels", type=str, default="5,15,50,100,200")
    p.add_argument("--k", type=int, default=20, help="Parallel seeds per level")
    p.add_argument("--seed", type=int, default=0, help="Base PRNG seed")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else (
        _PROJ_ROOT / "results" / f"grid_sensitivity_vectorized__{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    training = dict(_BASE_TRAINING)
    training["seed"] = args.seed

    all_rows: list[dict] = []
    for n_prices in levels:
        combo = _combo_key(n_prices)
        print(f"\n=== {combo} — K={args.k} seeds, {training['total_steps']:,} steps ===")
        results = run_vectorized(
            env_config=dict(_BASE_ENV),
            agent_config=_agent_config(n_prices),
            training_config=training,
            K=args.k,
            output_dir=out_root / combo,
        )
        for r in results:
            all_rows.append({
                "combo":     combo,
                "n_prices":  n_prices,
                "seed":      r["seed_idx"],
                "ci":        round(r["ci"], 4),
                "nash_dist": round(r["nash_dist"], 4),
                "ce_gap":    r["ce_gap"],
                "status":    "completed",
            })
        _write_aggregate(out_root / "aggregate", all_rows, levels)

    print(f"\nDone. Output: {out_root}")


def _write_aggregate(agg_dir: Path, rows: list[dict], levels: list[int]) -> None:
    agg_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for n_prices in sorted(levels):
        group = [r for r in rows if r["n_prices"] == n_prices]
        if not group:
            continue
        ci_vals = [r["ci"] for r in group]
        summary_rows.append({
            "n_prices":       n_prices,
            "n_states":       n_prices ** 2,
            "n_seeds":        len(group),
            "ci_mean":        round(float(np.mean(ci_vals)), 4),
            "ci_std":         round(float(np.std(ci_vals)), 4),
            "ci_median":      round(float(np.median(ci_vals)), 4),
            "pct_colluding":  round(sum(1 for c in ci_vals if c > 0.5) / len(ci_vals) * 100, 1),
            "ce_gap_mean":    float("nan"),
            "nash_dist_mean": round(float(np.mean([r["nash_dist"] for r in group])), 4),
        })

    with open(agg_dir / "summary_table.csv", "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    with open(agg_dir / "all_seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
