"""
EXP-02 — Grid Granularity Sensitivity Sweep.

Hypothesis: Finer price grids → larger state space → less off-path exploration
→ less collusion. CI should decline monotonically with n_prices.

Protocol (Klein 2021, RAND):
  - Fixed: N=2, rival_prices, Q-learning, all Calvano params
  - Vary: n_prices ∈ [5, 15, 50, 100, 200]
  - 20 seeds per level  →  100 runs total

State space per level (n_actions^2 with memory_length=2):
  n=5:   25 states,  5 actions  (very coarse)
  n=15:  225 states, 15 actions (Calvano baseline)
  n=50: 2500 states, 50 actions
  n=100: 10000 states, 100 actions
  n=200: 40000 states, 200 actions

Output structure:
  results/grid_sensitivity__<timestamp>/
    progress.json
    n_prices_5/seed_000/ ... seed_019/
    n_prices_15/...
    ...
    aggregate/
      summary_table.csv   ← n_prices | ci_mean | ci_std | pct_colluding | ...
"""

from __future__ import annotations

import argparse
import csv
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

from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
from src.metrics.multi_agent import final_metrics

_RESULTS_DIR = _PROJ_ROOT / "results"

# ---------------------------------------------------------------------------
# Grid levels to sweep
# ---------------------------------------------------------------------------

_N_PRICES_LEVELS = [5, 15, 50, 100, 200]

_BASE_ENV = {
    "type":   "bertrand",
    "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
    "cost":   {"c": 1.0},
    "market": {"n_firms": 2, "grid_extend": 1.0},
    "state":  {"info_structure": "rival_prices"},
}

_BASE_TRAINING = {
    "total_steps":   2_000_000,
    "eval_interval":    10_000,
    "eval_window":     200_000,
    "early_stop":      {"enabled": False},   # keep consistent with Calvano replication
}


def _agent_config(n_prices: int) -> dict:
    return {
        "type":                "q_learning",
        "n_actions":           n_prices,
        "memory_length":       2,          # N=2, rival_prices → state = (p_A, p_B)
        "q_init":              "opponent_optimized",
        "alpha":               0.15,
        "gamma":               0.99,
        "epsilon_start":       1.0,
        "epsilon_end":         0.01,
        "epsilon_decay_steps": 1_150_000,
        "epsilon_decay_type":  "geometric",
    }


def _combo_key(n_prices: int) -> str:
    return f"n_prices_{n_prices}"


# ---------------------------------------------------------------------------
# Helpers (copied from sweep_phase2b to avoid import coupling)
# ---------------------------------------------------------------------------

def _load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _mark_progress(progress: dict, path: Path, combo: str, seed: str, status: str) -> None:
    progress.setdefault(combo, {})[seed] = status
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def _load_seed_history(seed_dir: Path) -> list[dict]:
    csv_path = seed_dir / "metrics_history.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _row_from_disk(seed_dir: Path, combo: str, n_prices: int, seed: int,
                   bm: dict, tc: dict) -> dict | None:
    history = _load_seed_history(seed_dir)
    if not history:
        return None
    try:
        fm = final_metrics(history, tc["eval_window"], tc["eval_interval"], bm, n=2)
        return {
            "combo": combo, "n_prices": n_prices, "seed": seed,
            "status": "completed", "elapsed_s": 0.0, **fm
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Worker (top-level so it's picklable)
# ---------------------------------------------------------------------------

def _run_one_seed(task: dict) -> dict:
    import sys, time
    from pathlib import Path

    proj_root = task["proj_root"]
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.training.multi_agent_trainer import run as trainer_run
    from src.metrics.multi_agent import final_metrics
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv

    seed_dir = Path(task["seed_dir"])
    seed_dir.mkdir(parents=True, exist_ok=True)

    meta = {"combo": task["combo"], "n_prices": task["n_prices"], "seed": task["seed"]}

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_f = open(seed_dir / "training.log", "w", buffering=1)
    sys.stdout = sys.stderr = log_f

    t0 = time.time()
    try:
        trainer_run(
            env_config=task["env_cfg"],
            agent_config=task["agent_cfg"],
            training_config=task["tc"],
            full_config=task["full_config"],
            output_dir=seed_dir,
        )
        elapsed = time.time() - t0

        bm = MultiAgentBertrandEnv(task["env_cfg"]).benchmarks()
        history = _load_seed_history(seed_dir)
        if not history:
            return {**meta, "status": "failed", "reason": "empty history",
                    "elapsed_s": round(elapsed, 1)}

        fm = final_metrics(
            history, task["eval_window"], task["eval_interval"], bm, n=2
        )
        return {**meta, "status": "completed", "elapsed_s": round(elapsed, 1), **fm}

    except Exception as exc:
        return {**meta, "status": "failed", "reason": str(exc),
                "elapsed_s": round(elapsed, 1)}

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_f.close()


# ---------------------------------------------------------------------------
# Aggregate output
# ---------------------------------------------------------------------------

def _write_summary_csv(agg_dir: Path, rows: list[dict]) -> None:
    from itertools import groupby
    import numpy as np

    # Group by n_prices
    summary_rows = []
    completed = [r for r in rows if r.get("status") == "completed"]
    for n_prices in sorted(_N_PRICES_LEVELS):
        group = [r for r in completed if r.get("n_prices") == n_prices]
        if not group:
            continue
        ci_vals = [r["ci"] for r in group]
        summary_rows.append({
            "n_prices":        n_prices,
            "n_states":        n_prices ** 2,
            "n_seeds":         len(group),
            "ci_mean":         round(float(np.mean(ci_vals)), 4),
            "ci_std":          round(float(np.std(ci_vals)), 4),
            "ci_median":       round(float(np.median(ci_vals)), 4),
            "pct_colluding":   round(sum(1 for c in ci_vals if c > 0.5) / len(ci_vals) * 100, 1),
            "ce_gap_mean":     round(float(np.mean([r.get("ce_gap", 0) for r in group])), 4),
            "nash_dist_mean":  round(float(np.mean([r.get("nash_dist", 0) for r in group])), 4),
        })

    with open(agg_dir / "summary_table.csv", "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Full seed-level CSV
    all_cols = sorted({k for r in completed for k in r.keys()})
    with open(agg_dir / "all_seeds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)

    print(f"\nSummary by n_prices:")
    print(f"{'n_prices':>10} {'n_states':>10} {'ci_mean':>9} {'ci_std':>8} "
          f"{'pct_coll%':>10} {'n_seeds':>8}")
    for r in summary_rows:
        print(f"{r['n_prices']:>10} {r['n_states']:>10} {r['ci_mean']:>9.4f} "
              f"{r['ci_std']:>8.4f} {r['pct_colluding']:>10.1f} {r['n_seeds']:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-02: Grid granularity sensitivity sweep")
    p.add_argument("--n_seeds",   type=int, default=20, help="Seeds per n_prices level")
    p.add_argument("--seed_start",type=int, default=0)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--steps",     type=int, default=None, help="Override total_steps")
    p.add_argument("--levels",    type=str, default=None,
                   help="Comma-separated n_prices levels, e.g. 5,15,50")
    p.add_argument("--resume",    type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume:
        sweep_dir = Path(args.resume).resolve()
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Resume path not found: {sweep_dir}")
        print(f"Resuming from: {sweep_dir}")
    else:
        sweep_dir = _RESULTS_DIR / f"grid_sensitivity__{timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

    progress_path = sweep_dir / "progress.json"
    progress      = _load_progress(progress_path)

    seeds        = list(range(args.seed_start, args.seed_start + args.n_seeds))
    total_steps  = args.steps or _BASE_TRAINING["total_steps"]
    n_prices_levels = (
        [int(x) for x in args.levels.split(",")]
        if args.levels else _N_PRICES_LEVELS
    )

    # Build flat task list
    pending:      list[dict] = []
    skipped_rows: list[dict] = []

    for n_prices in n_prices_levels:
        key       = _combo_key(n_prices)
        env_cfg   = dict(_BASE_ENV)
        agent_cfg = _agent_config(n_prices)
        train_cfg = {**_BASE_TRAINING, "total_steps": total_steps}
        bm        = MultiAgentBertrandEnv(env_cfg).benchmarks()
        combo_dir = sweep_dir / key
        combo_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            seed_dir = combo_dir / f"seed_{seed:03d}"
            if progress.get(key, {}).get(str(seed)) == "completed":
                row = _row_from_disk(seed_dir, key, n_prices, seed, bm, train_cfg)
                if row is not None:
                    skipped_rows.append(row)
                continue

            tc = {**train_cfg, "seed": seed}
            pending.append({
                "proj_root":     str(_PROJ_ROOT),
                "combo":         key,
                "n_prices":      n_prices,
                "seed":          seed,
                "seed_dir":      str(seed_dir),
                "env_cfg":       env_cfg,
                "agent_cfg":     agent_cfg,
                "tc":            tc,
                "full_config":   {
                    "experiment_name": f"grid_sensitivity_{key}",
                    "environment":     env_cfg,
                    "agent":           agent_cfg,
                    "training":        tc,
                },
                "eval_window":   train_cfg["eval_window"],
                "eval_interval": train_cfg["eval_interval"],
                "bm":            bm,
            })

    total_runs = len(n_prices_levels) * len(seeds)
    print(f"\n{'='*70}")
    print(f"EXP-02: Grid Granularity Sensitivity Sweep")
    print(f"  Levels  : {n_prices_levels}")
    print(f"  Grid    : {len(n_prices_levels)} levels × {len(seeds)} seeds = {total_runs} total")
    print(f"  Skipped : {len(skipped_rows)} already completed")
    print(f"  Pending : {len(pending)}")
    print(f"  Workers : {args.n_workers}")
    print(f"  Steps   : {total_steps:,} per run")
    print(f"  Output  : {sweep_dir.relative_to(_PROJ_ROOT)}/")
    print(f"{'='*70}\n")

    all_rows: list[dict] = list(skipped_rows)
    done_count = len(skipped_rows)

    if pending:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp_ctx) as executor:
            future_to_task = {executor.submit(_run_one_seed, t): t for t in pending}

            for future in as_completed(future_to_task):
                task       = future_to_task[future]
                done_count += 1

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "combo": task["combo"], "n_prices": task["n_prices"],
                        "seed": task["seed"], "status": "failed",
                        "reason": str(exc), "elapsed_s": 0.0,
                    }

                status = result.get("status", "failed")
                _mark_progress(progress, progress_path,
                               result["combo"], str(result["seed"]), status)

                elapsed = result.get("elapsed_s", 0.0)
                tag = (f"[{done_count:03d}/{total_runs}] "
                       f"n_prices={result['n_prices']} seed={result['seed']}")

                if status == "completed":
                    all_rows.append(result)
                    print(
                        f"  {tag}  "
                        f"CI={result['ci']:+.3f}  "
                        f"CE-gap={result['ce_gap']:.4f}  "
                        f"dyn={result['dynamics_label']:<22}  [{elapsed:.0f}s]",
                        flush=True,
                    )
                else:
                    print(
                        f"  {tag}  FAILED: {result.get('reason', '?')}  [{elapsed:.0f}s]",
                        flush=True,
                    )

    # Aggregate
    agg_dir = sweep_dir / "aggregate"
    agg_dir.mkdir(exist_ok=True)
    if all_rows:
        _write_summary_csv(agg_dir, all_rows)

    print(f"\nGrid sensitivity sweep complete → {sweep_dir.relative_to(_PROJ_ROOT)}/")


if __name__ == "__main__":
    main()
