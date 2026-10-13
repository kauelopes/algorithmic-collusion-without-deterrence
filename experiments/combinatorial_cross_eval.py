"""
Combinatorial Cross-Evaluation — EXP-01 extension.

Takes all agent_0 files from a completed (or partial) cross_deploy training
directory and evaluates every cross-env pair: any two agents that were NOT
trained together.

Sampling: when --max_evals is given, randomly sample that many combinations
(reproducible via --sample_seed). This lets you cap wall-clock time.

Rule of thumb: ~30s per eval on 4 workers → 240 evals/2h per worker batch.
With 4 workers: ~960 evals in 2h. Default max_evals=240 keeps a safe margin.

Usage:
    python experiments/combinatorial_cross_eval.py \\
        --training_dir results/cross_deploy__20260621_203040/training \\
        --config configs/experiments/cross_deploy_bertrand.yaml \\
        --max_evals 240 \\
        --n_workers 4

    # Run all combinations (760 for 40 agents):
    python experiments/combinatorial_cross_eval.py \\
        --training_dir results/cross_deploy__20260621_203040/training \\
        --config configs/experiments/cross_deploy_bertrand.yaml \\
        --n_workers 4

    # Resume interrupted run:
    python experiments/combinatorial_cross_eval.py \\
        --resume results/combinatorial_cross_eval__<timestamp>

Output:
    results/combinatorial_cross_eval__<timestamp>/
        progress.json
        eval/<envA>_x_<envB>/  metrics_history.csv, summary.md
        summary.csv
        summary.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from src.general_utils import config_loader

_RESULTS_DIR = _PROJ_ROOT / "results"


# ---------------------------------------------------------------------------
# Agent discovery
# ---------------------------------------------------------------------------

def _discover_agents(training_dir: Path) -> list[dict]:
    """
    Scan training_dir for completed runs (those with an agents/ subdir).
    Returns a list of dicts: {env_label, agent_idx, path}.
    Only agent_0 is included (one representative per training env).
    """
    agents = []
    for env_dir in sorted(training_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        npz = env_dir / "agents" / "agent_0.npz"
        if npz.exists():
            agents.append({
                "env_label": env_dir.name,   # e.g. "pair_003_A"
                "agent_idx": 0,
                "path": str(npz),
            })
    return agents


def _build_combinations(agents: list[dict], max_evals: int | None, sample_seed: int) -> list[tuple[dict, dict]]:
    """
    All C(n,2) pairs where env_label differs.
    If max_evals < total, sample randomly (reproducibly via sample_seed).
    """
    all_pairs = [
        (a, b)
        for a, b in itertools.combinations(agents, 2)
        if a["env_label"] != b["env_label"]
    ]
    if max_evals is not None and max_evals < len(all_pairs):
        rng = random.Random(sample_seed)
        all_pairs = rng.sample(all_pairs, max_evals)
    return all_pairs


# ---------------------------------------------------------------------------
# Eval worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _eval_one(task: dict) -> dict:
    import sys
    from pathlib import Path

    proj_root = task["proj_root"]
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.training.multi_agent_trainer import eval_run
    from src.agents.q_learning import load_state as load_q_state

    out_dir = Path(task["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        state_a = load_q_state(task["path_a"])
        state_b = load_q_state(task["path_b"])

        result = eval_run(
            env_config=task["env_cfg"],
            agent_config=task["agent_cfg"],
            states=[state_a, state_b],
            eval_steps=int(task["eval_steps"]),
            eval_interval=int(task["eval_interval"]),
            seed=int(task["eval_seed"]),
            output_dir=out_dir,
        )
        result.pop("history", None)
        return {
            "label":     task["label"],
            "env_a":     task["env_a"],
            "env_b":     task["env_b"],
            "status":    "completed",
            "ci":        result.get("ci", float("nan")),
            "nash_dist": result.get("nash_dist", float("nan")),
            "ce_gap":    result.get("ce_gap", float("nan")),
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as exc:
        import traceback
        return {
            "label":     task["label"],
            "env_a":     task["env_a"],
            "env_b":     task["env_b"],
            "status":    "failed",
            "reason":    str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.time() - t0, 1),
            "ci":        float("nan"),
            "nash_dist": float("nan"),
            "ce_gap":    float("nan"),
        }


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress(run_dir: Path) -> dict:
    p = run_dir / "progress.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_progress(run_dir: Path, progress: dict) -> None:
    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(run_dir: Path, results: list[dict]) -> None:
    import numpy as np

    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        return

    csv_path = run_dir / "summary.csv"
    fieldnames = ["env_a", "env_b", "ci", "nash_dist", "ce_gap", "elapsed_s", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(completed, key=lambda r: (r["env_a"], r["env_b"])))

    ci_arr = [r["ci"] for r in completed if not (r["ci"] != r["ci"])]  # filter NaN
    ci_np  = np.array(ci_arr) if ci_arr else np.array([])

    def fmt(arr):
        if len(arr) == 0:
            return "—"
        return f"{arr.mean():.4f} ±{arr.std():.4f}  (n={len(arr)})"

    pct_above_half = (ci_np > 0.5).mean() * 100 if len(ci_np) > 0 else float("nan")
    pct_below_zero = (ci_np < 0.0).mean() * 100 if len(ci_np) > 0 else float("nan")

    md = [
        "# Combinatorial Cross-Evaluation — Results",
        "",
        "Deploys agent_0 from every training env against agent_0 from every other env.",
        "Agents were never in the same training environment.",
        "",
        "## Summary",
        "",
        f"- Evaluations completed: **{len(completed)}**",
        f"- CI cross-env: **{fmt(ci_np)}**",
        f"- % pairs with CI > 0.5 (strong collusion): **{pct_above_half:.1f}%**",
        f"- % pairs with CI < 0.0 (below Nash):        **{pct_below_zero:.1f}%**",
        "",
        "## Interpretation",
        "",
        "If CI_cross_env << CI_within_pair (≈ 0.62 from EXP-01 baseline),",
        "the collusion learned during co-training does not generalize to new partners.",
        "This is consistent with a Self-Confirming Equilibrium (SCE) interpretation,",
        "not a Nash Equilibrium strategy that is robust to partner identity.",
        "",
        "## CI Distribution",
        "",
        f"| Range         | Count | % |",
        f"|---------------|-------|---|",
    ]
    if len(ci_np) > 0:
        bins = [
            ("< 0.0  (below Nash)", ci_np < 0.0),
            ("0.0 – 0.3",           (ci_np >= 0.0) & (ci_np < 0.3)),
            ("0.3 – 0.5",           (ci_np >= 0.3) & (ci_np < 0.5)),
            ("0.5 – 0.7",           (ci_np >= 0.5) & (ci_np < 0.7)),
            ("> 0.7  (strong)",     ci_np >= 0.7),
        ]
        for label, mask in bins:
            n = mask.sum()
            pct = n / len(ci_np) * 100
            md.append(f"| {label:20s} | {n:5d} | {pct:.1f}% |")

    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nSummary written to {run_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combinatorial cross-evaluation of all trained agent_0 pairs"
    )
    parser.add_argument("--training_dir", type=str, default=None,
                        help="Path to training/ subdir of a cross_deploy run")
    parser.add_argument("--config", type=str,
                        default="configs/experiments/cross_deploy_bertrand.yaml")
    parser.add_argument("--max_evals", type=int, default=None,
                        help="Cap number of eval pairs (random sample). "
                             "None = run all C(n,2) combinations. "
                             "~240 fits in ~2h with 4 workers.")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="RNG seed for sampling when --max_evals is set")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing run directory")
    args = parser.parse_args()

    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv

    cfg = config_loader.load(args.config)
    env_cfg   = cfg["environment"]
    agent_cfg = dict(cfg["agent"])
    eval_cfg  = cfg["eval"]

    env = MultiAgentBertrandEnv(env_cfg)
    agent_cfg["action_min"] = env.p_min
    agent_cfg["q_max"]      = env.p_max

    # Resolve run directory
    if args.resume:
        run_dir = Path(args.resume).resolve()
        assert run_dir.exists(), f"Resume dir not found: {run_dir}"
        # Load training_dir from metadata if not provided
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            training_dir = Path(meta["training_dir"])
        else:
            assert args.training_dir, "--training_dir required when resuming without meta.json"
            training_dir = Path(args.training_dir).resolve()
        print(f"Resuming from {run_dir}")
    else:
        assert args.training_dir, "--training_dir is required"
        training_dir = Path(args.training_dir).resolve()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _RESULTS_DIR / f"combinatorial_cross_eval__{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    # Save metadata for resume
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "training_dir": str(training_dir),
            "config":       args.config,
            "max_evals":    args.max_evals,
            "sample_seed":  args.sample_seed,
        }, indent=2))

    progress = _load_progress(run_dir)

    # Discover agents
    agents = _discover_agents(training_dir)
    if not agents:
        print(f"No completed agents found in {training_dir}")
        sys.exit(1)

    print(f"\nFound {len(agents)} trained agent_0 files in {training_dir}")

    # Build combinations
    pairs = _build_combinations(agents, args.max_evals, args.sample_seed)
    total = len(pairs)
    total_possible = len(agents) * (len(agents) - 1) // 2

    print(f"Total possible cross-env combinations: {total_possible}")
    if args.max_evals is not None and args.max_evals < total_possible:
        print(f"Sampling {total} combinations (--max_evals={args.max_evals}, seed={args.sample_seed})")
    else:
        print(f"Running all {total} combinations")
    print(f"n_workers: {args.n_workers}")

    # Build tasks (skip already done)
    tasks = []
    for a, b in pairs:
        label = f"{a['env_label']}_x_{b['env_label']}"
        if progress.get(f"eval.{label}") == "completed":
            continue
        tasks.append({
            "label":        label,
            "env_a":        a["env_label"],
            "env_b":        b["env_label"],
            "path_a":       a["path"],
            "path_b":       b["path"],
            "output_dir":   str(eval_dir / label),
            "proj_root":    str(_PROJ_ROOT),
            "env_cfg":      env_cfg,
            "agent_cfg":    agent_cfg,
            "eval_steps":   int(eval_cfg["steps"]),
            "eval_interval": int(eval_cfg["interval"]),
            "eval_seed":    int(eval_cfg["seed"]),
        })

    already_done = total - len(tasks)
    print(f"\n  {already_done} already completed, {len(tasks)} to run")
    print(f"{'='*60}\n")

    results = []

    # Load already-done results from progress
    for a, b in pairs:
        label = f"{a['env_label']}_x_{b['env_label']}"
        if progress.get(f"eval.{label}") == "completed":
            results.append({
                "label":     label,
                "env_a":     a["env_label"],
                "env_b":     b["env_label"],
                "status":    "completed",
                "ci":        progress.get(f"ci.{label}", float("nan")),
                "nash_dist": float("nan"),
                "ce_gap":    float("nan"),
                "elapsed_s": 0,
            })

    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.n_workers,
            mp_context=__import__("multiprocessing").get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_eval_one, t): t["label"] for t in tasks}
            done_count = already_done
            for fut in as_completed(futures):
                r = fut.result()
                done_count += 1
                status = r["status"]
                ci     = r.get("ci", float("nan"))
                elapsed = r.get("elapsed_s", 0)
                pct    = done_count / total * 100
                print(
                    f"  [{done_count:4d}/{total}] {r['label']:45s}  "
                    f"CI={ci:.4f}  ({elapsed:.0f}s)  [{pct:.0f}%]"
                )
                if status == "completed":
                    progress[f"eval.{r['label']}"]  = "completed"
                    progress[f"ci.{r['label']}"]    = ci
                else:
                    print(f"    FAILED: {r.get('reason', '?')}")
                results.append(r)
                _save_progress(run_dir, progress)

    _write_summary(run_dir, results)

    import numpy as np
    ci_vals = np.array([r["ci"] for r in results
                        if r["status"] == "completed" and r["ci"] == r["ci"]])
    print(f"\n{'='*60}")
    print(f"Done. {len(ci_vals)} evals completed.")
    if len(ci_vals) > 0:
        print(f"CI cross-env: {ci_vals.mean():.4f} ±{ci_vals.std():.4f}")
    print(f"Results in {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
