"""
EXP-01: Cross-Deploy Experiment — Training Dynamics vs. Market Dynamics.

Tests whether collusion learned via co-training (Calvano setup) is dyadic
(calibrated to the training partner) or generalizes to any competent agent.

Three phases:
  1. Training  — 40 parallel runs (20 pairs × 2 environments):
                  pair n: envA(seed=n), envB(seed=n+20), both using Calvano Q-learning
  2. Cross-eval — deploy agent_0(envA_n) × agent_0(envB_n): never-met partners
  3. Within-eval — deploy agent_0(envA_n) × agent_1(envA_n): original training partners

Expected results:
  H1a: CI within  ≈ 0.62 (collusion maintained with original partner)
  H1b: CI cross   << 0.62 (collusion collapses with new partner)

Usage:
    python experiments/cross_deploy_experiment.py \\
        --config configs/experiments/cross_deploy_bertrand.yaml \\
        --n_pairs 20 --n_workers 4

    # Resume after interrupted training:
    python experiments/cross_deploy_experiment.py \\
        --config configs/experiments/cross_deploy_bertrand.yaml \\
        --resume results/cross_deploy__<timestamp>

Output:
    results/cross_deploy__<timestamp>/
      progress.json
      training/pair_000_A/  agents/agent_0.npz, agent_1.npz, metrics_history.csv ...
      training/pair_000_B/  ...
      eval/within_pair_000/  metrics_history.csv, summary.md
      eval/cross_pair_000/   metrics_history.csv, summary.md
      summary.csv
      summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np

from src.general_utils import config_loader
# NOTE: MultiAgentBertrandEnv and load_q_state import JAX at module level.
# Both are imported lazily inside main() so the parent process never triggers
# JAX initialisation — only the spawned worker processes do (same pattern as
# sweep_calvano_replication.py and sweep_phase2b.py).

_RESULTS_DIR = _PROJ_ROOT / "results"


# ---------------------------------------------------------------------------
# Training worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _train_one(task: dict) -> dict:
    """Train one (pair, env) job in a child process. Mirrors sweep_phase2b.py."""
    import sys
    import time
    from pathlib import Path

    proj_root = task["proj_root"]
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.training.multi_agent_trainer import run as trainer_run
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    from src.metrics.multi_agent import final_metrics

    seed_dir = Path(task["seed_dir"]).resolve()
    seed_dir.mkdir(parents=True, exist_ok=True)

    orig_out, orig_err = sys.stdout, sys.stderr
    log_f = open(seed_dir / "training.log", "w", buffering=1)
    sys.stdout = sys.stderr = log_f

    t0 = time.time()
    label = task["label"]
    try:
        trainer_run(
            env_config=task["env_cfg"],
            agent_config=task["agent_cfg"],
            training_config=task["training_cfg"],
            full_config=task.get("full_config"),
            output_dir=seed_dir,
        )
        elapsed = time.time() - t0

        bm = MultiAgentBertrandEnv(task["env_cfg"]).benchmarks()
        csv_path = seed_dir / "metrics_history.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path) as f:
                rows = list(_csv.DictReader(f))
            history = [{k: float(v) if v not in ("", "True", "False") else v
                        for k, v in r.items()} for r in rows]
        else:
            history = []

        if not history:
            return {"label": label, "status": "failed", "reason": "empty history",
                    "elapsed_s": round(elapsed, 1)}

        fm = final_metrics(
            history, task["eval_window"], task["eval_interval"], bm, task["n_firms"]
        )
        return {"label": label, "status": "completed",
                "elapsed_s": round(elapsed, 1), **fm}

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        return {"label": label, "status": "failed", "reason": str(exc),
                "traceback": traceback.format_exc(), "elapsed_s": round(elapsed, 1)}
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_f.close()


# ---------------------------------------------------------------------------
# Eval worker (sequential — fast)
# ---------------------------------------------------------------------------

def _eval_pair(
    kind: str,          # "within" or "cross"
    pair_idx: int,
    state_a: object,    # QLearningState for agent_0
    state_b: object,    # QLearningState for agent_1 (within) or agent_0 from envB (cross)
    env_cfg: dict,
    agent_cfg: dict,
    eval_cfg: dict,
    output_dir: Path,
) -> dict:
    """Run one eval episode and return CI metrics."""
    from src.training.multi_agent_trainer import eval_run

    result = eval_run(
        env_config=env_cfg,
        agent_config=agent_cfg,
        states=[state_a, state_b],
        eval_steps=int(eval_cfg["steps"]),
        eval_interval=int(eval_cfg["interval"]),
        seed=int(eval_cfg["seed"]),
        output_dir=output_dir,
    )
    result.pop("history", None)  # don't carry full history in summary
    result["kind"] = kind
    result["pair"] = pair_idx
    return result


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress(run_dir: Path) -> dict:
    p = run_dir / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_progress(run_dir: Path, progress: dict) -> None:
    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(run_dir: Path, training_results: dict, eval_results: list[dict]) -> None:
    rows = []
    within = {r["pair"]: r for r in eval_results if r["kind"] == "within"}
    cross   = {r["pair"]: r for r in eval_results if r["kind"] == "cross"}

    for n in sorted(within):
        tr_A = training_results.get(f"pair_{n:03d}_A", {})
        tr_B = training_results.get(f"pair_{n:03d}_B", {})
        rows.append({
            "pair":          f"{n:03d}",
            "ci_train_A":    round(tr_A.get("ci", float("nan")), 4),
            "ci_train_B":    round(tr_B.get("ci", float("nan")), 4),
            "ci_within":     round(within[n].get("ci", float("nan")), 4),
            "ci_cross":      round(cross.get(n, {}).get("ci", float("nan")), 4),
            "nash_within":   round(within[n].get("nash_dist", float("nan")), 4),
            "nash_cross":    round(cross.get(n, {}).get("nash_dist", float("nan")), 4),
        })

    csv_path = run_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        return

    ci_within_arr = np.array([r["ci_within"] for r in rows if not np.isnan(r["ci_within"])])
    ci_cross_arr  = np.array([r["ci_cross"]  for r in rows if not np.isnan(r["ci_cross"])])
    ci_trainA_arr = np.array([r["ci_train_A"] for r in rows if not np.isnan(r["ci_train_A"])])
    ci_trainB_arr = np.array([r["ci_train_B"] for r in rows if not np.isnan(r["ci_train_B"])])

    def fmt(arr):
        if len(arr) == 0:
            return "—"
        return f"{arr.mean():.4f} ±{arr.std():.4f} (n={len(arr)})"

    md = [
        "# EXP-01: Cross-Deploy Experiment — Results",
        "",
        "## Hypothesis",
        "- **H1a:** CI within-pair ≈ 0.62 — collusion persists with original training partner",
        "- **H1b:** CI cross-pair << 0.62 — collusion collapses with new partner",
        "- **Evidence for SCE:** if H1a holds and H1b holds, collusion is dyadic (training-path artifact)",
        "",
        "## Summary",
        "",
        "| Condition | Mean CI | std | n |",
        "|-----------|---------|-----|---|",
        f"| Training (Env A) | {fmt(ci_trainA_arr)} |",
        f"| Training (Env B) | {fmt(ci_trainB_arr)} |",
        f"| **Within-pair deploy** | {fmt(ci_within_arr)} |",
        f"| **Cross-pair deploy**  | {fmt(ci_cross_arr)}  |",
        "",
        "## Per-pair Results",
        "",
        "| Pair | CI train A | CI train B | CI within | CI cross | Nash within | Nash cross |",
        "|------|-----------|-----------|----------|---------|------------|-----------|",
    ]
    for r in rows:
        md.append(
            f"| {r['pair']} | {r['ci_train_A']} | {r['ci_train_B']} "
            f"| **{r['ci_within']}** | **{r['ci_cross']}** "
            f"| {r['nash_within']} | {r['nash_cross']} |"
        )

    md += [
        "",
        "## Interpretation",
        "",
        "If CI_cross << CI_within, the co-trained strategy is **not** Nash — it is a "
        "Self-Confirming Equilibrium calibrated to the specific training trajectory.",
        "The high price behavior does not transfer to a new partner, which would be "
        "expected if it were a genuinely learned market strategy.",
    ]

    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nSummary written to {run_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EXP-01: Cross-deploy experiment")
    parser.add_argument("--config", default="configs/experiments/cross_deploy_bertrand.yaml")
    parser.add_argument("--n_pairs",   type=int, default=20)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--resume",    type=str, default=None,
                        help="Resume from existing run dir")
    args = parser.parse_args()

    # Lazy imports — keep parent process JAX-free (CUDA/fork safety)
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    from src.agents.q_learning import load_state as load_q_state

    cfg = config_loader.load(args.config)
    env_cfg      = cfg["environment"]
    agent_cfg    = cfg["agent"]
    training_cfg = cfg["training"]
    eval_cfg     = cfg["eval"]
    n_firms      = int(env_cfg["market"]["n_firms"])

    # Sync agent grid bounds with env
    env = MultiAgentBertrandEnv(env_cfg)
    agent_cfg = dict(agent_cfg)
    agent_cfg["action_min"] = env.p_min
    agent_cfg["q_max"]      = env.p_max

    # Create run directory
    if args.resume:
        run_dir = Path(args.resume).resolve()
        assert run_dir.exists(), f"Resume dir not found: {run_dir}"
        print(f"Resuming from {run_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _RESULTS_DIR / f"cross_deploy__{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {run_dir}")

    train_dir = run_dir / "training"
    eval_dir  = run_dir / "eval"
    train_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    progress = _load_progress(run_dir)

    # ── Phase 1: Training ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Training {args.n_pairs * 2} jobs ({args.n_pairs} pairs × 2 envs)")
    print(f"  config: {args.config}")
    print(f"  n_workers: {args.n_workers}")
    print(f"{'='*60}\n")

    training_tasks = []
    for n in range(args.n_pairs):
        # B-seed offset is FIXED at 20 (docstring: envB(seed=n+20)). It was
        # previously `n + args.n_pairs`, which made seeds depend on the CLI
        # flag and produced duplicated agents across resumes with different
        # --n_pairs (pair_008_B == pair_000_B, both seed 20).
        for env_id, seed in [("A", n), ("B", n + 20)]:
            label = f"pair_{n:03d}_{env_id}"
            seed_dir = train_dir / label
            if progress.get(f"training.{label}") == "completed":
                print(f"  [skip] {label} already completed")
                continue
            tc = dict(training_cfg)
            tc["seed"] = seed
            training_tasks.append({
                "label":       label,
                "seed_dir":    str(seed_dir),
                "proj_root":   str(_PROJ_ROOT),
                "env_cfg":     env_cfg,
                "agent_cfg":   dict(agent_cfg),
                "training_cfg": tc,
                "full_config": {**cfg, "training": tc},
                "eval_window":  int(training_cfg.get("eval_window", 200_000)),
                "eval_interval": int(training_cfg.get("eval_interval", 10_000)),
                "n_firms":     n_firms,
            })

    if training_tasks:
        training_results_new = {}
        with ProcessPoolExecutor(
            max_workers=args.n_workers,
            mp_context=__import__("multiprocessing").get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_train_one, t): t["label"] for t in training_tasks}
            for fut in as_completed(futures):
                label = futures[fut]
                result = fut.result()
                status = result.get("status", "failed")
                ci = result.get("ci", float("nan"))
                elapsed = result.get("elapsed_s", 0)
                print(
                    f"  [{status}] {label:20s}  CI={ci:.4f}  ({elapsed:.0f}s)"
                )
                if status == "completed":
                    progress[f"training.{label}"] = "completed"
                    training_results_new[label] = result
                else:
                    print(f"    reason: {result.get('reason', '?')}")
                _save_progress(run_dir, progress)

        # persist training results
        tr_json = run_dir / "training_results.json"
        existing = json.loads(tr_json.read_text()) if tr_json.exists() else {}
        existing.update(training_results_new)
        tr_json.write_text(json.dumps(existing, indent=2))
    else:
        print("  All training jobs already completed.")

    # Load all training results; for pairs trained in a previous invocation
    # (training_results.json absent or incomplete — the cause of the nan
    # ci_train_A/B columns), fall back to the per-pair training summary.md.
    tr_json = run_dir / "training_results.json"
    training_results = json.loads(tr_json.read_text()) if tr_json.exists() else {}
    for n in range(args.n_pairs):
        for env_id in ("A", "B"):
            label = f"pair_{n:03d}_{env_id}"
            if label not in training_results:
                s = train_dir / label / "summary.md"
                ci = _read_metric_from_summary(s, "Collusion Index")
                if not np.isnan(ci):
                    training_results[label] = {"ci": ci}

    # ── Phase 2 & 3: Evaluation ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2/3: Evaluating {args.n_pairs} within-pair and {args.n_pairs} cross-pair episodes")
    print(f"  eval_steps={eval_cfg['steps']}, seed={eval_cfg['seed']}")
    print(f"{'='*60}\n")

    eval_results = []
    for n in range(args.n_pairs):
        agent_npz_A0 = train_dir / f"pair_{n:03d}_A" / "agents" / "agent_0.npz"
        agent_npz_A1 = train_dir / f"pair_{n:03d}_A" / "agents" / "agent_1.npz"
        agent_npz_B0 = train_dir / f"pair_{n:03d}_B" / "agents" / "agent_0.npz"

        if not (agent_npz_A0.exists() and agent_npz_A1.exists() and agent_npz_B0.exists()):
            missing = [p for p in [agent_npz_A0, agent_npz_A1, agent_npz_B0]
                       if not p.exists()]
            print(f"  [skip pair {n:03d}] missing agent files: {[str(m) for m in missing]}")
            continue

        state_A0 = load_q_state(agent_npz_A0)
        state_A1 = load_q_state(agent_npz_A1)
        state_B0 = load_q_state(agent_npz_B0)

        # Within-pair eval
        within_key = f"eval.within_{n:03d}"
        if progress.get(within_key) == "completed":
            print(f"  [skip] within_pair_{n:03d} already evaluated")
            # load from existing summary
            s = eval_dir / f"within_pair_{n:03d}" / "summary.md"
            eval_results.append({"kind": "within", "pair": n,
                                  "ci": _read_ci_from_summary(s),
                                  "nash_dist": _read_metric_from_summary(s, "Nash dist:")})
        else:
            print(f"  within_pair_{n:03d}: A₀ × A₁ (original partners)  ", end="", flush=True)
            t0 = time.time()
            r = _eval_pair("within", n, state_A0, state_A1,
                           env_cfg, agent_cfg, eval_cfg,
                           eval_dir / f"within_pair_{n:03d}")
            elapsed = time.time() - t0
            print(f"CI={r['ci']:.4f}  ({elapsed:.0f}s)")
            eval_results.append(r)
            progress[within_key] = "completed"
            _save_progress(run_dir, progress)

        # Cross-pair eval
        cross_key = f"eval.cross_{n:03d}"
        if progress.get(cross_key) == "completed":
            print(f"  [skip] cross_pair_{n:03d} already evaluated")
            s = eval_dir / f"cross_pair_{n:03d}" / "summary.md"
            eval_results.append({"kind": "cross", "pair": n,
                                  "ci": _read_ci_from_summary(s),
                                  "nash_dist": _read_metric_from_summary(s, "Nash dist:")})
        else:
            print(f"  cross_pair_{n:03d}:  A₀ × B₀ (never-met partners)  ", end="", flush=True)
            t0 = time.time()
            r = _eval_pair("cross", n, state_A0, state_B0,
                           env_cfg, agent_cfg, eval_cfg,
                           eval_dir / f"cross_pair_{n:03d}")
            elapsed = time.time() - t0
            print(f"CI={r['ci']:.4f}  ({elapsed:.0f}s)")
            eval_results.append(r)
            progress[cross_key] = "completed"
            _save_progress(run_dir, progress)

    # ── Summary ───────────────────────────────────────────────────────────────
    if eval_results:
        _write_summary(run_dir, training_results, eval_results)

    print(f"\n{'='*60}")
    print(f"EXP-01 complete. Results in {run_dir}")
    print(f"{'='*60}\n")


def _read_metric_from_summary(path: Path, key: str) -> float:
    """Extract the first float after `key` in a summary.md line.

    Works for both eval summaries ('- **CI:** 0.0714') and training
    summaries ('| Collusion Index | 0.6282 | ...'). The old split(":")
    parser choked on the bold markers and returned nan for every line.
    """
    try:
        for line in path.read_text().splitlines():
            if key in line:
                m = re.search(rf"{re.escape(key)}[^\d\-+]*([-+]?\d+\.?\d*)", line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return float("nan")


def _read_ci_from_summary(path: Path) -> float:
    return _read_metric_from_summary(path, "CI:")


if __name__ == "__main__":
    main()
