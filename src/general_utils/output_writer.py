"""
Writes all output files for a completed experiment run.

Output folder: results/{experiment_name}__{seed}__{YYYYMMDD_HHMMSS}/
  config.yaml
  summary.md
  metrics_history.csv
  final_metrics.csv
  tables/benchmarks.md
  tables/convergence.md
  tables/welfare.md
  trajectories/quantity_stats.csv
  trajectories/reward_stats.csv
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.equilibrium import collusion_index, convergence_step, dynamics_label
from src.metrics.welfare import consumer_surplus, producer_surplus, welfare_table

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR  = _PROJECT_ROOT / "results"


def write(config: dict, history: list[dict]) -> Path:
    """
    Write the full output folder for a completed run.

    Parameters
    ----------
    config  : fully-resolved experiment config (from config_loader.load)
    history : list of per-eval-window dicts collected by the trainer
              Each dict must have: step, window_mean_q, window_std_q,
              window_mean_reward, window_mean_price, nash_dist,
              collusion_index, epsilon, greedy_q, greedy_dist

    Returns
    -------
    Path to the created output folder.
    """
    seed      = config["training"]["seed"]
    exp_name  = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder    = _RESULTS_DIR / f"{exp_name}__{seed}__{timestamp}"

    (folder / "tables").mkdir(parents=True, exist_ok=True)
    (folder / "trajectories").mkdir(parents=True, exist_ok=True)
    (folder / "plots").mkdir(parents=True, exist_ok=True)

    benchmarks = _get_benchmarks(config)
    final      = _final_window(config, history, benchmarks)

    _write_config(folder, config)
    _write_metrics_history(folder, history)
    _write_trajectories(folder, history)
    _write_final_metrics(folder, config, final, benchmarks)
    _write_table_benchmarks(folder, benchmarks)
    _write_table_convergence(folder, final)
    _write_table_welfare(folder, config, final, benchmarks)
    _write_plots(folder, history, benchmarks, config)
    _write_summary(folder, config, final, benchmarks, history)
    _append_experiments_log(config, folder, final)

    return folder


# ---------------------------------------------------------------------------
# Helpers — data extraction
# ---------------------------------------------------------------------------

def _get_benchmarks(config: dict) -> dict:
    from src.econ_utils.benchmarks import nash_symmetric, monopoly, competitive
    e = config["environment"]
    a, b, c, n = e["demand"]["a"], e["demand"]["b"], e["cost"]["c"], e["market"]["n_firms"]
    ne   = nash_symmetric(a, b, c, n)
    mono = monopoly(a, b, c)
    comp = competitive(a, b, c)
    return {
        "q_NE":        ne["q_per_firm"],
        "Q_NE":        ne["Q_total"],
        "p_NE":        ne["price"],
        "pi_NE":       ne["profit_per_firm"],
        "Q_monopoly":  mono["Q_total"],
        "p_monopoly":  mono["price"],
        "pi_monopoly": mono["profit_total"],
        "Q_competitive": comp["Q_total"],
        "p_competitive": comp["price"],
        "a": a, "b": b, "c": c, "n": n,
    }


def _final_window(config: dict, history: list[dict], benchmarks: dict) -> dict:
    """Compute convergence metrics over the last eval_window steps."""
    eval_window   = config["training"].get("eval_window", 10000)
    eval_interval = config["training"]["eval_interval"]
    n_tail        = max(1, eval_window // eval_interval)
    tail          = history[-n_tail:]

    mean_q   = float(np.mean([r["window_mean_q"]     for r in tail]))
    std_q    = float(np.mean([r["window_std_q"]      for r in tail]))
    mean_r   = float(np.mean([r["window_mean_reward"] for r in tail]))
    mean_p   = float(np.mean([r["window_mean_price"]  for r in tail]))

    q_NE  = benchmarks["q_NE"]
    Q_NE  = benchmarks["Q_NE"]
    Q_M   = benchmarks["Q_monopoly"]
    n     = benchmarks["n"]
    Q_rl  = mean_q * n + (n - 1) * q_NE  # learner Q + fixed opponents

    # Use learner Q relative to single-firm benchmarks
    Q_rl_total = mean_q + (n - 1) * q_NE
    ci = collusion_index(Q_rl_total, Q_NE, Q_M)

    q_series = np.array([r["window_mean_q"]  for r in history])
    nd_series = np.array([r["nash_dist"]      for r in history])
    steps     = np.array([r["step"]           for r in history])
    threshold = 0.05 * q_NE

    a, b, c = benchmarks["a"], benchmarks["b"], benchmarks["c"]
    price = max(0.0, a - b * Q_rl_total)
    cs = consumer_surplus(Q_rl_total, a, b)
    ps = producer_surplus([mean_q] + [q_NE] * (n - 1), price, c)

    return {
        "final_mean_q":          round(mean_q, 6),
        "final_std_q":           round(std_q, 6),
        "final_mean_reward":     round(mean_r, 6),
        "final_mean_price":      round(mean_p, 6),
        "nash_dist_final":       round(abs(mean_q - q_NE), 6),
        "greedy_q":              history[-1].get("greedy_q", float("nan")),
        "greedy_dist":           history[-1].get("greedy_dist", float("nan")),
        "collusion_index_final": round(ci, 6),
        "consumer_surplus":      round(cs, 6),
        "producer_surplus":      round(ps, 6),
        "total_welfare":         round(cs + ps, 6),
        "dynamics_label":        dynamics_label(q_series, q_NE),
        "converged":             bool(abs(mean_q - q_NE) < threshold),
        "convergence_step":      convergence_step(nd_series, steps, threshold),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_config(folder: Path, config: dict) -> None:
    with open(folder / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _write_metrics_history(folder: Path, history: list[dict]) -> None:
    if not history:
        return
    with open(folder / "metrics_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _write_trajectories(folder: Path, history: list[dict]) -> None:
    q_fields = ["step", "window_mean_q", "window_std_q",
                 "window_min_q", "window_max_q"]
    r_fields = ["step", "window_mean_reward", "window_std_reward"]

    with open(folder / "trajectories" / "quantity_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=q_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    with open(folder / "trajectories" / "reward_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=r_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _write_final_metrics(
    folder: Path, config: dict, final: dict, benchmarks: dict
) -> None:
    row = {
        "experiment_name":       config.get("experiment_name", ""),
        "seed":                  config["training"]["seed"],
        "agent_type":            config["agent"].get("type", ""),
        "n_firms":               benchmarks["n"],
        "info_structure":        config["environment"]["state"]["info_structure"],
        "memory_length":         config["environment"]["state"]["memory_length"],
        **final,
        "q_NE":                  benchmarks["q_NE"],
        "q_monopoly_per_firm":   benchmarks["Q_monopoly"] / benchmarks["n"],
    }
    with open(folder / "final_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def _write_table_benchmarks(folder: Path, bm: dict) -> None:
    lines = [
        "# Analytical Benchmarks\n",
        "| Benchmark    | Q (total) | Q (per firm) | Price  | Profit/firm |",
        "|-------------|-----------|-------------|--------|-------------|",
        f"| Nash         | {bm['Q_NE']:.4f}    | {bm['q_NE']:.4f}       | {bm['p_NE']:.4f} | {bm['pi_NE']:.4f}      |",
        f"| Monopoly     | {bm['Q_monopoly']:.4f}    | {bm['Q_monopoly']/bm['n']:.4f}       | {bm['p_monopoly']:.4f} | {bm['pi_monopoly']/bm['n']:.4f}      |",
        f"| Competitive  | {bm['Q_competitive']:.4f}    | {bm['Q_competitive']/bm['n']:.4f}       | {bm['p_competitive']:.4f} | 0.0000      |",
    ]
    (folder / "tables" / "benchmarks.md").write_text("\n".join(lines) + "\n")


def _write_table_convergence(folder: Path, final: dict) -> None:
    converged_str = "Yes" if final["converged"] else "No"
    conv_step = final["convergence_step"] if final["convergence_step"] >= 0 else "did not converge"
    ci_val = final["collusion_index_final"]
    ci_interp = (
        "none"   if ci_val < 0.05 else
        "mild"   if ci_val < 0.25 else
        "strong" if ci_val < 0.75 else
        "full"
    )
    lines = [
        "# Convergence Statistics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Final mean quantity (behavioural) | {final['final_mean_q']:.6f} |",
        f"| Final greedy quantity | {final['greedy_q']:.6f} |",
        f"| Nash distance (behavioural) | {final['nash_dist_final']:.6f} |",
        f"| Nash distance (greedy) | {final['greedy_dist']:.6f} |",
        f"| Collusion index | {ci_val:.4f} ({ci_interp}) |",
        f"| Converged to Nash | {converged_str} |",
        f"| Convergence step | {conv_step} |",
        f"| Dynamics label | {final['dynamics_label']} |",
    ]
    (folder / "tables" / "convergence.md").write_text("\n".join(lines) + "\n")


def _write_table_welfare(
    folder: Path, config: dict, final: dict, bm: dict
) -> None:
    wt = welfare_table(
        Q_rl=final["final_mean_q"] + (bm["n"] - 1) * bm["q_NE"],
        n=bm["n"],
        benchmarks=bm,
        a=bm["a"], b=bm["b"], c=bm["c"],
    )
    lines = [
        "# Welfare Comparison\n",
        "| Outcome | Consumer Surplus | Producer Surplus | Total Welfare |",
        "|---------|-----------------|-----------------|---------------|",
    ]
    for label, vals in wt.items():
        lines.append(f"| {label:<11} | {vals['CS']:.6f}         | {vals['PS']:.6f}         | {vals['W']:.6f}       |")
    (folder / "tables" / "welfare.md").write_text("\n".join(lines) + "\n")


def _write_plots(
    folder: Path, history: list[dict], bm: dict, config: dict
) -> None:
    """Generate training curve plots and save to plots/training_curves.png."""
    steps     = [r["step"]           for r in history]
    mean_q    = [r["window_mean_q"]  for r in history]
    greedy_q  = [r["greedy_q"]       for r in history]
    nash_dist = [r["nash_dist"]      for r in history]
    gd_dist   = [r["greedy_dist"]    for r in history]
    ci        = [r["collusion_index"] for r in history]

    q_NE  = bm["q_NE"]
    q_M   = bm["Q_monopoly"] / bm["n"]
    q_C   = bm["Q_competitive"] / bm["n"]
    agent = config["agent"].get("type", "agent")
    exp   = config.get("experiment_name", "")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"{exp}  |  agent: {agent}  |  seed: {config['training']['seed']}",
                 fontsize=11, y=0.98)

    # ── Panel 1: Quantity trajectory ──────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, mean_q,   color="#2196F3", lw=1.5, label="mean_q (behavioural)", alpha=0.85)
    ax.plot(steps, greedy_q, color="#FF5722", lw=1.5, label="greedy_q (deterministic)", alpha=0.85)
    ax.axhline(q_NE, color="black",  lw=1.0, ls="--", label=f"Nash q*={q_NE:.3f}")
    ax.axhline(q_M,  color="#9C27B0", lw=0.8, ls=":",  label=f"Monopoly q={q_M:.3f}")
    ax.axhline(q_C,  color="#4CAF50", lw=0.8, ls=":",  label=f"Competitive q={q_C:.3f}")
    ax.set_ylabel("Quantity")
    ax.set_ylim(0, max(q_C * 1.1, max(mean_q) * 1.1))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Nash distance ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, nash_dist, color="#2196F3", lw=1.5, label="nash_dist (behavioural)", alpha=0.85)
    ax.plot(steps, gd_dist,   color="#FF5722", lw=1.5, label="greedy_dist (deterministic)", alpha=0.85)
    ax.axhline(0.05 * q_NE, color="black", lw=0.8, ls="--",
               label=f"5% threshold ({0.05*q_NE:.4f})")
    ax.set_ylabel("Nash Distance")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 3: Collusion index ───────────────────────────────────────────
    ax = axes[2]
    ax.plot(steps, ci, color="#009688", lw=1.5, label="Collusion Index", alpha=0.85)
    ax.axhline(0.0, color="black",   lw=0.8, ls="--", label="Nash (CI=0)")
    ax.axhline(1.0, color="#9C27B0", lw=0.8, ls=":",  label="Monopoly (CI=1)")
    ax.set_ylabel("Collusion Index")
    ax.set_xlabel("Training Step")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(folder / "plots" / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    folder: Path, config: dict, final: dict, bm: dict, history: list[dict]
) -> None:
    e   = config["environment"]
    tr  = config["training"]
    ag  = config["agent"]
    n   = bm["n"]
    q_NE = bm["q_NE"]

    ci_val = final["collusion_index_final"]
    ci_interp = (
        "none"   if ci_val < 0.05 else
        "mild"   if ci_val < 0.25 else
        "strong" if ci_val < 0.75 else
        "full"
    )
    conv_step = final["convergence_step"] if final["convergence_step"] >= 0 else "did not converge"
    converged_str = "Yes" if final["converged"] else "No"

    first_q = history[0]["window_mean_q"]  if history else float("nan")
    last_q  = history[-1]["window_mean_q"] if history else float("nan")
    trend   = "decreasing" if last_q < first_q - 0.01 else "stable" if abs(last_q - first_q) < 0.01 else "increasing"

    lines = [
        f"# Experiment: {config.get('experiment_name', 'unnamed')}\n",
        "## Configuration",
        f"- Agent: {ag.get('type', '')}, Seed: {tr['seed']}, Steps: {tr['total_steps']}",
        f"- Market: n={n}, demand=linear(a={e['demand']['a']}, b={e['demand']['b']}), cost=linear(c={e['cost']['c']})",
        f"- Info structure: {e['state']['info_structure']}, Memory: {e['state']['memory_length']}",
        "",
        "## Analytical Benchmarks",
        "| Benchmark    | Q/firm  | Price   | Profit/firm |",
        "|-------------|---------|---------|-------------|",
        f"| Nash         | {bm['q_NE']:.4f}  | {bm['p_NE']:.4f}  | {bm['pi_NE']:.4f}      |",
        f"| Monopoly     | {bm['Q_monopoly']/n:.4f}  | {bm['p_monopoly']:.4f}  | {bm['pi_monopoly']/n:.4f}      |",
        f"| Competitive  | {bm['Q_competitive']/n:.4f}  | {bm['p_competitive']:.4f}  | 0.0000      |",
        "",
        "## Convergence Results",
        f"- Final mean quantity (behavioural): {final['final_mean_q']:.4f}  (Nash: {q_NE:.4f}, deviation: {(final['final_mean_q']-q_NE)/q_NE*100:.1f}%)",
        f"- Final greedy quantity: {final['greedy_q']:.4f}  (greedy dist: {final['greedy_dist']:.4f})",
        f"- Collusion Index: {ci_val:.4f} — interpretation: {ci_interp}",
        f"- Converged to Nash: {converged_str} (threshold: 5% of q_NE = {0.05*q_NE:.4f})",
        f"- Convergence step: {conv_step}",
        f"- Dynamics classification: {final['dynamics_label']}",
        "",
        "## Welfare at Convergence",
        "| Metric             | RL Agent  | Nash Eq.  | Monopoly  | Competitive |",
        "|-------------------|-----------|-----------|-----------|-------------|",
    ]

    wt = welfare_table(
        Q_rl=final["final_mean_q"] + (n - 1) * q_NE,
        n=n, benchmarks=bm, a=bm["a"], b=bm["b"], c=bm["c"],
    )
    lines += [
        f"| Consumer Surplus  | {wt['RL']['CS']:.4f}    | {wt['Nash']['CS']:.4f}    | {wt['Monopoly']['CS']:.4f}    | {wt['Competitive']['CS']:.4f}       |",
        f"| Producer Surplus  | {wt['RL']['PS']:.4f}    | {wt['Nash']['PS']:.4f}    | {wt['Monopoly']['PS']:.4f}    | {wt['Competitive']['PS']:.4f}       |",
        f"| Total Welfare     | {wt['RL']['W']:.4f}    | {wt['Nash']['W']:.4f}    | {wt['Monopoly']['W']:.4f}    | {wt['Competitive']['W']:.4f}       |",
        "",
        "## Training Curves",
        "",
        "![Training Curves](plots/training_curves.png)",
        "",
        "### Summary",
        f"- Quantity trend: {trend} (start: {first_q:.4f} → end: {last_q:.4f})",
        f"- Final 10% of training: mean_q={final['final_mean_q']:.4f}, std_q={final['final_std_q']:.4f}",
        f"- Git commit: {config.get('_meta', {}).get('git_commit', 'unknown')}",
    ]

    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def _append_experiments_log(config: dict, folder: Path, final: dict) -> None:
    log_path = _PROJECT_ROOT / "research" / "experiments_log.md"
    log_path.parent.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    exp_name  = config.get("experiment_name", "unnamed")
    agent     = config["agent"].get("type", "")
    n         = config["environment"]["market"]["n_firms"]
    info      = config["environment"]["state"]["info_structure"]
    ci        = final["collusion_index_final"]
    converged = final["converged"]
    dyn       = final["dynamics_label"]
    try:
        rel_path = folder.resolve().relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        rel_path = folder

    entry = (
        f"\n## {exp_name} — {timestamp}\n"
        f"- Agent: {agent} | n_firms: {n} | info: {info}\n"
        f"- Result folder: {rel_path}/\n"
        f"- Key result: CI={ci:.4f}, converged={converged}, dynamics={dyn}\n"
    )

    with open(log_path, "a") as f:
        f.write(entry)
