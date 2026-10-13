"""
Output writer for multi-agent Phase 2 experiments.

Output folder: results/{experiment_name}__{seed}__{YYYYMMDD_HHMMSS}/
  config.yaml
  summary.md
  metrics_history.csv
  tables/
    equilibrium.md      — CI, CE-gap, dynamics_label
    per_agent.md        — per-agent final quantities and rewards
  plots/
    training_curves.png
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.metrics.multi_agent import final_metrics as compute_final

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR  = _PROJECT_ROOT / "results"


def _env_type(config: dict) -> str:
    return config.get("environment", {}).get("type", "cournot")


def _action_refs(bm: dict, N: int, env_type: str) -> tuple[float, float, float]:
    """Return (ref_NE, ref_M, ref_C) per-firm action benchmarks for plots."""
    if env_type in ("bertrand", "matrix"):
        # `matrix` (p7): as âncoras são ÍNDICES DE AÇÃO — a não-cooperativa faz o papel de
        # Nash e a cooperativa o de monopólio. Ver RepeatedMatrixGameEnv.benchmarks().
        return bm["p_NE"], bm["p_monopoly"], bm["p_competitive"]
    else:
        q_M = bm["Q_monopoly"] / N
        q_C = bm["Q_competitive"] / N
        return bm["q_NE"], q_M, q_C


def write(
    config: dict,
    history: list[dict],
    benchmarks: dict,
    output_dir: Path | None = None,
    cce_data: dict | None = None,
) -> Path:
    """
    Write output folder for a completed multi-agent run.

    Parameters
    ----------
    config      : fully-resolved experiment config
    history     : list of per-eval dicts from multi_agent_trainer
    benchmarks  : env.benchmarks() dict
    output_dir  : explicit destination folder; auto-generated if None

    Returns
    -------
    Path to created output folder.
    """
    if output_dir is not None:
        folder = Path(output_dir)
    else:
        seed      = config["training"]["seed"]
        exp_name  = config.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder    = _RESULTS_DIR / f"{exp_name}__{seed}__{timestamp}"

    (folder / "plots").mkdir(parents=True, exist_ok=True)
    (folder / "tables").mkdir(parents=True, exist_ok=True)

    N        = config["environment"]["market"]["n_firms"]
    eval_w   = config["training"].get("eval_window", 50000)
    eval_i   = config["training"]["eval_interval"]
    env_type = _env_type(config)
    final    = compute_final(history, eval_w, eval_i, benchmarks, N)

    _write_config(folder, config)
    _write_metrics_history(folder, history)
    _write_plots(folder, history, benchmarks, config, N, env_type)
    _write_table_equilibrium(folder, final, benchmarks)
    _write_table_per_agent(folder, final, benchmarks, N, env_type)
    _write_summary(folder, config, history, benchmarks, final, N, env_type)
    _append_experiments_log(config, folder, final, benchmarks, N)
    if cce_data is not None:
        _write_cce_plots(folder, cce_data, benchmarks, N, env_type)
        _write_cce_table(folder, cce_data, benchmarks, N, env_type)

    return folder


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_config(folder: Path, config: dict) -> None:
    with open(folder / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _write_metrics_history(folder: Path, history: list[dict]) -> None:
    if not history:
        return
    with open(folder / "metrics_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _write_table_equilibrium(folder: Path, final: dict, bm: dict) -> None:
    ci    = final["ci"]
    ceg   = final["ce_gap"]
    dyn   = final["dynamics_label"]
    nd    = final["nash_dist"]
    sym   = final["sym_q"]

    ci_interp = _ci_label(ci)
    ceg_interp = "CE-consistent (no incentive to deviate)" if ceg < 0.01 else "deviation incentive present"

    lines = [
        "# Equilibrium Identification\n",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| Collusion Index | {ci:.4f} | {ci_interp} |",
        f"| Nash Distance (mean) | {nd:.4f} | distance from q_NE={bm['q_NE']:.4f} |",
        f"| CE-gap | {ceg:.4f} | {ceg_interp} |",
        f"| Inter-agent symmetry (std) | {sym:.4f} | 0 = symmetric play |",
        f"| Dynamics label | {dyn} | — |",
        "",
        "## Classification Logic",
        "",
        "| Condition | Label |",
        "|-----------|-------|",
        "| autocorr(Q_greedy) > 0.90 AND CI < 0.10 | nash_fixed_point |",
        "| autocorr(Q_greedy) > 0.90 AND CI > 0.20 | collusive_fixed_point |",
        "| 0.30 < autocorr(Q_greedy) ≤ 0.90 | cycling |",
        "| std(Q_greedy) > 0.20 × Q_NE | unstable |",
        "| otherwise | converging |",
    ]
    (folder / "tables" / "equilibrium.md").write_text("\n".join(lines) + "\n")


def _write_table_per_agent(
    folder: Path, final: dict, bm: dict, N: int, env_type: str = "cournot"
) -> None:
    if env_type == "matrix":
        ref_NE = bm["p_NE"]
        a_label = "action"
    elif env_type == "bertrand":
        ref_NE = bm["p_NE"]
        a_label = "price"
    else:
        ref_NE = bm["q_NE"]
        a_label = "quantity"

    lines = [
        "# Per-Agent Final Metrics\n",
        f"| Agent | mean_{a_label} (window) | greedy_{a_label} | Nash dist (greedy) |",
        "|-------|----------------|----------|-------------------|",
    ]
    for i in range(N):
        mq = final.get(f"mean_q_{i}", float("nan"))
        gq = final.get(f"greedy_q_{i}", float("nan"))
        nd = abs(gq - ref_NE)
        lines.append(f"| {i} | {mq:.6f} | {gq:.6f} | {nd:.6f} |")
    (folder / "tables" / "per_agent.md").write_text("\n".join(lines) + "\n")


def _write_plots(
    folder: Path,
    history: list[dict],
    bm: dict,
    config: dict,
    N: int,
    env_type: str = "cournot",
) -> None:
    steps = [r["step"]            for r in history]
    ci    = [r["collusion_index"] for r in history]
    nd    = [r["nash_dist"]       for r in history]
    ceg   = [r["ce_gap"]          for r in history]

    ref_NE, ref_M, ref_C = _action_refs(bm, N, env_type)
    a_label = {"bertrand": "Price", "matrix": "Action"}.get(env_type, "Quantity")
    agent   = config["agent"].get("type", "agent")
    info    = config["environment"]["state"]["info_structure"]
    exp     = config.get("experiment_name", "")

    # Painéis extras (p3, etapa 3) — só quando o history tem as chaves, para manter
    # compatibilidade com CSVs antigos re-renderizados.
    has_profit = f"mean_reward_0" in history[0]
    has_td     = ("td_error_mean_0" in history[0]) or ("epsilon" in history[0])
    n_rows = 3 + int(has_profit) + int(has_td)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.3 * n_rows), sharex=True)
    fig.suptitle(
        f"{exp}  |  agent={agent}  |  info={info}  |  N={N}  |  seed={config['training']['seed']}",
        fontsize=10, y=0.98,
    )

    # Panel 1: per-agent greedy actions + cross-agent mean ± std band
    ax = axes[0]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]
    for i in range(N):
        color = colors[i % len(colors)]
        gkey  = f"greedy_q_{i}"
        mkey  = f"mean_q_{i}"
        if gkey in history[0]:
            gqs = [r[gkey] for r in history]
            ax.plot(steps, gqs, color=color, lw=1.2,
                    label=f"agent {i} greedy", alpha=0.85)
        if mkey in history[0]:
            mqs = [r[mkey] for r in history]
            ax.plot(steps, mqs, color=color, lw=0.8, ls="--",
                    label=f"agent {i} mean", alpha=0.45)
            ax.annotate(
                f"{mqs[-1]:.3f}",
                xy=(steps[-1], mqs[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7, color=color, va="center",
            )

    # Cross-agent mean and ±1 std of per-agent mean prices
    if f"mean_q_0" in history[0]:
        agent_means = np.array(
            [[r[f"mean_q_{i}"] for i in range(N)] for r in history]
        )  # shape (T, N)
        cross_mean = agent_means.mean(axis=1)
        cross_std  = agent_means.std(axis=1)
        ax.plot(steps, cross_mean, color="black", lw=2.0, ls="-",
                label=f"mean of means", zorder=5)
        ax.fill_between(
            steps,
            cross_mean - cross_std,
            cross_mean + cross_std,
            color="black", alpha=0.12, label="±1 std (agents)",
        )

    ax.axhline(ref_NE, color="black",   lw=1.0, ls="--", label=f"Nash={ref_NE:.3f}")
    ax.axhline(ref_M,  color="#9C27B0", lw=0.8, ls=":",  label=f"Monopoly/firm={ref_M:.3f}")
    ax.axhline(ref_C,  color="#009688", lw=0.8, ls=":",  label=f"Comp/firm={ref_C:.3f}")
    ax.set_ylabel(f"{a_label} (greedy / mean ± std)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: CI and Nash distance
    ax = axes[1]
    ax.plot(steps, ci, color="#2196F3", lw=1.5, label="Collusion Index", alpha=0.85)
    ax.plot(steps, nd, color="#FF5722", lw=1.5, label="Nash Distance (mean)", alpha=0.85)
    ax.axhline(0.0, color="black",   lw=0.8, ls="--", label="Nash (CI=0)")
    ax.axhline(1.0, color="#9C27B0", lw=0.8, ls=":",  label="Monopoly (CI=1)")
    ax.set_ylabel("CI / Nash Dist")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: CE-gap
    ax = axes[2]
    ax.plot(steps, ceg, color="#009688", lw=1.5, label="CE-gap", alpha=0.85)
    ax.axhline(0.0, color="black", lw=0.8, ls="--", label="CE-gap=0 (equilibrium)")
    ax.set_ylabel("CE-gap")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    row = 3

    # Panel 4 (p3): lucro por agente na janela, contra π_NE e π_M por firma
    if has_profit:
        ax = axes[row]; row += 1
        for i in range(N):
            key = f"mean_reward_{i}"
            if key in history[0]:
                vals = [r[key] for r in history]
                ax.plot(steps, vals, color=colors[i % len(colors)], lw=1.2,
                        label=f"agent {i} profit", alpha=0.85)
        pi_ne = bm.get("pi_NE")
        pi_m  = bm.get("pi_monopoly_per_firm")
        if pi_ne is not None:
            ax.axhline(float(pi_ne), color="black", lw=1.0, ls="--",
                       label=f"π_NE={float(pi_ne):.4f}")
        if pi_m is not None:
            ax.axhline(float(pi_m), color="#9C27B0", lw=0.8, ls=":",
                       label=f"π_M/firm={float(pi_m):.4f}")
        ax.set_ylabel("Profit (window mean)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Panel 5 (p3): a "loss" do tabular — |TD-error| janelado (escala log) + ε/α
    if has_td:
        ax = axes[row]; row += 1
        any_td = False
        for i in range(N):
            key = f"td_error_mean_{i}"
            if key in history[0]:
                vals = [r[key] for r in history]
                if np.any(np.isfinite(vals)):
                    any_td = True
                ax.plot(steps, vals, color=colors[i % len(colors)], lw=1.2,
                        label=f"agent {i} |TD| mean", alpha=0.85)
        if any_td:
            ax.set_yscale("log")
        ax.set_ylabel("|TD-error| (window)")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        if "epsilon" in history[0]:
            ax2.plot(steps, [r["epsilon"] for r in history], color="#607D8B",
                     lw=1.0, ls="--", label="epsilon", alpha=0.7)
        if "alpha_0" in history[0]:
            ax2.plot(steps, [r["alpha_0"] for r in history], color="#795548",
                     lw=1.0, ls=":", label="alpha (agent 0)", alpha=0.7)
        ax2.set_ylabel("ε / α", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Training Step")

    plt.tight_layout()
    fig.savefig(folder / "plots" / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    folder: Path,
    config: dict,
    history: list[dict],
    bm: dict,
    final: dict,
    N: int,
    env_type: str = "cournot",
) -> None:
    agent = config["agent"].get("type", "")
    info  = config["environment"]["state"]["info_structure"]
    seed  = config["training"]["seed"]
    steps = config["training"]["total_steps"]

    ci_val  = final["ci"]
    ceg_val = final["ce_gap"]
    dyn     = final["dynamics_label"]

    ci_interp  = _ci_label(ci_val)
    ceg_interp = "CE-consistent" if ceg_val < 0.01 else "deviation incentive present"

    interpretation = _interpret(ci_val, ceg_val, dyn, agent, info, N)

    greedy_last = [final.get(f"greedy_q_{i}", float("nan")) for i in range(N)]

    if env_type == "matrix":
        # p7 — jogo 2×2 repetido: não há demanda nem custo; o "mercado" é a matriz.
        ref_NE    = bm["p_NE"]
        mkt_line  = (f"- Game: {bm.get('game_name', '?')} 2×2 repetido, N={N}, "
                     f"T={bm['T']:.2f} R={bm['R']:.2f} P={bm['P']:.2f} S={bm['S']:.2f}")
        bm_lines  = [
            f"- N0 (jogo uniforme): CR={bm['cr_random']:.3f} · "
            f"π̄={bm['pi_random']:.4f} · IC_π={bm['ic_pi_random']:.4f}",
            f"- Ações: {bm['p_monopoly']:.0f} = cooperativa · {bm['p_NE']:.0f} = não-cooperativa",
            f"- CR definido neste ambiente: {bm['cr_defined']}",
        ]
        a_label = "Action"
    elif env_type == "bertrand":
        ref_NE    = bm["p_NE"]
        d         = config["environment"]["demand"]
        mkt_line  = f"- Market: N={N}, demand=logit(a={d['a']:.2f}, μ={d['mu']:.3f}), c={config['environment']['cost']['c']:.2f}"
        bm_lines  = [
            f"- Nash: p/firm={bm['p_NE']:.4f}, q/firm={bm['q_NE']:.4f}, π/firm={bm['pi_NE']:.4f}",
            f"- Monopoly: p/firm={bm['p_monopoly']:.4f}, q/firm={bm['q_monopoly_per_firm']:.4f}",
            f"- Competitive: p/firm={bm['p_competitive']:.4f}",
        ]
        a_label = "Price"
    else:
        ref_NE    = bm["q_NE"]
        d         = config["environment"]["demand"]
        mkt_line  = f"- Market: N={N}, demand=linear(a={d['a']:.1f}, b={d['b']:.1f}), c={config['environment']['cost']['c']:.1f}"
        bm_lines  = [
            f"- Nash: q/firm={bm['q_NE']:.4f}, Q={bm['Q_NE']:.4f}, p={bm['p_NE']:.4f}, π/firm={bm['pi_NE']:.4f}",
            f"- Monopoly: Q={bm['Q_monopoly']:.4f}, p={bm['p_monopoly']:.4f}",
            f"- Competitive: Q={bm['Q_competitive']:.4f}, p={bm['p_competitive']:.4f}",
        ]
        a_label = "Quantity"

    lines = [
        f"# Experiment: {config.get('experiment_name', 'unnamed')}\n",
        "## Configuration",
        f"- Agent: {agent} | Seed: {seed} | Steps: {steps}",
        mkt_line,
        f"- Info structure: {info}",
        "",
        "## Analytical Benchmarks",
        *bm_lines,
        "",
        "## Equilibrium Results (final window)",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| Collusion Index | {ci_val:.4f} | {ci_interp} |",
        f"| Nash Distance (mean) | {final['nash_dist']:.4f} | — |",
        f"| CE-gap | {ceg_val:.4f} | {ceg_interp} |",
        f"| Symmetry (std greedy) | {final['sym_q']:.4f} | — |",
        f"| Dynamics | {dyn} | — |",
        "",
        f"## Per-Agent Greedy {a_label}s",
        "",
        f"| Agent | greedy_{a_label.lower()} | Nash dist |",
        "|-------|----------|-----------|",
    ]
    for i in range(N):
        gq = greedy_last[i]
        lines.append(f"| {i} | {gq:.4f} | {abs(gq - ref_NE):.4f} |")

    # Profit (final window) — a métrica primária do p3; CI é secundário.
    last = history[-1] if history else {}
    if any(f"mean_reward_{i}" in last for i in range(N)):
        pi_ne = float(bm.get("pi_NE", float("nan")))
        pi_m  = float(bm.get("pi_monopoly_per_firm", float("nan")))
        profs = [float(last.get(f"mean_reward_{i}", float("nan"))) for i in range(N)]
        lines += [
            "",
            "## Profit (final window)",
            "",
            f"π_NE/firm = {pi_ne:.4f} · π_M/firm = {pi_m:.4f}",
            "",
            "| Agent | π̄ (window) | π̄/π_M | π̄ − π_NE |",
            "|-------|------------|--------|-----------|",
        ]
        for i in range(N):
            frac = profs[i] / pi_m if pi_m and np.isfinite(pi_m) else float("nan")
            lines.append(f"| {i} | {profs[i]:.4f} | {frac:.4f} | {profs[i] - pi_ne:+.4f} |")
        if N == 2 and all(np.isfinite(p) for p in profs):
            lines += [
                "",
                f"Assimetria: Δπ = {profs[0] - profs[1]:+.4f} "
                f"(razão {profs[0] / profs[1]:.3f})" if profs[1] else "",
            ]
        # curvas de treino — diagnóstico da triagem
        td_bits = []
        for i in range(N):
            td = last.get(f"td_error_mean_{i}")
            if td is not None and np.isfinite(td):
                td_bits.append(f"|TD|_{i} = {td:.5f}")
        eps = last.get("epsilon")
        if eps is not None and np.isfinite(eps):
            td_bits.append(f"ε_final = {eps:.4f}")
        if td_bits:
            lines += ["", "Final da janela: " + " · ".join(td_bits)]

    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Training Curves",
        "",
        "![Training Curves](plots/training_curves.png)",
        "",
        "See [tables/equilibrium.md](tables/equilibrium.md) for full classification detail.",
        "See [tables/per_agent.md](tables/per_agent.md) for per-agent breakdown.",
    ]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def _append_experiments_log(
    config: dict,
    folder: Path,
    final: dict,
    bm: dict,
    N: int,
) -> None:
    log_path = _PROJECT_ROOT / "research" / "experiments_log.md"
    log_path.parent.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    exp_name  = config.get("experiment_name", "unnamed")
    agent     = config["agent"].get("type", "")
    info      = config["environment"]["state"]["info_structure"]
    try:
        rel_path = folder.resolve().relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        rel_path = folder

    entry = (
        f"\n## {exp_name} — {timestamp}\n"
        f"- Agent: {agent} | N_firms: {N} | info: {info} (Phase 2)\n"
        f"- Result folder: {rel_path}/\n"
        f"- Key result: CI={final['ci']:.4f}, CE-gap={final['ce_gap']:.4f}, "
        f"dynamics={final['dynamics_label']}\n"
    )
    with open(log_path, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CCE characterisation
# ---------------------------------------------------------------------------

def _write_cce_plots(
    folder: Path,
    cce_data: dict,
    bm: dict,
    N: int,
    env_type: str,
) -> None:
    colors    = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]
    ref_NE, ref_M, ref_C = _action_refs(bm, N, env_type)
    a_label   = "Price" if env_type == "bertrand" else "Quantity"
    cf_grid   = cce_data["cf_grid"]
    actions   = cce_data["actions"]       # list[list[float]], (N, T)
    exp_cf    = cce_data["expected_cf"]   # list[np.ndarray], (N, 50)
    mean_prof = cce_data["mean_profits"]  # list[float]

    # --- Plot 1: action distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(N):
        ax.hist(
            actions[i], bins=30, alpha=0.55,
            color=colors[i % len(colors)], label=f"agent {i}",
            density=True,
        )
    ax.axvline(ref_NE, color="black",   lw=1.2, ls="--", label=f"Nash={ref_NE:.3f}")
    ax.axvline(ref_M,  color="#9C27B0", lw=1.0, ls=":",  label=f"Monopoly={ref_M:.3f}")
    ax.axvline(ref_C,  color="#009688", lw=1.0, ls=":",  label=f"Comp={ref_C:.3f}")
    ax.set_xlabel(a_label)
    ax.set_ylabel("Density")
    ax.set_title("Action Distribution — Final Eval Window")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(folder / "plots" / "action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: deviation gain curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(N):
        gain = exp_cf[i] - mean_prof[i]
        ax.plot(cf_grid, gain, color=colors[i % len(colors)], lw=1.5, label=f"agent {i}")
        max_gain = float(np.max(gain))
        argmax   = float(cf_grid[np.argmax(gain)])
        ax.annotate(
            f"max={max_gain:+.4f}\n@{argmax:.3f}",
            xy=(argmax, max_gain),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color=colors[i % len(colors)],
        )
    ax.axhline(0.0, color="black", lw=1.2, ls="--", label="CCE threshold (gain=0)")
    ax.set_xlabel(f"Deviation {a_label}")
    ax.set_ylabel("E[π(p′, a₋ᵢ)] − E[π(aᵢ, a₋ᵢ)]")
    ax.set_title("Deviation Gain Curve — CCE Verification")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(folder / "plots" / "deviation_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: joint distribution heatmap (N=2 only) ---
    if N == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        h, xedges, yedges = np.histogram2d(actions[0], actions[1], bins=20)
        h = h / h.sum()
        im = ax.imshow(
            h.T, origin="lower", aspect="auto", cmap="Blues",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        plt.colorbar(im, ax=ax, label="Relative frequency")
        ax.scatter([ref_NE], [ref_NE], color="black",   s=60, zorder=5, label=f"Nash ({ref_NE:.3f})")
        ax.scatter([ref_M],  [ref_M],  color="#9C27B0", s=60, zorder=5, label=f"Monopoly ({ref_M:.3f})")
        diag = np.linspace(min(xedges[0], yedges[0]), max(xedges[-1], yedges[-1]), 100)
        ax.plot(diag, diag, color="gray", lw=0.8, ls="--", alpha=0.6, label="Symmetric play")
        ax.set_xlabel(f"Agent 0 {a_label}")
        ax.set_ylabel(f"Agent 1 {a_label}")
        ax.set_title("Joint Action Distribution — Final Eval Window")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(folder / "plots" / "joint_action_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _write_cce_table(
    folder: Path,
    cce_data: dict,
    bm: dict,
    N: int,
    env_type: str,
) -> None:
    ref_NE    = bm["p_NE"] if env_type == "bertrand" else bm["q_NE"]
    a_label   = "price" if env_type == "bertrand" else "quantity"
    cf_grid   = cce_data["cf_grid"]
    actions   = cce_data["actions"]
    exp_cf    = cce_data["expected_cf"]
    mean_prof = cce_data["mean_profits"]

    lines = [
        "# CCE Verification\n",
        f"Deviation gain = E[π(p′, a_{{-i}})] − E[π(aᵢ, a_{{-i}})] averaged over final eval window.",
        "CCE holds if max deviation gain ≤ 0 for all agents.\n",
        f"| Agent | Mean {a_label} | Mean profit | Max deviation gain | Best deviation {a_label} | CCE? |",
        "|-------|------------|-------------|-------------------|----------------------|------|",
    ]
    all_ok = True
    for i in range(N):
        gain      = exp_cf[i] - mean_prof[i]
        max_gain  = float(np.max(gain))
        best_dev  = float(cf_grid[np.argmax(gain)])
        mean_act  = float(np.mean(actions[i]))
        cce_ok    = max_gain <= 0.0
        if not cce_ok:
            all_ok = False
        mark = "✓" if cce_ok else "✗"
        lines.append(
            f"| {i} | {mean_act:.4f} | {mean_prof[i]:.4f} "
            f"| {max_gain:+.4f} | {best_dev:.4f} | {mark} |"
        )

    lines += [
        "",
        f"**Overall CCE: {'✓ satisfied' if all_ok else '✗ not satisfied'}**",
        "",
        f"Nash reference {a_label}: {ref_NE:.4f}",
    ]
    (folder / "tables" / "cce.md").write_text("\n".join(lines) + "\n")


def _ci_label(ci: float) -> str:
    if ci < 0.05:
        return "Nash (no collusion)"
    if ci < 0.25:
        return "mild collusion"
    if ci < 0.75:
        return "strong collusion"
    return "full collusion (near monopoly)"


def _interpret(
    ci: float, ce_gap: float, dyn: str, agent: str, info: str, N: int
) -> str:
    """Generate a one-paragraph plain-language interpretation of the result."""
    ci_desc  = _ci_label(ci)
    eq_desc  = "consistent with a correlated equilibrium" if ce_gap < 0.01 else \
               "not consistent with equilibrium — agents have incentive to deviate"

    if dyn in ("nash_fixed_point", "collusive_fixed_point", "fixed_point"):
        dyn_desc = "converged to a stationary point"
    elif dyn == "cycling":
        dyn_desc = "showing cyclic dynamics"
    elif dyn == "unstable":
        dyn_desc = "unstable (high variance, no convergence)"
    else:
        dyn_desc = "still converging"

    return (
        f"With **{agent}** agents under **{info}** information and **N={N}** firms, "
        f"the system is {dyn_desc}. "
        f"The collusion index is **{ci:.3f}** ({ci_desc}), and the outcome is {eq_desc} "
        f"(CE-gap = {ce_gap:.4f}). "
        f"{'High CI with low CE-gap indicates a genuine collusive equilibrium, not merely a transient high-price phase.' if ci >= 0.25 and ce_gap < 0.01 else ''}"
        f"{'Low CI with low CE-gap is consistent with Nash convergence.' if ci < 0.10 and ce_gap < 0.01 else ''}"
        f"{'Positive CE-gap means at least one agent would benefit from deviating unilaterally — the current distribution is not an equilibrium.' if ce_gap >= 0.01 else ''}"
    ).strip()
