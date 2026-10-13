"""
Policy functions dos agentes treinados — figuras estilo economia.

1. fig_reaction_functions — as "funções de reação" aprendidas: preço escolhido
   em t como função do preço do rival em t−1, com o próprio preço em t−1 fixado
   no nível on-path de cada agente. Spaghetti dos 18 agentes (agent_0 de cada
   par) + best-response estático analítico + identidade (45°).

2. fig_policy_heatmap — a política completa de um par representativo:
   heatmap 15×15 (próprio preço t−1 × preço do rival t−1 → preço escolhido),
   com o atrator conjunto marcado.

Convenção de estado (do trainer): s = idx(p0)·n + idx(p1) para AMBOS os
agentes. Para o agente 0, "próprio" é o primeiro slot; para o agente 1, o
segundo — os heatmaps abaixo já fazem essa transposição.

Outputs: analysis/figures/fig_reaction_functions.{png,pdf},
         analysis/figures/fig_policy_heatmap.{png,pdf}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


ROOT = _repo_root()
PAIRS = ROOT / "results" / "cross_deploy__20260621_203040" / "training"
OUT = ROOT / "projects" / "p1_deterrence" / "analysis" / "figures"

BLUE, RED, GRAY, ORANGE = "#2196F3", "#FF5722", "#757575", "#FF9800"
P_NE, P_M = 1.4729, 1.9250
A, MU, C, A0 = 2.0, 0.25, 1.0, 0.0

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def _load(pair_dir: Path):
    cfg = yaml.safe_load((pair_dir / "config.yaml").read_text())
    n = int(cfg["agent"]["n_actions"])
    q = [np.load(pair_dir / "agents" / f"agent_{i}.npz")["q_table"]
         for i in (0, 1)]
    # grade: reconstruída dos benchmarks (grid_extend=1.0)
    extend = float(cfg["environment"]["market"].get("grid_extend", 1.0))
    p_min = max(C, P_NE - extend * (P_M - P_NE))
    p_max = P_M + extend * (P_M - P_NE)
    grid = np.linspace(p_min, p_max, n)
    return q, grid, n


def _attractor(q0, q1, n: int) -> tuple[int, int]:
    """Índices (i0, i1) do atrator conjunto (após burn-in greedy)."""
    pol0, pol1 = q0.argmax(axis=1), q1.argmax(axis=1)
    s = 0
    for _ in range(1000):
        s = pol0[s] * n + pol1[s]
    return s // n, s % n


def _static_br_curve(grid: np.ndarray) -> np.ndarray:
    e_out = np.exp(A0 / MU)
    br = []
    for pr in grid:
        e_r = np.exp((A - pr) / MU)
        profits = [(p - C) * np.exp((A - p) / MU) /
                   (np.exp((A - p) / MU) + e_r + e_out) for p in grid]
        br.append(grid[int(np.argmax(profits))])
    return np.array(br)


# ------------------------------------------------------------------
# 1. Funções de reação aprendidas
# ------------------------------------------------------------------

def fig_reaction():
    pair_dirs = sorted(d for d in PAIRS.iterdir()
                       if (d / "agents" / "agent_0.npz").exists())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    grid_ref = None
    for d in pair_dirs:
        q, grid, n = _load(d)
        grid_ref = grid
        i0, i1 = _attractor(q[0], q[1], n)
        # agente 0: s = own*n + rival; fatia em own = preço on-path
        curve = grid[[int(q[0][i0 * n + j].argmax()) for j in range(n)]]
        ax.plot(grid, curve, color=BLUE, lw=1.0, alpha=0.35, zorder=2)
        ax.plot(grid[i1], curve[i1], "o", color=RED, ms=4.5, zorder=4,
                mec="white", mew=0.6)

    br = _static_br_curve(grid_ref)
    ax.plot(grid_ref, br, color="#263238", lw=2, ls="--", zorder=3,
            label="Static best response")
    ax.plot(grid_ref, grid_ref, color=GRAY, lw=1, alpha=0.6, zorder=1,
            label="45° line")
    ax.axhline(P_NE, color=GRAY, lw=0.9, ls="--", alpha=0.7)
    ax.axvline(P_NE, color=GRAY, lw=0.9, ls="--", alpha=0.7)
    ax.text(grid_ref[-1], P_NE + 0.015, "$p^{NE}$", ha="right", color=GRAY,
            fontsize=9)

    ax.plot([], [], color=BLUE, lw=1.2, alpha=0.7,
            label="Learned policy (18 agents),\nown price fixed on-path")
    ax.plot([], [], "o", color=RED, ms=5, mec="white",
            label="On-path state (training partner)")
    ax.set_xlabel("Rival's price in $t-1$")
    ax.set_ylabel("Price chosen in $t$")
    ax.set_title("Learned reaction functions vs static best response",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, "fig_reaction_functions")


# ------------------------------------------------------------------
# 2. Heatmap da política — par representativo
# ------------------------------------------------------------------

def fig_heatmap(pair_name: str = "pair_000_A"):
    q, grid, n = _load(PAIRS / pair_name)
    i0, i1 = _attractor(q[0], q[1], n)

    # matriz [own, rival] respeitando a convenção s = idx(p0)*n + idx(p1)
    mats = []
    m0 = np.array([[grid[int(q[0][o * n + r].argmax())] for r in range(n)]
                   for o in range(n)])
    m1 = np.array([[grid[int(q[1][r * n + o].argmax())] for r in range(n)]
                   for o in range(n)])
    mats = [(m0, i0, i1, "Agent 0"), (m1, i1, i0, "Agent 1")]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, (m, own_i, riv_i, title) in zip(axes, mats):
        im = ax.imshow(m, origin="lower", cmap="Blues",
                       vmin=grid[0], vmax=grid[-1],
                       extent=(grid[0], grid[-1], grid[0], grid[-1]),
                       aspect="auto")
        ax.plot(grid[riv_i], grid[own_i], "*", color=RED, ms=13,
                mec="white", mew=0.8, label="On-path state (attractor)")
        for v, ls in ((P_NE, "--"), (P_M, ":")):
            ax.axhline(v, color=GRAY, lw=0.9, ls=ls, alpha=0.8)
            ax.axvline(v, color=GRAY, lw=0.9, ls=ls, alpha=0.8)
        ax.set_xlabel("Rival's price in $t-1$")
        ax.set_title(f"{title} — greedy policy", fontsize=10)
    axes[0].set_ylabel("Own price in $t-1$")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label("Price chosen in $t$", fontsize=9)
    fig.suptitle(f"Policy functions of a co-trained pair ({pair_name}) — "
                 "dashed: $p^{NE}$, dotted: $p^{M}$", fontsize=10.5)
    _save(fig, "fig_policy_heatmap")


if __name__ == "__main__":
    print("Gerando figuras de policy function:")
    fig_reaction()
    fig_heatmap()
