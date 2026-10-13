"""
Regera fig_grim_vs_nash a partir do Exp. 12 (overlay com init cooperativo,
results/overlay_coop_init__20260807/) — substitui a versao confundida
(grim disparado em t=0). Mostra o preco do agente vs grim, nash e monopoly,
com o passo de trigger do grim marcado.

Output: analysis/figures/fig_grim_vs_nash.{png,pdf}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


ROOT = _repo_root()
IN_DIR = ROOT / "results" / "overlay_coop_init__20260807"
OUT = ROOT / "projects" / "p1_deterrence" / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GRAY, GREEN = "#2196F3", "#FF5722", "#757575", "#4CAF50"
P_NE, P_M = 1.4729, 1.9250

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def main() -> None:
    summary = pd.read_csv(IN_DIR / "summary.csv")
    pair_dirs = [IN_DIR / p for p in summary["pair"]]

    n_show = 6
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.8), sharey=True)

    # Rótulo dos painéis: uma letra por par de treino (cada diretório pair_XXX_{A,B}
    # é um run independente = um par co-treinado distinto; _A/_B NÃO são os dois
    # agentes de um par), número = qual agente do par (a bateria avalia o 1º).
    for i, (ax, d) in enumerate(zip(axes.ravel(), pair_dirs[:n_show])):
        grim = pd.read_csv(d / "overlay_grim_player.csv")
        nash = pd.read_csv(d / "overlay_nash_player.csv")
        mono = pd.read_csv(d / "overlay_monopoly_player.csv")

        ax.axhline(P_NE, color=GRAY, lw=1, ls="--", zorder=1)
        ax.axhline(P_M, color=GRAY, lw=1, ls=":", zorder=1)
        ax.plot(nash["t"], nash["price_agent"], color=BLUE, lw=1.6,
                label="vs nash", solid_capstyle="round")
        ax.plot(mono["t"], mono["price_agent"], color=GREEN, lw=1.6,
                label="vs monopoly", solid_capstyle="round")
        ax.plot(grim["t"], grim["price_agent"], color=RED, lw=1.4,
                ls=(0, (4, 3)), label="vs grim (cooperative)")

        trig = summary.loc[summary["pair"] == d.name,
                           "grim_trigger_step"].iloc[0]
        if trig >= 0:
            ax.axvline(trig, color=RED, lw=0.9, alpha=0.5)
            ax.text(trig + 0.6, ax.get_ylim()[0] + 0.02, f"trigger t={trig}",
                    color=RED, fontsize=7.5, va="bottom")

        ax.set_title(f"agent {chr(ord('A') + i)}1", fontsize=9)
        ax.set_xlim(0, 40)

    for ax in axes[-1]:
        ax.set_xlabel("Step", fontsize=9)
    axes[0][0].set_ylabel("Price")
    axes[1][0].set_ylabel("Price")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Trained agent's price vs grim (starting cooperative), "
                 "constant Nash and constant monopoly — cooperative "
                 "initialization", fontsize=10.5)
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_grim_vs_nash.{ext}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)

    n_trig = int((summary["grim_trigger_step"] >= 0).sum())
    print(f"fig_grim_vs_nash regerada ({n_trig}/{len(summary)} pares com "
          f"trigger; primeiros {n_show} exibidos)")


if __name__ == "__main__":
    main()
