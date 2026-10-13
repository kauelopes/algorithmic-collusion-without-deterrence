"""
Figuras: impulse-response (à la Calvano), overlay grim vs nash, CE-gap interpretável.

Inputs:
  results/impulse_response__20260807/   (impulse.csv, overlay_*.csv por par)
  analysis/data/interpretable_ce_gap.csv

Outputs (png + pdf):
  analysis/figures/fig_impulse_response.{png,pdf}
  analysis/figures/fig_grim_vs_nash.{png,pdf}
  analysis/figures/fig_ce_gap_interpretable.{png,pdf}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
IR_DIR = ROOT / "results" / "impulse_response__20260807"
OUT = ROOT / "projects" / "p1_deterrence" / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta do projeto (plot_cross_deploy_analysis.py)
BLUE, RED, GRAY, GREEN = "#2196F3", "#FF5722", "#757575", "#4CAF50"
P_NE, P_M = 1.4729, 1.9250

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def _benchmark_lines(ax):
    ax.axhline(P_NE, color=GRAY, lw=1, ls="--", zorder=1)
    ax.axhline(P_M, color=GRAY, lw=1, ls=":", zorder=1)


# ------------------------------------------------------------------
# 1. Impulse response — média entre os 18 pares
# ------------------------------------------------------------------

def _load_impulses(base: Path):
    dfs = [pd.read_csv(d / "impulse.csv") for d in sorted(base.iterdir())
           if (d / "impulse.csv").exists()]
    t = dfs[0]["t"].values
    p0 = np.stack([df["price_0"].values for df in dfs])   # desviante
    p1 = np.stack([df["price_1"].values for df in dfs])   # rival
    return t, p0, p1


def fig_impulse():
    panels = [("Within training pair", IR_DIR, True),
              ("Cross-paired (never trained together)", IR_DIR / "cross", False)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (title, base, annotate) in zip(axes, panels):
        t, p0, p1 = _load_impulses(base)
        _benchmark_lines(ax)
        ax.axvline(0, color=GRAY, lw=0.8, alpha=0.6)
        for arr, color, label in ((p0, RED, "Deviating agent"),
                                  (p1, BLUE, "Rival")):
            mean = arr.mean(axis=0)
            lo = np.quantile(arr, 0.25, axis=0)
            hi = np.quantile(arr, 0.75, axis=0)
            ax.fill_between(t, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(t, mean, color=color, lw=2, label=label)
        if annotate:
            ax.annotate("forced deviation\n(static best response)",
                        xy=(0, float(p0[:, np.argmax(t == 0)].mean())),
                        xytext=(18, 1.50), fontsize=8.5, color=GRAY,
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
        ax.set_xlim(t[0], t[-1] + 12)
        ax.text(t[-1] + 1, P_NE, "$p^{NE}$", va="center", color=GRAY, fontsize=9)
        ax.text(t[-1] + 1, P_M, "$p^{M}$", va="center", color=GRAY, fontsize=9)
        ax.set_xlabel("Steps relative to forced deviation")
        ax.set_title(f"{title} (18)", fontsize=10)
    axes[0].set_ylabel("Price")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Impulse response: mean and interquartile range", fontsize=10.5)
    fig.tight_layout()
    _save(fig, "fig_impulse_response")


# ------------------------------------------------------------------
# 1b. IRF estilo economia — 2 choques × 2 braços, média ± 1 d.p.
# ------------------------------------------------------------------

def fig_irf():
    csv_path = ROOT / "results" / "irf__20260807" / "irf_data.csv"
    if not csv_path.exists():
        print("  (irf_data.csv ausente — pulando fig_irf)")
        return
    df = pd.read_csv(csv_path)

    shock_titles = {"down": "One-period undercut (static best response)",
                    "up":   "One-period price raise (to $p^{M}$)"}
    arm_titles = {"within": "Co-trained pair",
                  "cross": "Independently trained (cross-paired)"}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex=True, sharey=True)
    for r, shock in enumerate(("down", "up")):
        for c, arm in enumerate(("within", "cross")):
            ax = axes[r][c]
            sub = df[(df["shock"] == shock) & (df["arm"] == arm)]
            piv0 = sub.pivot_table(index=["pair", "sim"], columns="t",
                                   values="price_0")
            piv1 = sub.pivot_table(index=["pair", "sim"], columns="t",
                                   values="price_1")
            t = piv0.columns.values

            _benchmark_lines(ax)
            ax.axvline(0, color=GRAY, lw=0.8, alpha=0.6)
            for piv, color, label in ((piv0, RED, "Shocked agent"),
                                      (piv1, BLUE, "Rival")):
                m, s = piv.mean(axis=0).values, piv.std(axis=0).values
                ax.fill_between(t, m - s, m + s, color=color, alpha=0.15, lw=0)
                ax.plot(t, m, color=color, lw=1.8, label=label)

            if r == 0:
                ax.set_title(arm_titles[arm], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{shock_titles[shock]}\n\nPrice", fontsize=9)
            if c == 1:
                ax.text(t[-1] + 0.5, P_NE, "$p^{NE}$", va="center",
                        color=GRAY, fontsize=8.5)
                ax.text(t[-1] + 0.5, P_M, "$p^{M}$", va="center",
                        color=GRAY, fontsize=8.5)
            ax.set_xlim(t[0], t[-1] + 4)

    for ax in axes[1]:
        ax.set_xlabel("Periods relative to shock")
    axes[0][0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("Impulse response functions — mean ± 1 s.d. "
                 "(18 pairs × 10 simulations)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, "fig_irf")


# ------------------------------------------------------------------
# 2. Overlay grim vs nash — grade de pares
# ------------------------------------------------------------------

def fig_overlay():
    pair_dirs = [d for d in sorted(IR_DIR.iterdir())
                 if (d / "overlay_grim_player.csv").exists()]
    n_show = min(6, len(pair_dirs))
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.6), sharey=True)

    for ax, d in zip(axes.ravel(), pair_dirs[:n_show]):
        grim = pd.read_csv(d / "overlay_grim_player.csv")
        nash = pd.read_csv(d / "overlay_nash_player.csv")
        _benchmark_lines(ax)
        ax.plot(nash["t"], nash["price_agent"], color=BLUE, lw=2.4,
                label="vs nash", solid_capstyle="round")
        ax.plot(grim["t"], grim["price_agent"], color=RED, lw=1.2,
                ls=(0, (4, 3)), label="vs grim")
        ax.set_title(d.name.replace("pair_", "agent "), fontsize=9)
        ax.set_xlim(0, 60)

    for ax in axes[-1]:
        ax.set_xlabel("Step", fontsize=9)
    axes[0][0].set_ylabel("Price")
    axes[1][0].set_ylabel("Price")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Trained agent's price path: grim trigger vs constant Nash opponent",
                 fontsize=10.5)
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, "fig_grim_vs_nash")

    # Quantificação: fração de pares com trajetórias idênticas
    same = 0
    for d in pair_dirs:
        g = pd.read_csv(d / "overlay_grim_player.csv")["price_agent"].values
        n = pd.read_csv(d / "overlay_nash_player.csv")["price_agent"].values
        same += int(np.allclose(g, n))
    print(f"  → trajetórias grim ≡ nash em {same}/{len(pair_dirs)} agentes")
    return same, len(pair_dirs)


# ------------------------------------------------------------------
# 3. CE-gap interpretável
# ------------------------------------------------------------------

def fig_ce_gap():
    df = pd.read_csv(ROOT / "projects" / "p1_deterrence" / "analysis" / "interpretable_ce_gap.csv")
    df = df.sort_values("gain_pct").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(range(len(df)), df["gain_pct"], color=BLUE, width=0.72)
    mean = df["gain_pct"].mean()
    ax.axhline(mean, color=RED, lw=1.4, ls="--")
    ax.text(0.4, mean + 1.0, f"mean {mean:.1f}%", color=RED, fontsize=9)
    ax.set_xlabel("Trained agents (sorted)")
    ax.set_ylabel("Foregone profit (%)")
    ax.set_title("Profit left on the table: exact best response vs converged play,\n"
                 "against the actual training partner", fontsize=10.5)
    ax.set_xticks([])
    _save(fig, "fig_ce_gap_interpretable")


if __name__ == "__main__":
    print("Gerando figuras:")
    fig_impulse()
    fig_irf()
    fig_overlay()
    fig_ce_gap()
