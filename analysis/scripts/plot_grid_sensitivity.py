"""
EXP-02: Plot grid granularity sensitivity results.

Reads: results/grid_sensitivity__<timestamp>/aggregate/summary_table.csv
       (or results/grid_sensitivity__<timestamp>/aggregate/all_seeds.csv)

Produces:
  figures/fig_grid_sensitivity_ci.pdf   — CI mean ± std vs log(n_prices)
  figures/fig_grid_sensitivity_dist.pdf — CI distributions per level (violin)

Usage:
  python analysis/scripts/plot_grid_sensitivity.py --sweep_dir results/grid_sensitivity__*/
  python analysis/scripts/plot_grid_sensitivity.py --sweep_dir results/grid_sensitivity__20260622_...
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_NE = 1.4729
P_M  = 1.9250

BLUE   = "#2196F3"
RED    = "#FF5722"
GRAY   = "#757575"
PURPLE = "#9C27B0"


def load_data(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary_table.csv and all_seeds.csv from aggregate dir."""
    agg = sweep_dir / "aggregate"
    summary = pd.read_csv(agg / "summary_table.csv")
    seeds   = pd.read_csv(agg / "all_seeds.csv")
    return summary, seeds


def plot_ci_line(summary: pd.DataFrame, seeds: pd.DataFrame, out: Path) -> None:
    """CI mean ± 95% CI band vs log n_prices, plus individual seed scatter."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_vals = summary["n_prices"].values
    mu     = summary["ci_mean"].values
    sd     = summary["ci_std"].values
    n_s    = summary["n_seeds"].values
    se95   = 1.96 * sd / np.sqrt(n_s)

    # Seed-level scatter
    jitter_rng = np.random.default_rng(42)
    for _, row in seeds.iterrows():
        x_jitter = np.log10(row["n_prices"]) + jitter_rng.uniform(-0.04, 0.04)
        ax.scatter(x_jitter, row["ci"], color=BLUE, alpha=0.25, s=10, zorder=1)

    # Mean line + 95% CI band
    x = np.log10(n_vals)
    ax.plot(x, mu, color=BLUE, lw=2.5, marker="o", ms=7, zorder=3, label="Mean CI")
    ax.fill_between(x, mu - se95, mu + se95, color=BLUE, alpha=0.2, label="95% CI")

    # Baseline marker
    baseline_idx = list(n_vals).index(15) if 15 in n_vals else None
    if baseline_idx is not None:
        ax.scatter([x[baseline_idx]], [mu[baseline_idx]],
                   marker="D", color="black", s=60, zorder=4, label="Calvano baseline (n=15)")

    ax.axhline(0.0, color=GRAY,   lw=1.0, ls="--", label="Nash (CI=0)")
    ax.axhline(1.0, color=PURPLE, lw=0.8, ls=":",  label="Monopoly (CI=1)")

    ax.set_xlabel("log₁₀(n_prices)", fontsize=11)
    ax.set_ylabel("Collusion Index (CI)", fontsize=11)
    ax.set_title("Grid Granularity vs Collusion\n"
                 "Finer grids → less off-path coverage → less collusion",
                 fontsize=10)
    ax.set_xticks(np.log10(n_vals))
    ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.15, 1.05)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_ci_violin(seeds: pd.DataFrame, out: Path) -> None:
    """CI distribution per n_prices level as violin + strip plot."""
    n_levels = sorted(seeds["n_prices"].unique())
    data = [seeds[seeds["n_prices"] == n]["ci"].values for n in n_levels]

    fig, ax = plt.subplots(figsize=(9, 5))

    parts = ax.violinplot(data, positions=range(len(n_levels)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.4)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Scatter overlay
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        jx = rng.uniform(-0.1, 0.1, size=len(d))
        ax.scatter(i + jx, d, color=BLUE, alpha=0.5, s=15, zorder=3)

    ax.axhline(0.0, color=GRAY,   lw=1.2, ls="--", label="Nash (CI=0)")
    ax.axhline(1.0, color=PURPLE, lw=0.8, ls=":",  label="Monopoly (CI=1)")

    ax.set_xticks(range(len(n_levels)))
    ax.set_xticklabels([f"n={n}" for n in n_levels], fontsize=9)
    ax.set_xlabel("Price grid size (n_prices)", fontsize=11)
    ax.set_ylabel("Collusion Index (CI)", fontsize=11)
    ax.set_title("CI Distribution by Grid Size\n(N=2, Q-learning, rival_prices, 20 seeds each)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.15, 1.05)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot EXP-02 grid sensitivity results")
    p.add_argument("--sweep_dir", required=True,
                   help="Path to grid_sensitivity__ results dir (glob OK)")
    p.add_argument("--out_dir",   default=str(OUT_DIR))
    args = p.parse_args()

    # Resolve glob
    candidates = sorted(glob.glob(args.sweep_dir))
    if not candidates:
        raise FileNotFoundError(f"No directories match: {args.sweep_dir}")
    sweep_dir = Path(candidates[-1])  # most recent if glob
    print(f"Using sweep dir: {sweep_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, seeds = load_data(sweep_dir)
    print(summary.to_string(index=False))

    plot_ci_line(summary, seeds, out_dir / "fig_grid_sensitivity_ci.pdf")
    plot_ci_violin(seeds, out_dir / "fig_grid_sensitivity_dist.pdf")


if __name__ == "__main__":
    main()
