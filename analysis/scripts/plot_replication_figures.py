"""
Generate replication figures for Seção 2 do paper (FIG-SET-A).

Usage:
    python plot_replication_figures.py [--sweep-dir RESULTS/calvano_replication__*]

Outputs (figures/):
    fig_replication_learning_curve.pdf   — CI médio ± 1std ao longo do treinamento
    fig_replication_ci_hist.pdf          — Distribuição de CI final (bimodalidade)
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


ROOT = _repo_root()
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Calvano (2020) reported interval and our replication result
CALVANO_CI_LOW, CALVANO_CI_HIGH = 0.60, 0.85
OUR_CI_MEAN = 0.6245


def find_sweep_dir(pattern: str | None) -> Path:
    if pattern:
        return Path(pattern)
    candidates = sorted(glob.glob(str(RESULTS / "calvano_replication__*")))
    dirs = [p for p in candidates if Path(p).is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No calvano_replication__ dirs found in {RESULTS}")
    return Path(dirs[-1])  # most recent


def load_histories(sweep_dir: Path) -> list[pd.DataFrame]:
    frames = []
    for seed_dir in sorted(sweep_dir.glob("seed_*")):
        csv = seed_dir / "metrics_history.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed"] = seed_dir.name
            frames.append(df)
    return frames


def plot_learning_curve(frames: list[pd.DataFrame], out_dir: Path) -> None:
    if not frames:
        print("  [skip] no metrics_history.csv found")
        return

    # align on common steps
    all_steps = sorted(set.intersection(*[set(df["step"].values) for df in frames]))
    ci_matrix = np.array([
        df[df["step"].isin(all_steps)].set_index("step")["collusion_index"].reindex(all_steps).values
        for df in frames
    ])

    steps = np.array(all_steps)
    mu = np.nanmean(ci_matrix, axis=0)
    sd = np.nanstd(ci_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.fill_between(steps / 1e6, mu - sd, mu + sd, alpha=0.25, color="#2166ac", label="±1 std")
    ax.plot(steps / 1e6, mu, color="#2166ac", linewidth=1.8, label=f"Mean CI (n={len(frames)} seeds)")

    ax.axhline(OUR_CI_MEAN, color="#d73027", linewidth=1.0, linestyle="--",
               label=f"Final mean CI = {OUR_CI_MEAN:.4f}")
    ax.axhspan(CALVANO_CI_LOW, CALVANO_CI_HIGH, alpha=0.08, color="green",
               label=f"Calvano target [{CALVANO_CI_LOW}, {CALVANO_CI_HIGH}]")

    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Collusion index (CI)")
    ax.set_title("Calvano replication — CI learning curve")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        path = out_dir / f"fig_replication_learning_curve{ext}"
        fig.savefig(path)
        print(f"  saved {path.name}")
    plt.close(fig)


def plot_ci_histogram(frames: list[pd.DataFrame], out_dir: Path) -> None:
    if not frames:
        print("  [skip] no metrics_history.csv found")
        return

    # use final CI per seed
    final_ci = np.array([df["collusion_index"].iloc[-1] for df in frames])
    n = len(final_ci)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # histogram
    counts, bins, _ = ax.hist(
        final_ci, bins=20,
        color="#4575b4", alpha=0.6, edgecolor="white", linewidth=0.5,
        label=f"Seed CIs (n={n})",
    )

    # KDE overlay
    x_kde = np.linspace(final_ci.min() - 0.05, final_ci.max() + 0.05, 300)
    try:
        kde = gaussian_kde(final_ci, bw_method=0.25)
        scale = counts.max() / kde(x_kde).max()
        ax.plot(x_kde, kde(x_kde) * scale, color="#d73027", linewidth=1.8, label="KDE")
    except Exception:
        pass

    # reference lines
    ax.axvspan(CALVANO_CI_LOW, CALVANO_CI_HIGH, alpha=0.10, color="green",
               label=f"Calvano target [{CALVANO_CI_LOW}, {CALVANO_CI_HIGH}]")
    ax.axvline(OUR_CI_MEAN, color="#d73027", linewidth=1.0, linestyle="--",
               label=f"Mean = {OUR_CI_MEAN:.4f}")

    ax.set_xlabel("Collusion index (CI)")
    ax.set_ylabel("Count")
    ax.set_title("Calvano replication — Final CI distribution")
    ax.legend(framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in [".pdf", ".png"]:
        path = out_dir / f"fig_replication_ci_hist{ext}"
        fig.savefig(path)
        print(f"  saved {path.name}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default=None,
                        help="Path to calvano_replication sweep dir (default: latest in results/)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sweep_dir = find_sweep_dir(args.sweep_dir)
    print(f"Using sweep dir: {sweep_dir}")

    frames = load_histories(sweep_dir)
    print(f"  loaded {len(frames)} seed histories")

    print("Generating figures...")
    plot_learning_curve(frames, args.out_dir)
    plot_ci_histogram(frames, args.out_dir)
    print(f"\nDone. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
