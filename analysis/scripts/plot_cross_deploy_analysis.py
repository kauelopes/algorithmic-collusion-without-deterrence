"""
Cross-deploy price trajectory analysis.

Loads eval episodes from /tmp/cross_deploy_analysis/ and produces:
  1. fig_cross_deploy_trajectories.pdf — within vs cross price dynamics (12-panel grid)
  2. fig_cross_deploy_ci_comparison.pdf — CI bar chart within vs cross
  3. fig_cross_deploy_taxonomy.pdf — 3 canonical pattern illustrations

Benchmarks (Calvano N=2, logit demand a=2, mu=0.25, c=1):
  p_NE = 1.4729, p_M = 1.9250, p_min = 1.0209, p_max = 2.3770
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = _repo_root() / Path("results/cross_deploy__20260621_203040/eval")
OUT_DIR  = _repo_root() / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Calvano benchmarks
P_NE  = 1.4729
P_M   = 1.9250
P_MIN = 1.0209

BLUE   = "#2196F3"
RED    = "#FF5722"
GREEN  = "#4CAF50"
PURPLE = "#9C27B0"
GRAY   = "#757575"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    episodes = {}
    for d in sorted(DATA_DIR.iterdir()):
        csv = d / "metrics_history.csv"
        if csv.exists():
            # Normalise names: within_pair_000 → within_000, cross_pair_000 → cross_000
            key = d.name.replace("within_pair_", "within_").replace("cross_pair_", "cross_")
            episodes[key] = pd.read_csv(csv)
    return episodes


def classify_episode(df: pd.DataFrame) -> str:
    """Classify cross-pair trajectory into behavioral pattern."""
    sym_q = df["sym_q"].mean()
    ci    = df["collusion_index"].mean()
    # Check for cycling: greedy_q_0 has more than 1 unique value in cycle
    cycle_len = df["greedy_q_0"].nunique()
    if cycle_len > 1:
        return f"limit_cycle_p{cycle_len}"
    elif sym_q < 0.01:
        return "symmetric_fixed"
    else:
        return "asymmetric_fixed"


# ------------------------------------------------------------------
# Figure 1: 12-panel price trajectory grid
# ------------------------------------------------------------------

def plot_trajectories(episodes: dict):
    within = {k: v for k, v in episodes.items() if k.startswith("within")}
    cross  = {k: v for k, v in episodes.items() if k.startswith("cross")}

    n_pairs = max(len(within), len(cross))
    fig, axes = plt.subplots(2, n_pairs, figsize=(n_pairs * 2.2, 6), sharey=True)
    fig.suptitle(
        "Price Trajectories: Within-pair vs Cross-pair Greedy Deployment\n"
        r"$p_{NE}$=1.473 (Nash)   $p_M$=1.925 (Monopoly/firm)",
        fontsize=11, y=1.01
    )

    def _plot_one(ax, df, title, condition):
        steps = df["step"] / 1000  # in thousands
        # Use action_mean (actual avg price played per window) — more informative than greedy snapshot
        ax.plot(steps, df["mean_q_0"], color=BLUE,  lw=1.5, label="Agent 0", zorder=3)
        ax.plot(steps, df["mean_q_1"], color=RED,   lw=1.5, label="Agent 1", zorder=3)
        ax.axhline(P_NE, color=GRAY,   lw=1.0, ls="--", zorder=2)
        ax.axhline(P_M,  color=PURPLE, lw=0.8, ls=":",  zorder=2)
        ci_val = df["collusion_index"].mean()
        ax.set_title(f"{title}\nCI={ci_val:.3f}", fontsize=8)
        ax.set_ylim(P_MIN - 0.05, P_M + 0.15)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#E3F2FD" if condition == "within" else "#FBE9E7")
        ax.tick_params(labelsize=7)

    for col, (name, df) in enumerate(sorted(within.items())):
        seed_label = name.replace("within_", "s").replace("_", "/")
        _plot_one(axes[0, col], df, seed_label, "within")

    for col, (name, df) in enumerate(sorted(cross.items())):
        seed_label = name.replace("cross_", "").replace("_", "×s")
        _plot_one(axes[1, col], df, seed_label, "cross")

    axes[0, 0].set_ylabel("Within-pair\nPrice", fontsize=9)
    axes[1, 0].set_ylabel("Cross-pair\nPrice", fontsize=9)

    # Shared x label
    for ax in axes[1]:
        ax.set_xlabel("Step (k)", fontsize=8)

    # Shared legend
    h = [
        mpatches.Patch(color=BLUE,  label="Agent 0"),
        mpatches.Patch(color=RED,   label="Agent 1"),
        plt.Line2D([0], [0], color=GRAY,   ls="--", label=f"Nash ({P_NE})"),
        plt.Line2D([0], [0], color=PURPLE, ls=":",  label=f"Monopoly ({P_M})"),
        mpatches.Patch(color="#E3F2FD", label="Within-pair (control)"),
        mpatches.Patch(color="#FBE9E7", label="Cross-pair (test)"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_cross_deploy_trajectories.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ------------------------------------------------------------------
# Figure 2: CI comparison + individual bars
# ------------------------------------------------------------------

def plot_ci_comparison(episodes: dict):
    within_ci = [df["collusion_index"].mean()
                 for k, df in episodes.items() if k.startswith("within")]
    cross_ci  = [df["collusion_index"].mean()
                 for k, df in episodes.items() if k.startswith("cross")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- left: individual bars per episode ----
    ax = axes[0]
    names_w = sorted(k.replace("within_", "") for k in episodes if k.startswith("within"))
    names_c = sorted(k.replace("cross_", "").replace("_", "×") for k in episodes if k.startswith("cross"))
    x_w = np.arange(len(within_ci))
    x_c = np.arange(len(cross_ci))

    ax.bar(x_w - 0.2, within_ci, width=0.35, color=BLUE, alpha=0.8, label="Within-pair")
    ax.bar(x_w + 0.2, cross_ci,  width=0.35, color=RED,  alpha=0.8, label="Cross-pair")
    ax.axhline(0, color="black",  lw=0.8, ls="--", label="Nash (CI=0)")
    ax.axhline(1, color=PURPLE,   lw=0.8, ls=":",  label="Monopoly (CI=1)")
    ax.set_xticks(x_w)
    ax.set_xticklabels([f"p{i}" for i in range(len(within_ci))], fontsize=9)
    ax.set_ylabel("Collusion Index (CI)")
    ax.set_title("CI per Episode Pair", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- right: summary bar within vs cross ----
    ax = axes[1]
    mu_w, sd_w = np.mean(within_ci), np.std(within_ci)
    mu_c, sd_c = np.mean(cross_ci),  np.std(cross_ci)
    collapse_pct = (1 - mu_c / mu_w) * 100 if mu_w > 0 else 0

    bars = ax.bar(["Within-pair\n(training partner)", "Cross-pair\n(unseen partner)"],
                  [mu_w, mu_c], color=[BLUE, RED], alpha=0.85,
                  yerr=[sd_w, sd_c], capsize=6, error_kw={"elinewidth": 1.5})

    ax.axhline(0, color="black",  lw=0.8, ls="--", label="Nash (CI=0)")
    ax.axhline(1, color=PURPLE,   lw=0.8, ls=":",  label="Monopoly (CI=1)")

    # Annotate bars with mean ± std
    for bar, mu, sd in zip(bars, [mu_w, mu_c], [sd_w, sd_c]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sd + 0.03,
                f"{mu:.3f}±{sd:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Annotate collapse
    ax.annotate(
        f"↓ {collapse_pct:.0f}% collapse",
        xy=(1, mu_c), xytext=(0.5, mu_c + 0.15),
        fontsize=10, color="darkred", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkred"),
        ha="center"
    )

    ax.set_ylabel("Mean Collusion Index (CI)")
    n = len(within_ci)
    ax.set_title(f"Within-pair vs Cross-pair CI\n({n} pairs, final run)", fontsize=10)
    ax.set_ylim(-0.2, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_cross_deploy_ci_comparison.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()
    print(f"  Within CI: {mu_w:.3f} ± {sd_w:.3f}")
    print(f"  Cross  CI: {mu_c:.3f} ± {sd_c:.3f}")
    print(f"  Collapse:  {collapse_pct:.1f}%")


# ------------------------------------------------------------------
# Figure 3: Taxonomy of cross-pair outcomes (3 canonical patterns)
# ------------------------------------------------------------------

def plot_taxonomy(episodes: dict):
    """Show 3 canonical cross-pair behavioral patterns as a 3-panel illustration."""
    # Pick representatives from actual 9-pair run:
    #   within_000  → control (CI≈0.661, symmetric fixed point)
    #   cross_004   → sub-Nash (CI≈-0.143, price war / asymmetric fixed point)
    #   cross_000   → limit cycle (2 unique greedy prices, CI≈0.071)
    panels = [
        ("within_000", "Symmetric Fixed Point\n(Within-pair control, CI=0.661)", "within"),
        ("cross_004",  "Sub-Nash Fixed Point\n(Cross-pair: price war, CI=−0.143)", "cross"),
        ("cross_000",  "Limit Cycle (period-2)\n(Cross-pair: failed coordination, CI=0.071)", "cross"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        "Behavioral Taxonomy: Within-pair Coordination vs Cross-pair Outcomes",
        fontsize=11
    )

    for ax, (name, title, condition) in zip(axes, panels):
        df = episodes[name]
        steps = df["step"] / 1000
        ax.plot(steps, df["mean_q_0"], color=BLUE, lw=2.0, label="Agent 0", zorder=3)
        ax.plot(steps, df["mean_q_1"], color=RED,  lw=2.0, label="Agent 1", zorder=3)
        ax.axhline(P_NE, color=GRAY,   lw=1.2, ls="--",
                   label=f"Nash ({P_NE})", zorder=2)
        ax.axhline(P_M,  color=PURPLE, lw=1.0, ls=":",
                   label=f"Monopoly ({P_M})", zorder=2)

        ci_val = df["collusion_index"].mean()
        sym_q  = df["sym_q"].mean()
        cycle  = df["greedy_q_0"].nunique()

        # Annotation box with key metrics
        txt = f"CI = {ci_val:.3f}\n|Δp| = {sym_q:.3f}\n#prices_0 = {cycle}"
        ax.text(0.97, 0.04, txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Step (k)", fontsize=9)
        ax.set_ylim(P_MIN - 0.05, P_M + 0.20)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        bg = "#E3F2FD" if condition == "within" else "#FBE9E7"
        ax.set_facecolor(bg)

    axes[0].set_ylabel("Price", fontsize=10)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_cross_deploy_taxonomy.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ------------------------------------------------------------------
# Figure 4: Detailed limit cycle decomposition
# ------------------------------------------------------------------

def plot_limit_cycle_detail(episodes: dict):
    """
    Show the period-2 limit cycle (cross_000) alongside a within-pair control (within_000),
    with time-series and phase portraits.
    Only cross_000 exhibits a limit cycle in the 9-pair run.
    """
    df_cycle   = episodes["cross_000"].copy()
    df_control = episodes["within_000"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(
        "Limit Cycle Anatomy in Cross-pair Greedy Deployment\n"
        "Deterministic price oscillation — no convergence to Nash or collusive equilibrium",
        fontsize=10
    )

    def _plot_trajectory(ax, df, name, condition):
        steps = df["step"] / 1000
        ax.plot(steps, df["mean_q_0"], color=BLUE, lw=1.8, label="Agent 0")
        ax.plot(steps, df["mean_q_1"], color=RED,  lw=1.8, label="Agent 1")
        ax.axhline(P_NE, color=GRAY,   lw=1.2, ls="--", label=f"Nash ({P_NE})")
        ax.axhline(P_M,  color=PURPLE, lw=0.8, ls=":",  label=f"Monopoly ({P_M})")
        ci_val = df["collusion_index"].mean()
        ax.set_title(f"{name}  (CI={ci_val:.3f})", fontsize=9)
        ax.set_ylabel("Price (greedy snapshot)", fontsize=8)
        ax.set_xlabel("Step (k)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#E3F2FD" if condition == "within" else "#FBE9E7")
        cycle_len = df["greedy_q_0"].nunique()
        if cycle_len > 1:
            ax.text(0.02, 0.96, f"Cycle length: {cycle_len} price levels",
                    transform=ax.transAxes, va="top", fontsize=8, color="darkred")

    def _plot_phase_portrait(ax, df, name, condition):
        p0 = df["mean_q_0"].values
        p1 = df["mean_q_1"].values
        ax.scatter(p0, p1, c=range(len(p0)), cmap="Blues", s=15, zorder=3)
        for i in range(min(len(p0) - 1, 200)):
            ax.annotate("", xy=(p0[i+1], p1[i+1]), xytext=(p0[i], p1[i]),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
        ax.axvline(P_NE, color=GRAY, lw=1.0, ls="--", alpha=0.6)
        ax.axhline(P_NE, color=GRAY, lw=1.0, ls="--", alpha=0.6, label=f"Nash ({P_NE})")
        ax.scatter([P_NE], [P_NE], marker="*", color=GRAY, s=80, zorder=5, label="Nash eq.")
        ax.set_xlabel("Agent 0 price (greedy)", fontsize=8)
        ax.set_ylabel("Agent 1 price (greedy)", fontsize=8)
        ax.set_title(f"Phase portrait: {name}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#E3F2FD" if condition == "within" else "#FBE9E7")

    _plot_trajectory(axes[0, 0], df_control, "within_000 (within-pair control)", "within")
    _plot_trajectory(axes[1, 0], df_cycle,   "cross_000  (limit cycle, CI=0.071)", "cross")
    _plot_phase_portrait(axes[0, 1], df_control, "within_000", "within")
    _plot_phase_portrait(axes[1, 1], df_cycle,   "cross_000",  "cross")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_cross_deploy_limit_cycles.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary(episodes: dict):
    print("\n" + "="*80)
    print("CROSS-DEPLOY SMOKE TEST SUMMARY")
    print("="*80)
    print(f"{'Episode':<20} {'Type':<10} {'Mean CI':>8} {'sym_q':>8} "
          f"{'p0 mean':>8} {'p1 mean':>8} {'pattern':<20}")
    print("-"*80)
    for name, df in sorted(episodes.items()):
        kind   = "within" if name.startswith("within") else "cross"
        ci     = df["collusion_index"].mean()
        sym_q  = df["sym_q"].mean()
        p0     = df["greedy_q_0"].mean()
        p1     = df["greedy_q_1"].mean()
        n_uniq = df["greedy_q_0"].nunique()
        if kind == "within":
            pattern = "symmetric_fixed"
        elif n_uniq > 1:
            pattern = f"limit_cycle_p{n_uniq}"
        elif sym_q < 0.01:
            pattern = "symmetric_fixed"
        else:
            pattern = "asymmetric_fixed"
        print(f"{name:<20} {kind:<10} {ci:>8.3f} {sym_q:>8.3f} "
              f"{p0:>8.4f} {p1:>8.4f} {pattern:<20}")
    print("-"*80)
    within_ci = [df["collusion_index"].mean() for k, df in episodes.items()
                 if k.startswith("within")]
    cross_ci  = [df["collusion_index"].mean() for k, df in episodes.items()
                 if k.startswith("cross")]
    print(f"\nWithin-pair CI:  {np.mean(within_ci):.3f} ± {np.std(within_ci):.3f}")
    print(f"Cross-pair  CI:  {np.mean(cross_ci):.3f} ± {np.std(cross_ci):.3f}")
    print(f"CI collapse:     {(1 - np.mean(cross_ci) / np.mean(within_ci))*100:.1f}%")
    print(f"\nCross-pair pattern breakdown:")
    cross_eps = {k: v for k, v in episodes.items() if k.startswith("cross")}
    n_cycle   = sum(1 for df in cross_eps.values() if df["greedy_q_0"].nunique() > 1)
    n_asym    = sum(1 for df in cross_eps.values()
                    if df["greedy_q_0"].nunique() == 1 and df["sym_q"].mean() > 0.01)
    n_sym     = len(cross_eps) - n_cycle - n_asym
    print(f"  Limit cycles:        {n_cycle}")
    print(f"  Asymmetric fixed:    {n_asym}")
    print(f"  Symmetric fixed:     {n_sym}")
    print("="*80)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    episodes = load_all()
    print(f"Loaded {len(episodes)} episodes: {sorted(episodes.keys())}")
    print_summary(episodes)
    plot_trajectories(episodes)
    plot_ci_comparison(episodes)
    plot_taxonomy(episodes)
    plot_limit_cycle_detail(episodes)
    print("\nAll figures saved to figures/")
