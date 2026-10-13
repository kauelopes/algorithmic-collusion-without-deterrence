"""
Report generator for strategy test episodes.

Produces:
  strategy_test_summary.md       — markdown table with CI per strategy opponent
  plots/strategy_grid.png        — grid of price-dynamics subplots (overview)
  strategies/{name}.md           — per-strategy rich report with:
      plots/{name}_trajectory.png    price & profit time series
      plots/{name}_histogram.png     price distribution
      plots/{name}_acf.png           serial autocorrelation of trained-agent prices
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_ci(episode: dict, p_NE: float, p_monopoly: float) -> float:
    mean_price = float(np.mean(episode["price_a"] + episode["price_b"]))
    denom = p_monopoly - p_NE
    if abs(denom) < 1e-10:
        return 0.0
    return (mean_price - p_NE) / denom


def _ci_label(ci: float) -> str:
    if ci < 0.05:
        return "Nash"
    if ci < 0.25:
        return "mild collusion"
    if ci < 0.75:
        return "strong collusion"
    return "near monopoly"


def _rolling_mean(arr: list[float], window: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


def _acf(arr: list[float], max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, dtype=float)
    a = a - a.mean()
    var = np.var(a)
    lags = np.arange(1, max_lag + 1)
    acf_vals = np.array([
        np.mean(a[lag:] * a[:-lag]) / var if var > 1e-12 else 0.0
        for lag in lags
    ])
    return lags, acf_vals


def _ccf(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation between a and b at lags -max_lag … +max_lag.
    Positive lag k: corr(a[k:], b[:-k]) — a responds to b k steps later.
    Negative lag k: corr(a[:-|k|], b[|k|:]) — b responds to a |k| steps later.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.std(a) * np.std(b)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_vals = []
    for lag in lags:
        if lag == 0:
            v = np.mean(a * b)
        elif lag > 0:
            v = np.mean(a[lag:] * b[:-lag])
        else:
            v = np.mean(a[:lag] * b[-lag:])
        ccf_vals.append(v / denom if denom > 1e-12 else 0.0)
    return lags, np.array(ccf_vals)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    episode_results: list[dict],
    benchmarks: dict,
    output_dir: Path,
    config: dict,
) -> Path:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    strategies_dir = output_dir / "strategies"
    plots_dir.mkdir(parents=True, exist_ok=True)
    strategies_dir.mkdir(parents=True, exist_ok=True)

    _write_overview_grid(episode_results, benchmarks, plots_dir)
    _write_summary(episode_results, benchmarks, output_dir, config)

    for ep in episode_results:
        _write_strategy_report(ep, benchmarks, output_dir, config)

    return output_dir


# ---------------------------------------------------------------------------
# Overview grid (unchanged behaviour)
# ---------------------------------------------------------------------------

def _write_overview_grid(
    episode_results: list[dict],
    benchmarks: dict,
    plots_dir: Path,
) -> None:
    p_NE = float(benchmarks["p_NE"])
    p_monopoly = float(benchmarks["p_monopoly"])

    n = len(episode_results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Trained Agent vs. Strategy Opponents — Price Dynamics", fontsize=12, y=1.01)

    for idx, ep in enumerate(episode_results):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        steps = ep["steps"]
        ci = _episode_ci(ep, p_NE, p_monopoly)

        ax.plot(steps, ep["price_a"], color="#2196F3", lw=0.8, label="trained agent", alpha=0.75)
        ax.plot(steps, ep["price_b"], color="#FF6F00", lw=0.8, label="strategy", alpha=0.75)
        ax.axhline(p_NE,       color="black",   lw=1.0, ls="--", label=f"p_NE={p_NE:.3f}")
        ax.axhline(p_monopoly, color="#7B1FA2", lw=0.8, ls="--", label=f"p_M={p_monopoly:.3f}")

        ax.set_title(f"{ep['strategy_name']}  CI={ci:.3f}", fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Price", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(plots_dir / "strategy_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _write_summary(
    episode_results: list[dict],
    benchmarks: dict,
    output_dir: Path,
    config: dict,
) -> None:
    p_NE = float(benchmarks["p_NE"])
    p_monopoly = float(benchmarks["p_monopoly"])

    lines = [
        "# Strategy Test Summary\n",
        f"Experiment: `{config.get('experiment_name', 'strategy_test')}`\n",
        "| Strategy | Mean Price (trained) | Mean Price (opponent) | CI | Notes |",
        "|----------|---------------------|----------------------|-----|-------|",
    ]

    for ep in episode_results:
        name = ep["strategy_name"]
        mean_a = float(np.mean(ep["price_a"]))
        mean_b = float(np.mean(ep["price_b"]))
        ci = _episode_ci(ep, p_NE, p_monopoly)
        lines.append(
            f"| [{name}](strategies/{name}.md) | {mean_a:.4f} | {mean_b:.4f} "
            f"| {ci:.4f} | {_ci_label(ci)} |"
        )

    lines += [
        "",
        f"Benchmarks: p_NE={p_NE:.4f}, p_monopoly={p_monopoly:.4f}",
        "",
        "![Strategy Grid](plots/strategy_grid.png)",
    ]

    (output_dir / "strategy_test_summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Per-strategy rich report
# ---------------------------------------------------------------------------

def _write_strategy_report(
    ep: dict,
    benchmarks: dict,
    output_dir: Path,
    config: dict,
) -> None:
    name = ep["strategy_name"]
    p_NE      = float(benchmarks["p_NE"])
    p_monopoly = float(benchmarks["p_monopoly"])
    p_comp     = float(benchmarks.get("p_competitive", benchmarks.get("p_NE")))

    price_a  = np.asarray(ep["price_a"],  dtype=float)
    price_b  = np.asarray(ep["price_b"],  dtype=float)
    profit_a = np.asarray(ep["profit_a"], dtype=float)
    profit_b = np.asarray(ep["profit_b"], dtype=float)
    steps    = np.asarray(ep["steps"],    dtype=int)
    n_steps  = len(steps)

    ci         = _episode_ci(ep, p_NE, p_monopoly)
    ci_a_only  = (np.mean(price_a) - p_NE) / (p_monopoly - p_NE) if abs(p_monopoly - p_NE) > 1e-10 else 0.0
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # -- 1. Price trajectory --------------------------------------------------
    _plot_trajectory(
        steps, price_a, price_b, profit_a, profit_b,
        p_NE, p_monopoly, name, ci,
        plots_dir / f"{name}_trajectory.png",
    )

    # -- 2. Price histogram ---------------------------------------------------
    _plot_histogram(
        price_a, price_b, p_NE, p_monopoly, p_comp, name,
        plots_dir / f"{name}_histogram.png",
    )

    # -- 3. ACF ---------------------------------------------------------------
    max_lag = min(100, n_steps // 10)
    _plot_acf(price_a, max_lag, name, plots_dir / f"{name}_acf.png")

    # -- 4. CCF ---------------------------------------------------------------
    _plot_ccf(price_a, price_b, max_lag, name, plots_dir / f"{name}_ccf.png")

    # -- 5. Markdown ----------------------------------------------------------
    _write_strategy_md(
        ep, benchmarks, output_dir, config,
        ci, ci_a_only, price_a, price_b, profit_a,
    )


_TRAJECTORY_WINDOW = 100  # max steps shown in trajectory plots


def _plot_trajectory(
    steps, price_a, price_b, profit_a, profit_b,
    p_NE, p_monopoly, name, ci, out_path,
) -> None:
    # Show only the first _TRAJECTORY_WINDOW steps for readability
    n_show = min(_TRAJECTORY_WINDOW, len(steps))
    steps    = steps[:n_show]
    price_a  = price_a[:n_show]
    price_b  = price_b[:n_show]
    profit_a = profit_a[:n_show]
    profit_b = profit_b[:n_show]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"Trained Agent vs {name}  —  CI={ci:.3f}  (first {n_show} steps)",
        fontsize=12,
    )

    # Price panel
    ax1.plot(steps, price_a, color="#2196F3", lw=1.2, alpha=0.85, label="trained agent")
    ax1.plot(steps, price_b, color="#FF6F00", lw=1.2, alpha=0.85, label="strategy")
    ax1.axhline(p_NE,       color="black",   lw=1.1, ls="--", label=f"p_NE={p_NE:.3f}")
    ax1.axhline(p_monopoly, color="#7B1FA2", lw=0.9, ls=":",  label=f"p_M={p_monopoly:.3f}")
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # Profit panel
    ax2.plot(steps, profit_a, color="#2196F3", lw=1.2, alpha=0.85, label="trained agent")
    ax2.plot(steps, profit_b, color="#FF6F00", lw=1.2, alpha=0.85, label="strategy")
    ax2.set_ylabel("Profit")
    ax2.set_xlabel("Step")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(
    price_a, price_b, p_NE, p_monopoly, p_comp, name, out_path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    bins = 60
    ax.hist(price_a, bins=bins, color="#2196F3", alpha=0.6, label="trained agent", density=True)
    ax.hist(price_b, bins=bins, color="#FF6F00", alpha=0.5, label="strategy",      density=True)

    ax.axvline(p_NE,       color="black",   lw=1.4, ls="--", label=f"p_NE={p_NE:.3f}")
    ax.axvline(p_monopoly, color="#7B1FA2", lw=1.2, ls=":",  label=f"p_M={p_monopoly:.3f}")
    ax.axvline(p_comp,     color="#388E3C", lw=1.0, ls="-.", label=f"p_comp={p_comp:.3f}")

    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.set_title(f"Price Distribution — vs {name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ccf(price_a, price_b, max_lag, name, out_path) -> None:
    lags, ccf_vals = _ccf(price_a, price_b, max_lag)
    ci_bound = 1.96 / math.sqrt(len(price_a))

    colors = np.where(np.abs(ccf_vals) > ci_bound, "#1565C0", "#90CAF9")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags, ccf_vals, color=colors, alpha=0.85, width=0.8)
    ax.axhline(0,          color="black",   lw=0.8)
    ax.axhline( ci_bound,  color="#D32F2F", lw=1.0, ls="--",
                label=f"±1.96/√n ({ci_bound:.3f})")
    ax.axhline(-ci_bound,  color="#D32F2F", lw=1.0, ls="--")
    ax.axvline(0,          color="grey",    lw=0.8, ls=":")
    ax.set_xlabel("Lag  (positive → trained leads strategy)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"Cross-correlation: Trained Agent ↔ Strategy — vs {name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_acf(price_a, max_lag, name, out_path) -> None:
    lags, acf_vals = _acf(price_a, max_lag)
    ci_bound = 1.96 / math.sqrt(len(price_a))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, acf_vals, color="#2196F3", alpha=0.75, width=0.8)
    ax.axhline(0,          color="black", lw=0.8)
    ax.axhline( ci_bound,  color="#D32F2F", lw=1.0, ls="--", label=f"±1.96/√n ({ci_bound:.3f})")
    ax.axhline(-ci_bound,  color="#D32F2F", lw=1.0, ls="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(f"Serial Autocorrelation of Trained-Agent Price — vs {name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_strategy_md(
    ep, benchmarks, output_dir, config,
    ci, ci_a_only, price_a, price_b, profit_a,
) -> None:
    name       = ep["strategy_name"]
    p_NE       = float(benchmarks["p_NE"])
    p_monopoly = float(benchmarks["p_monopoly"])

    mean_a  = float(np.mean(price_a))
    std_a   = float(np.std(price_a))
    mean_b  = float(np.mean(price_b))
    std_b   = float(np.std(price_b))
    mean_pi = float(np.mean(profit_a))
    std_pi  = float(np.std(profit_a))

    # Fraction of steps where trained agent undercuts / matches / overcuts strategy
    n = len(price_a)
    undercut = float(np.sum(price_a < price_b - 1e-6) / n)
    overcut  = float(np.sum(price_a > price_b + 1e-6) / n)
    match    = 1.0 - undercut - overcut

    # Price convergence: std of last 10% vs first 10%
    tail = max(1, n // 10)
    vol_early = float(np.std(price_a[:tail]))
    vol_late  = float(np.std(price_a[-tail:]))

    lags, acf_vals = _acf(list(price_a), min(10, n // 10))
    acf_str = "  ".join(f"lag{l}={v:.3f}" for l, v in zip(lags[:5], acf_vals[:5]))

    lines = [
        f"# Strategy Report: {name}",
        "",
        f"Experiment: `{config.get('experiment_name', 'strategy_test')}`  "
        f"— {len(price_a):,} steps",
        "",
        "## Summary statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| CI (joint mean) | {ci:.4f} |",
        f"| CI (trained only) | {ci_a_only:.4f} |",
        f"| Label | {_ci_label(ci)} |",
        f"| Mean price — trained | {mean_a:.4f} ± {std_a:.4f} |",
        f"| Mean price — strategy | {mean_b:.4f} ± {std_b:.4f} |",
        f"| p_NE | {p_NE:.4f} |",
        f"| p_monopoly | {p_monopoly:.4f} |",
        f"| Mean profit — trained | {mean_pi:.4f} ± {std_pi:.4f} |",
        f"| Steps undercutting | {undercut:.1%} |",
        f"| Steps matching | {match:.1%} |",
        f"| Steps overcutting | {overcut:.1%} |",
        f"| Price volatility (early 10%) | {vol_early:.4f} |",
        f"| Price volatility (late 10%) | {vol_late:.4f} |",
        f"| ACF lags 1–5 | {acf_str} |",
        "",
        "## Price & Profit Trajectory",
        "",
        f"![Trajectory](../plots/{name}_trajectory.png)",
        "",
        "Raw series (thin, transparent) and rolling mean (bold). "
        "Dashed lines mark Nash and monopoly benchmarks.",
        "",
        "## Price Distribution",
        "",
        f"![Histogram](../plots/{name}_histogram.png)",
        "",
        "Density histogram of prices over the full episode. "
        "Vertical lines: Nash (black dashed), monopoly (purple dotted), competitive (green dash-dot).",
        "",
        "## Serial Autocorrelation (Trained Agent Prices)",
        "",
        f"![ACF](../plots/{name}_acf.png)",
        "",
        "Bars show ACF at each lag. Red dashed lines are the 95% significance bounds (±1.96/√n). "
        "Significant positive autocorrelation indicates price persistence / trending; "
        "negative autocorrelation indicates oscillation.",
        "",
        "## Cross-correlation: Trained Agent ↔ Strategy",
        "",
        f"![CCF](../plots/{name}_ccf.png)",
        "",
        "Cross-correlation at lags −max_lag … +max_lag. "
        "Positive lag k: trained-agent price at t+k correlates with strategy price at t (strategy leads). "
        "Negative lag k: trained-agent price at t correlates with strategy price at t+|k| (trained leads). "
        "Dark blue bars exceed the 95% significance bound.",
        "",
        "---",
        "",
        f"← [Back to summary](../strategy_test_summary.md)",
    ]

    out_path = output_dir / "strategies" / f"{name}.md"
    out_path.write_text("\n".join(lines) + "\n")
