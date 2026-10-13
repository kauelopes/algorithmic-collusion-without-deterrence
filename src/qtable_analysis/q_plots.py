"""
q_plots.py — Funções de visualização para análise de Q-tables (Calvano replication)

Recebe dados já calculados (matrizes, trajetórias, pontos fixos) e produz plots.
Nenhuma função faz cálculo pesado — use q_commons.py para isso.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .q_commons import P_NE, P_M, ci


# ── Helpers visuais internos ──────────────────────────────────────────────────

def _ref_lines(ax: Axes, p_ne: float, p_m: float,
               orientation: str = "both", color_ne="royalblue", color_m="mediumpurple"):
    """Adiciona linhas de referência Nash e Monopólio ao eixo."""
    for val, color, ls in [(p_ne, color_ne, "--"), (p_m, color_m, ":")]:
        kw = dict(color=color, ls=ls, lw=1.0, alpha=0.7)
        if orientation in ("h", "both"):
            ax.axhline(val, label=f"Nash ({p_ne})" if val == p_ne else f"Monop. ({p_m})", **kw)
        if orientation in ("v", "both"):
            ax.axvline(val, **kw)


def _price_ticks(ax: Axes, price_grid: np.ndarray, step: int = 2):
    """Aplica ticks de preço formatados nos dois eixos."""
    ticks = list(range(0, len(price_grid), step))
    labels = [f"{price_grid[k]:.2f}" for k in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)


def _mark_fps(ax: Axes, fps: list[dict], price_space: bool = True):
    """
    Sobrepõe pontos fixos conjuntos no eixo.

    price_space=True  → coordenadas (p_B, p_A)  (para plots em preço)
    price_space=False → coordenadas (j, i)        (para imshow em índice)
    """
    if not fps:
        return
    xs = [fp["p_B"] if price_space else fp["j"] for fp in fps]
    ys = [fp["p_A"] if price_space else fp["i"] for fp in fps]
    ax.scatter(xs, ys, marker="*", s=280, color="gold", edgecolors="black",
               linewidths=1.2, zorder=10, label=f"Ponto fixo (n={len(fps)})")


# ── Política Greedy ───────────────────────────────────────────────────────────

def plot_policy_map(
    greedy_matrix: np.ndarray,
    price_grid: np.ndarray,
    fps: list[dict] | None = None,
    title: str = "Política Greedy",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap RdYlGn da política greedy com linhas Nash/Monop e FPs marcados."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    p_lo, p_hi = price_grid[0], price_grid[-1]
    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))

    im = ax.imshow(greedy_matrix, origin="lower", cmap="RdYlGn",
                   vmin=p_lo, vmax=p_hi, aspect="auto")
    plt.colorbar(im, ax=ax, label="Preço greedy escolhido")

    for idx, color, label in [
        (ne_idx, "royalblue",    f"Nash ({P_NE})"),
        (pm_idx, "mediumpurple", f"Monop. ({P_M})"),
    ]:
        ax.axhline(idx, color=color, lw=1.2, ls="--", alpha=0.7, label=label)
        ax.axvline(idx, color=color, lw=1.2, ls="--", alpha=0.7)

    if fps:
        n = len(price_grid)
        _mark_fps(ax, fps, price_space=False)

    _price_ticks(ax, price_grid)
    ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper left")
    return ax


def plot_best_response_curve(
    greedy_fn,
    price_grid: np.ndarray,
    own_price: float | None = None,
    title: str = "Curva de Best-Response",
    config: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Curva de best-response com eixo y em CI.
    Se config fornecido, adiciona lucro π no eixo y direito.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    p_lo, p_hi = price_grid[0], price_grid[-1]
    own = own_price if own_price is not None else P_M
    br = np.array([greedy_fn(own, rp) for rp in price_grid])

    ax.plot(price_grid, br, "o-", color="steelblue", lw=2,
            label=f"Greedy (own={own:.3f})")
    ax.plot([p_lo, p_hi], [p_lo, p_hi], color="dimgray", lw=1.0, ls="-",
            alpha=0.5, label="Identidade (y=x)")
    ax.axhline(P_NE, color="royalblue",    ls="--", lw=1.2, label=f"Nash ({P_NE})")
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1.2, label=f"Monop. ({P_M})")
    ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço escolhido")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0)

    if config is not None:
        from .q_commons import compute_profit
        profits = np.array([compute_profit(br[j], price_grid[j], config)
                            for j in range(len(price_grid))])
        ax2 = ax.twinx()
        ax2.plot(price_grid, profits, "s--", color="darkorange", lw=1.5,
                 ms=5, alpha=0.85, label="Lucro π")
        ax2.set_ylabel("Lucro π", color="darkorange", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=8)
        ax2.legend(fontsize=8, loc="lower right")

    unique = len(np.unique(br))
    print(f"Preços únicos retornados: {unique}  "
          f"({'invariante — SCE puro' if unique == 1 else 'varia com rival'})")
    return ax


# ── Q-values ──────────────────────────────────────────────────────────────────

def plot_qvalue_heatmaps(
    max_q: np.ndarray,
    spread: np.ndarray,
    price_grid: np.ndarray,
    title: str = "",
) -> Figure:
    """Dois painéis: max Q-value e spread (max - 2º max) por estado."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))

    for ax, mat, cmap, label, t in [
        (axes[0], max_q,  "viridis", "max Q(s,·)",    "Max Q-value por estado"),
        (axes[1], spread, "hot",     "Q_max − Q_2nd", "Spread (max − 2º max)"),
    ]:
        im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=label)
        c1, c2 = ("white", "yellow") if cmap == "viridis" else ("cyan", "yellow")
        ax.axhline(ne_idx, color=c1, lw=1.2, ls="--", alpha=0.7, label="Nash")
        ax.axvline(ne_idx, color=c1, lw=1.2, ls="--", alpha=0.7)
        ax.axhline(pm_idx, color=c2, lw=1.2, ls=":",  alpha=0.7, label="Monop.")
        ax.axvline(pm_idx, color=c2, lw=1.2, ls=":",  alpha=0.7)
        _price_ticks(ax, price_grid)
        ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
        ax.set_title(t); ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_onpath_offpath(
    max_q: np.ndarray,
    spread: np.ndarray,
    n_actions: int,
    title: str = "",
) -> Figure:
    """Histogramas on-path (diagonal ±2) vs off-path para max Q e spread."""
    on_mask  = np.array([[abs(i - j) <= 2 for j in range(n_actions)]
                         for i in range(n_actions)])
    off_mask = ~on_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mat, t, xlabel in [
        (axes[0], max_q,  "Distribuição de Max Q-values", "max Q(s,·)"),
        (axes[1], spread, "Distribuição de Spread",       "Q_max − Q_2nd"),
    ]:
        ax.hist(mat[on_mask],  bins=20, alpha=0.7, color="steelblue",
                label=f"On-path (n={on_mask.sum()})")
        ax.hist(mat[off_mask], bins=20, alpha=0.7, color="tomato",
                label=f"Off-path (n={off_mask.sum()})")
        ax.set_xlabel(xlabel); ax.set_ylabel("Frequência")
        ax.set_title(t); ax.legend(); ax.grid(True, alpha=0.3)

    print(f"Max Q  — on-path:  {max_q[on_mask].mean():.4f} ± {max_q[on_mask].std():.4f}")
    print(f"Max Q  — off-path: {max_q[off_mask].mean():.4f} ± {max_q[off_mask].std():.4f}")
    print(f"Spread — on-path:  {spread[on_mask].mean():.4f}")
    print(f"Spread — off-path: {spread[off_mask].mean():.4f}")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ── Pontos fixos ──────────────────────────────────────────────────────────────

def plot_self_fixed_states(
    self_fps: list[dict],
    greedy_matrix: np.ndarray,
    price_grid: np.ndarray,
    title: str = "",
) -> Figure:
    """Dois painéis: mapa de política com FPs individuais + CI dos FPs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))
    p_lo, p_hi = price_grid[0], price_grid[-1]

    ax = axes[0]
    im = ax.imshow(greedy_matrix, origin="lower", cmap="RdYlGn",
                   vmin=p_lo, vmax=p_hi, aspect="auto")
    plt.colorbar(im, ax=ax, label="Preço greedy")
    for idx, color, label in [
        (ne_idx, "royalblue",    "Nash"),
        (pm_idx, "mediumpurple", "Monop."),
    ]:
        ax.axhline(idx, color=color, lw=1, ls="--", alpha=0.6, label=label)
        ax.axvline(idx, color=color, lw=1, ls="--", alpha=0.6)
    if self_fps:
        ax.scatter([fp["j_rival"] for fp in self_fps],
                   [fp["i_own"]   for fp in self_fps],
                   marker="x", s=60, color="black", linewidths=1.5, label="Ponto fixo individual")
    _price_ticks(ax, price_grid)
    ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
    ax.set_title(f"Pontos fixos individuais (n={len(self_fps)})\ngreedy(own,rival)==own")
    ax.legend(fontsize=8)

    ax = axes[1]
    if self_fps:
        sc = ax.scatter(
            [fp["rival"] for fp in self_fps],
            [fp["own"]   for fp in self_fps],
            c=[fp["ci_own"] for fp in self_fps],
            cmap="RdYlGn", vmin=-0.5, vmax=1.2, s=80,
            edgecolors="k", linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label="CI")
    ax.axhline(P_NE, color="gray",        ls="--", lw=1, label=f"Nash ({P_NE})")
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1, label=f"Monop. ({P_M})")
    ax.axvline(P_NE, color="gray",        ls="--", lw=0.8, alpha=0.5)
    ax.axvline(P_M,  color="mediumpurple", ls=":",  lw=0.8, alpha=0.5)
    ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
    ax.set_title("CI nos pontos fixos individuais")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_joint_fps(
    fps: list[dict],
    price_grid: np.ndarray,
    title: str = "Pontos Fixos Conjuntos",
    ax: Axes | None = None,
) -> Axes:
    """Scatter dos pontos fixos conjuntos no espaço de preços (p_B × p_A)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    p_lo, p_hi = price_grid[0], price_grid[-1]

    ax.set_xlim(p_lo - 0.05, p_hi + 0.05)
    ax.set_ylim(p_lo - 0.05, p_hi + 0.05)
    _ref_lines(ax, P_NE, P_M)
    ax.plot([p_lo, p_hi], [p_lo, p_hi], color="lightgray", lw=0.8,
            ls="-", alpha=0.5, label="Diagonal")

    if fps:
        sc = ax.scatter(
            [fp["p_B"] for fp in fps], [fp["p_A"] for fp in fps],
            c=[fp["ci_mean"] for fp in fps],
            cmap="RdYlGn", vmin=-0.5, vmax=1.2,
            s=200, edgecolors="black", linewidths=1.5, zorder=5,
        )
        plt.colorbar(sc, ax=ax, label="CI médio")
        for fp in fps:
            ax.annotate(f"({fp['p_A']:.3f}, {fp['p_B']:.3f})\nCI={fp['ci_mean']:.2f}",
                        (fp["p_B"], fp["p_A"]), textcoords="offset points",
                        xytext=(8, 4), fontsize=7)
    else:
        ax.text(0.5, 0.5, "Nenhum ponto fixo\n(sistema oscila)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="tomato", fontweight="bold")

    ax.set_xlabel("p_B"); ax.set_ylabel("p_A")
    ax.set_title(title); ax.legend(fontsize=8)
    return ax


# ── Trajetórias e dinâmica ────────────────────────────────────────────────────

def plot_price_trajectories(
    ha: np.ndarray,
    hb: np.ndarray,
    price_grid: np.ndarray,
    fps: list[dict] | None = None,
    labels: tuple[str, str] = ("A", "B"),
    title: str = "",
    ci_mean: float | None = None,
    profit_a: np.ndarray | None = None,
    profit_b: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Trajetórias temporais de preço de dois agentes.
    Se profit_a/profit_b fornecidos, adiciona lucro no eixo y direito.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    p_lo, p_hi = price_grid[0], price_grid[-1]
    t = np.arange(len(ha))

    ax.plot(t, ha, color="steelblue", lw=1.5, label=f"Preço {labels[0]}")
    ax.plot(t, hb, color="tomato",    lw=1.5, label=f"Preço {labels[1]}")
    ax.axhline(P_NE, color="gray",        ls="--", lw=1, alpha=0.7, label=f"Nash ({P_NE})")
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1, alpha=0.7, label=f"Monop. ({P_M})")

    if fps:
        for fp in fps:
            ax.axhline(fp["p_A"], color="steelblue", lw=0.8, ls=":", alpha=0.4)
            ax.axhline(fp["p_B"], color="tomato",    lw=0.8, ls=":", alpha=0.4)

    ax.set_ylim(p_lo - 0.05, p_hi + 0.05)
    ax.set_xlabel("Período"); ax.set_ylabel("Preço")
    ci_str = f"  |  CI_médio={ci_mean:.3f}" if ci_mean is not None else ""
    ax.set_title(title + ci_str)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    if profit_a is not None and profit_b is not None:
        ax2 = ax.twinx()
        ax2.plot(t, profit_a, color="steelblue", lw=1.0, ls="--", alpha=0.55,
                 label=f"π {labels[0]}")
        ax2.plot(t, profit_b, color="tomato",    lw=1.0, ls="--", alpha=0.55,
                 label=f"π {labels[1]}")
        ax2.set_ylabel("Lucro π", color="dimgray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="dimgray", labelsize=8)
        ax2.legend(fontsize=7, loc="upper right")

    return ax


def plot_phase_diagram(
    ha: np.ndarray,
    hb: np.ndarray,
    price_grid: np.ndarray,
    fps: list[dict] | None = None,
    warmup: int = 0,
    title: str = "Diagrama de Fase",
    color_by_ci: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """
    Trajetória no espaço de preços (p_B × p_A) com ponto inicial/final.
    Se color_by_ci=True, colore cada passo pelo CI médio instantâneo.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    p_lo, p_hi = price_grid[0], price_grid[-1]

    ax.set_xlim(p_lo - 0.05, p_hi + 0.05)
    ax.set_ylim(p_lo - 0.05, p_hi + 0.05)
    _ref_lines(ax, P_NE, P_M)
    ax.plot([p_lo, p_hi], [p_lo, p_hi], color="lightgray", lw=0.8, alpha=0.5,
            label="Diagonal")

    if warmup > 0:
        ax.plot(hb[:warmup], ha[:warmup], color="lightgray", lw=0.8, alpha=0.4)

    if color_by_ci:
        ci_vals = (np.vectorize(ci)(ha[warmup:]) + np.vectorize(ci)(hb[warmup:])) / 2
        sc = ax.scatter(hb[warmup:], ha[warmup:], c=ci_vals, cmap="RdYlGn",
                        vmin=-0.5, vmax=1.2, s=25, alpha=0.85, zorder=4)
        plt.colorbar(sc, ax=ax, label="CI médio")
    else:
        ax.plot(hb[warmup:], ha[warmup:], color="steelblue", lw=1.8, alpha=0.8,
                label="Trajetória")

    ax.scatter([hb[warmup]],  [ha[warmup]],  marker="o", s=100, color="green",
               zorder=5, label="Início")
    ax.scatter([hb[-1]], [ha[-1]], marker="*", s=200, color="black",
               zorder=5, label="Fim")

    _mark_fps(ax, fps or [], price_space=True)

    ax.set_xlabel("p_B"); ax.set_ylabel("p_A")
    ax.set_title(title); ax.legend(fontsize=7)
    return ax


def plot_play_against(
    result: dict,
    title: str = "",
    config: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Visualiza resultado de play_against(): trajetórias de preço.
    Se config fornecido, adiciona lucro π no eixo y direito.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    t = range(len(result["own_prices"]))
    ax.plot(list(t), result["own_prices"],   "o-",  color="steelblue", lw=2, label="Agente (preço)")
    ax.plot(list(t), result["rival_prices"], "s--", color="tomato",    lw=2, label="Rival (preço)")
    ax.axhline(P_NE, color="gray",        ls="--", lw=1, label=f"Nash ({P_NE})")
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1, label=f"Monop. ({P_M})")
    ax.set_xlabel("Período"); ax.set_ylabel("Preço")
    ci_str = f"  |  CI_médio={result['ci_mean']:.4f}"
    ax.set_title((title + ci_str).strip(" |"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    if config is not None:
        from .q_commons import compute_profit
        profits = [compute_profit(op, rp, config)
                   for op, rp in zip(result["own_prices"], result["rival_prices"])]
        ax2 = ax.twinx()
        ax2.plot(list(t), profits, "D-", color="darkorange", lw=1.5,
                 ms=5, alpha=0.85, label="Lucro π")
        ax2.set_ylabel("Lucro π", color="darkorange", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=8)
        ax2.legend(fontsize=8, loc="lower right")
    return ax


# ── Bacia de atração ──────────────────────────────────────────────────────────

def plot_basin_heatmap(
    basin: np.ndarray,
    price_grid: np.ndarray,
    fps: list[dict] | None = None,
    title: str = "Bacia de Atração",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap RdYlGn do CI médio final por condição inicial."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))

    im = ax.imshow(basin, origin="lower", cmap="RdYlGn",
                   vmin=-0.5, vmax=1.2, aspect="auto")
    plt.colorbar(im, ax=ax, label="CI médio final")
    ax.axhline(ne_idx, color="royalblue", ls="--", lw=1, alpha=0.8, label="Nash")
    ax.axvline(ne_idx, color="royalblue", ls="--", lw=1, alpha=0.8)
    ax.axhline(pm_idx, color="white",     ls=":",  lw=1, alpha=0.7, label="Monop.")
    ax.axvline(pm_idx, color="white",     ls=":",  lw=1, alpha=0.7)
    _mark_fps(ax, fps or [], price_space=False)

    _price_ticks(ax, price_grid, step=3)
    ax.set_xlabel("Preço inicial B"); ax.set_ylabel("Preço inicial A")
    ax.set_title(title); ax.legend(fontsize=7)
    return ax


# ── Comparação de políticas ───────────────────────────────────────────────────

def plot_policy_comparison(
    gm_a: np.ndarray,
    gm_b: np.ndarray,
    price_grid: np.ndarray,
    fps_a: list[dict] | None = None,
    fps_b: list[dict] | None = None,
    labels: tuple[str, str] = ("A", "B"),
    title: str = "Comparação de Políticas",
) -> Figure:
    """Três painéis: política A, política B, diferença A − B."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    p_lo, p_hi = price_grid[0], price_grid[-1]
    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))

    for ax, mat, fps, lbl in [
        (axes[0], gm_a, fps_a, labels[0]),
        (axes[1], gm_b, fps_b, labels[1]),
    ]:
        im = ax.imshow(mat, origin="lower", cmap="RdYlGn",
                       vmin=p_lo, vmax=p_hi, aspect="auto")
        plt.colorbar(im, ax=ax, label="Preço escolhido")
        ax.axhline(ne_idx, color="royalblue",    lw=1, ls="--", alpha=0.6, label="Nash")
        ax.axvline(ne_idx, color="royalblue",    lw=1, ls="--", alpha=0.6)
        ax.axhline(pm_idx, color="white",        lw=1, ls=":",  alpha=0.6, label="Monop.")
        ax.axvline(pm_idx, color="white",        lw=1, ls=":",  alpha=0.6)
        _mark_fps(ax, fps or [], price_space=False)
        _price_ticks(ax, price_grid, step=3)
        ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
        ax.set_title(f"Política greedy {lbl}"); ax.legend(fontsize=7)

    ax = axes[2]
    diff = gm_a - gm_b
    vmax_diff = np.abs(diff).max()
    im = ax.imshow(diff, origin="lower", cmap="coolwarm", aspect="auto",
                   vmin=-vmax_diff, vmax=vmax_diff)
    plt.colorbar(im, ax=ax, label=f"GM_{labels[0]} − GM_{labels[1]}")
    ax.axhline(ne_idx, color="black", lw=1, ls="--", alpha=0.6, label="Nash")
    ax.axvline(ne_idx, color="black", lw=1, ls="--", alpha=0.6)
    _price_ticks(ax, price_grid, step=3)
    ax.set_xlabel("Preço rival"); ax.set_ylabel("Preço próprio")
    ax.set_title(f"Incompatibilidade: {labels[0]} − {labels[1]}")
    ax.legend(fontsize=7)

    print(f"Diferença média |GM_A - GM_B|: {np.abs(diff).mean():.4f}")
    print(f"Estados onde políticas coincidem: {(np.abs(diff) < 1e-9).sum()} / {diff.size}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_attractor_basins(
    attractor_id: np.ndarray,
    attractors: list[dict],
    price_grid: np.ndarray,
    title: str = "Bacias de Atração por Atrator",
    ax: Axes | None = None,
) -> Axes:
    """
    Heatmap N×N onde cada célula é colorida pelo atrator que ela converge.
    Uma cor distinta por atrator (até 10 via tab10).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    n_attr = max(len(attractors), 1)
    cmap   = plt.cm.get_cmap("tab10", n_attr)

    ax.imshow(attractor_id, origin="lower", cmap=cmap,
              vmin=-0.5, vmax=n_attr - 0.5, aspect="auto",
              interpolation="nearest")

    ne_idx = int(np.argmin(np.abs(price_grid - P_NE)))
    pm_idx = int(np.argmin(np.abs(price_grid - P_M)))
    ax.axhline(ne_idx, color="white", ls="--", lw=1.0, alpha=0.8)
    ax.axvline(ne_idx, color="white", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(pm_idx, color="white", ls=":",  lw=1.0, alpha=0.8)
    ax.axvline(pm_idx, color="white", ls=":",  lw=1.0, alpha=0.8)

    patches = []
    for a in sorted(attractors, key=lambda x: x["id"]):
        tipo  = "FP" if a["period"] == 1 else f"Ciclo-{a['period']}"
        label = f"#{a['id']} {tipo}  CI={a['ci_mean']:.2f}  bacia={a['basin_fraction']:.0%}"
        patches.append(Patch(facecolor=cmap(a["id"]), edgecolor="k",
                             linewidth=0.5, label=label))

    _price_ticks(ax, price_grid, step=3)
    ax.set_xlabel("Preço inicial B"); ax.set_ylabel("Preço inicial A")
    ax.set_title(title)
    ax.legend(handles=patches, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    return ax


def plot_attractor_characterization(
    attractors: list[dict],
    price_grid: np.ndarray,
    title: str = "Caracterização dos Atratores",
) -> Figure:
    """
    Dois painéis:
      Esquerda — ciclos desenhados no espaço de preços (p_B × p_A), uma cor por atrator.
      Direita  — barras horizontais de CI_mean com whiskers [CI_min, CI_max] e
                 anotação do tamanho de bacia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_attr = max(len(attractors), 1)
    cmap   = plt.cm.get_cmap("tab10", n_attr)
    p_lo, p_hi = price_grid[0], price_grid[-1]

    # ── Painel esquerdo: ciclos no espaço de preços ───────────────────────────
    ax = axes[0]
    ax.set_xlim(p_lo - 0.05, p_hi + 0.05)
    ax.set_ylim(p_lo - 0.05, p_hi + 0.05)
    ax.axhline(P_NE, color="royalblue",    ls="--", lw=1.0, alpha=0.6,
               label=f"Nash ({P_NE})")
    ax.axvline(P_NE, color="royalblue",    ls="--", lw=1.0, alpha=0.6)
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1.0, alpha=0.6,
               label=f"Monop. ({P_M})")
    ax.axvline(P_M,  color="mediumpurple", ls=":",  lw=1.0, alpha=0.6)
    ax.plot([p_lo, p_hi], [p_lo, p_hi], color="lightgray", lw=0.8, alpha=0.5,
            label="Diagonal")

    for a in attractors:
        color = cmap(a["id"])
        tipo  = "FP" if a["period"] == 1 else f"Ciclo-{a['period']}"
        cycle = a["cycle"]  # list of (p_A, p_B)
        p_bs  = [pb for _, pb in cycle]
        p_as  = [pa for pa, _ in cycle]

        if a["period"] == 1:
            ax.scatter(p_bs, p_as, s=200, color=color, edgecolors="black",
                       linewidths=1.2, zorder=6,
                       label=f"#{a['id']} {tipo}  CI={a['ci_mean']:.2f}  "
                             f"bacia={a['basin_fraction']:.0%}")
        else:
            # Ciclo fechado: conecta último → primeiro
            xs = p_bs + [p_bs[0]]
            ys = p_as + [p_as[0]]
            ax.plot(xs, ys, "o-", color=color, lw=2.0, ms=7, zorder=5,
                    label=f"#{a['id']} {tipo}  CI={a['ci_mean']:.2f}  "
                          f"bacia={a['basin_fraction']:.0%}")
            # Seta de direção no primeiro segmento
            if len(xs) >= 2:
                ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_xlabel("p_B"); ax.set_ylabel("p_A")
    ax.set_title("Atratores no espaço de preços")
    ax.legend(fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0)

    # ── Painel direito: CI com range e bacia ─────────────────────────────────
    ax = axes[1]
    sorted_attrs = sorted(attractors, key=lambda x: x["id"])
    y_pos   = np.arange(len(sorted_attrs))
    colors  = [cmap(a["id"]) for a in sorted_attrs]

    ax.barh(y_pos, [a["ci_mean"] for a in sorted_attrs],
            color=colors, alpha=0.8, height=0.5, zorder=3)

    for y, a in zip(y_pos, sorted_attrs):
        # Whisker [CI_min, CI_max]
        ax.plot([a["ci_min"], a["ci_max"]], [y, y],
                color="black", lw=2.0, zorder=4)
        ax.plot([a["ci_min"]], [y], "|", color="black", ms=8, mew=1.5)
        ax.plot([a["ci_max"]], [y], "|", color="black", ms=8, mew=1.5)
        # Anotação de bacia
        x_ann = max(a["ci_max"], a["ci_mean"]) + 0.04
        ax.text(x_ann, y, f"bacia={a['basin_fraction']:.0%}",
                va="center", fontsize=8)

    ax.axvline(0, color="royalblue",    ls="--", lw=1.0, alpha=0.7, label="Nash (CI=0)")
    ax.axvline(1, color="mediumpurple", ls=":",  lw=1.0, alpha=0.7, label="Monop. (CI=1)")
    ax.set_xlim(-0.6, 1.6)
    ytick_labels = []
    for a in sorted_attrs:
        tipo = "FP" if a["period"] == 1 else f"Ciclo-{a['period']}"
        ytick_labels.append(f"#{a['id']} {tipo}")
    ax.set_yticks(y_pos); ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("Collusion Index")
    ax.set_title("CI médio ± range  |  bacia de atração")
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_trajectories_grid(
    results: list[dict],
    price_grid: np.ndarray,
    fps: list[dict] | None = None,
    figsize_per_row: tuple[int, int] = (10, 3.2),
) -> Figure:
    """
    Grade de trajetórias para múltiplas condições iniciais.

    results: lista de dicts com 'ha', 'hb', 'label', 'seed', 'ci_final', 'ci_mean'
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * n),
                             sharex=False)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ci_mean_val = r.get("ci_mean", None)
        plot_price_trajectories(
            r["ha"], r["hb"], price_grid,
            fps=fps,
            labels=r.get("labels", ("A", "B")),
            title=f"Init: {r.get('label', '')} (seed={r.get('seed', 0)})",
            ci_mean=ci_mean_val,
            ax=ax,
        )
    axes[-1].set_xlabel("Período")
    fig.tight_layout()
    return fig
