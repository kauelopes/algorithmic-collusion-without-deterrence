"""
q_vector_field.py — Campo vetorial de políticas Q-table

Interpreta o sistema dinâmico de dois agentes greedy como um campo vetorial
no espaço de preços (p_A × p_B).

Vetor em cada estado (p_A, p_B):
    v_A = greedy_A(p_A, p_B) - p_A   (quanto A quer mudar seu preço)
    v_B = greedy_B(p_B, p_A) - p_B   (quanto B quer mudar seu preço)

Pontos fixos e nullclines
─────────────────────────
  Nullcline de A  (dA=0): A ficaria parado, B ainda pode se mover  → marcador azul ◆
  Nullcline de B  (dB=0): B ficaria parado, A ainda pode se mover  → marcador vermelho ◆
  Ponto fixo conjunto (dA=0 ∧ dB=0): ambos param simultaneamente   → estrela dourada ★

A intersecção das duas nullclines = pontos fixos conjuntos.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .q_commons import make_greedy_fn


# ── Campo vetorial ────────────────────────────────────────────────────────────

def compute_vector_field(
    q_a: np.ndarray,
    q_b: np.ndarray,
    price_grid: np.ndarray,
    idx_a: int = 0,
    idx_b: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o campo vetorial (dA, dB) no grid N×N de preços.

    dA[i, j] = greedy_A(price_grid[i], price_grid[j]) - price_grid[i]
    dB[i, j] = greedy_B(price_grid[j], price_grid[i]) - price_grid[j]

    Eixo 0 = p_A (linhas/y), eixo 1 = p_B (colunas/x).

    `idx_a` / `idx_b`: slots de treino das Q-tables — ver
    q_commons.make_greedy_fn. Default (0, 1) = par within.
    """
    n = len(price_grid)
    greedy_a = make_greedy_fn(q_a, price_grid, idx_a)
    greedy_b = make_greedy_fn(q_b, price_grid, idx_b)

    dA = np.zeros((n, n))
    dB = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pa, pb = price_grid[i], price_grid[j]
            dA[i, j] = greedy_a(pa, pb) - pa
            dB[i, j] = greedy_b(pb, pa) - pb

    return dA, dB


# ── Detecção de nullclines e pontos fixos ─────────────────────────────────────

def find_nullclines(
    dA: np.ndarray,
    dB: np.ndarray,
    price_grid: np.ndarray,
) -> dict[str, list[dict]]:
    """
    Classifica cada estado do grid em três categorias:

      'joint'  — dA=0 ∧ dB=0  : ponto fixo conjunto (ambos param)
      'a_null' — dA=0 ∧ dB≠0  : nullcline de A (só A ficaria parado)
      'b_null' — dA≠0 ∧ dB=0  : nullcline de B (só B ficaria parado)

    Retorna dict com listas de dicts {'p_A', 'p_B', 'i', 'j'}.
    """
    joint, a_null, b_null = [], [], []
    for i in range(len(price_grid)):
        for j in range(len(price_grid)):
            a_zero = abs(dA[i, j]) < 1e-9
            b_zero = abs(dB[i, j]) < 1e-9
            entry = {"p_A": price_grid[i], "p_B": price_grid[j], "i": i, "j": j}
            if a_zero and b_zero:
                joint.append(entry)
            elif a_zero:
                a_null.append(entry)
            elif b_zero:
                b_null.append(entry)
    return {"joint": joint, "a_null": a_null, "b_null": b_null}


def find_fixed_points_from_field(
    dA: np.ndarray,
    dB: np.ndarray,
    price_grid: np.ndarray,
) -> list[dict]:
    """Atalho: retorna apenas os pontos fixos conjuntos (dA=0 ∧ dB=0)."""
    return find_nullclines(dA, dB, price_grid)["joint"]


def print_nullcline_summary(
    nc: dict[str, list[dict]],
    P_NE: float,
    P_M: float,
) -> None:
    """Imprime resumo legível das nullclines."""
    joint  = nc["joint"]
    a_null = nc["a_null"]
    b_null = nc["b_null"]
    print(f"  Pontos fixos conjuntos (★) : {len(joint)}")
    for fp in joint:
        ci_a = (fp["p_A"] - P_NE) / (P_M - P_NE)
        ci_b = (fp["p_B"] - P_NE) / (P_M - P_NE)
        print(f"    p_A={fp['p_A']:.4f} (CI={ci_a:.3f})  "
              f"p_B={fp['p_B']:.4f} (CI={ci_b:.3f})")
    print(f"  Nullcline A (◆ azul)       : {len(a_null)} estados  (A pára, B move)")
    print(f"  Nullcline B (◆ vermelho)   : {len(b_null)} estados  (B pára, A move)")
    if not joint:
        print("  → Sem ponto fixo: nullclines não se intersectam → sistema oscila.")


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _add_reference_lines(ax: Axes, price_grid: np.ndarray, P_NE: float, P_M: float):
    p_lo, p_hi = price_grid[0], price_grid[-1]
    ax.axhline(P_NE, color="royalblue",    ls="--", lw=1.0, alpha=0.7,
               label=f"Nash ({P_NE:.3f})")
    ax.axvline(P_NE, color="royalblue",    ls="--", lw=1.0, alpha=0.7)
    ax.axhline(P_M,  color="mediumpurple", ls=":",  lw=1.0, alpha=0.7,
               label=f"Monop. ({P_M:.3f})")
    ax.axvline(P_M,  color="mediumpurple", ls=":",  lw=1.0, alpha=0.7)
    ax.plot([p_lo, p_hi], [p_lo, p_hi], color="lightgray", lw=0.8, alpha=0.6,
            label="Diagonal")


def _add_nullclines(
    ax: Axes,
    nc: dict[str, list[dict]],
    show_nullclines: bool = True,
):
    if show_nullclines:
        if nc["a_null"]:
            ax.scatter(
                [p["p_B"] for p in nc["a_null"]],
                [p["p_A"] for p in nc["a_null"]],
                marker="D", s=45, color="steelblue", edgecolors="black",
                linewidths=0.6, alpha=0.75, zorder=8,
                label=f"Nullcline A  dA=0 (n={len(nc['a_null'])})",
            )
        if nc["b_null"]:
            ax.scatter(
                [p["p_B"] for p in nc["b_null"]],
                [p["p_A"] for p in nc["b_null"]],
                marker="D", s=45, color="tomato", edgecolors="black",
                linewidths=0.6, alpha=0.75, zorder=8,
                label=f"Nullcline B  dB=0 (n={len(nc['b_null'])})",
            )
    if nc["joint"]:
        ax.scatter(
            [p["p_B"] for p in nc["joint"]],
            [p["p_A"] for p in nc["joint"]],
            marker="*", s=320, color="gold", edgecolors="black", linewidths=1.2,
            zorder=10, label=f"Ponto fixo (n={len(nc['joint'])})",
        )


# ── Funções principais de plot ────────────────────────────────────────────────

def plot_vector_field(
    dA: np.ndarray,
    dB: np.ndarray,
    price_grid: np.ndarray,
    P_NE: float,
    P_M: float,
    nullclines: dict[str, list[dict]] | None = None,
    mode: str = "normalized",
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "plasma",
    show_nullclines: bool = True,
) -> Axes:
    """
    Plota o campo vetorial do sistema de dois agentes.

    Parameters
    ----------
    dA, dB        : (N, N) — saída de compute_vector_field()
    price_grid    : grade de preços
    P_NE, P_M     : referências econômicas
    nullclines    : saída de find_nullclines() — marca zeros do campo
    mode          : 'normalized' | 'exact' | 'adjusted'
    show_nullclines : se False, mostra apenas pontos fixos (sem nullclines parciais)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    X, Y = np.meshgrid(price_grid, price_grid)
    U = dB   # horizontal: mudança em p_B
    V = dA   # vertical:   mudança em p_A

    magnitude = np.sqrt(U**2 + V**2)
    max_mag   = magnitude.max()
    cell_size = (price_grid[-1] - price_grid[0]) / (len(price_grid) - 1)

    if mode == "normalized":
        safe = magnitude.copy(); safe[safe == 0] = 1.0
        U_plot, V_plot = U / safe, V / safe
        scale_kw   = {"scale": len(price_grid) / 0.8}
        label_mode = "Normalizado (direção)"
    elif mode == "exact":
        U_plot, V_plot = U, V
        scale_kw   = {}
        label_mode = "Módulo exato (Δpreço)"
    elif mode == "adjusted":
        factor = (cell_size * 0.75 / max_mag) if max_mag > 0 else 1.0
        U_plot, V_plot = U * factor, V * factor
        scale_kw   = {"scale": 1.0, "scale_units": "xy", "angles": "xy"}
        label_mode = "Módulo ajustado (escala visual)"
    else:
        raise ValueError(f"mode deve ser 'normalized', 'exact' ou 'adjusted'. "
                         f"Recebeu: {mode!r}")

    norm = mcolors.Normalize(vmin=0, vmax=max_mag if max_mag > 0 else 1)
    colors_flat = plt.cm.get_cmap(cmap)(norm(magnitude.ravel()))
    ax.quiver(X, Y, U_plot, V_plot, color=colors_flat, alpha=0.85, **scale_kw)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="|Δp| real")

    _add_reference_lines(ax, price_grid, P_NE, P_M)
    if nullclines is not None:
        _add_nullclines(ax, nullclines, show_nullclines=show_nullclines)

    p_lo, p_hi = price_grid[0], price_grid[-1]
    ax.set_xlim(p_lo - cell_size * 0.5, p_hi + cell_size * 0.5)
    ax.set_ylim(p_lo - cell_size * 0.5, p_hi + cell_size * 0.5)
    ax.set_xlabel("p_B"); ax.set_ylabel("p_A")
    ax.set_title(title or f"Campo Vetorial — {label_mode}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return ax


def plot_vector_field_trio(
    dA: np.ndarray,
    dB: np.ndarray,
    price_grid: np.ndarray,
    P_NE: float,
    P_M: float,
    nullclines: dict[str, list[dict]] | None = None,
    suptitle: str = "Campo Vetorial — Sistema de Dois Agentes",
    figsize: tuple[int, int] = (18, 6),
    cmap: str = "plasma",
    show_nullclines: bool = True,
) -> Figure:
    """Os três modos lado a lado: normalizado, exato e ajustado."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, mode, title in [
        (axes[0], "normalized", "Normalizado\n(direção do movimento)"),
        (axes[1], "exact",      "Módulo exato\n(Δp real em cada passo)"),
        (axes[2], "adjusted",   "Módulo ajustado\n(escala visual)"),
    ]:
        plot_vector_field(
            dA, dB, price_grid, P_NE, P_M,
            nullclines=nullclines, mode=mode, ax=ax, title=title,
            cmap=cmap, show_nullclines=show_nullclines,
        )
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_streamlines(
    dA: np.ndarray,
    dB: np.ndarray,
    price_grid: np.ndarray,
    P_NE: float,
    P_M: float,
    nullclines: dict[str, list[dict]] | None = None,
    ax: Axes | None = None,
    title: str = "Streamlines",
    density: float = 1.2,
    cmap: str = "plasma",
    show_nullclines: bool = True,
) -> Axes:
    """Streamlines — ideal para visualizar bacias de atração e ciclos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    U, V      = dB, dA
    magnitude = np.sqrt(U**2 + V**2)

    ax.streamplot(
        price_grid, price_grid, U, V,
        color=magnitude, cmap=cmap,
        density=density, linewidth=1.5, arrowsize=1.2,
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(0, magnitude.max() or 1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="|Δp| real")

    _add_reference_lines(ax, price_grid, P_NE, P_M)
    if nullclines is not None:
        _add_nullclines(ax, nullclines, show_nullclines=show_nullclines)

    p_lo, p_hi = price_grid[0], price_grid[-1]
    ax.set_xlim(p_lo, p_hi); ax.set_ylim(p_lo, p_hi)
    ax.set_xlabel("p_B"); ax.set_ylabel("p_A")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    return ax
