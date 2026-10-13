"""Cover figure for the companion site: within-pair vs cross-pair attractor mass.

Reads analysis/data/attractor_cross_heatmaps.npz (written by
attractor_cross_prediction.py) and draws the two 15 x 15 heatmaps as a single
wide picture with one shared colour scale, short labels and the competitive
and monopoly prices marked. Output: docs/figures/fig_hero_heatmaps.png (+ .pdf).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
P_NE, P_M = 1.4729, 1.9250

PAPER = "#fbfaf6"; INK = "#16181a"; INK2 = "#4f5256"; MUTE = "#86888b"; RULE = "#dcd9d0"; BLUE = "#1f77b4"
cmap = LinearSegmentedColormap.from_list("paperblue", ["#ffffff", "#dbe8f4", "#9ec5e6", "#4b93c9", BLUE, "#15537d"])


def main() -> None:
    d = np.load(ROOT / "analysis" / "data" / "attractor_cross_heatmaps.npz")
    Hw, Hx, grid = d["H_within"], d["H_cross"], d["grid"]
    n = len(grid); step = grid[1] - grid[0]
    ext = [grid[0] - step / 2, grid[-1] + step / 2]
    vmax = max(Hw.max(), Hx.max())

    plt.rcParams.update({"font.family": ["Archivo Narrow", "Roboto Condensed", "Arial Narrow", "DejaVu Sans"],
                         "font.size": 11, "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(12.4, 5.6), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.30, left=0.06, right=0.94, top=0.80, bottom=0.12)
    panels = [
        (fig.add_subplot(gs[0, 0]), Hw, "With the partner it trained with", "1,024 pairs · frozen policies"),
        (fig.add_subplot(gs[0, 1]), Hx, "With a stranger trained the same way", "2,000 random matchings · same policies"),
    ]
    for ax, H, title, sub in panels:
        ax.set_facecolor("#ffffff")
        im = ax.imshow(H.T, origin="lower", cmap=cmap, extent=ext + ext, aspect="equal", vmin=0, vmax=vmax, interpolation="nearest")
        ax.plot(ext, ext, color=RULE, lw=0.8, zorder=2)
        for ref, lab in ((P_NE, "competitive"), (P_M, "monopoly")):
            ax.axvline(ref, color=MUTE, ls=(0, (4, 3)), lw=0.9, zorder=3)
            ax.axhline(ref, color=MUTE, ls=(0, (4, 3)), lw=0.9, zorder=3)
            ax.text(ref + 0.006, ext[1] - 0.012, lab, color=MUTE, fontsize=9.5, ha="left", va="top", rotation=90)
        ax.set_xlim(ext); ax.set_ylim(ext)
        ax.set_xticks([1.5, 1.6, 1.7, 1.8, 1.9]); ax.set_yticks([1.5, 1.6, 1.7, 1.8, 1.9])
        ax.tick_params(length=3, labelsize=10)
        ax.set_xlabel("firm 1, long-run price", color=INK2, fontsize=11)
        if ax is panels[0][0]:
            ax.set_ylabel("firm 2, long-run price", color=INK2, fontsize=11)
        ax.set_title(title, color=INK, fontsize=15, fontweight="semibold", loc="left", pad=24)
        ax.text(0, 1.025, sub, transform=ax.transAxes, color=MUTE, fontsize=10.5, ha="left", va="bottom")
        for sp in ax.spines.values(): sp.set_color(RULE)

    # a seta no vão entre os painéis: as mesmas políticas, outro parceiro
    b0 = panels[0][0].get_position(); b1 = panels[1][0].get_position()
    xm = (b0.x1 + b1.x0) / 2; ym = (b0.y0 + b0.y1) / 2
    fig.text(xm, ym + 0.03, "→", ha="center", va="center", fontsize=28, color=MUTE)
    fig.text(xm, ym - 0.04, "swap the\npartner", ha="center", va="center", fontsize=10, color=MUTE, linespacing=1.2)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("share of pairs", color=INK2, fontsize=10.5)
    cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=9.5, color=INK2, labelcolor=INK2)

    fig.text(0.06, 0.95, "Where frozen pricing policies end up", color=MUTE, fontsize=11.5, ha="left")
    out = ROOT / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
    for e, dpi in (("png", 170), ("pdf", 170)):
        fig.savefig(out / f"fig_hero_heatmaps.{e}", dpi=dpi, facecolor=PAPER)
    print("ok ->", out / "fig_hero_heatmaps.png")


if __name__ == "__main__":
    main()
