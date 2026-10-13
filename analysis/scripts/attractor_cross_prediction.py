"""
Previsão do cross-pair a partir da distribuição de atratores .

Pergunta: a dispersão dos atratores entre treinos explica o colapso de CI
quando agentes de treinos diferentes são pareados (0.674 -> 0.126 no paper)?

Método: M pareamentos sorteados (p, q), p != q, entre os P pares do piloto.
O agente do slot 0 do par p joga com o agente do slot 1 do par q, os dois
congelados. Como s = idx(p0)·n + idx(p1) para os dois slots, o mapa conjunto é
F(s) = argmax(Q[p,0])[s]·n + argmax(Q[q,1])[s]. O CI de um pareamento é a média
dos preços no ciclo terminal, ponderada pela bacia (225 estados iniciais, peso
igual). A MESMA medida aplicada dentro do par (p, p) dá a linha de base.

Também: CI cruzado contra a distância entre os atratores próprios dos dois
pares (|CI_p − CI_q|), para ver se pares "parecidos" se coordenam melhor.

Saídas (analysis/):
  data/attractor_cross_prediction.csv      uma linha por pareamento
  figures/fig_attractor_cross_prediction.{png,pdf}
  reports/attractor_cross_prediction.md

Uso: python analysis/scripts/attractor_cross_prediction.py [run_dir] [--m 2000]
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qtable_analysis.q_commons import P_NE, P_M, ci as ci_of  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attractor_distribution import load_run, terminal_cycles  # noqa: E402

OUT = ROOT / "analysis"


def longrun_ci(pol_a: np.ndarray, pol_b: np.ndarray, grid: np.ndarray, n: int, H=None):
    """CI do preço médio de longo prazo, ponderado pela bacia (225 inícios).
    Se `H` (n×n) é dado, acumula nele a massa de atrator ponderada pela bacia."""
    F = pol_a * n + pol_b
    cyc_id, cyc_len, cycles = terminal_cycles(F)
    S = n * n
    tot = 0.0
    for s0 in range(S):
        cyc = cycles[cyc_id[s0]]
        pm = np.mean([grid[s // n] for s in cyc] + [grid[s % n] for s in cyc])
        tot += pm / S
        if H is not None:
            w = 1.0 / (S * len(cyc))
            for s in cyc:
                H[s // n, s % n] += w
    return ci_of(tot), len(cycles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else sorted((ROOT / "results").glob("attractor_pilot__*"))[-1]
    cfg, grid, Q = load_run(run_dir)
    P, _, S, n = Q.shape
    pol = Q.argmax(axis=3)  # (P, 2, S)

    # linha de base: dentro do par, mesma medida
    H_within = np.zeros((n, n))
    within = np.array([longrun_ci(pol[p, 0], pol[p, 1], grid, n, H_within)[0] for p in range(P)])
    H_cross = np.zeros((n, n))

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.m):
        p, q = rng.choice(P, size=2, replace=False)
        c01, k01 = longrun_ci(pol[p, 0], pol[q, 1], grid, n, H_cross)  # slot 0 de p × slot 1 de q
        rows.append({"p": int(p), "q": int(q), "ci_cross": round(float(c01), 4), "n_attractors": k01,
                     "ci_within_p": round(float(within[p]), 4), "ci_within_q": round(float(within[q]), 4),
                     "gap_within": round(float(abs(within[p] - within[q])), 4)})
    cross = np.array([r["ci_cross"] for r in rows]); gap = np.array([r["gap_within"] for r in rows])
    natt = np.array([r["n_attractors"] for r in rows])

    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / "attractor_cross_prediction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(cross, gap)
    # quartis da distância entre atratores próprios
    qs = np.quantile(gap, [0.25, 0.5, 0.75])
    bins = np.digitize(gap, qs)
    by_bin = [(float(cross[bins == b].mean()), int((bins == b).sum())) for b in range(4)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.hist(within, bins=np.linspace(-0.3, 1.1, 43), color="#1f77b4", alpha=.85, label=f"within pair (n={P})", density=True)
    ax.hist(cross, bins=np.linspace(-0.3, 1.1, 43), color="#7f7f7f", alpha=.75, label=f"cross pair (n={args.m})", density=True)
    ax.axvline(within.mean(), color="#1f77b4", ls="--", lw=1); ax.axvline(cross.mean(), color="#7f7f7f", ls="--", lw=1)
    ax.axvline(0, color="0.3", lw=.8); ax.axvline(1, color="0.3", lw=.8, ls=":")
    ax.set_xlabel("collusion index (basin-weighted long-run price)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Within {within.mean():.3f} → cross {cross.mean():.3f} ({(cross.mean() / within.mean() - 1) * 100:+.0f}%)", fontsize=10)
    ax = axes[1]
    ax.scatter(gap, cross, s=6, alpha=.35, color="#1f77b4")
    ax.set_xlabel("|CI of pair p − CI of pair q| (each within its own pair)"); ax.set_ylabel("cross-pair CI")
    ax.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.1e})", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(OUT / "figures" / f"fig_attractor_cross_prediction.{e}", dpi=150)
    plt.close(fig)

    # ---- heatmap: dentro do par vs cruzado (massa ponderada pela bacia) ----
    Hw = H_within / H_within.sum(); Hx = H_cross / H_cross.sum()
    np.savez(OUT / "data" / "attractor_cross_heatmaps.npz", H_within=Hw, H_cross=Hx, grid=grid)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    ext = [grid[0] - (grid[1] - grid[0]) / 2, grid[-1] + (grid[1] - grid[0]) / 2]
    vmax = max(Hw.max(), Hx.max())
    for ax, H, title in ((axes[0], Hw, f"Within pair (n = {P}): basin-weighted attractor mass"),
                         (axes[1], Hx, f"Cross pair (n = {args.m} random matchings)")):
        im = ax.imshow(H.T, origin="lower", cmap="Blues", extent=ext + ext, aspect="equal", vmin=0, vmax=vmax)
        for ref in (P_NE, P_M):
            ax.axvline(ref, color="0.4", ls="--", lw=.8); ax.axhline(ref, color="0.4", ls="--", lw=.8)
        ax.plot(ext, ext, color="0.6", lw=.6)
        ax.set_xlabel("firm 1 price"); ax.set_ylabel("firm 2 price"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03, label="share of mass")
    fig.suptitle("Where frozen policies end up: the same agents with their training partner vs. with a stranger", fontsize=10.5)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(OUT / "figures" / f"fig_attractor_heatmap_cross.{e}", dpi=150)
    plt.close(fig)
    diag_w = float(np.trace(Hw)); diag_x = float(np.trace(Hx))
    top_x = np.dstack(np.unravel_index(np.argsort(-Hx, axis=None)[:5], Hx.shape))[0]

    lines = [
        "# Previsão do cross-pair a partir da distribuição de atratores", "",
        f"Run: `{run_dir.name}` · {P} pares · {args.m} pareamentos sorteados (slot 0 de p × slot 1 de q) · medida: CI do preço médio no ciclo terminal, ponderado pela bacia (225 inícios)", "",
        "| | média | mediana | sd | < Nash (CI<0) |", "|---|---|---|---|---|",
        f"| dentro do par (mesma medida) | **{within.mean():.3f}** | {np.median(within):.3f} | {within.std():.3f} | {np.mean(within < 0) * 100:.1f}% |",
        f"| entre pares sorteados | **{cross.mean():.3f}** | {np.median(cross):.3f} | {cross.std():.3f} | {np.mean(cross < 0) * 100:.1f}% |",
        "",
        f"- Queda prevista: **{(1 - cross.mean() / within.mean()) * 100:.1f}%** (paper, 12 pareamentos: 0.674 → 0.126, −81%; 40 pareamentos: 0.109, 30% abaixo de Nash).",
        f"- Pareamentos cruzados com um único atrator: {np.mean(natt == 1) * 100:.1f}% (atratores por pareamento: média {natt.mean():.2f}).",
        f"- CI cruzado vs distância entre os atratores próprios: Spearman ρ = {rho:.3f}, p = {pval:.1e}.",
        "- CI cruzado por quartil da distância |CI_p − CI_q| (do mais parecido ao mais diferente): " + " · ".join(f"{m:.3f} (n={k})" for m, k in by_bin),
        "", "## Heatmap cruzado (massa ponderada pela bacia)", "",
        f"- massa na diagonal: dentro do par {diag_w * 100:.1f}% · cruzado {diag_x * 100:.1f}%",
        "- células mais frequentes no cruzado: " + " · ".join(f"({grid[i]:.3f}, {grid[j]:.3f}) {Hx[i, j] * 100:.1f}%" for i, j in top_x),
        "", "Figuras: `figures/fig_attractor_cross_prediction.png`, `figures/fig_attractor_heatmap_cross.png`. Dados: `data/attractor_cross_prediction.csv`, `data/attractor_cross_heatmaps.npz`.",
    ]
    OUT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "attractor_cross_prediction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
