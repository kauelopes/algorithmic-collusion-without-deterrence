"""
Distribuição dos atratores conjuntos sobre as 15 × 15 células .

Entrada: results/attractor_pilot__<ts>/batch_*/q_tables.npy
         (K, 2, 225, 15), produzido por experiments/attractor_pilot.py.

Convenção de estado (q_commons.make_greedy_fn, verificada em 2026-08-10):
s = idx(p0)·n + idx(p1) para OS DOIS agentes. Logo, com pol_a = argmax(q_a) e
pol_b = argmax(q_b) sobre s, o mapa conjunto é F(s) = pol_a[s]·n + pol_b[s].

Para cada par:
  - atrator alcançado a partir do canto cooperativo (p_M, p_M);
  - atrator terminal de CADA um dos 225 estados (massa ponderada pela bacia),
    número de atratores distintos e bacia dominante;
  - comprimento do ciclo, simetria (i == j em ponto fixo), CI do preço médio.

Saídas (analysis/):
  data/attractor_pilot_pairs.csv          uma linha por par
  data/attractor_pilot_heatmaps.npz       H_corner, H_basin (15 × 15), grid
  figures/fig_attractor_heatmap.{png,pdf} dois painéis
  figures/fig_attractor_convergence.{png,pdf}  TV entre metades vs N
  reports/attractor_pilot.md              números do piloto

Uso: python analysis/scripts/attractor_distribution.py <run_dir>
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qtable_analysis.q_commons import P_NE, P_M, ci as ci_of  # noqa: E402

OUT = ROOT / "analysis"


def load_run(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    n = int(cfg["agent"]["n_actions"])
    ext = float(cfg["environment"]["market"]["grid_extend"])
    c = float(cfg["environment"]["cost"]["c"])
    p_min = max(c, P_NE - ext * (P_M - P_NE)); p_max = P_M + ext * (P_M - P_NE)
    grid = np.linspace(p_min, p_max, n)
    qs = [np.load(b / "q_tables.npy") for b in sorted(run_dir.glob("batch_*")) if (b / "q_tables.npy").exists()]
    Q = np.concatenate(qs, axis=0)  # (P, 2, n*n, n)
    return cfg, grid, Q


def terminal_cycles(F: np.ndarray):
    """Para cada estado, o conjunto (tupla ordenada) do ciclo terminal."""
    S = len(F)
    # itera S passos: garante que todo estado esteja no ciclo
    x = np.arange(S)
    for _ in range(S):
        x = F[x]
    # x[s] está no ciclo terminal de s; o ciclo é identificado pelo menor estado nele
    cyc_id = np.empty(S, dtype=np.int64); cyc_len = np.empty(S, dtype=np.int64)
    cache = {}
    for s in range(S):
        y = x[s]
        if y not in cache:
            members = [y]; z = F[y]
            while z != y:
                members.append(z); z = F[z]
            m = min(members); cache[y] = (m, len(members), tuple(sorted(members)))
            for mm in members: cache[mm] = cache[y]
        cyc_id[s], cyc_len[s], _ = cache[y]
    cycles = {v[0]: v[2] for v in cache.values()}
    return cyc_id, cyc_len, cycles


def main(run_dir: Path):
    cfg, grid, Q = load_run(run_dir)
    P, _, S, n = Q.shape
    i_c = int(np.argmin(np.abs(grid - P_M)))
    s_c = i_c * n + i_c
    H_corner = np.zeros((n, n)); H_basin = np.zeros((n, n))
    rows = []
    for p in range(P):
        pol_a = Q[p, 0].argmax(axis=1); pol_b = Q[p, 1].argmax(axis=1)
        F = pol_a * n + pol_b
        cyc_id, cyc_len, cycles = terminal_cycles(F)
        # atrator do canto cooperativo
        corner = cycles[cyc_id[s_c]]
        for s in corner:
            H_corner[s // n, s % n] += 1.0 / len(corner)
        # massa ponderada pela bacia: cada estado inicial pesa 1/S, repartido pelo ciclo
        for s0 in range(S):
            cyc = cycles[cyc_id[s0]]
            w = 1.0 / (S * len(cyc))
            for s in cyc:
                H_basin[s // n, s % n] += w
        n_att = len(cycles)
        dom = max(np.bincount(cyc_id, minlength=S).max() / S, 0.0)
        ii = [s // n for s in corner]; jj = [s % n for s in corner]
        p_mean = float(np.mean([grid[a] for a in ii] + [grid[b] for b in jj]))
        rows.append({
            "pair": p, "cycle_len": len(corner), "symmetric": int(len(corner) == 1 and ii[0] == jj[0]),
            "i": ii[0], "j": jj[0], "p0": round(float(grid[ii[0]]), 4), "p1": round(float(grid[jj[0]]), 4),
            "ci_mean": round(ci_of(p_mean), 4), "n_attractors": n_att, "dominant_basin": round(float(dom), 4),
            "corner_in_dominant": int(cyc_id[s_c] == np.bincount(cyc_id, minlength=S).argmax()),
        })

    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / "attractor_pilot_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.savez(OUT / "data" / "attractor_pilot_heatmaps.npz", H_corner=H_corner, H_basin=H_basin, grid=grid)

    # ---- estatísticas ----
    cl = np.array([r["cycle_len"] for r in rows]); sym = np.array([r["symmetric"] for r in rows])
    cim = np.array([r["ci_mean"] for r in rows]); na = np.array([r["n_attractors"] for r in rows])
    dom = np.array([r["dominant_basin"] for r in rows])
    Hc = H_corner / H_corner.sum(); Hb = H_basin / H_basin.sum()
    diag_mass = float(np.trace(Hc))
    top = np.dstack(np.unravel_index(np.argsort(-Hc, axis=None)[:5], Hc.shape))[0]

    # ---- convergência: distância de variação total entre metades disjuntas ----
    rng = np.random.default_rng(0)
    def heat_of(idx):
        H = np.zeros((n, n))
        for p in idx:
            r = rows[p]
            # usa só o ponto representativo para velocidade; ciclos entram pelo 1º estado
            H[r["i"], r["j"]] += 1
        return H / max(H.sum(), 1)
    conv = []
    for N in [64, 128, 256, 512, 1024]:
        if N > P: break
        tvs = []
        for _ in range(20):
            perm = rng.permutation(P)[:N]
            a, b = perm[: N // 2], perm[N // 2:]
            tvs.append(0.5 * np.abs(heat_of(a) - heat_of(b)).sum())
        conv.append((N, float(np.mean(tvs)), float(np.std(tvs))))

    # ---- figuras ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    ext = [grid[0] - (grid[1] - grid[0]) / 2, grid[-1] + (grid[1] - grid[0]) / 2]
    for ax, H, title in ((axes[0], Hc, "Attractor reached from the cooperative corner"),
                         (axes[1], Hb, "Basin-weighted attractor mass (all 225 starts)")):
        im = ax.imshow(H.T, origin="lower", cmap="Blues", extent=ext + ext, aspect="equal")
        for ref, lab in ((P_NE, "Nash"), (P_M, "monopoly")):
            ax.axvline(ref, color="0.4", ls="--", lw=.8); ax.axhline(ref, color="0.4", ls="--", lw=.8)
        ax.plot(ext, ext, color="0.6", lw=.6)
        ax.set_xlabel("firm 1 price"); ax.set_ylabel("firm 2 price"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03, label="share of pairs")
    fig.suptitle(f"Joint attractors of {P} co-trained pairs (grid_extend {cfg['environment']['market']['grid_extend']}, "
                 f"{cfg['training']['total_steps'] / 1e6:.0f}M steps)", fontsize=10.5)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(OUT / "figures" / f"fig_attractor_heatmap.{e}", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    Ns = [c[0] for c in conv]; m = [c[1] for c in conv]; sd = [c[2] for c in conv]
    ax.errorbar(Ns, m, yerr=sd, marker="o", color="#1f77b4", capsize=3)
    ref = m[0] * np.sqrt(Ns[0] / np.array(Ns))
    ax.plot(Ns, ref, ls="--", color="0.5", label=r"$\propto 1/\sqrt{N}$")
    ax.set_xscale("log", base=2); ax.set_xlabel("pairs used (two disjoint halves)"); ax.set_ylabel("total variation between halves")
    ax.set_title("Does the attractor distribution converge?", fontsize=10); ax.legend(frameon=False)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(OUT / "figures" / f"fig_attractor_convergence.{e}", dpi=150)
    plt.close(fig)

    # ---- relatório ----
    OUT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    lines = [
        "# Piloto: distribuição dos atratores conjuntos", "",
        f"Run: `{run_dir.name}` · {P} pares · grid_extend {cfg['environment']['market']['grid_extend']} · "
        f"{cfg['training']['total_steps']:,} passos · script `attractor_distribution.py`", "",
        "## Forma do atrator (a partir do canto cooperativo)", "",
        f"- ponto fixo: **{np.mean(cl == 1) * 100:.1f}%** · ciclo de período 2: {np.mean(cl == 2) * 100:.1f}% · período ≥ 3: {np.mean(cl >= 3) * 100:.1f}%",
        f"- pontos fixos simétricos (i = j): **{sym.sum()} de {int((cl == 1).sum())}** ({sym.sum() / max((cl == 1).sum(), 1) * 100:.1f}% dos pontos fixos)",
        f"- massa na diagonal do heatmap: {diag_mass * 100:.1f}%",
        f"- CI do atrator: média **{cim.mean():.3f}**, mediana {np.median(cim):.3f}, sd {cim.std():.3f}, min {cim.min():.3f}, max {cim.max():.3f}",
        "", "## Estrutura de bacia", "",
        f"- atratores por par: média {na.mean():.2f} · um único atrator em **{np.mean(na == 1) * 100:.1f}%** dos pares",
        f"- bacia dominante: média **{dom.mean() * 100:.1f}%** · ≥ 99% em {np.mean(dom >= .99) * 100:.1f}% dos pares",
        f"- o atrator do canto cooperativo é o dominante em {np.mean([r['corner_in_dominant'] for r in rows]) * 100:.1f}% dos pares",
        "", "## Células mais frequentes (canto cooperativo)", "", "| firma 1 | firma 2 | share |", "|---|---|---|",
    ]
    for i, j in top:
        lines.append(f"| {grid[i]:.4f} (CI {ci_of(grid[i]):.2f}) | {grid[j]:.4f} (CI {ci_of(grid[j]):.2f}) | {Hc[i, j] * 100:.1f}% |")
    lines += ["", "## Convergência (variação total entre duas metades disjuntas)", "", "| N | TV média | sd |", "|---|---|---|"]
    for N, mu, s in conv:
        lines.append(f"| {N} | {mu:.3f} | {s:.3f} |")
    lines += ["", "Figuras: `figures/fig_attractor_heatmap.png`, `figures/fig_attractor_convergence.png`. "
              "Dados: `data/attractor_pilot_pairs.csv`, `data/attractor_pilot_heatmaps.npz`."]
    (OUT / "reports" / "attractor_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        runs = sorted((ROOT / "results").glob("attractor_pilot__*"))
        if not runs: sys.exit("nenhum run attractor_pilot__* encontrado")
        main(runs[-1])
    else:
        main(Path(sys.argv[1]))
