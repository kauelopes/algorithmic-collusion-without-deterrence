"""Export the mean vector field of a population of frozen pairs for the site.

Each co-trained pair defines a deterministic map on the 225 joint states
(p0_{t-1}, p1_{t-1}). Averaging Q-tables across pairs is meaningless: each pair
is its own convention, and the argmax of a sum is a policy nobody played. What
can be averaged is the map itself: for every cell, the mean displacement in
price units over the population, and the share of pairs whose attractor
(reached from the cooperative corner) contains the cell.

Input: results/attractor_pilot__<ts>/batch_*/q_tables.npy, (K, 2, 225, 15),
       on Calvano's grid (grid_extend 0.1). State s = i*n + j for both agents,
       with i the index of firm 1's last price and j firm 2's.

Output: docs/data/population.json with
  run, n_pairs   provenance
  grid, p_ne, p_m
  dx[i][j]       mean price change of firm 1 from cell (i, j), over the pairs
  dy[i][j]       same for firm 2
  mag[i][j]      mean Euclidean length of the step, in price units
  att[i][j]      share of pairs whose corner attractor contains (i, j)

Usage: python analysis/scripts/export_population_json.py [run_dir]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.qtable_analysis.q_commons import P_NE, P_M  # noqa: E402
from attractor_distribution import load_run  # noqa: E402


def main(run_dir: Path):
    _, grid, Q = load_run(run_dir)
    P, _, S, n = Q.shape
    pol0 = Q[:, 0].argmax(axis=2)            # (P, S)
    pol1 = Q[:, 1].argmax(axis=2)
    i_s = np.arange(S) // n; j_s = np.arange(S) % n
    dx = (grid[pol0] - grid[i_s][None, :])   # (P, S)
    dy = (grid[pol1] - grid[j_s][None, :])
    mag = np.sqrt(dx ** 2 + dy ** 2)

    i_c = int(np.argmin(np.abs(grid - P_M)))
    att = np.zeros(S)
    for p in range(P):
        F = pol0[p] * n + pol1[p]
        s = i_c * n + i_c; seen = {}
        t = 0
        while s not in seen:
            seen[s] = t; s = int(F[s]); t += 1
        for q, k in seen.items():
            if k >= seen[s]:
                att[q] += 1
    att /= P

    def grid2(v, nd=4):
        return [[round(float(v[i * n + j]), nd) for j in range(n)] for i in range(n)]
    out = {
        "run": run_dir.name, "n_pairs": int(P),
        "grid": [round(float(p), 4) for p in grid],
        "p_ne": round(P_NE, 4), "p_m": round(P_M, 4),
        "dx": grid2(dx.mean(axis=0)), "dy": grid2(dy.mean(axis=0)),
        "mag": grid2(mag.mean(axis=0)), "att": grid2(att),
    }
    dst = ROOT / "docs" / "data" / "population.json"
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{run_dir.name}: {P} pairs, max attractor share {att.max():.3f}, "
          f"mean |step| {mag.mean():.3f} -> {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    runs = sorted((ROOT / "results").glob("attractor_pilot__*"))
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    elif runs:
        main(runs[-1])
    else:
        sys.exit("no results/attractor_pilot__* found; pass the run directory")
