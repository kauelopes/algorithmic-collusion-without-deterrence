"""Export one converged pair as a JSON joint policy for the companion site.

For each co-trained pair, the two frozen greedy policies define a deterministic
map on the 225 joint states (p0_{t-1}, p1_{t-1}). The site animates that map.

Output: docs/data/pair.json with
  grid        15 prices
  p_ne, p_m   Nash and monopoly benchmarks
  next0[i][j] index of agent 0's greedy price when its own last price is i
              and the rival's is j
  next1[j][i] same for agent 1 (own = j, rival = i)
  attractor   the cycle reached from the cooperative corner
  basin_share fraction of the 225 states that reach that same cycle

Usage: python analysis/scripts/export_policy_json.py [pair_dir]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qtable_analysis.q_commons import (  # noqa: E402
    build_greedy_matrix, load_agent_and_grid, P_NE, P_M,
)


def joint_map(next0, next1, n):
    def step(i, j):
        return next0[i][j], next1[j][i]
    return step


def cycle_from(step, i, j, n):
    seen = {}
    t = 0
    while (i, j) not in seen:
        seen[(i, j)] = t
        i, j = step(i, j)
        t += 1
    start = seen[(i, j)]
    cyc = [s for s, k in sorted(seen.items(), key=lambda kv: kv[1]) if k >= start]
    return cyc


def main(pair_dir: Path):
    q0, grid, _ = load_agent_and_grid(pair_dir, 0)
    q1, _, _ = load_agent_and_grid(pair_dir, 1)
    n = len(grid)
    g0 = build_greedy_matrix(q0, grid, agent_idx=0)   # [own, rival] -> price
    g1 = build_greedy_matrix(q1, grid, agent_idx=1)
    def idx(p):
        return int(np.argmin(np.abs(grid - p)))
    next0 = [[idx(g0[i, j]) for j in range(n)] for i in range(n)]
    next1 = [[idx(g1[j, i]) for i in range(n)] for j in range(n)]
    step = joint_map(next0, next1, n)

    i_c = int(np.argmin(np.abs(grid - P_M)))
    attractor = cycle_from(step, i_c, i_c, n)
    att = set(attractor)
    reach = 0
    for i in range(n):
        for j in range(n):
            if any(s in att for s in cycle_from(step, i, j, n)):
                reach += 1
    out = {
        "pair": pair_dir.name,
        "grid": [round(float(p), 4) for p in grid],
        "p_ne": round(P_NE, 4), "p_m": round(P_M, 4),
        "next0": next0, "next1": next1,
        "attractor": [[int(a), int(b)] for a, b in attractor],
        "basin_share": round(reach / (n * n), 4),
    }
    dst = ROOT / "docs" / "data" / "pair.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{pair_dir.name}: attractor length {len(attractor)}, basin share {out['basin_share']:.3f} -> {dst}")


if __name__ == "__main__":
    default = ROOT / "results/cross_deploy__20260621_203040/training/pair_000_A"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
