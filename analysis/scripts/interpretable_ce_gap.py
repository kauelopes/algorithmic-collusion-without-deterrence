"""
CE-gap em unidades interpretáveis: % de lucro deixado na mesa.

Para cada par de Q-tables treinadas, calcula:
  1. V_path  — valor descontado do agente i seguindo a política greedy conjunta
               a partir do atrator convergido (policy evaluation exata);
  2. V_BR    — valor da best-response ÓTIMA do agente i contra a política greedy
               do parceiro (value iteration exata no MDP de 225 estados — o
               parceiro REAGE, não é ação contrafactual de 1 período);
  3. gain%   — (V_BR − V_path) / V_path: quanto lucro o agente ganharia se
               jogasse otimamente contra o parceiro que de fato tem.

Contexto de escala: lucros por período Nash (π_NE) e monopólio/2 (π_M) do logit.

Usage (dentro do container):
  python analysis/scripts/interpretable_ce_gap.py \
      [--pairs_dir results/cross_deploy__20260621_203040/training] \
      [--extra_run results/calvano_careful__42__20260624_213325]

Outputs:
  analysis/data/interpretable_ce_gap.csv  (uma linha por agente)
  stdout: tabela resumo
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


_PROJ_ROOT = _repo_root()
sys.path.insert(0, str(_PROJ_ROOT))


# ---------------------------------------------------------------------------
# Logit payoffs (analytic, numpy only)
# ---------------------------------------------------------------------------

def _payoff_matrix(grid: np.ndarray, a: float, mu: float, c: float,
                   a0: float) -> np.ndarray:
    """PI[i, j] = lucro do agente jogando grid[i] contra grid[j]."""
    e = np.exp((a - grid) / mu)
    e_out = np.exp(a0 / mu)
    denom = e[:, None] + e[None, :] + e_out
    return (grid[:, None] - c) * e[:, None] / denom


def _grid_from_config(env_cfg: dict, n_actions: int):
    """Reconstrói a grade de preços e benchmarks sem instanciar o env (numpy)."""
    from src.econ_utils.benchmarks import logit_benchmarks
    d, c = env_cfg["demand"], float(env_cfg["cost"]["c"])
    bm = logit_benchmarks(a=float(d["a"]), mu=float(d["mu"]), c=c,
                          a0=float(d.get("a0", 0.0)),
                          n=int(env_cfg["market"]["n_firms"]))
    extend = float(env_cfg["market"].get("grid_extend", 1.0))
    p_ne, p_m = float(bm["p_NE"]), float(bm["p_monopoly"])
    p_min = max(c, p_ne - extend * (p_m - p_ne))
    p_max = p_m + extend * (p_m - p_ne)
    return np.linspace(p_min, p_max, n_actions), bm


# ---------------------------------------------------------------------------
# Exact DP on the 225-state MDP
# ---------------------------------------------------------------------------

def _attractor_state(pol0: np.ndarray, pol1: np.ndarray, n: int,
                     start: int = 0, burn: int = 1000) -> int:
    s = start
    for _ in range(burn):
        s = pol0[s] * n + pol1[s]
    return int(s)


def _policy_value(pol0, pol1, PI, gamma: float, n: int, agent: int) -> np.ndarray:
    """V(s) do agente `agent` quando ambos seguem (pol0, pol1) — avaliação exata.

    Convenção de estado: s = idx(p0) * n + idx(p1) para os dois agentes.
    """
    V = np.zeros(n * n)
    own, riv = (pol0, pol1) if agent == 0 else (pol1, pol0)
    r   = PI[own, riv]                      # r(s) — lucro do agente
    nxt = pol0 * n + pol1                   # s' determinístico (sempre p0, p1)
    for _ in range(10_000):
        V_new = r + gamma * V[nxt]
        if np.max(np.abs(V_new - V)) < 1e-12:
            break
        V = V_new
    return V


def _best_response_value(pol_partner, PI, gamma: float, n: int, agent: int):
    """V*(s) do agente `agent` contra a política fixa do parceiro — VI exata.

    O parceiro REAGE: sua ação em s' depende da ação do agente em s.
    """
    S = n * n
    V = np.zeros(S)
    b = pol_partner[np.arange(S)]                    # ação do parceiro em s
    r_sa = PI[:, b].T                                # (S, n): lucro(s, a)
    nxt = np.empty((S, n), dtype=np.int64)
    for a_i in range(n):
        # s' = (p0', p1'): nosso preço ocupa o slot do nosso índice
        nxt[:, a_i] = a_i * n + b if agent == 0 else b * n + a_i
    for _ in range(10_000):
        Q = r_sa + gamma * V[nxt]
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-12:
            break
        V = V_new
    return V, Q.argmax(axis=1)


def analyze_pair(pair_dir: Path) -> list[dict]:
    config = yaml.safe_load((pair_dir / "config.yaml").read_text())
    agent_cfg, env_cfg = config["agent"], config["environment"]
    n     = int(agent_cfg["n_actions"])
    gamma = float(agent_cfg["gamma"])
    grid, bm = _grid_from_config(env_cfg, n)
    PI = _payoff_matrix(grid, float(env_cfg["demand"]["a"]),
                        float(env_cfg["demand"]["mu"]),
                        float(env_cfg["cost"]["c"]),
                        float(env_cfg["demand"].get("a0", 0.0)))

    q = [np.load(pair_dir / "agents" / f"agent_{i}.npz")["q_table"]
         for i in (0, 1)]
    pol = [qt.argmax(axis=1) for qt in q]

    pi_ne = _payoff_matrix(np.array([bm["p_NE"]]), float(env_cfg["demand"]["a"]),
                           float(env_cfg["demand"]["mu"]),
                           float(env_cfg["cost"]["c"]),
                           float(env_cfg["demand"].get("a0", 0.0)))[0, 0]

    rows = []
    s_star = _attractor_state(pol[0], pol[1], n)
    for i in (0, 1):
        V_path = _policy_value(pol[0], pol[1], PI, gamma, n, agent=i)
        V_br, _ = _best_response_value(pol[1 - i], PI, gamma, n, agent=i)
        vp, vb = float(V_path[s_star]), float(V_br[s_star])
        per_period_path = vp * (1 - gamma)
        per_period_br   = vb * (1 - gamma)
        rows.append({
            "pair": pair_dir.name, "agent": i,
            "profit_path": round(per_period_path, 5),
            "profit_br":   round(per_period_br, 5),
            "gain_pct":    round(100 * (vb - vp) / vp, 2) if vp > 0 else float("nan"),
            "profit_nash": round(float(pi_ne), 5),
            "path_vs_nash_pct": round(100 * (per_period_path - pi_ne) / pi_ne, 2),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", type=str,
                   default="results/cross_deploy__20260621_203040/training")
    p.add_argument("--extra_run", type=str,
                   default="results/calvano_careful__42__20260624_213325")
    p.add_argument("--out", type=str,
                   default="analysis/data/interpretable_ce_gap.csv")
    args = p.parse_args()

    pairs_dir = _PROJ_ROOT / args.pairs_dir
    pair_dirs = sorted(d for d in pairs_dir.iterdir()
                       if (d / "agents" / "agent_0.npz").exists())
    extra = _PROJ_ROOT / args.extra_run
    if (extra / "agents" / "agent_0.npz").exists():
        pair_dirs.append(extra)

    all_rows = []
    for d in pair_dirs:
        try:
            all_rows.extend(analyze_pair(d))
        except Exception as exc:
            print(f"  {d.name}: ERRO {exc}")

    out = _PROJ_ROOT / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    gains = [r["gain_pct"] for r in all_rows if np.isfinite(r["gain_pct"])]
    print(f"\nAgentes analisados: {len(all_rows)}")
    print(f"Ganho de best-response sobre o caminho convergido:")
    print(f"  média {np.mean(gains):.1f}%  mediana {np.median(gains):.1f}%  "
          f"min {np.min(gains):.1f}%  max {np.max(gains):.1f}%")
    print(f"CSV → {out}")


if __name__ == "__main__":
    main()
