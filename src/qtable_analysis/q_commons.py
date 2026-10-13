"""
q_commons.py — Funções compartilhadas para análise de Q-tables (Calvano replication)

Constantes, carregamento, helpers de grid, pontos fixos e simulação.
Importado pelos notebooks e por q_vector_field.py.
"""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from typing import Callable

# ── Constantes econômicas ─────────────────────────────────────────────────────

P_NE: float = 1.4729   # preço Nash-Equilibrium (Calvano 2020)
P_M:  float = 1.9250   # preço de monopólio


def ci(p: float, p_ne: float = P_NE, p_m: float = P_M) -> float:
    """Collusion Index: 0 = Nash, 1 = Monopólio."""
    return (p - p_ne) / (p_m - p_ne)


# ── Grid e carregamento ───────────────────────────────────────────────────────

def price_grid_from_config(config: dict) -> np.ndarray:
    """Reconstrói grade de preços a partir do config.yaml do experimento."""
    n      = config["agent"]["n_actions"]
    extend = config["environment"]["market"]["grid_extend"]
    c      = config["environment"]["cost"]["c"]
    p_min  = max(c, P_NE - extend * (P_M - P_NE))
    p_max  = P_M + extend * (P_M - P_NE)
    return np.linspace(p_min, p_max, n)


def load_agent(run_dir: str | Path, agent_idx: int) -> tuple[np.ndarray, dict]:
    """
    Carrega Q-table e config de um run.

    Returns
    -------
    q_table : np.ndarray shape (n_states, n_actions)
    config  : dict com o conteúdo do config.yaml
    """
    run_dir = Path(run_dir)
    data    = np.load(run_dir / "agents" / f"agent_{agent_idx}.npz")
    config  = yaml.safe_load((run_dir / "config.yaml").read_text())
    return data["q_table"], config


def load_agent_and_grid(
    run_dir: str | Path,
    agent_idx: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Atalho: carrega Q-table, reconstrói PRICE_GRID e retorna o config.

    Returns
    -------
    q          : np.ndarray shape (n_states, n_actions)
    price_grid : np.ndarray shape (n_actions,)
    config     : dict
    """
    q, config = load_agent(run_dir, agent_idx)
    price_grid = price_grid_from_config(config)
    return q, price_grid, config


# ── Helpers greedy ────────────────────────────────────────────────────────────

def make_greedy_fn(
    q: np.ndarray,
    price_grid: np.ndarray,
    agent_idx: int = 0,
) -> Callable:
    """
    Retorna closure `greedy(p_own, p_rival) -> float` para a Q-table `q`.
    Usado por q_vector_field.py e pelos notebooks.

    `agent_idx` é o SLOT em que a Q-table foi treinada, e é obrigatório para
    ler a tabela corretamente.

    Convenção de estado (src/agents/q_learning.py::_obs_to_state_idx):
    sob `rival_prices` o ambiente entrega o MESMO vetor `[p_0, ..., p_{N-1}]`
    a todos os agentes, indexado por FIRMA, e o índice de estado é

        s = idx(p_0) * n + idx(p_1)

    para os dois agentes. Logo o dígito mais significativo é o preço próprio
    para o agente 0 e o preço do RIVAL para o agente 1 — a tabela do agente 1
    é indexada (rival, own), não (own, rival).

    Verificado empiricamente (2026-08-10): simular os 12 pares within de
    `cross_deploy__20260621_203040` com esta convenção reproduz o `ci_within`
    registrado em summary.csv com erro absoluto médio de 0.0003 (11/12 exatos
    até a 4ª casa); a convenção antiga (own sempre major) erra por 0.44.
    """
    n = len(price_grid)
    p_lo, p_hi = price_grid[0], price_grid[-1]
    own_is_major = (agent_idx == 0)

    def _p2i(p: float) -> int:
        return int(np.clip(round((p - p_lo) / (p_hi - p_lo) * (n - 1)), 0, n - 1))

    def _i2p(i: int) -> float:
        return p_lo + i / (n - 1) * (p_hi - p_lo)

    def greedy(p_own: float, p_rival: float) -> float:
        i_own, i_riv = _p2i(p_own), _p2i(p_rival)
        s = i_own * n + i_riv if own_is_major else i_riv * n + i_own
        return _i2p(int(np.argmax(q[s])))

    return greedy


def build_greedy_matrix(
    q: np.ndarray,
    price_grid: np.ndarray,
    agent_idx: int = 0,
) -> np.ndarray:
    """Matriz N×N: greedy_matrix[i, j] = greedy_price quando own=i, rival=j."""
    greedy = make_greedy_fn(q, price_grid, agent_idx)
    n = len(price_grid)
    return np.array([
        [greedy(price_grid[i], price_grid[j]) for j in range(n)]
        for i in range(n)
    ])


def build_max_q_matrix(
    q: np.ndarray,
    n_actions: int,
    agent_idx: int = 0,
) -> np.ndarray:
    """Matriz N×N do max Q-value por estado, no eixo [own, rival]."""
    m = q.max(axis=1).reshape(n_actions, n_actions)
    return m if agent_idx == 0 else m.T


def build_spread_matrix(
    q: np.ndarray,
    n_actions: int,
    agent_idx: int = 0,
) -> np.ndarray:
    """Matriz N×N do spread (max - 2º max) Q-value por estado, eixo [own, rival]."""
    sorted_q = np.sort(q, axis=1)
    m = (sorted_q[:, -1] - sorted_q[:, -2]).reshape(n_actions, n_actions)
    return m if agent_idx == 0 else m.T


# ── Pontos fixos ──────────────────────────────────────────────────────────────

def find_self_fixed_states(
    q: np.ndarray,
    price_grid: np.ndarray,
    agent_idx: int = 0,
) -> list[dict]:
    """
    Encontra estados (p_own, p_rival) onde greedy(own, rival) == own.
    O agente ficaria parado nesse preço dado que o rival está em p_rival.

    Retorna lista de dicts: own, rival, i_own, j_rival, ci_own.
    """
    greedy = make_greedy_fn(q, price_grid, agent_idx)
    results = []
    for i, p_own in enumerate(price_grid):
        for j, p_rival in enumerate(price_grid):
            if abs(greedy(p_own, p_rival) - p_own) < 1e-9:
                results.append({
                    "own":     p_own,
                    "rival":   p_rival,
                    "i_own":   i,
                    "j_rival": j,
                    "ci_own":  ci(p_own),
                })
    return results


def find_joint_fixed_points(
    q_a: np.ndarray,
    q_b: np.ndarray,
    price_grid: np.ndarray,
    verbose: bool = False,
    idx_a: int = 0,
    idx_b: int = 1,
) -> list[dict]:
    """
    Pares (p_A, p_B) onde ambos os agentes ficam parados simultaneamente:
      greedy_A(p_A, p_B) == p_A  ∧  greedy_B(p_B, p_A) == p_B

    `idx_a` / `idx_b` são os SLOTS em que cada Q-table foi treinada — ver
    make_greedy_fn. O default (0, 1) é o caso within-pair (agent_0 × agent_1
    do mesmo run). Para cross-pair entre duas tabelas de agent_0, passe
    idx_b=0.

    Retorna lista de dicts: p_A, p_B, i, j, ci_A, ci_B, ci_mean.
    """
    greedy_a = make_greedy_fn(q_a, price_grid, idx_a)
    greedy_b = make_greedy_fn(q_b, price_grid, idx_b)
    results = []
    for i, p_a in enumerate(price_grid):
        for j, p_b in enumerate(price_grid):
            if abs(greedy_a(p_a, p_b) - p_a) < 1e-9 and \
               abs(greedy_b(p_b, p_a) - p_b) < 1e-9:
                ci_a = ci(p_a); ci_b = ci(p_b)
                results.append({
                    "p_A": p_a, "p_B": p_b, "i": i, "j": j,
                    "ci_A": ci_a, "ci_B": ci_b,
                    "ci_mean": (ci_a + ci_b) / 2,
                })
                if verbose:
                    print(f"  FP: p_A={p_a:.4f} CI={ci_a:.3f}  "
                          f"p_B={p_b:.4f} CI={ci_b:.3f}  "
                          f"CI_mean={(ci_a+ci_b)/2:.3f}")
    if verbose and not results:
        print("  Nenhum ponto fixo — nullclines não se intersectam.")
    return results


def print_fixed_points(fps: list[dict], label: str = "") -> None:
    """Imprime resumo formatado dos pontos fixos conjuntos."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Pontos fixos conjuntos: {len(fps)}")
    for fp in fps:
        print(f"  p_A={fp['p_A']:.4f} CI={fp['ci_A']:.3f}  "
              f"p_B={fp['p_B']:.4f} CI={fp['ci_B']:.3f}  "
              f"CI_mean={fp['ci_mean']:.3f}")
    if not fps:
        print("  → Nenhum (sistema oscila / limit cycle).")


# ── Simulação ─────────────────────────────────────────────────────────────────

def simulate_joint(
    q_a: np.ndarray,
    q_b: np.ndarray,
    price_grid: np.ndarray,
    n_periods: int = 200,
    init_a: float | None = None,
    init_b: float | None = None,
    seed: int = 0,
    idx_a: int = 0,
    idx_b: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula n_periods de interação greedy entre dois agentes.
    Cada agente age com base no preço do outro no período anterior.

    `idx_a` / `idx_b` são os SLOTS de treino das Q-tables — ver make_greedy_fn.
    Com o default (0, 1) esta função reproduz a dinâmica do trainer para um
    par within (verificado contra `ci_within` em cross_deploy, erro 0.0003).

    Returns
    -------
    hist_a, hist_b : np.ndarray shape (n_periods,)
    """
    rng = np.random.default_rng(seed)
    p_a = init_a if init_a is not None else float(rng.choice(price_grid))
    p_b = init_b if init_b is not None else float(rng.choice(price_grid))

    greedy_a = make_greedy_fn(q_a, price_grid, idx_a)
    greedy_b = make_greedy_fn(q_b, price_grid, idx_b)

    hist_a, hist_b = [], []
    for _ in range(n_periods):
        p_a, p_b = greedy_a(p_a, p_b), greedy_b(p_b, p_a)
        hist_a.append(p_a); hist_b.append(p_b)
    return np.array(hist_a), np.array(hist_b)


def play_against(
    q: np.ndarray,
    rival_prices: list[float],
    price_grid: np.ndarray,
    init_own: float | None = None,
    verbose: bool = True,
    agent_idx: int = 0,
) -> dict:
    """
    Simula o agente greedy `q` contra a sequência `rival_prices`.

    Returns
    -------
    dict com 'own_prices', 'rival_prices', 'ci', 'ci_mean'.
    """
    greedy = make_greedy_fn(q, price_grid, agent_idx)
    own = init_own if init_own is not None else float(np.median(price_grid))

    own_hist, ci_hist = [], []
    if verbose:
        print(f"{'t':>4}  {'own':>8}  {'rival':>8}  {'CI':>6}")
        print("-" * 32)

    for t, rp in enumerate(rival_prices):
        own = greedy(own, rp)
        ci_val = ci(own)
        own_hist.append(own); ci_hist.append(ci_val)
        if verbose:
            print(f"{t:>4}  {own:>8.4f}  {rp:>8.4f}  {ci_val:>6.3f}")

    ci_mean = float(np.mean(ci_hist))
    if verbose:
        print(f"{'CI médio:':>24} {ci_mean:.4f}")
    return {
        "own_prices":   own_hist,
        "rival_prices": list(rival_prices),
        "ci":           ci_hist,
        "ci_mean":      ci_mean,
    }


def detect_cycle(
    ha: np.ndarray,
    hb: np.ndarray,
    warmup: int = 200,
    max_period: int = 30,
) -> int | None:
    """
    Detecta o período do ciclo nas trajetórias após `warmup` passos.
    Retorna o período (int) ou None se não encontrar ciclo.
    """
    traj = list(zip(np.round(ha[warmup:], 6), np.round(hb[warmup:], 6)))
    n = len(traj)
    for period in range(1, max_period + 1):
        if all(traj[k] == traj[k % period] for k in range(n)):
            return period
    return None


def compute_basin(
    q_a: np.ndarray,
    q_b: np.ndarray,
    price_grid: np.ndarray,
    n_warm: int = 100,
    idx_a: int = 0,
    idx_b: int = 1,
) -> np.ndarray:
    """
    Mapa N×N de CI médio final para cada condição inicial (p_A₀, p_B₀).

    `idx_a` / `idx_b`: slots de treino das Q-tables — ver make_greedy_fn.

    Returns
    -------
    basin : np.ndarray shape (N, N)  — CI médio em (ha[-1], hb[-1])
    """
    n = len(price_grid)
    basin = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ha, hb = simulate_joint(
                q_a, q_b, price_grid, n_periods=n_warm,
                init_a=price_grid[i], init_b=price_grid[j],
                idx_a=idx_a, idx_b=idx_b,
            )
            basin[i, j] = (ci(ha[-1]) + ci(hb[-1])) / 2
    return basin


# ── Lucro instantâneo ─────────────────────────────────────────────────────────

def compute_profit(p_own: float, p_rival: float, config: dict) -> float:
    """
    Lucro por período no modelo Bertrand logit (Calvano 2020, duopólio).

    q_i = exp((a - p_i)/μ) / (exp(a0/μ) + exp((a-p_i)/μ) + exp((a-p_j)/μ))
    π_i = (p_i - c) * q_i
    """
    d  = config["environment"]["demand"]
    a  = float(d["a"])
    mu = float(d["mu"])
    a0 = float(d.get("a0", 0.0))
    c  = float(config["environment"]["cost"]["c"])
    logit_i = np.exp((a - p_own)   / mu)
    logit_j = np.exp((a - p_rival) / mu)
    denom   = np.exp(a0 / mu) + logit_i + logit_j
    return float((p_own - c) * logit_i / denom)


def profit_trajectory(
    ha: np.ndarray,
    hb: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o lucro instantâneo de cada agente ao longo da trajetória de preços.

    Returns
    -------
    pi_a, pi_b : np.ndarray shape (T,)
    """
    pi_a = np.array([compute_profit(pa, pb, config) for pa, pb in zip(ha, hb)])
    pi_b = np.array([compute_profit(pb, pa, config) for pa, pb in zip(ha, hb)])
    return pi_a, pi_b


# ── Atratores ─────────────────────────────────────────────────────────────────

def find_all_attractors(
    q_a: np.ndarray,
    q_b: np.ndarray,
    price_grid: np.ndarray,
    idx_a: int = 0,
    idx_b: int = 1,
) -> tuple[list[dict], np.ndarray]:
    """
    Enumera todos os atratores (ciclos e pontos fixos) do sistema determinístico
    (Q_A, Q_B) e mapeia a bacia de atração de cada um.

    `idx_a` / `idx_b` são os SLOTS de treino das Q-tables — ver make_greedy_fn.
    O default (0, 1) é o caso within-pair. Para dois agent_0 de runs distintos
    (cross-pair), passe idx_b=0.

    O espaço de estados é finito (N×N), portanto qualquer trajetória
    necessariamente entra num ciclo periódico. Esta função detecta todos os
    ciclos distintos e atribui cada condição inicial ao seu atrator.

    Returns
    -------
    attractors : list[dict]
        Cada dict contém:
          id, cycle (list[(p_A, p_B)]), cycle_idx (list[(i, j)]),
          period, ci_mean, ci_min, ci_max,
          basin_indices (list[(i, j)]), basin_fraction (float)
    attractor_id : np.ndarray shape (N, N), dtype int
        attractor_id[i, j] = índice do atrator para condição inicial (i, j)
    """
    n = len(price_grid)
    p_lo, p_hi = price_grid[0], price_grid[-1]

    def _p2i(p: float) -> int:
        return int(np.clip(round((p - p_lo) / (p_hi - p_lo) * (n - 1)), 0, n - 1))

    greedy_a = make_greedy_fn(q_a, price_grid, idx_a)
    greedy_b = make_greedy_fn(q_b, price_grid, idx_b)

    def _step(ci_: int, cj_: int) -> tuple[int, int]:
        pa, pb = price_grid[ci_], price_grid[cj_]
        return _p2i(greedy_a(pa, pb)), _p2i(greedy_b(pb, pa))

    attractor_id: np.ndarray = np.full((n, n), -1, dtype=int)
    attractors:   list[dict] = []
    cycle_to_id:  dict       = {}

    for i0 in range(n):
        for j0 in range(n):
            if attractor_id[i0, j0] != -1:
                continue

            history: list[tuple[int, int]] = []
            visited: dict[tuple[int, int], int] = {}
            cur_i, cur_j = i0, j0

            while True:
                if attractor_id[cur_i, cur_j] != -1:
                    # Chegou num estado já mapeado — backfill da trajetória
                    tid = int(attractor_id[cur_i, cur_j])
                    for hi, hj in history:
                        attractor_id[hi, hj] = tid
                        attractors[tid]["basin_indices"].append((hi, hj))
                    break

                if (cur_i, cur_j) in visited:
                    # Ciclo detectado na trajetória atual
                    cs = visited[(cur_i, cur_j)]
                    cycle_idx = history[cs:]
                    cycle_key = frozenset(cycle_idx)

                    if cycle_key not in cycle_to_id:
                        tid = len(attractors)
                        cycle_to_id[cycle_key] = tid
                        cycle_prices = [
                            (price_grid[ii], price_grid[jj])
                            for ii, jj in cycle_idx
                        ]
                        ci_vals = [(ci(pa) + ci(pb)) / 2 for pa, pb in cycle_prices]
                        attractors.append({
                            "id":            tid,
                            "cycle":         cycle_prices,
                            "cycle_idx":     cycle_idx,
                            "period":        len(cycle_idx),
                            "ci_mean":       float(np.mean(ci_vals)),
                            "ci_min":        float(np.min(ci_vals)),
                            "ci_max":        float(np.max(ci_vals)),
                            "basin_indices": [],
                        })

                    tid = cycle_to_id[cycle_key]
                    for hi, hj in history:
                        attractor_id[hi, hj] = tid
                        attractors[tid]["basin_indices"].append((hi, hj))
                    break

                visited[(cur_i, cur_j)] = len(history)
                history.append((cur_i, cur_j))
                cur_i, cur_j = _step(cur_i, cur_j)

    total = n * n
    for attr in attractors:
        attr["basin_fraction"] = len(attr["basin_indices"]) / total

    return attractors, attractor_id


def print_attractor_summary(
    attractors: list[dict],
    config: dict | None = None,
) -> None:
    """
    Imprime tabela resumida de todos os atratores, ordenada por bacia decrescente.

    Métricas exibidas:
      Sempre  : ID, Tipo, Período, CI_médio, CI_min, CI_max, Amplitude, Bacia
      + config: π_A (lucro médio A), π_B (lucro médio B), |Δπ| (assimetria)

    Amplitude = range de preços dentro do ciclo — 0 para ponto fixo.
    |Δπ| = |π_A − π_B|, mede quão assimétrico é o lucro entre os agentes.
    """
    with_profit = config is not None

    header = (f"{'ID':>3}  {'Tipo':>12}  {'Per.':>4}  {'CI_méd':>6}  "
              f"{'CI_min':>6}  {'CI_max':>6}  {'Ampl.':>5}")
    if with_profit:
        header += f"  {'π_A':>6}  {'π_B':>6}  {'|Δπ|':>5}"
    header += f"  {'Bacia':>7}"
    print(header)
    print("─" * (len(header) + 2))

    for a in sorted(attractors, key=lambda x: -x["basin_fraction"]):
        tipo  = "Ponto fixo" if a["period"] == 1 else f"Ciclo-{a['period']}"
        cycle = a["cycle"]  # list of (p_A, p_B)

        # Amplitude: range de preços dos dois agentes dentro do ciclo
        pa_vals = [p for p, _ in cycle]
        pb_vals = [p for _, p in cycle]
        ampl = max(max(pa_vals) - min(pa_vals), max(pb_vals) - min(pb_vals))

        row = (f"{a['id']:>3}  {tipo:>12}  {a['period']:>4}  "
               f"{a['ci_mean']:>6.3f}  {a['ci_min']:>6.3f}  {a['ci_max']:>6.3f}  "
               f"{ampl:>5.3f}")

        if with_profit:
            pi_a = float(np.mean([compute_profit(pa, pb, config) for pa, pb in cycle]))
            pi_b = float(np.mean([compute_profit(pb, pa, config) for pa, pb in cycle]))
            row += f"  {pi_a:>6.4f}  {pi_b:>6.4f}  {abs(pi_a - pi_b):>5.4f}"

        row += f"  {a['basin_fraction']:>6.1%}"
        print(row)
