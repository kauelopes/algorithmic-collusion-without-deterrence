"""
Equilibrium metrics for Phase 2 multi-agent experiments.

Works for both Cournot (quantity competition) and Bertrand (price competition).
All functions are pure (no side effects, no env dependency).

Metrics implemented:
  collusion_index        — CI = (Q_NE - Q_rl) / (Q_NE - Q_M)   [Cournot]
  collusion_index_price  — CI = (p_rl - p_NE) / (p_M - p_NE)   [Bertrand, proxy]
  collusion_index_profit — Δ  = (π - π_NE) / (π_M - π_NE)      [a definição do Calvano]
  discounted_value       — Σ γ^t π_t, o que o agente de fato maximiza
  nash_distance_joint    — mean |action_i - ref_NE| across agents
  inter_agent_symmetry   — std(action_i) across agents
  dynamics_label         — classification of greedy action series
  final_metrics          — aggregate tail-window summary (env-agnostic)

Reference: PRD_phase2.md and PRD_phase2_bertrand.md, Section 4.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.metrics.equilibrium import collusion_index as _ci_scalar


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def collusion_index(Q_rl: float, Q_NE: float, Q_M: float) -> float:
    """
    CI = (Q_NE - Q_rl) / (Q_NE - Q_M)   [Cournot — quantity-based]

    0 → Nash.  1 → cartel.  Negative → over-production.
    """
    return _ci_scalar(Q_rl, Q_NE, Q_M)


def collusion_index_price(p_rl: float, p_NE: float, p_M: float) -> float:
    """
    CI = (p_rl - p_NE) / (p_M - p_NE)   [Bertrand — price-based]

    0 → Nash.  1 → joint monopoly.  Negative → below-Nash pricing.
    """
    denom = p_M - p_NE
    if abs(denom) < 1e-12:
        return 0.0
    return float((p_rl - p_NE) / denom)


def collusion_index_profit(pi_rl: float, pi_NE: float, pi_M: float) -> float:
    """
    Δ = (π̄ - π^NE) / (π^M - π^NE)   [baseado em LUCRO]

    0 → Nash.  1 → monopólio conjunto.  Negativo → lucro abaixo do competitivo.

    **Esta é a definição do Calvano et al. (2020)**; `collusion_index_price` é um
    proxy em preço. Os dois quase coincidem quando o resultado é SIMÉTRICO, mas
    divergem muito quando os preços descasam — que é sistematicamente o caso em
    avaliação cross-pair (agentes que nunca treinaram juntos).

    Medido no p1 (`cross_deploy__20260621_203040`, 12 pares):

        within  |Δp|=0.020  →  CI preço +0.674   CI lucro **+0.842**
        cross   |Δp|=0.111  →  CI preço +0.117   CI lucro **−0.037**

    O viés é sistemático nas duas pontas: em resultado simétrico o preço
    SUBESTIMA (a curva de lucro é côncava, então 67% do caminho em preço rendem
    84% em lucro); em resultado assimétrico o preço SUPERESTIMA (a firma cara
    perde participação sem compensação de margem). Nas duas pontas, o índice de
    preço faz o colapso within→cross parecer MENOR do que é: −82,6% em preço
    contra −104,3% em lucro.

    Reporte sempre os dois, com |Δp| junto — é ele que explica a divergência.
    """
    denom = pi_M - pi_NE
    if abs(denom) < 1e-12:
        return 0.0
    return float((pi_rl - pi_NE) / denom)


def discounted_value(profits: Sequence[float], gamma: float) -> float:
    """Valor presente `Σ_t γ^t π_t` de uma trajetória de lucros.

    É a grandeza que o agente de fato maximiza, e é diretamente comparável aos
    Q-valores. Para uma trajetória estacionária vale `π̄/(1−γ)`; para uma cíclica,
    depende da fase, e é por isso que somamos a trajetória em vez de usar a média.

    Truncamento: com γ=0.95, 100 períodos capturam 99,4% da soma infinita.

    Referências úteis (Calvano, γ=0.95): `V_Nash = π_NE/(1−γ) = 4.459`,
    `V_Monopólio = π_M/(1−γ) = 6.750`.
    """
    arr = np.asarray(profits, dtype=np.float64)
    return float((arr * gamma ** np.arange(len(arr))).sum())


def nash_distance_joint(action_list: Sequence[float], ref_NE: float) -> float:
    """Mean absolute deviation from symmetric Nash action (q or p) across agents."""
    arr = np.array(action_list, dtype=np.float64)
    return float(np.mean(np.abs(arr - ref_NE)))


def inter_agent_symmetry(q_greedy_list: Sequence[float]) -> float:
    """
    std(q_greedy_i) across agents.

    Approaches 0 at any symmetric equilibrium (Nash or collusive).
    Large values indicate one agent has become dominant.
    """
    return float(np.std(np.array(q_greedy_list, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Dynamics classification (PRD Section 4.4)
# ---------------------------------------------------------------------------

def dynamics_label(
    Q_greedy_series: np.ndarray,
    Q_NE: float,
    ci_final: float,
) -> str:
    """
    Classify dynamics from total greedy quantity series.

    Uses the full series passed in. Caller is responsible for selecting
    the relevant window (typically the last eval_window steps).

    Classification rules (PRD 4.4):
      nash_fixed_point     : autocorr(lag-1) > 0.9  AND  CI < 0.10
      collusive_fixed_point: autocorr(lag-1) > 0.9  AND  CI > 0.20
      cycling              : 0.30 < autocorr(lag-1) <= 0.90
      unstable             : std(Q) > 0.20 * Q_NE
      converging           : otherwise

    Parameters
    ----------
    Q_greedy_series : 1-D array of total greedy quantity per eval step
    Q_NE            : symmetric Nash total quantity (n * q_NE)
    ci_final        : mean CI over the same window (for fixed-point subtype)
    """
    arr = np.array(Q_greedy_series, dtype=np.float64)
    if arr.shape[0] < 5:
        return "insufficient_data"

    std_Q = float(np.std(arr))
    if std_Q > 0.20 * Q_NE:
        return "unstable"

    autocorr = _lag1_autocorr(arr)

    if autocorr > 0.90:
        if ci_final < 0.10:
            return "nash_fixed_point"
        if ci_final > 0.20:
            return "collusive_fixed_point"
        return "fixed_point"

    if autocorr > 0.30:
        return "cycling"

    return "converging"


def _lag1_autocorr(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D array. Returns 0.0 if variance ≈ 0."""
    if arr.shape[0] < 2:
        return 0.0
    var = float(np.var(arr))
    if var < 1e-12:
        return 1.0   # constant series → perfectly autocorrelated
    centered = arr - arr.mean()
    return float(np.mean(centered[:-1] * centered[1:]) / var)


# ---------------------------------------------------------------------------
# Aggregated final-window summary
# ---------------------------------------------------------------------------

def final_metrics(
    history: list[dict],
    eval_window: int,
    eval_interval: int,
    benchmarks: dict,
    N: int,
) -> dict:
    """
    Aggregate metrics over the final eval_window steps.

    Parameters
    ----------
    history       : full per-eval-step history from the trainer
    eval_window   : number of steps in the final window (e.g. 50000)
    eval_interval : steps between eval records (e.g. 10000)
    benchmarks    : env.benchmarks() dict
    N             : number of agents

    Returns
    -------
    dict with keys:
      ci, nash_dist, ce_gap, sym_q, dynamics_label,
      mean_q_{i}, greedy_q_{i} for i in 0..N-1
    """
    n_tail = max(1, eval_window // eval_interval)
    tail   = history[-n_tail:]

    # Env-agnostic: detect Bertrand by presence of "p_NE" key (no "Q_NE")
    if "p_NE" in benchmarks and "Q_NE" not in benchmarks:
        # Bertrand: greedy_q_{i} stores greedy price; use N*p_NE as scale
        ref_NE_total = N * benchmarks["p_NE"]
    else:
        ref_NE_total = benchmarks["Q_NE"]

    ci_val   = float(np.mean([r["collusion_index"] for r in tail]))
    nd_val   = float(np.mean([r["nash_dist"]       for r in tail]))
    ceg_val  = float(np.mean([r["ce_gap"]          for r in tail]))
    sym_val  = float(np.mean([r["sym_q"]           for r in tail]))

    # Greedy action series summed over agents (quantity total or price total)
    Q_greedy_series = np.array([
        sum(r.get(f"greedy_q_{i}", 0.0) for i in range(N))
        for r in history
    ])
    dyn = dynamics_label(Q_greedy_series[-n_tail:], ref_NE_total, ci_val)

    out: dict = {
        "ci":             round(ci_val,  6),
        "nash_dist":      round(nd_val,  6),
        "ce_gap":         round(ceg_val, 6),
        "sym_q":          round(sym_val, 6),
        "dynamics_label": dyn,
    }
    for i in range(N):
        out[f"mean_q_{i}"]   = round(float(np.mean([r.get(f"mean_q_{i}", 0.0)   for r in tail])), 6)
        out[f"greedy_q_{i}"] = round(float(np.mean([r.get(f"greedy_q_{i}", 0.0) for r in tail])), 6)

    return out
