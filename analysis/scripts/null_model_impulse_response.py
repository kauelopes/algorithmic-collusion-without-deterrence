"""
Modelo nulo: mapas aleatórios reproduzem "queda-e-retorno" e bacias grandes?

Contexto: como s_t = a_{t-1} (o estado É a ação conjunta do período anterior),
qualquer par de políticas greedy fixas define um mapa determinístico
F: S -> S num conjunto finito (225 estados, N=2). Todo mapa desse tipo é
eventualmente periódico (ponto fixo ou ciclo) — fato combinatório trivial
(pigeonhole), não uma propriedade aprendida. Isso levanta uma pergunta que os
Experimentos 6/7 (impulse-response, `grim` ≡ `nash`) não respondem sozinhos:
o "retorno ao preço pré-desvio" documentado lá é evidência de uma regra de
punição aprendida, ou é a geometria genérica de bacias de atração de QUALQUER
mapa determinístico num espaço de estados finito — inclusive um aleatório?

Método: dois testes, ambos comparando os 18 pares REAIS co-treinados
(`results/cross_deploy__20260621_203040/training/`) contra um modelo nulo de
pares de Q-tables aleatórias (mesma forma 225x15, valores i.i.d. uniformes no
MESMO intervalo observado nas Q-tables reais — a mesma construção do
`analysis/notebooks/q_random_explorer.ipynb`, Experimento 5).

  Teste A — Bacia do atrator dominante: enumera TODOS os atratores do sistema
  conjunto (Q_a, Q_b) via busca exaustiva nos 225x225 pares de estado inicial
  (`q_commons.find_all_attractors`) e reporta a fração de estados que caem no
  maior atrator. Se pares aleatórios já têm bacias dominantes tipicamente
  grandes, isso é combinatória de mapa aleatório, não seleção de equilíbrio.

  Teste B — Protocolo de impulse-response (o de Calvano/Experimento 6),
  reimplementado em numpy puro sobre as mesmas Q-tables: assenta no atrator,
  força 1 período de desvio (best-response estática ao rival), solta de volta
  ao greedy, e mede se e quando o par retorna EXATAMENTE ao preço pré-desvio.
  Compara taxa e velocidade de retorno, pares reais vs. modelo nulo.

Outputs:
  analysis/figures/fig_null_model_basin_fraction.{png,pdf}
  analysis/figures/fig_null_model_return_rate.{png,pdf}
  analysis/data/null_model_impulse_response.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from src.qtable_analysis.q_commons import (  # noqa: E402
    load_agent_and_grid, make_greedy_fn, find_all_attractors,
    compute_profit, P_NE, P_M,
)

PAIRS_DIR = ROOT / "results" / "cross_deploy__20260621_203040" / "training"
OUT_FIG   = ROOT / "projects" / "p1_deterrence" / "analysis" / "figures"
OUT_DATA  = ROOT / "projects" / "p1_deterrence" / "analysis" / "data"
OUT_DATA.mkdir(parents=True, exist_ok=True)

BLUE, RED, GRAY, ORANGE = "#2196F3", "#FF5722", "#757575", "#FF9800"

N_RANDOM_BASIN  = 50    # pares aleatórios p/ Teste A (find_all_attractors, ~1s cada)
N_RANDOM_IMPULSE = 100  # pares aleatórios p/ Teste B (barato, roda em ms)
N_INITS_PER_PAIR = 5    # condições iniciais aleatórias por par (real e nulo)
SETTLE, POST      = 500, 100   # mesma janela do Experimento 6 (settle=500, post=100)
RETURN_TOL        = 1e-9       # "retorno exato" — casamento na grade discreta

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


# ---------------------------------------------------------------------------
# Carregamento dos 18 pares reais
# ---------------------------------------------------------------------------

# A figura canônica do draft (EVIDENCE.md, Table 2c: 70.0% = 63/90, 4.9 vs 13.4
# steps) foi gerada com a população ORIGINAL de 18 runs (pair_000..pair_008);
# os 6 runs pair_009..pair_011 foram adicionados depois (extensão pós-audit de
# seeds) e não entram no null model. Sem este corte os números divergem dos do
# paper (24 runs -> 83/120 = 69.2%).
PAIR_LIMIT = 18

def load_real_pairs():
    pairs = []
    for d in sorted(PAIRS_DIR.iterdir())[:PAIR_LIMIT]:
        if not (d / "agents" / "agent_0.npz").exists():
            continue
        q0, grid, config = load_agent_and_grid(d, agent_idx=0)
        q1, _, _          = load_agent_and_grid(d, agent_idx=1)
        pairs.append({"name": d.name, "q0": q0, "q1": q1, "grid": grid, "config": config})
    return pairs


# ---------------------------------------------------------------------------
# Modelo nulo — Q-tables aleatórias, mesmo intervalo de valores dos reais
# ---------------------------------------------------------------------------

def random_pair(rng, n_states, n_actions, q_low, q_high):
    q0 = rng.uniform(q_low, q_high, size=(n_states, n_actions))
    q1 = rng.uniform(q_low, q_high, size=(n_states, n_actions))
    return q0, q1


# ---------------------------------------------------------------------------
# Segundo nulo — Q na inicialização (Calvano tipo 'O'), o "não-treinado" real
# ---------------------------------------------------------------------------
#
# O nulo uniforme aleatório e o nulo de inicialização são opostos, e a distinção
# importa: `opponent_optimized` replica UMA linha em todos os n_states estados,
# logo a política inicial é CONSTANTE (independente do estado) e o par tem
# trivialmente um único atrator global. Já uma Q uniforme tem argmax independente
# por linha — máxima dependência de estado, muitos atratores. Comparar os pares
# treinados só com o nulo uniforme responde a pergunta errada.

def init_pair(config, price_grid, n_states, rng=None, noise=0.0):
    """Par de Q-tables na inicialização 'O' de Calvano, opcionalmente perturbado."""
    from src.agents.q_learning import opponent_optimized_qtable

    n = len(price_grid)
    e = config["environment"]
    row = opponent_optimized_qtable(
        n, float(price_grid[0]), float(price_grid[-1]), 1,
        float(config["agent"]["gamma"]),
        float(e["demand"]["a"]), float(e["demand"]["mu"]),
        float(e["cost"]["c"]), float(e["demand"].get("a0", 0.0)),
    )
    base = np.tile(row, (n_states, 1))
    if noise <= 0.0 or rng is None:
        return base.copy(), base.copy()
    return (base + rng.normal(0, noise, base.shape),
            base + rng.normal(0, noise, base.shape))


# ---------------------------------------------------------------------------
# Teste B — impulse-response em numpy puro
# ---------------------------------------------------------------------------

def static_best_response(p_rival, price_grid, config):
    profits = [compute_profit(p, p_rival, config) for p in price_grid]
    return float(price_grid[int(np.argmax(profits))])


def run_impulse(q_a, q_b, price_grid, config, settle, post, seed,
                idx_a=0, idx_b=1):
    """Assenta, força 1 desvio, solta greedy. Retorna (pre_a, pre_b, return_step).

    idx_a/idx_b: slot de treino de cada Q-table. O agente 1 tem a tabela
    indexada (rival, own) — ver q_commons.make_greedy_fn. Até 2026-08-10 esta
    função lia a tabela do agente 1 transposta, o que embaralhava a política
    do braço REAL e o aproximava artificialmente do nulo aleatório.
    """
    rng = np.random.default_rng(seed)
    p_a = float(rng.choice(price_grid))
    p_b = float(rng.choice(price_grid))
    greedy_a = make_greedy_fn(q_a, price_grid, idx_a)
    greedy_b = make_greedy_fn(q_b, price_grid, idx_b)

    for _ in range(settle):
        p_a, p_b = greedy_a(p_a, p_b), greedy_b(p_b, p_a)
    pre_a, pre_b = p_a, p_b

    # t0: agente A forçado à best-response estática; B age normalmente
    forced   = static_best_response(p_b, price_grid, config)
    b_normal = greedy_b(p_b, p_a)
    p_a, p_b = forced, b_normal

    return_step = None
    for t in range(1, post + 1):
        p_a, p_b = greedy_a(p_a, p_b), greedy_b(p_b, p_a)
        if abs(p_a - pre_a) < RETURN_TOL and abs(p_b - pre_b) < RETURN_TOL:
            return_step = t
            break

    return pre_a, pre_b, return_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    real_pairs = load_real_pairs()
    print(f"{len(real_pairs)} pares reais carregados de {PAIRS_DIR.name}/")

    n_states  = real_pairs[0]["q0"].shape[0]
    n_actions = real_pairs[0]["q0"].shape[1]
    all_q = np.concatenate([np.concatenate([p["q0"], p["q1"]]) for p in real_pairs])
    q_low, q_high = float(all_q.min()), float(all_q.max())
    print(f"Range de Q real: [{q_low:.3f}, {q_high:.3f}]  (usado para amostrar o nulo)")
    config_ref = real_pairs[0]["config"]
    grid_ref   = real_pairs[0]["grid"]

    rows = []  # linhas do CSV: group, pair, seed, pre_a, pre_b, return_step, returned

    # -----------------------------------------------------------------
    # Teste B — impulse-response: reais
    # -----------------------------------------------------------------
    print("\nTeste B (impulse-response) — pares reais...")
    for p in real_pairs:
        for s in range(N_INITS_PER_PAIR):
            pre_a, pre_b, ret = run_impulse(p["q0"], p["q1"], p["grid"], p["config"],
                                            SETTLE, POST, seed=s)
            rows.append({"group": "real", "pair": p["name"], "seed": s,
                        "pre_a": pre_a, "pre_b": pre_b,
                        "return_step": ret if ret is not None else -1,
                        "returned": ret is not None})

    # -----------------------------------------------------------------
    # Teste B — impulse-response: modelo nulo
    # -----------------------------------------------------------------
    print("Teste B (impulse-response) — modelo nulo...")
    rng_master = np.random.default_rng(2026)
    for k in range(N_RANDOM_IMPULSE):
        q0, q1 = random_pair(rng_master, n_states, n_actions, q_low, q_high)
        for s in range(N_INITS_PER_PAIR):
            pre_a, pre_b, ret = run_impulse(q0, q1, grid_ref, config_ref,
                                            SETTLE, POST, seed=1000 * k + s)
            rows.append({"group": "random", "pair": f"rand_{k:03d}", "seed": s,
                        "pre_a": pre_a, "pre_b": pre_b,
                        "return_step": ret if ret is not None else -1,
                        "returned": ret is not None})

    import csv
    with open(OUT_DATA / "null_model_impulse_response.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {OUT_DATA / 'null_model_impulse_response.csv'}")

    real_ret   = [r["returned"] for r in rows if r["group"] == "real"]
    rand_ret   = [r["returned"] for r in rows if r["group"] == "random"]
    real_steps = [r["return_step"] for r in rows if r["group"] == "real" and r["returned"]]
    rand_steps = [r["return_step"] for r in rows if r["group"] == "random" and r["returned"]]

    rate_real, rate_rand = np.mean(real_ret), np.mean(rand_ret)
    print(f"\nTaxa de retorno exato (janela {POST} steps):")
    print(f"  Real:   {rate_real:.1%}  ({sum(real_ret)}/{len(real_ret)})"
         f"  |  step médio entre os que retornam: {np.mean(real_steps):.2f}")
    print(f"  Nulo:   {rate_rand:.1%}  ({sum(rand_ret)}/{len(rand_ret)})"
         f"  |  step médio entre os que retornam: {np.mean(rand_steps):.2f}")

    table = np.array([[sum(real_ret), len(real_ret) - sum(real_ret)],
                      [sum(rand_ret), len(rand_ret) - sum(rand_ret)]])
    odds, p_fisher = stats.fisher_exact(table)
    print(f"  Fisher exact (taxa de retorno, real vs nulo): odds={odds:.3f}  p={p_fisher:.4g}")

    if real_steps and rand_steps:
        u_stat, p_mw = stats.mannwhitneyu(real_steps, rand_steps, alternative="two-sided")
        print(f"  Mann-Whitney (step de retorno | retornou, real vs nulo): U={u_stat:.1f}  p={p_mw:.4g}")

    # -----------------------------------------------------------------
    # Teste A — bacia do atrator dominante
    # -----------------------------------------------------------------
    print("\nTeste A (bacia do atrator dominante) — pares reais...")
    basin_real = []
    period_real = []
    n_attr_real = []
    for p in real_pairs:
        attrs, _ = find_all_attractors(p["q0"], p["q1"], p["grid"])
        dom = max(attrs, key=lambda a: a["basin_fraction"])
        basin_real.append(dom["basin_fraction"])
        period_real.append(dom["period"])
        n_attr_real.append(len(attrs))
        print(f"  {p['name']:>10}: {len(attrs)} atratores, dominante bacia="
             f"{dom['basin_fraction']:.1%}  período={dom['period']}  CI={dom['ci_mean']:.3f}")

    # -----------------------------------------------------------------
    # Teste A — segundo nulo: Q na inicialização ('O'), com e sem ruído
    # -----------------------------------------------------------------
    print("\nTeste A — nulo de INICIALIZAÇÃO (q_init tipo 'O')...")
    rng_init = np.random.default_rng(4242)
    init_rows = []
    for noise in (0.0, 0.05, 0.20, 0.50):
        reps = 1 if noise == 0.0 else 20
        bas, per = [], []
        for _ in range(reps):
            qi0, qi1 = init_pair(config_ref, grid_ref, n_states,
                                 rng=rng_init, noise=noise)
            at, _ = find_all_attractors(qi0, qi1, grid_ref)
            dom = max(at, key=lambda a: a["basin_fraction"])
            bas.append(dom["basin_fraction"])
            per.append(len(at))
        init_rows.append((noise, float(np.mean(per)), float(np.mean(bas))))
        tag = "puro" if noise == 0.0 else f"+ ruído N(0,{noise})"
        print(f"  q_init {tag:<18}: {np.mean(per):>4.2f} atratores, "
              f"bacia dominante = {np.mean(bas):.1%}")

    print(f"\nTeste A (bacia do atrator dominante) — modelo nulo ({N_RANDOM_BASIN} pares)...")
    rng_master2 = np.random.default_rng(777)
    basin_rand, period_rand, n_attr_rand = [], [], []
    for k in range(N_RANDOM_BASIN):
        q0, q1 = random_pair(rng_master2, n_states, n_actions, q_low, q_high)
        attrs, _ = find_all_attractors(q0, q1, grid_ref)
        dom = max(attrs, key=lambda a: a["basin_fraction"])
        basin_rand.append(dom["basin_fraction"])
        period_rand.append(dom["period"])
        n_attr_rand.append(len(attrs))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{N_RANDOM_BASIN}...")

    print(f"\nBacia do atrator dominante:")
    print(f"  Real: média={np.mean(basin_real):.1%}  [{min(basin_real):.1%}, {max(basin_real):.1%}]")
    print(f"  Nulo: média={np.mean(basin_rand):.1%}  [{min(basin_rand):.1%}, {max(basin_rand):.1%}]")
    u2, p_mw2 = stats.mannwhitneyu(basin_real, basin_rand, alternative="two-sided")
    print(f"  Mann-Whitney (bacia dominante, real vs nulo): U={u2:.1f}  p={p_mw2:.4g}")

    percentile = 100.0 * np.mean([b < np.mean(basin_real) for b in basin_rand])
    print(f"  Média real cai no percentil {percentile:.0f} da distribuição nula")

    # -----------------------------------------------------------------
    # Quadro comparativo — os três regimes
    # -----------------------------------------------------------------
    print("\nQuadro comparativo (fragmentação do espaço de estados):")
    print(f"  {'regime':<34} {'#atratores':>10} {'bacia dom.':>11}")
    print("  " + "-" * 57)
    print(f"  {'q_init puro (política constante)':<34} "
          f"{init_rows[0][1]:>10.2f} {init_rows[0][2]:>11.1%}")
    for noise, per, bas in init_rows[1:]:
        print(f"  {'q_init + ruído N(0,' + str(noise) + ')':<34} {per:>10.2f} {bas:>11.1%}")
    print(f"  {'PARES TREINADOS':<34} "
          f"{np.mean(n_attr_real):>10.2f} {np.mean(basin_real):>11.1%}")
    print(f"  {'Q uniforme aleatória':<34} "
          f"{np.mean(n_attr_rand):>10.2f} {np.mean(basin_rand):>11.1%}")
    print("\n  Leitura: os pares treinados ficam no limite do nulo de INICIALIZAÇÃO"
          "\n  (atrator global único), não no do nulo uniforme aleatório.")

    # -----------------------------------------------------------------
    # Figuras
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.hist(basin_rand, bins=15, color=GRAY, alpha=0.7, label=f"Random (n={N_RANDOM_BASIN})")
    for i, b in enumerate(basin_real):
        ax.axvline(b, color=BLUE, lw=1.2, alpha=0.6,
                   label="Trained (18 pairs)" if i == 0 else None)
    ax.axvline(np.mean(basin_rand), color=GRAY, lw=2, ls="--")
    ax.axvline(np.mean(basin_real), color=RED, lw=2, ls="-", label="Trained mean")
    ax.set_xlabel("Fraction of states in the dominant attractor's basin")
    ax.set_ylabel("Number of pairs (null)")
    ax.set_title("Test A: dominant-basin size", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar(["Trained\n(18 pairs)", "Random\n(100 pairs)"], [rate_real, rate_rand],
          color=[BLUE, GRAY], width=0.55)
    for i, v in enumerate([rate_real, rate_rand]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"Exact-return rate within {POST} steps")
    ax.set_title(f"Test B: impulse response (Fisher p={p_fisher:.3g})", fontsize=10.5)

    fig.suptitle("Null model: random maps on the same 225-state space",
                fontsize=11, y=1.03)
    _save(fig, "fig_null_model_basin_fraction")

    # Segunda figura: distribuição do step de retorno (só quem retornou)
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
    bins = np.arange(1, POST + 2) - 0.5
    ax2.hist(real_steps, bins=bins, color=BLUE, alpha=0.6, density=True,
            label=f"Trained (returned: {sum(real_ret)}/{len(real_ret)})")
    ax2.hist(rand_steps, bins=bins, color=GRAY, alpha=0.6, density=True,
            label=f"Random (returned: {sum(rand_ret)}/{len(rand_ret)})")
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Steps until exact return to the pre-shock price")
    ax2.set_ylabel("Density (among returning trials)")
    ax2.set_title("Test B: return speed, trained vs random pairs", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5)
    _save(fig2, "fig_null_model_return_rate")

    print("\nOK.")


if __name__ == "__main__":
    main()
