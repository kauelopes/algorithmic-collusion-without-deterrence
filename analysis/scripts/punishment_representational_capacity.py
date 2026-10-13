"""
Capacidade representacional de punição no esquema de estados de Calvano.

Pergunta: a matriz de política s = (p_próprio_{t-1}, p_rival_{t-1}) -> p_t
(15x15 = 225 estados, memory_length=2 no código deste projeto, equivalente
ao Memory=1 do Fortran de Calvano) É CAPAZ, estruturalmente, de representar
uma lógica de punição-e-reversão? Ou a arquitetura já proíbe esse tipo de
estratégia, tornando o "achado" de Calvano uma coincidência impossível de
outra forma?

Método: prova construtiva. Construímos duas políticas por regra fixa (não
aprendidas) no MESMO espaço de estados de 225 células usado pelos agentes
reais (mesma grade, mesmo grid_extend=1.0, mesmos benchmarks, mesma
convenção de indexação de src/agents/q_learning.py), materializadas como
Q-tables (Q=1 na ação escolhida, Q=0 nas demais) e simuladas no MESMO
ambiente (src/environment/multi_agent_bertrand_env.py) contra uma sequência
de preço do rival com um desvio único forçado.

  - "reactive"  : pune (Nash) sempre que o preço do rival no período anterior
                  estava abaixo do limiar; senão coopera. Não usa o próprio
                  preço passado — só 1 dígito de memória do rival.
  - "grim"      : usa o próprio preço passado como flag de fase: uma vez que
                  a própria ação vira Nash, o estado seguinte terá
                  own_{t-1}=Nash, e a regra "se own_{t-1}==Nash, jogue Nash"
                  a mantém lá para sempre (punição permanente, sem perdão).

Se ambas simulam corretamente punição+reversão (reactive) e punição
permanente (grim) dentro do MESMO esquema de estados usado pelos agentes
reais, a hipótese de impossibilidade estrutural fica refutada para
`rival_prices`. O que resta then é comparar essa política "capaz por
construção" com o heatmap REALMENTE aprendido (fig_policy_heatmap.png,
sal-e-pimenta, ja gerado por plot_policy_functions.py) para isolar
representação de aprendizado.

Outputs:
  analysis/figures/fig_constructive_punishment_trajectories.{png,pdf}
  analysis/figures/fig_constructive_policy_heatmap.{png,pdf}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
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

from src.agents import q_learning
from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv

OUT = ROOT / "projects" / "p1_deterrence" / "analysis" / "figures"

BLUE, RED, GRAY, ORANGE, GREEN = "#2196F3", "#FF5722", "#757575", "#FF9800", "#4CAF50"

# Mesmos parâmetros e grid_extend=1.0 dos 18 pares co-treinados do
# cross_deploy__20260621_203040 (ver config.yaml de pair_000_A) — mesma
# arena onde o heatmap real (sal-e-pimenta) foi medido.
A, MU, C, A0 = 2.0, 0.25, 1.0, 0.0
N_ACTIONS = 15

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def _env_and_grid():
    config = {
        "demand": {"a": A, "mu": MU, "a0": A0},
        "cost": {"c": C},
        "market": {"n_firms": 2, "grid_extend": 1.0, "init_mode": "p_min"},
        "state": {"info_structure": "rival_prices"},
    }
    env = MultiAgentBertrandEnv(config)
    grid = np.linspace(env.p_min, env.p_max, N_ACTIONS)
    return env, grid


def _idx(price, grid):
    return int(np.argmin(np.abs(grid - price)))


def build_qtable(grid, p_NE, p_c, threshold, mode):
    """Constrói Q-table (225, 15) por regra fixa — não é aprendida.

    Estado s = idx(own_{t-1}) * 15 + idx(rival_{t-1}), na convenção de
    src/agents/q_learning.py._obs_to_state_idx aplicada ao agente que ocupa
    o slot 0 do ambiente (obs = [p0, p1] = [own, rival] para esse slot).
    """
    n = len(grid)
    i_NE = _idx(p_NE, grid)
    i_c = _idx(p_c, grid)
    Q = np.zeros((n * n, n), dtype=np.float64)

    for own_i in range(n):
        for riv_i in range(n):
            s = own_i * n + riv_i
            riv_p = grid[riv_i]
            own_was_NE = (own_i == i_NE)

            if mode == "reactive":
                # 1 dígito de memória do rival, esquece a própria fase —
                # pune enquanto o rival estiver abaixo do limiar, perdoa
                # assim que ele volta a cooperar. Não usa own_{t-1}.
                a_idx = i_NE if riv_p < threshold else i_c
            elif mode == "grim":
                # own_{t-1} funciona como flag de fase: uma vez punido,
                # o estado seguinte tem own_{t-1}=NE e a regra o mantém lá.
                if own_was_NE:
                    a_idx = i_NE
                elif riv_p < threshold:
                    a_idx = i_NE
                else:
                    a_idx = i_c
            else:
                raise ValueError(mode)

            Q[s, a_idx] = 1.0
    return Q


def simulate(Q, grid, env, rival_schedule, own_agent_idx=0, init_prev=None):
    """Roda o agente construído (slot 0) contra uma sequência de preço do
    rival IMPOSTA (não reativa a nós) — mesmo protocolo dos experimentos de
    impulse-response já existentes no projeto (desvio único forçado).

    `init_prev=(own_prev, riv_prev)` fixa o estado conjunto inicial. O default
    (None) mantém o comportamento original, o reset do ambiente (p_min, p_min).
    ⚠️ Com o reset, uma regra grim lê o próprio p_min inicial como fase de
    punição e nunca coopera — o mesmo artefato que a auditoria (ii) do paper
    evita iniciando no estado cooperativo. Para a figura do paper, passe
    (p_c, p_c).
    """
    config = {
        "n_actions": N_ACTIONS,
        "q_max": float(grid[-1]),
        "action_min": float(grid[0]),
        "memory_length": 2,
        "epsilon_start": 0.0, "epsilon_end": 0.0, "epsilon_decay_steps": 1,
        "gamma": 0.95, "alpha": 0.0,
    }
    state = q_learning.QLearningState(q_table=Q, epsilon=0.0, step=0)

    env.reset()
    own_prices, riv_prices = [], []
    # obs inicial: preços de reset (p_min, p_min), salvo `init_prev`
    if init_prev is None:
        own_prev, riv_prev = env.p_min, env.p_min
    else:
        own_prev, riv_prev = init_prev

    for riv_p in rival_schedule:
        obs = np.array([own_prev, riv_prev], dtype=np.float32)
        own_action = q_learning.greedy_action(state, obs, config)
        actions = [None, None]
        actions[own_agent_idx] = own_action
        actions[1 - own_agent_idx] = riv_p
        env.step(actions)
        own_prices.append(own_action)
        riv_prices.append(riv_p)
        own_prev, riv_prev = own_action, riv_p

    return np.array(own_prices), np.array(riv_prices)


def fig_trajectories(env, grid, p_NE, p_c, threshold):
    """Desvio único forçado do rival: coopera, desvia 5 períodos, retorna."""
    T_PRE, T_DEV, T_POST = 15, 5, 30
    schedule = (
        [p_c] * T_PRE + [env.p_min] * T_DEV + [p_c] * T_POST
    )
    t = np.arange(len(schedule))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    for ax, mode, title in (
        (axes[0], "reactive", "Reactive policy (1-lag trigger, forgiving)"),
        (axes[1], "grim", "Grim policy (own-price-as-phase-flag, permanent)"),
    ):
        Q = build_qtable(grid, p_NE, p_c, threshold, mode)
        # estado inicial cooperativo: o gatilho começa desarmado (como na
        # auditoria (ii) do paper); sem isso o grim pune desde t=0.
        own_p, riv_p = simulate(Q, grid, env, schedule, init_prev=(p_c, p_c))

        ax.plot(t, riv_p, color=GRAY, lw=1.6, ls="--", label="Rival price (scripted)")
        ax.plot(t, own_p, color=BLUE, lw=2.0, label="Constructed agent's price")
        ax.axhline(p_NE, color=RED, lw=0.8, ls=":", alpha=0.7)
        ax.axhline(p_c, color=GREEN, lw=0.8, ls=":", alpha=0.7)
        ax.axvspan(T_PRE, T_PRE + T_DEV, color=ORANGE, alpha=0.12,
                   label="Rival's one-shot deviation")
        ax.set_ylabel("Price")
        ax.set_title(title, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].set_xlabel("Period $t$")
    fig.suptitle(
        "Hand-built policies on the same 225-state matrix as the trained agents",
        fontsize=10, y=1.0)
    _save(fig, "fig_constructive_punishment_trajectories")


def fig_heatmaps(env, grid, p_NE, p_c, threshold):
    n = len(grid)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, mode, title in (
        (axes[0], "reactive", "Constructed: reactive (representable)"),
        (axes[1], "grim", "Constructed: grim (representable)"),
    ):
        Q = build_qtable(grid, p_NE, p_c, threshold, mode)
        pol = Q.argmax(axis=1)
        m = grid[pol].reshape(n, n)  # [own, rival] -> price
        im = ax.imshow(m, origin="lower", cmap="Blues",
                       vmin=grid[0], vmax=grid[-1],
                       extent=(grid[0], grid[-1], grid[0], grid[-1]),
                       aspect="auto")
        for v, ls in ((p_NE, "--"), (p_c, ":")):
            ax.axhline(v, color=GRAY, lw=0.9, ls=ls, alpha=0.8)
            ax.axvline(v, color=GRAY, lw=0.9, ls=ls, alpha=0.8)
        ax.set_xlabel("Rival's price in $t-1$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Own price in $t-1$")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label("Price chosen in $t$", fontsize=9)
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        "Hand-built policies show sharp threshold structure — contrast with\n"
        "the learned heatmap (fig_policy_heatmap.png), which is salt-and-pepper",
        fontsize=10.5)
    _save(fig, "fig_constructive_policy_heatmap")


if __name__ == "__main__":
    env, grid = _env_and_grid()
    p_NE = float(env.benchmarks()["p_NE"])
    p_M = float(env.benchmarks()["p_monopoly"])
    p_c = grid[_idx(p_M, grid)]          # preço "cooperativo" de referência = monopólio
    threshold = p_NE + 0.3 * (p_c - p_NE)  # mesmo espírito do defect_threshold dos strategy players

    print(f"p_NE={p_NE:.4f}  p_M={p_M:.4f}  p_c(grid)={p_c:.4f}  threshold={threshold:.4f}")
    print("Gerando figuras da prova construtiva:")
    fig_trajectories(env, grid, p_NE, p_c, threshold)
    fig_heatmaps(env, grid, p_NE, p_c, threshold)
