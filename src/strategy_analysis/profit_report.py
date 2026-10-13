"""
Ficha de agente — relatório markdown por run, centrado em LUCRO (p3).

Substitui a narrativa por CI do report antigo (`report.py`, congelado para o p1):
contra um oponente fixo o CI conjunto mistura o preço do oponente na média e não
informa nada — contra um `nash_player` constante o ótimo é um preço constante. As
grandezas desta ficha:

  π̄ próprio e do oponente, assimetria (Δπ, razão)   o que o CI escondia
  π̄ / π*_const                                       fração do melhor compromisso
                                                      constante contra ESTE oponente
                                                      (exato p/ oponente constante;
                                                      referência p/ os reativos — A15)
  discounted_value                                    a grandeza que o agente maximiza
  policy_value_score / action_quality                 ordem da decisão contra o teto
                                                      exato, nulo embaralhado ao lado
                                                      (só oponentes Markov-1)
  CI de lucro próprio                                 SECUNDÁRIO, por comparabilidade

Toda linha numérica carrega o nulo/teto ao lado — regra das 4 coisas da metodologia.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import jax
import numpy as np
import yaml

from src.agents.q_learning import n_internal
from src.metrics.multi_agent import collusion_index_profit, discounted_value
from src.qtable_analysis.optimal import (
    _MARKOV1_PLAYERS,
    action_quality,
    policy_of_agent,
    policy_value_score,
    solve_optimal_vs_B,
    strategy_player_response,
)
from src.strategy_analysis.runner import run_episode

ALL_STRATEGIES = [
    "nash_player", "competitive_player", "monopoly_player",
    "tit_for_tat_player", "grim_player", "wsls_player",
    "imitation_player", "imitation_plus_player",
    "random_player", "noise_nash_player",
]

SHORT = {"nash_player": "nash", "competitive_player": "compet",
         "monopoly_player": "monop", "tit_for_tat_player": "tft",
         "grim_player": "grim", "wsls_player": "wsls",
         "imitation_player": "imit", "imitation_plus_player": "imit+",
         "random_player": "rand", "noise_nash_player": "n-nash"}

_CONST_STEPS = 300     # simulação por preço constante (mesma escala do A15)
_CONST_WARMUP = 100


# ---------------------------------------------------------------------------
# Infra
# ---------------------------------------------------------------------------

def _build_env(env_config: dict):
    from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
    return MultiAgentBertrandEnv(env_config)


def _strategy_config(name: str, bm: dict, env_config: dict, env) -> dict:
    """Idêntico ao do pool de treino (a15/a14) — divergir mediria outro player."""
    p_NE, p_M = float(bm["p_NE"]), float(bm["p_monopoly"])
    return {
        "type": name,
        "p_NE": p_NE, "p_monopoly": p_M,
        "p_competitive": float(env_config["cost"]["c"]),
        "p_min": float(env.p_min), "p_max": float(env.p_max),
        "n_agents": int(env_config["market"]["n_firms"]),
        "own_idx": 1,
        "defect_threshold": 0.05,
        "noise_sigma": 0.05,
        "rate": 0.1, "adjust": 0.5,
        "shift_delta": (p_M - p_NE) / 7,
    }


def _const_episode_profit(env, price: float, b_mod, b_cfg, rng) -> float:
    """Lucro médio (pós-warmup) de jogar `price` em todo período contra o player.

    O player recebe `update()` — é onde grim arma o gatilho e wsls compara lucros;
    sem isso grim vira monopoly_player (lição do A15).
    """
    from src.agents.q_learning import Transition

    env.reset(seed=0)
    env._last_prices[0] = np.float32(price)
    b_state = b_mod.init(rng, b_cfg)
    obs_b = env._obs_for(1)
    profits = []
    for _ in range(_CONST_STEPS):
        rng, rb = jax.random.split(rng, 2)
        act_b, b_state = b_mod.select_action(b_state, obs_b, rb, b_cfg, training=False)
        nxt, rew, _, _ = env.step([price, float(act_b)])
        b_state, _ = b_mod.update(
            b_state,
            Transition(obs=obs_b, action=float(act_b), reward=float(rew[1]),
                       next_obs=nxt[1], done=False),
            b_cfg,
        )
        profits.append(float(rew[0]))
        obs_b = nxt[1]
    return float(np.mean(profits[_CONST_WARMUP:]))


def _best_const(env, grid, b_mod, b_cfg, rng) -> tuple[float, float]:
    """(π*_const, preço) — o melhor compromisso de preço constante da grade contra o
    player. Exato para oponente constante; para os reativos é a REFERÊNCIA do A15
    (não o teto dinâmico — esse exige induzir o oponente)."""
    best_pi, best_p = -np.inf, float("nan")
    for p in grid:
        v = _const_episode_profit(env, float(p), b_mod, b_cfg, rng)
        if v > best_pi:
            best_pi, best_p = v, float(p)
    return best_pi, best_p


# ---------------------------------------------------------------------------
# Ficha
# ---------------------------------------------------------------------------

def generate_ficha(
    run_dir: str | Path,
    n_steps: int = 1000,
    seed: int = 0,
    agent_idx: int = 0,
    strategies: list[str] | None = None,
    out_name: str = "ficha.md",
) -> Path:
    """Gera a ficha markdown do agente em `run_dir/ficha.md` (+ figura em plots/)."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    env_config = config["environment"]
    agent_config = config["agent"]
    gamma = float(agent_config["gamma"])
    strategies = strategies or ALL_STRATEGIES

    env = _build_env(env_config)
    bm = env.benchmarks()
    # bounds de ação que os trainers injetam em runtime (não vivem no YAML salvo)
    agent_config.setdefault("action_min", float(env.p_min))
    agent_config.setdefault("q_max", float(env.p_max))
    n_act = int(agent_config["n_actions"])
    grid = np.linspace(float(env.p_min), float(env.p_max), n_act)
    pi_NE = float(bm["pi_NE"])
    pi_M = float(bm.get("pi_monopoly_per_firm", float("nan")))

    agent_mod = importlib.import_module(f"src.agents.{agent_config['type']}")
    npz = run_dir / "agents" / f"agent_{agent_idx}.npz"
    state = agent_mod.load_state(npz)
    state = state._replace(epsilon=float(agent_config.get("epsilon_end", 0.01)))
    q_table = np.asarray(state.q_table)
    M = n_internal(agent_config)

    rng = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------ painel
    rows = []
    episodes = {}
    for name in strategies:
        s_mod = importlib.import_module(f"src.strategy_players.{name}")
        s_cfg = _strategy_config(name, bm, env_config, env)
        rng, r_ep, r_c = jax.random.split(rng, 3)
        s_state = s_mod.init(r_ep, s_cfg)
        ep = run_episode(env, agent_mod, state, agent_config,
                         s_mod, s_state, s_cfg, n_steps, r_ep,
                         trained_agent_idx=0)
        episodes[name] = ep
        pa = np.asarray(ep["profit_a"]); pb = np.asarray(ep["profit_b"])
        pi_star, p_star = _best_const(env, grid, s_mod, s_cfg, r_c)
        mean_a = float(pa.mean())
        rows.append({
            "opp": name,
            "pi_a": mean_a, "pi_a_std": float(pa.std()),
            "pi_b": float(pb.mean()),
            "delta": mean_a - float(pb.mean()),
            "ratio": mean_a / float(pb.mean()) if pb.mean() else float("nan"),
            "pi_star": pi_star, "p_star": p_star,
            "frac_star": mean_a / pi_star if pi_star else float("nan"),
            "disc": discounted_value(ep["profit_a"], gamma),
            "ci_profit_own": collusion_index_profit(mean_a, pi_NE, pi_M),
            "price_a": float(np.mean(ep["price_a"])),
            "price_b": float(np.mean(ep["price_b"])),
        })

    # -------------------------------------------------------- scores de decisão
    # ⚠️ `policy_of_agent`/`policy_value_score` vivem no espaço (i, j) de 15×15.
    # Com `obs_radices` (estado expandido) a Q-table tem outro layout e os scores
    # sairiam de linhas erradas SEM ERRO — melhor não computar do que computar errado
    # (instrumento não portado; registrar na COBERTURA como "não pode rodar").
    scores = []
    if (agent_config["type"] == "q_learning" and M == 1
            and agent_config.get("obs_radices") is None):
        pi_mat, _ = policy_of_agent(q_table, agent_idx, grid, 1)
        pi2d = pi_mat[:, :, 0]
        for name in strategies:
            if name not in _MARKOV1_PLAYERS:
                continue
            s_mod = importlib.import_module(f"src.strategy_players.{name}")
            s_cfg = _strategy_config(name, bm, env_config, env)
            B = strategy_player_response(s_mod, s_cfg, grid)
            sc = policy_value_score(pi2d, B, grid, config, gamma)
            Q_star, _ = solve_optimal_vs_B(B, grid, config, gamma)
            aq = action_quality(pi2d, Q_star)
            scores.append({"opp": name, **sc, "qualidade": aq["qualidade"],
                           "qualidade_nulo": aq["qualidade_nulo_uniforme"]})

    # ------------------------------------------------------------ curvas de treino
    treino = _training_section(run_dir)

    # ------------------------------------------------------------------- figura
    fig_rel = _painel_figure(run_dir, episodes, bm)

    # ----------------------------------------------------------------- markdown
    md = _render(run_dir, config, bm, rows, scores, treino, fig_rel,
                 n_steps, pi_NE, pi_M)
    out = run_dir / out_name
    out.write_text(md, encoding="utf-8")

    # sidecar JSON — o runner de triagem agrega daqui, sem parsear markdown
    import json
    fracs = [r["frac_star"] for r in rows if np.isfinite(r["frac_star"])]
    (run_dir / "ficha.json").write_text(json.dumps({
        "opponents": rows,
        "decision_scores": scores,
        "training": treino,
        "frac_star_mean": float(np.mean(fracs)) if fracs else float("nan"),
        "frac_star_min": float(np.min(fracs)) if fracs else float("nan"),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return out


def _training_section(run_dir: Path) -> dict:
    hist = run_dir / "metrics_history.csv"
    if not hist.exists():
        return {}
    rows = list(csv.DictReader(hist.open(encoding="utf-8")))
    if not rows:
        return {}

    def _f(row, key):
        try:
            return float(row.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    first, last = rows[0], rows[-1]
    out = {
        "n_evals": len(rows),
        "step_final": last.get("step"),
        "reward_first": [_f(first, f"mean_reward_{i}") for i in (0, 1)],
        "reward_last": [_f(last, f"mean_reward_{i}") for i in (0, 1)],
        "td_last": [_f(last, f"td_error_mean_{i}") for i in (0, 1)],
        "eps_last": _f(last, "epsilon"),
    }
    # step em que ε chegou perto do fim (≤ 1.05×ε_final) — diagnóstico de schedule
    eps_final = out["eps_last"]
    if np.isfinite(eps_final):
        for r in rows:
            if _f(r, "epsilon") <= eps_final * 1.05:
                out["eps_done_step"] = r.get("step")
                break
    return out


def _painel_figure(run_dir: Path, episodes: dict, bm: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(episodes)
    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.2 * nrows),
                             sharex=True, squeeze=False)
    T = 150   # o transiente é onde a reação aparece
    for k, name in enumerate(names):
        ax = axes[k // ncols][k % ncols]
        ep = episodes[name]
        ax.plot(ep["price_a"][:T], lw=1.1, label="agente", color="#2196F3")
        ax.plot(ep["price_b"][:T], lw=1.1, label=SHORT.get(name, name),
                color="#FF5722", alpha=0.8)
        ax.axhline(bm["p_NE"], color="black", lw=0.6, ls="--")
        ax.axhline(bm["p_monopoly"], color="#9C27B0", lw=0.6, ls=":")
        ax.set_title(SHORT.get(name, name), fontsize=9)
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(names), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Reação contra estratégias fixas — preços (primeiros 150 períodos)",
                 fontsize=11)
    plt.tight_layout()
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / "ficha_painel.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "plots/ficha_painel.png"


def _render(run_dir, config, bm, rows, scores, treino, fig_rel,
            n_steps, pi_NE, pi_M) -> str:
    exp = config.get("experiment_name", run_dir.name)
    lines = [
        f"# Ficha do agente — {exp}",
        "",
        f"Run: `{run_dir.name}` · episódio de avaliação: {n_steps} períodos, guloso.",
        f"Benchmarks: π_NE/firma = {pi_NE:.4f} · π_M/firma = {pi_M:.4f} · "
        f"p_NE = {float(bm['p_NE']):.4f} · p_M = {float(bm['p_monopoly']):.4f}",
        "",
        "> CI é métrica **secundária** nesta ficha. Contra oponente fixo o número que",
        "> informa é π̄/π*_const — a fração do melhor compromisso constante contra ELE.",
        "",
        "## Curvas de treino",
        "",
    ]
    if treino:
        r0, r1 = treino["reward_first"], treino["reward_last"]
        lines += [
            f"- π̄ por janela: {r0[0]:.4f} → **{r1[0]:.4f}** (agente 0); "
            f"{r0[1]:.4f} → {r1[1]:.4f} (agente/oponente 1)",
            f"- |TD| final: {treino['td_last'][0]:.5f} (agente 0) · "
            f"ε final: {treino['eps_last']:.4f}"
            + (f" · ε estabilizou no step {treino.get('eps_done_step')}"
               if treino.get("eps_done_step") else ""),
            f"- {treino['n_evals']} avaliações até o step {treino['step_final']}",
            "",
            "![Training curves](plots/training_curves.png)",
            "",
        ]
    else:
        lines += ["(sem metrics_history.csv no run)", ""]

    lines += [
        "## Reação contra estratégias fixas",
        "",
        f"![Painel]({fig_rel})",
        "",
        "| oponente | π̄ agente ± dp | π̄ opon. | Δπ | razão | π*_const (@p) | **π̄/π\\*** | V desc. | p̄ ag. | p̄ op. | CI-π próprio |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {SHORT.get(r['opp'], r['opp'])} "
            f"| {r['pi_a']:.4f} ± {r['pi_a_std']:.3f} "
            f"| {r['pi_b']:.4f} "
            f"| {r['delta']:+.4f} "
            f"| {r['ratio']:.3f} "
            f"| {r['pi_star']:.4f} (@{r['p_star']:.3f}) "
            f"| **{r['frac_star']:.3f}** "
            f"| {r['disc']:.3f} "
            f"| {r['price_a']:.3f} "
            f"| {r['price_b']:.3f} "
            f"| {r['ci_profit_own']:+.3f} |"
        )
    fracs = [r["frac_star"] for r in rows if np.isfinite(r["frac_star"])]
    if fracs:
        lines += [
            "",
            f"**π̄/π\\* médio no painel: {np.mean(fracs):.3f}** "
            f"(mín {np.min(fracs):.3f} em "
            f"{SHORT.get(rows[int(np.argmin(fracs))]['opp'])}).",
        ]

    if scores:
        lines += [
            "",
            "## Scores de decisão (oponentes Markov-1, teto exato por Bellman)",
            "",
            "`score_shuffle` = (V^π − V^emb)/(V* − V^emb); o nulo embaralhado preserva a",
            "marginal de ações. Score alto = bom **explorador** do oponente congelado.",
            "",
            "| oponente | score_shuffle | gap_bruto | frac>nulo | qualidade (A11) | nulo unif. |",
            "|---|---|---|---|---|---|",
        ]
        for s in scores:
            lines.append(
                f"| {SHORT.get(s['opp'], s['opp'])} "
                f"| **{s['score_shuffle']:+.3f}** "
                f"| {s['gap_bruto']:.4f} "
                f"| {s['frac_estados_acima_do_nulo']:.3f} "
                f"| {s['qualidade']:.3f} "
                f"| {s['qualidade_nulo']:.3f} |"
            )

    lines += [
        "",
        "---",
        "Gerado por `src/strategy_analysis/profit_report.py` (p3). "
        "π*_const é exato para oponente constante e referência (A15) para os reativos; "
        "o teto dinâmico contra reativos exige induzir o oponente e não está nesta ficha.",
        "",
    ]
    return "\n".join(lines)
