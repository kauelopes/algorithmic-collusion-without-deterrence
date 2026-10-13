"""
Impulse-response à la Calvano et al. (2020) + overlay grim vs nash.

Dois protocolos sobre agentes Q-learning treinados (Q-tables .npz):

1. IMPULSE (within-pair): os dois agentes do par jogam greedy até assentar;
   no step t0 o agente 0 é forçado a jogar o best-response estático ao último
   preço do parceiro (o desvio de Calvano); depois ambos voltam ao greedy.
   Registra a janela [t0 - pre, t0 + post]. É o experimento com que Calvano
   exibe o "reward–punishment scheme" — aqui rodado no nosso ambiente.

2. OVERLAY (vs oponentes fixos): o agente treinado joga greedy contra grim e
   contra nash a partir do mesmo estado inicial. Se as trajetórias coincidem,
   o agente não distingue o oponente que pune do que joga Nash estático.

Usage:
  python experiments/impulse_response.py \
      [--pairs_dir results/cross_deploy__20260621_203040/training] \
      [--n_pairs 18] [--settle 500] [--pre 25] [--post 100] \
      [--overlay_steps 200] [--output_dir DIR]

Outputs:
  results/impulse_response__TS/
    <pair>/impulse.csv        # step, price_0, price_1, profit_0, profit_1
    <pair>/overlay_<strat>.csv
    summary.csv               # por par: preço pré/pós-choque, retorno ao nível
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import yaml

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

from experiments.strategy_test import _build_env, _strategy_config  # noqa: E402
from src.agents import q_learning  # noqa: E402


# ---------------------------------------------------------------------------
# Payoff helpers (logit demand, analytic)
# ---------------------------------------------------------------------------

def _logit_profit(p_own: float, p_rival: float, a: float, mu: float,
                  c: float, a0: float) -> float:
    e_own   = np.exp((a - p_own) / mu)
    e_rival = np.exp((a - p_rival) / mu)
    e_out   = np.exp(a0 / mu)
    return (p_own - c) * e_own / (e_own + e_rival + e_out)


def _static_best_response(p_rival: float, grid: np.ndarray, demand: dict,
                          c: float) -> float:
    profits = [_logit_profit(p, p_rival, demand["a"], demand["mu"], c,
                             demand.get("a0", 0.0)) for p in grid]
    return float(grid[int(np.argmax(profits))])


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _load_pair(pair_dir: Path):
    config = yaml.safe_load((pair_dir / "config.yaml").read_text())
    env    = _build_env(config["environment"])

    agent_config = dict(config["agent"])
    agent_config.setdefault("action_min", float(env.p_min))
    agent_config.setdefault("q_max", float(env.p_max))

    states = []
    for i in (0, 1):
        st = q_learning.load_state(pair_dir / "agents" / f"agent_{i}.npz")
        states.append(st._replace(epsilon=0.0))
    return config, env, agent_config, states


# ---------------------------------------------------------------------------
# Episode loops
# ---------------------------------------------------------------------------

def _greedy_step(env, modules, states, configs, obs_list, rng, forced_price_0=None):
    """One synchronous step; optionally force agent 0's price."""
    rng, rng0, rng1 = jax.random.split(rng, 3)
    a0, states[0] = modules[0].select_action(states[0], obs_list[0], rng0,
                                             configs[0], training=False)
    a1, states[1] = modules[1].select_action(states[1], obs_list[1], rng1,
                                             configs[1], training=False)
    if forced_price_0 is not None:
        a0 = forced_price_0
    next_obs, rewards, _, _ = env.step([a0, a1])
    return next_obs, (float(a0), float(a1)), (float(rewards[0]), float(rewards[1])), rng


def run_impulse(env, agent_config, states, demand, cost_c, grid,
                settle: int, pre: int, post: int, seed: int) -> list[dict]:
    modules = [q_learning, q_learning]
    configs = [agent_config, agent_config]
    sts     = list(states)
    obs     = env.reset(seed=seed)
    rng     = jax.random.PRNGKey(seed)

    rows = []
    t0 = settle
    for t in range(settle + post):
        forced = None
        if t == t0:
            p_rival_last = rows[-1]["price_1"] if rows else float(env.p_min)
            forced = _static_best_response(p_rival_last, grid, demand, cost_c)
        obs, prices, profits, rng = _greedy_step(env, modules, sts, configs,
                                                 obs, rng, forced_price_0=forced)
        if t >= t0 - pre:
            rows.append({"t": t - t0, "price_0": prices[0], "price_1": prices[1],
                         "profit_0": profits[0], "profit_1": profits[1],
                         "forced": int(t == t0)})
        else:
            rows.append({"t": t - t0, "price_0": prices[0], "price_1": prices[1],
                         "profit_0": profits[0], "profit_1": profits[1],
                         "forced": 0})
    return [r for r in rows if r["t"] >= -pre]


def run_overlay(env_config, agent_config, state0, strategy_name: str,
                n_steps: int, seed: int) -> list[dict]:
    import importlib
    env = _build_env(env_config)
    bm  = env.benchmarks()
    bm["p_min"], bm["p_max"] = float(env.p_min), float(env.p_max)

    strat_module = importlib.import_module(f"src.strategy_players.{strategy_name}")
    strat_cfg    = _strategy_config(strategy_name, bm, env_config)
    rng          = jax.random.PRNGKey(seed)
    rng, rng_s   = jax.random.split(rng)
    strat_state  = strat_module.init(rng_s, strat_cfg)

    obs   = env.reset(seed=seed)
    st0   = state0
    rows  = []
    for t in range(n_steps):
        rng, rng0, rng1 = jax.random.split(rng, 3)
        a0, st0 = q_learning.select_action(st0, obs[0], rng0, agent_config,
                                           training=False)
        a1, strat_state = strat_module.select_action(strat_state, obs[1], rng1,
                                                     strat_cfg, training=False)
        obs, rewards, _, _ = env.step([a0, a1])
        rows.append({"t": t, "price_agent": float(a0), "price_opp": float(a1),
                     "profit_agent": float(rewards[0])})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Impulse response + grim/nash overlay")
    p.add_argument("--pairs_dir", type=str,
                   default="results/cross_deploy__20260621_203040/training")
    p.add_argument("--n_pairs", type=int, default=99)
    p.add_argument("--settle", type=int, default=500)
    p.add_argument("--pre", type=int, default=25)
    p.add_argument("--post", type=int, default=100)
    p.add_argument("--overlay_steps", type=int, default=200)
    p.add_argument("--overlay_strategies", nargs="+",
                   default=["grim_player", "nash_player"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    pairs_dir = (_PROJ_ROOT / args.pairs_dir if not Path(args.pairs_dir).is_absolute()
                 else Path(args.pairs_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else (
        _PROJ_ROOT / "results" / f"impulse_response__{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    pair_dirs = sorted(d for d in pairs_dir.iterdir()
                       if d.is_dir() and (d / "agents" / "agent_0.npz").exists()
                       and (d / "agents" / "agent_1.npz").exists())[: args.n_pairs]
    print(f"Pairs: {len(pair_dirs)}  |  Output: {out_root}")

    summary = []

    # ---- Modo CROSS: agent_0 do par i com agent_1 do par j = i+1 (cíclico) ----
    # Controle do impulse within-pair: se o padrão desvio→queda→retorno é um
    # reflexo interno à díade, ele não deve aparecer entre agentes que nunca
    # se viram. Mesmo protocolo, mesma janela.
    for i, pair_dir in enumerate(pair_dirs):
        other = pair_dirs[(i + 1) % len(pair_dirs)]
        name = f"cross__{pair_dir.name}__{other.name}"
        config, env, agent_config, states_i = _load_pair(pair_dir)
        _, _, _, states_j = _load_pair(other)
        states = [states_i[0], states_j[1]]
        demand = config["environment"]["demand"]
        cost_c = float(config["environment"]["cost"]["c"])
        grid   = np.linspace(float(env.p_min), float(env.p_max),
                             int(agent_config["n_actions"]))

        pair_out = out_root / "cross" / name
        pair_out.mkdir(parents=True, exist_ok=True)
        rows = run_impulse(env, agent_config, states, demand, cost_c, grid,
                           args.settle, args.pre, args.post, args.seed)
        with open(pair_out / "impulse.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        pre_rows  = [r for r in rows if r["t"] < 0]
        post_tail = [r for r in rows if r["t"] > args.post - 20]
        pre_p  = float(np.mean([r["price_0"] for r in pre_rows]))
        dev_p  = next(r["price_0"] for r in rows if r["forced"])
        tail_p = float(np.mean([r["price_0"] for r in post_tail]))
        rival_min_post = float(min(r["price_1"] for r in rows if 0 <= r["t"] <= 10))
        summary.append({"mode": "cross", "pair": name,
                        "pre_price_0": round(pre_p, 4),
                        "deviation_price": round(dev_p, 4),
                        "tail_price_0": round(tail_p, 4),
                        "rival_min_10steps": round(rival_min_post, 4),
                        "returns_to_pre": int(abs(tail_p - pre_p) < 1e-3)})
        print(f"  [cross] {name}: pre={pre_p:.3f} dev={dev_p:.3f} tail={tail_p:.3f}")

    # ---- Modo WITHIN: par co-treinado ----
    for pair_dir in pair_dirs:
        name = pair_dir.name
        config, env, agent_config, states = _load_pair(pair_dir)
        demand = config["environment"]["demand"]
        cost_c = float(config["environment"]["cost"]["c"])
        grid   = np.linspace(float(env.p_min), float(env.p_max),
                             int(agent_config["n_actions"]))

        pair_out = out_root / name
        pair_out.mkdir(parents=True, exist_ok=True)

        # 1. Impulse
        rows = run_impulse(env, agent_config, states, demand, cost_c, grid,
                           args.settle, args.pre, args.post, args.seed)
        with open(pair_out / "impulse.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        pre_rows  = [r for r in rows if r["t"] < 0]
        post_tail = [r for r in rows if r["t"] > args.post - 20]
        pre_p  = float(np.mean([r["price_0"] for r in pre_rows]))
        dev_p  = next(r["price_0"] for r in rows if r["forced"])
        tail_p = float(np.mean([r["price_0"] for r in post_tail]))
        rival_min_post = float(min(r["price_1"] for r in rows if 0 <= r["t"] <= 10))
        summary.append({"mode": "within", "pair": name,
                        "pre_price_0": round(pre_p, 4),
                        "deviation_price": round(dev_p, 4),
                        "tail_price_0": round(tail_p, 4),
                        "rival_min_10steps": round(rival_min_post, 4),
                        "returns_to_pre": int(abs(tail_p - pre_p) < 1e-3)})
        print(f"  {name}: pre={pre_p:.3f} dev={dev_p:.3f} "
              f"tail={tail_p:.3f} rival_min={rival_min_post:.3f}")

        # 2. Overlay vs fixed opponents (agent 0 only)
        for strat in args.overlay_strategies:
            orows = run_overlay(config["environment"], agent_config, states[0],
                                strat, args.overlay_steps, args.seed)
            with open(pair_out / f"overlay_{strat}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(orows[0].keys()))
                w.writeheader(); w.writerows(orows)

    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nSummary → {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
