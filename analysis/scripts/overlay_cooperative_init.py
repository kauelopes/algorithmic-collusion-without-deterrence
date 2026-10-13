"""
Overlay grim/nash/monopoly com INICIALIZACAO COOPERATIVA — corrige o confound
do overlay original (results/impulse_response__20260807/overlay_*.csv), no qual
o estado inicial aleatorio fazia o grim disparar em t=0 e tornar-se
comportamentalmente identico ao nash player (tautologia apontada na revisao,
achado CRITICAL 2 do devil's advocate, 2026-08-07).

Desenho: o estado conjunto inicial e forcado ao preco de grade mais proximo do
nivel on-path within-pair (1.785 > gatilho do grim 1.523), de modo que o grim
comeca COOPERATIVO (p_M). Oponentes: grim_player, nash_player, monopoly_player
(o monopoly e o controle: identico ao grim enquanto o grim nao dispara — a
diferenca de trajetoria do agente entre grim e monopoly isola a resposta a
punicao de fato executada).

Saida:
  results/overlay_coop_init__20260807/<pair>/overlay_<strat>.csv
  results/overlay_coop_init__20260807/summary.csv  (trigger step, identidades)

Uso (container tf_env, CPU):
  python analysis/scripts/overlay_cooperative_init.py \
      --sweep_dir results/cross_deploy__20260621_203040/training \
      [--n_steps 200] [--p_init 1.785] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import jax
import numpy as np
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

from experiments.impulse_response import (  # noqa: E402
    _build_env, _load_pair, _strategy_config,
)
from src.agents import q_learning  # noqa: E402

STRATEGIES = ["grim_player", "nash_player", "monopoly_player"]


def run_overlay_coop(env_config, agent_config, state0, strategy_name: str,
                     n_steps: int, p_init: float, seed: int) -> list[dict]:
    env = _build_env(env_config)
    bm = env.benchmarks()
    bm["p_min"], bm["p_max"] = float(env.p_min), float(env.p_max)

    strat_module = importlib.import_module(f"src.strategy_players.{strategy_name}")
    strat_cfg = _strategy_config(strategy_name, bm, env_config)
    rng = jax.random.PRNGKey(seed)
    rng, rng_s = jax.random.split(rng)
    strat_state = strat_module.init(rng_s, strat_cfg)

    env.reset(seed=seed)
    # inicializacao cooperativa: estado conjunto em p_init (acima do gatilho
    # do grim, p_NE + 0.05); a discretizacao de estado do agente faz o binning
    p0 = float(np.clip(p_init, env.p_min, env.p_max))
    env._last_prices = np.full(env.n, p0, dtype=np.float32)
    obs = [env._obs_for(i) for i in range(env.n)]

    st0 = state0
    rows = []
    for t in range(n_steps):
        rng, rng0, rng1 = jax.random.split(rng, 3)
        a0, st0 = q_learning.select_action(st0, obs[0], rng0, agent_config,
                                           training=False)
        a1, strat_state = strat_module.select_action(strat_state, obs[1], rng1,
                                                     strat_cfg, training=False)
        obs, rewards, _, _ = env.step([a0, a1])
        rows.append({"t": t, "price_agent": float(a0), "price_opp": float(a1),
                     "profit_agent": float(rewards[0]),
                     "p_init": p0})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", type=str,
                   default="results/cross_deploy__20260621_203040/training")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--p_init", type=float, default=1.785)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="results/overlay_coop_init__20260807")
    args = p.parse_args()

    sweep = (_PROJ_ROOT / args.sweep_dir if not Path(args.sweep_dir).is_absolute()
             else Path(args.sweep_dir))
    out_root = (_PROJ_ROOT / args.output_dir
                if not Path(args.output_dir).is_absolute()
                else Path(args.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    p_ne = 1.4729266599792261  # benchmark logit; usado so p/ deteccao de trigger

    summary = []
    pair_dirs = sorted(d for d in sweep.iterdir()
                       if (d / "agents" / "agent_0.npz").exists())
    for pair_dir in pair_dirs:
        config, env, agent_config, states = _load_pair(pair_dir)
        out_dir = out_root / pair_dir.name
        out_dir.mkdir(exist_ok=True)

        paths = {}
        trigger_step = None
        for strat in STRATEGIES:
            rows = run_overlay_coop(config["environment"], agent_config,
                                    states[0], strat, args.n_steps,
                                    args.p_init, args.seed)
            with open(out_dir / f"overlay_{strat}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            paths[strat] = [r["price_agent"] for r in rows]
            if strat == "grim_player":
                opp = [r["price_opp"] for r in rows]
                trigger_step = next(
                    (t for t, pr in enumerate(opp) if abs(pr - p_ne) < 1e-3),
                    None)

        ident_gn = all(abs(a - b) < 1e-9
                       for a, b in zip(paths["grim_player"],
                                       paths["nash_player"]))
        ident_gm = all(abs(a - b) < 1e-9
                       for a, b in zip(paths["grim_player"],
                                       paths["monopoly_player"]))
        summary.append({"pair": pair_dir.name,
                        "grim_trigger_step": (trigger_step
                                              if trigger_step is not None
                                              else -1),
                        "agent_identical_grim_nash": ident_gn,
                        "agent_identical_grim_monopoly": ident_gm})
        print(f"{pair_dir.name}: trigger={trigger_step} "
              f"grim==nash:{ident_gn} grim==monopoly:{ident_gm}")

    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\n{len(summary)} pares -> {out_root}")


if __name__ == "__main__":
    main()
