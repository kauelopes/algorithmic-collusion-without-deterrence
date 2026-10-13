"""
Strategy test — evaluate trained Q-learning / REINFORCE agents against fixed
strategy opponents in the Bertrand environment.

Hedge is excluded: it has no persistent state and is observation-agnostic by
design, so strategy-opponent tests carry no scientific interpretation.

For each (combo, seed), loads agents/agent_0.npz, runs a fixed-length greedy
episode against every strategy player, and aggregates results.

Usage
-----
  python experiments/strategy_test.py [options]

  --sweep_dir DIR       sweep folder to load agents from
                        (default: auto-detect latest bertrand_sweep_* or
                         calvano_replication__*)
  --combos C [C ...]    combos to evaluate (default: all non-hedge combos found)
  --n_seeds N           seeds per combo (default: 5)
  --seed_mode MODE      first | colluding | all  (default: first)
  --episode_steps N     steps per episode (default: 50000)
  --output_dir DIR      output folder (default: results/strategy_tests__TIMESTAMP)
  --strategies S [S ..] strategies to test (default: all available)

Output
------
  results/strategy_tests__YYYYMMDD_HHMMSS/
    aggregate_summary.csv    one row per (combo, seed, strategy)
    aggregate_report.md      pivot table CI × (combo × strategy)
    {combo}/
      {seed}/
        strategy_test_summary.md
        plots/strategy_grid.png
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import yaml

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

_RESULTS_DIR = _PROJ_ROOT / "results"

_ALL_STRATEGIES = [
    "nash_player",
    "monopoly_player",
    "competitive_player",
    "tit_for_tat_player",
    "grim_player",
    "wsls_player",
    "random_player",
    "noise_nash_player",
    "imitation_player",
    "imitation_plus_player",   # faltava — o CLI listava 9 dos 10 do pool `full`
]

_SUPPORTED_AGENT_TYPES = ("q_learning", "reinforce")


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _load_agent(seed_dir: Path, agent_config: dict):
    """Load trained agent from agents/agent_0.npz. Returns (module, state)."""
    agent_type = agent_config["type"]
    if agent_type not in _SUPPORTED_AGENT_TYPES:
        raise ValueError(f"Agent type '{agent_type}' not supported for strategy tests.")

    npz = seed_dir / "agents" / "agent_0.npz"
    if not npz.exists():
        raise FileNotFoundError(f"agents/agent_0.npz not found in {seed_dir}")

    module = importlib.import_module(f"src.agents.{agent_type}")

    if agent_type == "q_learning":
        state = module.load_state(npz)
        # Force greedy: set epsilon to minimum
        state = state._replace(epsilon=float(agent_config.get("epsilon_end", 0.01)))
    elif agent_type == "reinforce":
        state = module.load_state(npz, agent_config)

    return module, state


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _build_env(env_config: dict):
    if env_config.get("type", "bertrand") == "bertrand":
        from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
        return MultiAgentBertrandEnv(env_config)
    from src.environment.multi_agent_cournot_env import MultiAgentCournotEnv
    return MultiAgentCournotEnv(env_config)


# ---------------------------------------------------------------------------
# Strategy config builder
# ---------------------------------------------------------------------------

def _strategy_config(strategy_name: str, bm: dict, env_config: dict) -> dict:
    p_NE = float(bm["p_NE"])
    p_M  = float(bm["p_monopoly"])
    N    = int(env_config["market"]["n_firms"])

    env = _build_env(env_config)

    return {
        "type":             strategy_name,
        "p_NE":             p_NE,
        "p_monopoly":       p_M,
        "p_competitive":    float(env_config["cost"]["c"]),
        "p_min":            float(env.p_min),
        "p_max":            float(env.p_max),
        "n_agents":         N,
        "own_idx":          1,
        "defect_threshold": 0.05,
        "noise_sigma":      0.05,
        "rate": 0.1, "adjust": 0.5,          # imitation_plus — mesmo config do pool
        "shift_delta":      (p_M - p_NE) / 7,
    }


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def _pick_seeds(combo_dir: Path, n_seeds: int, mode: str) -> list[Path]:
    all_seeds = sorted(combo_dir.glob("seed_*"))
    if not all_seeds:
        return []

    if mode == "first":
        return all_seeds[:n_seeds]

    if mode == "all":
        return all_seeds if n_seeds <= 0 else all_seeds[:n_seeds]

    # colluding: highest final CI
    scored = []
    for sd in all_seeds:
        hist = sd / "metrics_history.csv"
        if not hist.exists():
            continue
        rows = list(csv.DictReader(hist.open()))
        if rows:
            scored.append((float(rows[-1]["collusion_index"]), sd))
    scored.sort(reverse=True)
    return [sd for _, sd in scored[:n_seeds]]


# ---------------------------------------------------------------------------
# CI helper
# ---------------------------------------------------------------------------

def _episode_ci(ep: dict, p_NE: float, p_M: float) -> float:
    mean_p = float(np.mean(ep["price_a"]))
    denom  = p_M - p_NE
    return (mean_p - p_NE) / denom if abs(denom) > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Single-seed evaluation
# ---------------------------------------------------------------------------

def _eval_seed(
    seed_dir: Path,
    strategies: list[str],
    episode_steps: int,
    rng: jax.Array,
) -> tuple[list[dict], dict]:
    """
    Run trained agent in seed_dir against each strategy.
    Returns (episodes, benchmarks).
    """
    from src.strategy_analysis.runner import run_episode
    from src.strategy_analysis.report import generate_report

    config = yaml.safe_load((seed_dir / "config.yaml").read_text())
    env    = _build_env(config["environment"])
    bm     = env.benchmarks()
    bm["p_min"] = float(env.p_min)
    bm["p_max"] = float(env.p_max)

    agent_config = dict(config["agent"])
    # Inject grid bounds from env (not persisted in config.yaml)
    agent_config.setdefault("action_min", float(env.p_min))
    agent_config.setdefault("q_max",      float(env.p_max))

    agent_module, agent_state = _load_agent(seed_dir, agent_config)

    episodes: list[dict] = []
    for strategy_name in strategies:
        strat_module = importlib.import_module(f"src.strategy_players.{strategy_name}")
        strat_cfg    = _strategy_config(strategy_name, bm, config["environment"])

        rng, rng_ep = jax.random.split(rng)
        strat_state  = strat_module.init(rng_ep, strat_cfg)

        ep = run_episode(
            env=env,
            agent_a_module=agent_module,
            agent_a_state=agent_state,
            agent_a_config=agent_config,
            agent_b_module=strat_module,
            agent_b_state=strat_state,
            agent_b_config=strat_cfg,
            n_steps=episode_steps,
            rng=rng_ep,
            trained_agent_idx=0,
        )
        ep["strategy_name"] = strategy_name
        episodes.append(ep)

    # Write per-seed report inside the seed's own folder
    generate_report(episodes, bm, seed_dir / "strategy_tests", config)

    return episodes, bm


# ---------------------------------------------------------------------------
# Aggregate writer
# ---------------------------------------------------------------------------

def _write_aggregate(
    rows: list[dict],
    out_dir: Path,
    strategies: list[str],
) -> None:
    # Raw CSV
    if rows:
        with open(out_dir / "aggregate_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    combos = sorted({r["combo"] for r in rows})
    short  = [s.replace("_player", "") for s in strategies]

    lines = [
        "# Strategy Test — Aggregate Report",
        "",
        "**CI** = (mean trained-agent price − p_NE) / (p_M − p_NE)",
        "Values are mean ± std across evaluated seeds.",
        "",
        "## Pivot: CI médio por (combo × estratégia)",
        "",
        "| Combo | " + " | ".join(short) + " |",
        "|---|" + "|".join(["---"] * len(strategies)) + "|",
    ]

    for combo in combos:
        cells = []
        for strat in strategies:
            vals = [r["ci"] for r in rows if r["combo"] == combo and r["strategy"] == strat]
            cells.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "—")
        lines.append(f"| {combo} | " + " | ".join(cells) + " |")

    lines += ["", "---", ""]

    for combo in combos:
        lines += [f"## {combo}", "",
                  "| Estratégia | CI médio | CI std | p̄ treinado | p̄ oponente | π̄ treinado |",
                  "|---|---|---|---|---|---|"]
        for strat in strategies:
            vals = [r for r in rows if r["combo"] == combo and r["strategy"] == strat]
            if not vals:
                continue
            ci  = [r["ci"] for r in vals]
            pa  = [r["mean_price_a"] for r in vals]
            pb  = [r["mean_price_b"] for r in vals]
            pra = [r["mean_profit_a"] for r in vals]
            lines.append(
                f"| {strat.replace('_player','')} "
                f"| {np.mean(ci):.4f} | {np.std(ci):.4f} "
                f"| {np.mean(pa):.4f} | {np.mean(pb):.4f} "
                f"| {np.mean(pra):.4f} |"
            )
        lines.append("")

    (out_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n")
    print(f"Aggregate report → {out_dir / 'aggregate_report.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy test for trained Bertrand agents")
    parser.add_argument("--sweep_dir",     type=str, default=None)
    parser.add_argument("--combos",        nargs="+", default=None)
    parser.add_argument("--n_seeds",       type=int, default=5)
    # default era "colluding" — selecionar seeds pelo CI final é viés de seleção
    # (armadilha: só os runs mais colusivos entravam no teste). "first" é neutro.
    parser.add_argument("--seed_mode",     type=str, default="first",
                        choices=["first", "colluding", "all"])
    parser.add_argument("--episode_steps", type=int, default=50_000)
    parser.add_argument("--output_dir",    type=str, default=None)
    parser.add_argument("--strategies",    nargs="+", default=None)
    args = parser.parse_args()

    # Resolve sweep dir
    if args.sweep_dir:
        sweep_dir = (Path(args.sweep_dir) if Path(args.sweep_dir).is_absolute()
                     else _RESULTS_DIR / args.sweep_dir)
    else:
        for pattern in ("bertrand_sweep__*", "calvano_replication__*"):
            candidates = sorted(_RESULTS_DIR.glob(pattern), reverse=True)
            if candidates:
                sweep_dir = candidates[0]
                break
        else:
            print("No sweep folder found. Use --sweep_dir.")
            sys.exit(1)

    # Resolve output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.output_dir) if args.output_dir and Path(args.output_dir).is_absolute()
               else _RESULTS_DIR / (args.output_dir or f"strategy_tests__{ts}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = args.strategies or _ALL_STRATEGIES

    # Resolve combos — skip hedge combos
    if args.combos:
        combos = args.combos
    else:
        all_dirs = sorted(d for d in sweep_dir.iterdir()
                          if d.is_dir() and any(d.glob("seed_*")))
        combos = [d.name for d in all_dirs if "hedge" not in d.name.lower()]
        if not combos:
            # Flat layout (calvano_replication: seeds at sweep root)
            if any(sweep_dir.glob("seed_*")):
                combos = [""]

    print(f"Sweep     : {sweep_dir.name}")
    print(f"Output    : {out_dir}")
    print(f"Combos    : {combos if combos != [''] else ['(flat)']}")
    print(f"Strategies: {[s.replace('_player','') for s in strategies]}")
    print(f"Seeds     : {args.n_seeds} ({args.seed_mode})")
    print(f"Ep. steps : {args.episode_steps:,}")
    print()

    all_rows: list[dict] = []
    rng = jax.random.PRNGKey(42)

    for combo in combos:
        combo_dir = sweep_dir / combo if combo else sweep_dir
        seed_dirs = _pick_seeds(combo_dir, args.n_seeds, args.seed_mode)
        if not seed_dirs:
            print(f"  {combo or '(flat)'}: no seeds — skip")
            continue

        # Skip if agent type is hedge
        first_cfg_path = seed_dirs[0] / "config.yaml"
        if first_cfg_path.exists():
            first_cfg = yaml.safe_load(first_cfg_path.read_text())
            if first_cfg.get("agent", {}).get("type") == "hedge":
                print(f"  {combo}: hedge agent — skip")
                continue

        print(f"  {combo or '(flat)'} [{len(seed_dirs)} seeds]")

        for seed_dir in seed_dirs:
            print(f"    {seed_dir.name} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                episodes, bm = _eval_seed(seed_dir, strategies, args.episode_steps, rng)
                rng, _ = jax.random.split(rng)

                p_NE = float(bm["p_NE"])
                p_M  = float(bm["p_monopoly"])

                for ep in episodes:
                    ci = _episode_ci(ep, p_NE, p_M)
                    all_rows.append({
                        "combo":         combo or "flat",
                        "seed":          seed_dir.name,
                        "strategy":      ep["strategy_name"],
                        "ci":            round(ci, 4),
                        "mean_price_a":  round(float(np.mean(ep["price_a"])), 4),
                        "mean_price_b":  round(float(np.mean(ep["price_b"])), 4),
                        "mean_profit_a": round(float(np.mean(ep["profit_a"])), 4),
                        "mean_profit_b": round(float(np.mean(ep["profit_b"])), 4),
                    })

                summary = "  ".join(
                    f"{ep['strategy_name'].replace('_player','')}="
                    f"{_episode_ci(ep, p_NE, p_M):.3f}"
                    for ep in episodes
                )
                print(f"done ({time.time()-t0:.0f}s)")
                print(f"      {summary}")

            except Exception as exc:
                print(f"ERROR: {exc}")

    if all_rows:
        _write_aggregate(all_rows, out_dir, strategies)
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
