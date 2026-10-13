"""
Piloto da distribuição de atratores .

Pergunta: rodando N treinos independentes de pares co-treinados, qual é a
distribuição dos atratores conjuntos sobre as 15 × 15 células de estado?
Ela converge (metades de seeds dão o mesmo heatmap)? Tem moda, diagonal, ciclos?

Protocolo = o dos pares do paper (configs/experiments/cross_deploy_bertrand.yaml:
α=0.15, γ=0.95, ε geométrico 1.0→0.01 em 1.15M passos, 2M passos, resets a cada
10k com estado inicial aleatório), MAS na grade de Calvano (grid_extend 0.1,
convenção desde 2026-08-12). Treino vetorizado em lotes de K pares; cada lote
usa uma seed base diferente (lote b → seed = seed0 + b).

Saída: results/attractor_pilot__<ts>/
  config.yaml, batch_<b>/q_tables.npy (K, 2, 225, 15) + vectorized_results.csv

Uso:
  python experiments/attractor_pilot.py --n_pairs 1024 --k 128 [--steps 2000000]
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))
from src.training.vectorized_q_trainer import run_vectorized  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Piloto: distribuição de atratores")
    p.add_argument("--n_pairs", type=int, default=1024)
    p.add_argument("--k", type=int, default=128, help="pares por lote vetorizado")
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--grid_extend", type=float, default=0.1)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    env_config = {
        "demand": {"a": 2.0, "mu": 0.25, "a0": 0.0},
        "cost": {"c": 1.0},
        "market": {"n_firms": 2, "grid_extend": args.grid_extend, "init_mode": "random"},
        "state": {"info_structure": "rival_prices"},
    }
    agent_config = {
        "type": "q_learning", "n_actions": 15, "memory_length": 2,
        "q_init": "opponent_optimized", "alpha": 0.15, "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.01,
        "epsilon_decay_steps": 1_150_000, "epsilon_decay_type": "geometric",
    }
    training_config = {
        "total_steps": args.steps, "eval_interval": 10_000,
        "eval_window": 200_000, "episode_length": 10_000, "seed": args.seed0,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else _PROJ_ROOT / "results" / f"attractor_pilot__{ts}"
    out.mkdir(parents=True, exist_ok=True)
    n_batches = (args.n_pairs + args.k - 1) // args.k
    (out / "config.yaml").write_text(yaml.safe_dump({
        "environment": env_config, "agent": agent_config, "training": training_config,
        "n_pairs": args.n_pairs, "k": args.k, "n_batches": n_batches, "seed0": args.seed0,
        "note": "lote b usa training.seed = seed0 + b",
    }, sort_keys=False), encoding="utf-8")

    t0 = time.time()
    for b in range(n_batches):
        bdir = out / f"batch_{b:03d}"
        if (bdir / "q_tables.npy").exists():
            print(f"[lote {b}] já existe, pulando"); continue
        cfg = dict(training_config); cfg["seed"] = args.seed0 + b
        print(f"\n[lote {b + 1}/{n_batches}] K={args.k}, seed={cfg['seed']}")
        run_vectorized(env_config, agent_config, cfg, K=args.k, output_dir=bdir)
        el = time.time() - t0
        print(f"  decorrido {el / 60:.1f} min; estimativa total {el / (b + 1) * n_batches / 60:.1f} min")
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
