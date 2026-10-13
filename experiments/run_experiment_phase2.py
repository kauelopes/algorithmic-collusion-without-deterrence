"""
CLI entry point for Phase 2 (multi-agent) experiments.

Usage (inside container):
    python experiments/run_experiment_phase2.py \
        --config configs/experiments/phase2_qlearning_n2_bandit.yaml

    python experiments/run_experiment_phase2.py \
        --config configs/experiments/phase2_qlearning_n2_bandit.yaml \
        --override training.seed=7 training.total_steps=100000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from src.general_utils import config_loader
from src.training.multi_agent_trainer import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Phase 2 multi-agent Cournot experiment.")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Dot-notation overrides, e.g. training.seed=7",
    )
    return parser.parse_args()


def _print_config(config: dict) -> None:
    meta = config.get("_meta", {})
    print(f"\n{'='*60}")
    print(f"[Phase 2] Experiment : {config.get('experiment_name', '(unnamed)')}")
    print(f"Timestamp  : {meta.get('timestamp', '')}")
    print(f"Git commit : {meta.get('git_commit', '')}")
    if meta.get("overrides_applied"):
        print(f"Overrides  : {meta['overrides_applied']}")
    print(f"{'='*60}")
    display = {k: v for k, v in config.items() if not k.startswith("_")}
    print(yaml.dump(display, default_flow_style=False, sort_keys=False))


def main() -> None:
    args = parse_args()
    config = config_loader.load(args.config, overrides=args.override)
    _print_config(config)
    run(
        env_config=config["environment"],
        agent_config=config["agent"],
        training_config=config["training"],
        full_config=config,
    )


if __name__ == "__main__":
    main()
