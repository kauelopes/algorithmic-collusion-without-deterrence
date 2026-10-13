"""
Config loader for experiment runs.

Resolves a composed experiment config from YAML files and CLI overrides
into a single, fully-resolved dict. The resolved dict contains everything
needed to reproduce a run — no references, no defaults hidden elsewhere.

Composition model
-----------------
An experiment YAML declares its sub-configs under a `defaults` key:

    defaults:
      environment: environment/linear_cournot
      agent:       agents/q_learning
      training:    training/base

Each value is a path relative to the `configs/` root (without .yaml).
The loader reads each file and stores it under the corresponding key.
Any additional top-level keys in the experiment file override the defaults.

CLI overrides
-------------
Dot-notation strings of the form  section.key=value  or
section.subsection.key=value. Values are auto-cast: int > float > bool > str.

    agent.alpha=0.05          → config["agent"]["alpha"] = 0.05
    training.total_steps=50000 → config["training"]["total_steps"] = 50000
"""

import copy
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Root of the project (one level above src/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR  = _PROJECT_ROOT / "configs"


def load(experiment_path: str, overrides: list[str] | None = None) -> dict:
    """
    Load and fully resolve an experiment config.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment YAML, absolute or relative to project root.
    overrides : list[str], optional
        CLI override strings, e.g. ["agent.alpha=0.05", "training.seed=7"].

    Returns
    -------
    dict
        Fully resolved config. Always contains:
          experiment_name, environment, agent, training,
          _meta.timestamp, _meta.git_commit, _meta.overrides_applied
    """
    exp_path = _resolve_path(experiment_path)
    exp_raw  = _read_yaml(exp_path)

    defaults = exp_raw.pop("defaults", {})
    config: dict = {}

    # Load each sub-config into its section
    for section, rel_path in defaults.items():
        sub_path = _CONFIGS_DIR / f"{rel_path}.yaml"
        config[section] = _read_yaml(sub_path)

    # Merge top-level keys from experiment file (experiment-level overrides)
    for key, value in exp_raw.items():
        if key in config and isinstance(value, dict):
            config[key] = _deep_merge(config[key], value)
        else:
            config[key] = value

    # Apply CLI overrides
    applied_overrides = []
    for override in (overrides or []):
        _apply_override(config, override)
        applied_overrides.append(override)

    # Attach metadata
    config["_meta"] = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "git_commit":       _git_commit(),
        "experiment_file":  str(exp_path.relative_to(_PROJECT_ROOT)),
        "overrides_applied": applied_overrides,
    }

    return config


def dump(config: dict, path: str) -> None:
    """Write fully-resolved config to a YAML file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = _PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config file not found: {path}")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_override(config: dict, override: str) -> None:
    """
    Parse 'section.key=value' and set the value in config.
    Raises ValueError for malformed overrides.
    """
    if "=" not in override:
        raise ValueError(f"Override must be 'key.path=value', got: {override!r}")

    key_path, raw_value = override.split("=", 1)
    keys = key_path.strip().split(".")
    value = _cast(raw_value.strip())

    node = config
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def _cast(value: str) -> Any:
    """Try to cast string to int, float, bool; fall back to str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
