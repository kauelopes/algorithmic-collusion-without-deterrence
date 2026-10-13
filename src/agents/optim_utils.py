"""
Modular optimizer builder for neural-network agents.

Usage:
    from src.agents.optim_utils import build_optimizer
    optimizer = build_optimizer(agent_config)

The optimizer is an optax GradientTransformation composed from:
  1. (optional) gradient clipping — clip_by_global_norm
  2. Adam with an optional learning-rate schedule

Supported lr_schedule types
---------------------------
constant      : fixed learning_rate (default when lr_schedule is absent)
linear        : linear decay from learning_rate → lr_min over decay_steps
cosine_decay  : cosine annealing from learning_rate → lr_min over decay_steps
warmup_cosine : linear warmup then cosine decay

Config keys (all optional except learning_rate)
-----------------------------------------------
learning_rate   : float  — base (initial) LR
grad_clip_norm  : float  — clip gradients to this global norm (omit → no clipping)
lr_schedule:
  type          : str    — constant | linear | cosine_decay | warmup_cosine
  decay_steps   : int    — steps over which LR decays
  lr_min        : float  — final LR value (default 0.0)
  warmup_steps  : int    — warm-up steps (warmup_cosine only, default 1000)
"""

import optax


def build_optimizer(agent_config: dict) -> optax.GradientTransformation:
    """
    Build an optax optimizer from agent_config.

    Returns an optax.GradientTransformation that can be used as:
        opt_state = optimizer.init(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
    """
    lr = float(agent_config["learning_rate"])
    lr_fn = _build_schedule(lr, agent_config.get("lr_schedule"))

    transforms = []

    clip_norm = agent_config.get("grad_clip_norm")
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(clip_norm)))

    transforms.append(optax.adam(learning_rate=lr_fn))

    return optax.chain(*transforms) if len(transforms) > 1 else transforms[0]


def optimizer_config_key(agent_config: dict) -> tuple:
    """
    Return a hashable key that uniquely identifies the optimizer configuration.
    Used to cache JIT-compiled training step functions per optimizer setup.
    """
    sched = agent_config.get("lr_schedule") or {}
    return (
        float(agent_config["learning_rate"]),
        sched.get("type", "constant"),
        int(sched.get("decay_steps", 0)),
        float(sched.get("lr_min", 0.0)),
        int(sched.get("warmup_steps", 0)),
        agent_config.get("grad_clip_norm"),
    )


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------

def _build_schedule(lr: float, schedule_config: dict | None):
    """Return a scalar LR or an optax schedule callable."""
    if not schedule_config:
        return lr

    stype       = schedule_config.get("type", "constant")
    decay_steps = int(schedule_config.get("decay_steps", 1))
    lr_min      = float(schedule_config.get("lr_min", 0.0))

    if stype == "constant":
        return lr

    if stype == "linear":
        return optax.linear_schedule(
            init_value=lr,
            end_value=lr_min,
            transition_steps=decay_steps,
        )

    if stype == "cosine_decay":
        alpha = lr_min / lr if lr > 0 else 0.0
        return optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=decay_steps,
            alpha=alpha,
        )

    if stype == "warmup_cosine":
        warmup_steps = int(schedule_config.get("warmup_steps", 1000))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=lr_min,
        )

    raise ValueError(
        f"Unknown lr_schedule type: '{stype}'. "
        "Choose from: constant, linear, cosine_decay, warmup_cosine"
    )
