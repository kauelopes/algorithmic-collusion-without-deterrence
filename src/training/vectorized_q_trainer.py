"""
Vectorized Q-learning trainer — K parallel seeds on GPU/CPU via JAX.

Runs K independent Q-learning experiments simultaneously using jax.lax.scan
and vmapped environment steps. Primary use case: Monte Carlo statistics across
many seeds (Calvano et al. use 400 independent runs).

Speedup vs. sequential: roughly linear in K for GPU-bound workloads.
For K=32 on GPU, typical 20-40× wall-clock speedup over the standard trainer.

Restrictions (by design — all other cases use multi_agent_trainer):
  - agent_type : q_learning only
  - env_type   : bertrand only
  - info_structure : rival_prices only (all agents observe the same price vector)

Entry point:
    from src.training.vectorized_q_trainer import run_vectorized
    results = run_vectorized(env_config, agent_config, training_config, K=32)

Returns: list of K dicts with keys {ci, nash_dist, ce_gap, final_step, seed_idx}
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
from src.agents.q_learning import opponent_optimized_qtable, _decay_epsilon
from src.metrics.multi_agent import (
    collusion_index_price as _compute_ci,
    nash_distance_joint as _compute_dist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_idx_fn(n_actions: int, q_max: float, action_min: float, memory_length: int):
    """Return jit+vmap state-index function: (K, memory_length) → (K,) int32."""
    span   = max(q_max - action_min, 1e-9)
    powers = jnp.array(
        [n_actions ** (memory_length - 1 - j) for j in range(memory_length)],
        dtype=jnp.int32,
    )

    def _single(prices_m: jnp.ndarray) -> jnp.ndarray:
        raw = jnp.clip((prices_m - action_min) / span * (n_actions - 1), 0, n_actions - 1)
        return jnp.sum(jnp.round(raw).astype(jnp.int32) * powers)

    return jax.jit(jax.vmap(_single))


def _precompute_epsilon_schedule(
    total_steps: int, eps_start: float, eps_end: float,
    eps_decay: int, eps_type: str,
) -> np.ndarray:
    return np.array(
        [_decay_epsilon(t, eps_start, eps_end, eps_decay, eps_type) for t in range(1, total_steps + 1)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_vectorized(
    env_config: dict,
    agent_config: dict,
    training_config: dict,
    K: int = 32,
    output_dir: Path | None = None,
) -> list[dict]:
    """
    Train K independent Q-learning seeds in parallel.

    Parameters
    ----------
    K : int
        Number of parallel seeds. K=1 is equivalent to the standard trainer.
        Recommended: K=32 for statistics, K=64 for large GPU.

    Returns
    -------
    list of K dicts: [{ci, nash_dist, ce_gap, final_step, seed_idx}, ...]
    """
    env = MultiAgentBertrandEnv(env_config)
    bm  = env.benchmarks()
    N   = int(env_config["market"]["n_firms"])

    if env_config["state"]["info_structure"] != "rival_prices":
        raise ValueError(
            "vectorized_q_trainer only supports info_structure='rival_prices'. "
            f"Got: {env_config['state']['info_structure']!r}"
        )

    agent_config           = dict(agent_config)
    agent_config["action_min"] = env.p_min
    agent_config["q_max"]      = env.p_max

    n_actions     = int(agent_config["n_actions"])
    memory_length = int(agent_config["memory_length"])
    n_states      = n_actions ** memory_length
    q_max         = float(env.p_max)
    action_min    = float(env.p_min)
    gamma         = float(agent_config["gamma"])
    alpha         = float(agent_config["alpha"])
    eps_start     = float(agent_config["epsilon_start"])
    eps_end       = float(agent_config["epsilon_end"])
    eps_decay_steps = int(agent_config["epsilon_decay_steps"])
    eps_type        = str(agent_config.get("epsilon_decay_type", "linear"))

    total_steps    = int(training_config["total_steps"])
    eval_interval  = int(training_config["eval_interval"])
    eval_window    = int(training_config.get("eval_window", 500_000))
    episode_length = training_config.get("episode_length")
    seed           = int(training_config["seed"])

    # Sanity check — obs_dim must equal memory_length for rival_prices
    if N != memory_length:
        raise ValueError(
            f"For rival_prices, memory_length ({memory_length}) must equal n_firms ({N}). "
            "Each agent's observation is the full N-price vector."
        )

    # --- Q-table initialisation (all K seeds share the same init) ---
    if agent_config.get("q_init") == "opponent_optimized":
        q_row = opponent_optimized_qtable(
            n_actions=n_actions, action_min=action_min, q_max=q_max,
            n_rivals=N - 1, gamma=gamma,
            a=float(env_config["demand"]["a"]),
            mu=float(env_config["demand"]["mu"]),
            c=float(env_config["cost"]["c"]),
            a0=float(env_config["demand"].get("a0", 0.0)),
        )
        q_init = jnp.tile(
            jnp.array(q_row, dtype=jnp.float32),
            (K, N, n_states, 1),
        )  # (K, N, n_states, n_actions)
    else:
        q_init = jnp.zeros((K, N, n_states, n_actions), dtype=jnp.float32)

    # --- Compiled helpers ---
    state_idx_fn = _make_state_idx_fn(n_actions, q_max, action_min, memory_length)
    batched_env  = jax.jit(jax.vmap(env._step_fn))  # (K, N) f32 → 3× (K, N)

    # Precomputed epsilon schedule as JAX array
    epsilons_np = _precompute_epsilon_schedule(
        total_steps, eps_start, eps_end, eps_decay_steps, eps_type
    )  # (total_steps,) float32

    # Scatter index arrays for Q-table updates (constant per run)
    k_flat = jnp.repeat(jnp.arange(K), N)   # (K*N,) — seed indices
    n_flat = jnp.tile(jnp.arange(N), K)     # (K*N,) — agent indices
    K_idx  = jnp.arange(K)[:, None]         # (K, 1) for fancy indexing
    N_idx  = jnp.arange(N)[None, :]         # (1, N)

    # --- scan step function (pure JAX, compiled once) ---
    # Carry: (q, prices, rng, prices_sum, rewards_sum, window_count, step_count)
    # Inputs per step: (epsilon,)   [shape: scalar]
    # No per-step random inputs — PRNG is in the carry

    ep_len         = jnp.int32(episode_length) if episode_length else None
    window_start   = jnp.int32(total_steps - eval_window)
    n_actions_f32  = jnp.float32(n_actions - 1)
    gamma_f32      = jnp.float32(gamma)
    alpha_f32      = jnp.float32(alpha)
    action_min_f32 = jnp.float32(action_min)
    q_max_f32      = jnp.float32(q_max)
    p_min_f32      = jnp.float32(env.p_min)
    p_max_f32      = jnp.float32(env.p_max)

    def _step(carry, epsilon):
        q, prices, rng, prices_sum, rewards_sum, window_count, step_count = carry

        # Random numbers for this step (PRNG in carry → reproducible)
        rng, rng_eps, rng_act = jax.random.split(rng, 3)
        eps_roll = jax.random.uniform(rng_eps, shape=(K, N))
        rand_act = jax.random.randint(rng_act, shape=(K, N), minval=0, maxval=n_actions)

        # State indices (shared across agents — rival_prices obs)
        s_k  = state_idx_fn(prices)                              # (K,)
        s_kn = jnp.broadcast_to(s_k[:, None], (K, N))           # (K, N)

        # Epsilon-greedy
        q_rows   = q[K_idx, N_idx, s_kn, :]                     # (K, N, n_actions)
        greedy_a = jnp.argmax(q_rows, axis=-1)                   # (K, N) int
        a_kn     = jnp.where(eps_roll < epsilon, rand_act, greedy_a).astype(jnp.int32)

        # Action indices → prices
        act_prices = action_min_f32 + a_kn.astype(jnp.float32) / n_actions_f32 * (q_max_f32 - action_min_f32)

        # Environment step (batched over K)
        _, _, profits = batched_env(act_prices)                  # (K, N)

        # Next state = prices just played (rival_prices: obs is current prices)
        s_next_k  = state_idx_fn(act_prices)                     # (K,)
        s_next_kn = jnp.broadcast_to(s_next_k[:, None], (K, N))

        # Q-learning update
        q_next_max = jnp.max(q[K_idx, N_idx, s_next_kn, :], axis=-1)   # (K, N)
        td_error   = profits + gamma_f32 * q_next_max - q[K_idx, N_idx, s_kn, a_kn]
        q = q.at[k_flat, n_flat, jnp.repeat(s_k, N), a_kn.ravel()].add(alpha_f32 * td_error.ravel())

        # Episode reset (handles memory_length > 1 by resetting obs)
        next_prices = act_prices
        if ep_len is not None:
            rng, rng_reset = jax.random.split(rng)
            reset_prices = jax.random.uniform(
                rng_reset, shape=(K, N), minval=p_min_f32, maxval=p_max_f32
            )
            should_reset = jnp.mod(step_count + 1, ep_len) == 0
            next_prices  = jax.lax.cond(
                should_reset,
                lambda: reset_prices,
                lambda: act_prices,
            )

        # Metric accumulation — only during last eval_window steps
        in_window    = step_count >= window_start
        prices_sum   = prices_sum  + jnp.where(in_window, act_prices, jnp.float32(0))
        rewards_sum  = rewards_sum + jnp.where(in_window, profits,    jnp.float32(0))
        window_count = window_count + jnp.int32(in_window)

        next_carry = (q, next_prices, rng, prices_sum, rewards_sum, window_count, step_count + 1)
        return next_carry, None

    # --- Training loop (chunked scan for progress reporting) ---
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)

    init_prices = jax.random.uniform(
        rng_init, shape=(K, N), minval=p_min_f32, maxval=p_max_f32
    ).astype(jnp.float32)

    carry = (
        q_init,
        init_prices,
        rng,
        jnp.zeros((K, N), dtype=jnp.float32),   # prices_sum
        jnp.zeros((K, N), dtype=jnp.float32),   # rewards_sum
        jnp.int32(0),                            # window_count
        jnp.int32(0),                            # step_count
    )

    n_chunks   = total_steps // eval_interval
    chunk_size = eval_interval
    report_every = max(1, n_chunks // 10)

    t0 = time.time()
    device_str = jax.devices()[0].platform.upper()
    print(f"\nVectorized Q-learning: K={K} parallel seeds — {device_str}")
    print(f"  Steps: {total_steps:,}  |  Q-table: {n_states}×{n_actions}/agent  |  N={N}")
    print(f"  {'step':>10}  {'mean_CI':>9}  {'std_CI':>8}  {'mean_dist':>10}  {'elapsed':>9}")
    print("  " + "-" * 52)

    for chunk in range(n_chunks):
        sl      = slice(chunk * chunk_size, (chunk + 1) * chunk_size)
        eps_arr = jnp.array(epsilons_np[sl])  # (chunk_size,)
        carry, _ = jax.lax.scan(_step, carry, eps_arr)

        if (chunk + 1) % report_every == 0 or chunk == n_chunks - 1:
            wc = int(carry[5])   # window_count
            if wc > 0:
                mean_p = np.array(carry[3]) / wc   # (K, N)
                ci_k   = np.array([
                    _compute_ci(float(np.mean(mean_p[k])), bm["p_NE"], bm["p_monopoly"])
                    for k in range(K)
                ])
                dist_k = np.array([
                    _compute_dist(mean_p[k].tolist(), bm["p_NE"]) for k in range(K)
                ])
                step_done = (chunk + 1) * chunk_size
                elapsed   = time.time() - t0
                print(
                    f"  {step_done:>10,}  {np.mean(ci_k):>9.4f}  {np.std(ci_k):>8.4f}"
                    f"  {np.mean(dist_k):>10.4f}  {elapsed:>8.1f}s"
                )

    # Handle remainder steps (if total_steps % eval_interval != 0)
    remainder = total_steps - n_chunks * eval_interval
    if remainder > 0:
        eps_arr = jnp.array(epsilons_np[-remainder:])
        carry, _ = jax.lax.scan(_step, carry, eps_arr)

    print("  " + "-" * 52)

    # --- Final metrics ---
    wc      = max(int(carry[5]), 1)
    mean_p  = np.array(carry[3]) / wc   # (K, N)
    q_final = np.array(carry[0])        # (K, N, n_states, n_actions)

    results = []
    for k in range(K):
        p_k  = mean_p[k].tolist()
        ci   = _compute_ci(float(np.mean(p_k)), bm["p_NE"], bm["p_monopoly"])
        dist = _compute_dist(p_k, bm["p_NE"])
        results.append({
            "ci":         float(ci),
            "nash_dist":  float(dist),
            "ce_gap":     float("nan"),  # not computed in vectorized mode
            "final_step": total_steps,
            "seed_idx":   k,
        })

    ci_arr   = [r["ci"]        for r in results]
    dist_arr = [r["nash_dist"] for r in results]
    total_elapsed = time.time() - t0
    print(
        f"  CI:   {np.mean(ci_arr):.4f} ± {np.std(ci_arr):.4f}\n"
        f"  Dist: {np.mean(dist_arr):.4f} ± {np.std(dist_arr):.4f}\n"
        f"  Time: {total_elapsed:.1f}s  ({total_steps * K / total_elapsed / 1e6:.2f}M steps/s across seeds)\n"
    )

    if output_dir is not None:
        _save(Path(output_dir), results, q_final)

    return results


def _save(output_dir: Path, results: list[dict], q_tables: np.ndarray) -> None:
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vectorized_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed_idx", "ci", "nash_dist", "ce_gap", "final_step"])
        writer.writeheader()
        writer.writerows(results)
    np.save(str(output_dir / "q_tables.npy"), q_tables)
    print(f"  Saved to {output_dir}")
