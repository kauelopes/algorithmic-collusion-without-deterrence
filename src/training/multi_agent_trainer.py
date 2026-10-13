"""
Multi-agent training loop — Phase 2 (Cournot and Bertrand).

Runs N independent RL agents learning simultaneously. The environment type
is selected from env_config["type"]: "cournot" (default) or "bertrand".

Agents interact only through the environment — no agent accesses a rival's
internal state. Supported: q_learning, reinforce, hedge.

Console output every eval_interval: CI, nash_dist, CE-gap, sym_q.

CE-gap (approximate): single-sample estimate using the last step's rival
actions as the conditioning context.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.multi_agent_cournot_env import MultiAgentCournotEnv
from src.environment.multi_agent_bertrand_env import MultiAgentBertrandEnv
from src.agents.q_learning import Transition
from src.metrics.multi_agent import (
    collusion_index as compute_ci_cournot,
    collusion_index_price as compute_ci_bertrand,
    nash_distance_joint,
    inter_agent_symmetry,
    final_metrics as compute_final_metrics,
)
from src.general_utils import multi_agent_output_writer


_N_CF_ACTIONS = 20   # grid size for CE-gap counterfactual evaluation


def run(
    env_config: dict,
    agent_config: dict,
    training_config: dict,
    full_config: dict | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Train N RL agents simultaneously in a multi-agent Cournot environment.

    Parameters
    ----------
    env_config       : environment config sub-dict
    agent_config     : agent config sub-dict (same module for all N agents)
    training_config  : training config sub-dict
    full_config      : fully-resolved config (written to output folder).
                       If None, no output folder is created.
    output_dir       : explicit output folder passed to the writer.
                       Overrides the auto-generated path when provided.
    """
    agent_type = agent_config.get("type", "q_learning")
    agent = importlib.import_module(f"src.agents.{agent_type}")

    # Oponente fixo (c5_tesoura): `agent.opponent: {type, params}` põe um strategy
    # player determinístico no ÚLTIMO slot. Ele não aprende, não entra em snapshots
    # nem na serialização final — o run tem um único learner (agent_0).
    _opp_spec = agent_config.get("opponent")
    _opp_slot: int | None = None
    _opp_mod = None
    if _opp_spec:
        _opp_mod = importlib.import_module(
            f"src.strategy_players.{_opp_spec['type']}"
        )

    total_steps   = int(training_config["total_steps"])
    eval_interval = int(training_config["eval_interval"])
    seed          = int(training_config["seed"])
    N             = int(env_config["market"]["n_firms"])

    env_type = env_config.get("type", "cournot")
    if env_type == "bertrand":
        env     = MultiAgentBertrandEnv(env_config)
        cf_grid = np.linspace(env.p_min, env.p_max, _N_CF_ACTIONS)
    else:
        env     = MultiAgentCournotEnv(env_config)
        cf_grid = np.linspace(0.0, float(env._q_max), _N_CF_ACTIONS)

    bm = env.benchmarks()

    # Shallow copy before any mutation — don't touch caller's dict
    agent_config = dict(agent_config)

    # Sync agent grid bounds with env (critical when grid_extend is used)
    if env_type == "bertrand":
        agent_config["action_min"] = env.p_min
        agent_config["q_max"]      = env.p_max
    elif env_type == "matrix":
        # índice de ação como valor: com [0, n_actions−1] o `_action_to_idx` do agente é a
        # identidade sobre {0.0, 1.0}. Sincronizado aqui para que config e env não possam
        # divergir (a lição do `grid_extend` do p1).
        agent_config["action_min"] = 0.0
        agent_config["q_max"]      = float(env.n_actions - 1)
    if agent_config.get("q_init") == "opponent_optimized" and agent_type == "q_learning":
        from src.agents.q_learning import opponent_optimized_qtable
        q_row = opponent_optimized_qtable(
            n_actions  = int(agent_config["n_actions"]),
            action_min = float(agent_config.get("action_min", 0.0)),
            q_max      = float(agent_config["q_max"]),
            n_rivals   = N - 1,
            gamma      = float(agent_config["gamma"]),
            a          = float(env_config["demand"]["a"]),
            mu         = float(env_config["demand"]["mu"]),
            c          = float(env_config["cost"]["c"]),
            a0         = float(env_config["demand"].get("a0", 0.0)),
        )
        agent_config["_q_init_table"] = q_row
    if agent_type in ("mb_rival", "pg_tabular"):
        # Rotas A e C do p4: o lucro de estágio é ANALÍTICO (a demanda logit é
        # conhecida); a única incerteza que o agente estima é a reação do rival.
        # Injetar aqui é o mesmo padrão do `_q_init_table` — o agente não conhece o
        # env, só a matriz. (`pg_tabular` a usa no termo de shaping do LOQA, que
        # precisa do `Q` do rival.)
        from src.qtable_analysis.optimal import stage_profits
        _grid = np.linspace(env.p_min, env.p_max, int(agent_config["n_actions"]))
        agent_config["_profit_matrix"] = stage_profits(_grid, {"environment": env_config})
    if agent_config.get("q_init") == "uniform":
        gamma = float(agent_config["gamma"])
        if "q_init_value" in agent_config:
            # valor explícito, para inicialização otimista — ver population_trainer
            agent_config["_q_init_value"] = float(agent_config["q_init_value"])
        else:
            pi_nash = float(bm.get("pi_NE", bm.get("profit_NE", 0.0)))
            agent_config["_q_init_value"] = pi_nash / (1.0 - gamma) if gamma < 1.0 else pi_nash

    # Initialise N independent agents with distinct seeds
    #
    # q_init: crafted (c4_estabilidade) — inicialização POR AGENTE a partir de .npz
    # fabricados (`q_init_files: [pathA, pathB]`; null = agente virgem com o fallback
    # uniform). É o único modo em que os dois agentes podem nascer diferentes.
    _q_files = agent_config.get("q_init_files")
    if agent_config.get("q_init") == "crafted":
        if not _q_files or len(_q_files) != N:
            raise ValueError(
                f"q_init crafted exige `q_init_files` com {N} entradas (null = virgem)"
            )
    if _opp_mod is not None:
        _opp_slot = N - 1
        if agent_config.get("q_init") == "crafted" and _q_files[_opp_slot]:
            raise ValueError(
                "o slot do oponente fixo não recebe Q-table (q_init_files[-1] deve ser null)"
            )
    rng = jax.random.PRNGKey(seed)
    states = []
    # Config POR SLOT (c5): cada agente carrega sua própria cópia com `own_slot` —
    # é o que permite máscara de exploração e oráculo do rival em coordenadas do slot.
    # `mods[i]` é o módulo que governa o slot i (learner, ou o strategy player fixo).
    cfgs: list[dict] = []
    mods = [agent] * N
    for _i in range(N):
        rng, init_rng = jax.random.split(rng)
        if _i == _opp_slot:
            cfg_opp = dict(_opp_spec.get("params", {}))
            cfg_opp.setdefault("p_NE", float(bm.get("p_NE", 0.0)))
            cfg_opp.setdefault("p_monopoly", float(bm.get("p_monopoly", 0.0)))
            cfg_opp.setdefault("own_idx", _i)
            cfg_opp.setdefault("n_agents", N)
            cfg_opp["own_slot"] = _i
            mods[_i] = _opp_mod
            cfgs.append(cfg_opp)
            states.append(_opp_mod.init(init_rng, cfg_opp))
            continue
        cfg_i = dict(agent_config)
        cfg_i["own_slot"] = _i
        if agent_config.get("q_init") == "crafted":
            f = _q_files[_i]
            if f:
                _data = np.load(f)
                cfg_i["_q_init_full"] = _data["q_table"]
                if "visits" in _data.files:
                    cfg_i["_q_init_visits"] = _data["visits"]
            else:
                cfg_i["q_init"] = "uniform"
                _gamma = float(agent_config["gamma"])
                cfg_i["_q_init_value"] = float(agent_config.get(
                    "q_init_value",
                    float(bm.get("pi_NE", 0.0)) / (1.0 - _gamma) if _gamma < 1.0
                    else float(bm.get("pi_NE", 0.0)),
                ))
        cfgs.append(cfg_i)
        states.append(agent.init(init_rng, cfg_i))
    _learner_slots = [k for k in range(N) if k != _opp_slot]

    obs_list = env.reset(seed=seed)

    # Per-window accumulators (one list per agent)
    win_rewards = [[] for _ in range(N)]
    win_actions = [[] for _ in range(N)]
    # a "curva de loss" do tabular: |td_error| por update, janelado como win_rewards
    win_td      = [[] for _ in range(N)]
    _last_alpha = [float("nan")] * N
    history: list[dict] = []

    # Joint action visit counts for N=2 Q-learning (FIG-01 heatmap)
    _n_act    = int(agent_config.get("n_actions", 0))
    _q_max    = float(agent_config.get("q_max", 1.0))
    _act_min  = float(agent_config.get("action_min", 0.0))
    _track_visits = (agent_type == "q_learning" and N == 2 and _n_act > 0)
    if _track_visits:
        _visit_counts    = np.zeros((_n_act, _n_act), dtype=np.int64)
        _visit_snapshots: dict[str, np.ndarray] = {}
        _snap_steps      = {50_000, 500_000}

    # Episode resets — if set, env.reset() is called every episode_length steps
    episode_length = training_config.get("episode_length", None)

    # Série de snapshots de Q-table (c4_estabilidade): grade explícita de steps
    # (tipicamente log-espaçada) em que os agentes são serializados em
    # `agents_snap/step_*/agent_{i}.npz` — a matéria-prima da medição de erosão.
    # Independente do sistema de checkpoint/resume.
    _q_snap_steps = {int(s) for s in (training_config.get("qtable_snapshot_steps") or [])}

    # Snapshot POR EVENTO (c4c): grava um snapshot na PRIMEIRA mudança do argmax da
    # linha de um estado sentinela (tipicamente o estado cooperativo) — fecha o
    # buraco da grade log perto de escapes tardios. `[i, j]` em índices da grade.
    _ev_state = training_config.get("qtable_snapshot_event_state")
    _ev_rows = None
    if _ev_state is not None and agent_type == "q_learning":
        _n_act_ev = int(agent_config["n_actions"])
        _i_ev, _j_ev = int(_ev_state[0]), int(_ev_state[1])
        # convenção de slot: agente 0 lê (own, rival); agente 1, (rival, own)
        _ev_rows = [_i_ev * _n_act_ev + _j_ev, _j_ev * _n_act_ev + _i_ev][:N]
        _ev_prev = {k: int(np.argmax(states[k].q_table[_ev_rows[k]]))
                    for k in _learner_slots}
        _ev_fired = {k: False for k in _learner_slots}

    # MLflow — gated by training.mlflow.enabled
    mlflow_cfg  = training_config.get("mlflow", {})
    _mlflow_on  = bool(mlflow_cfg.get("enabled", False))
    _mlflow_run = None
    if _mlflow_on:
        try:
            import mlflow as _mlflow
            uri = mlflow_cfg.get("tracking_uri") or None
            if uri:
                _mlflow.set_tracking_uri(uri)
            _mlflow.set_experiment(mlflow_cfg.get("experiment_name", "rl_equili"))
            _mlflow_run = _mlflow.start_run()
            _log_mlflow_params(_mlflow, full_config or {})
            print(f"  [MLflow run started: {_mlflow_run.info.run_id}]", flush=True)
        except ImportError:
            print("  [MLflow not installed — skipping experiment tracking]", flush=True)
            _mlflow_on = False
        except Exception as _e:
            print(f"  [MLflow init failed ({type(_e).__name__}: {_e}) — skipping tracking]", flush=True)
            _mlflow_on = False

    # Curriculum (expanding action space)
    curriculum_cfg = training_config.get("curriculum", {})
    prev_clip: tuple = (None, None)

    # Early stopping state — disable via training.early_stop.enabled: false
    es_cfg           = training_config.get("early_stop", {})
    es_enabled       = bool(es_cfg.get("enabled", True))
    es_ce_thr        = float(es_cfg.get("ce_gap_threshold",    0.005))
    es_stability_thr = float(es_cfg.get("stability_threshold", 0.005))
    es_patience      = int(es_cfg.get("patience",              5))
    es_min_steps     = int(es_cfg.get("min_steps",             0))
    es_count           = 0   # CCE stop: ce_gap + stability
    es_stable_count    = 0   # convergence stop: stability only
    es_prev_greedy_qs: list[float] | None = None

    _print_header(N)

    last_rewards = [0.0] * N   # updated each step for CE-gap reference
    stopped_early = False
    last_win_actions: list[list[float]] | None = None
    last_win_rewards: list[list[float]] | None = None

    # Pre-generate all random numbers for Q-learning to eliminate per-step JAX dispatch.
    # Two JAX calls replace ~(total_steps × N × 3) individual calls.
    # Qualquer agente que exponha `select_action_numpy` entra no caminho rápido; quem
    # não expõe cai no `select_action` com chave JAX (correto, só mais lento).
    _use_fast_rng = hasattr(agent, "select_action_numpy")
    if _use_fast_rng:
        from src.agents.q_learning import n_internal as _n_int
        _n_act_rng = int(agent_config.get("n_actions", 1)) * (
            _n_int(agent_config) if agent_type == "q_learning" else 1)
        rng, _eps_rng, _act_rng = jax.random.split(rng, 3)
        _eps_rolls = np.asarray(jax.random.uniform(_eps_rng, shape=(total_steps, N)))
        _rand_acts = np.asarray(
            jax.random.randint(_act_rng, shape=(total_steps, N), minval=0, maxval=_n_act_rng)
        )
        _select_fn = agent.select_action_numpy

    for step in range(1, total_steps + 1):
        # --- collect actions from all agents ---
        actions: list[float] = []
        new_states = []
        if _use_fast_rng:
            _si = step - 1
            for i in range(N):
                if i == _opp_slot:
                    a, ns = _opp_mod.select_action(
                        states[i], obs_list[i], None, cfgs[i], training=False
                    )
                else:
                    a, ns = _select_fn(
                        states[i], obs_list[i],
                        float(_eps_rolls[_si, i]),
                        int(_rand_acts[_si, i]),
                        cfgs[i], training=True,
                    )
                actions.append(a)
                new_states.append(ns)
        else:
            for i in range(N):
                rng, action_rng = jax.random.split(rng)
                a, ns = mods[i].select_action(
                    states[i], obs_list[i], action_rng, cfgs[i],
                    training=(i != _opp_slot),
                )
                actions.append(a)
                new_states.append(ns)
        states = new_states

        # --- curriculum clipping ---
        action_max = env.p_max if env_type == "bertrand" else env._q_max
        q_lo, q_hi = _get_curriculum_clip(step, curriculum_cfg, action_max)
        if q_lo is not None:
            actions = [float(np.clip(a, q_lo, q_hi)) for a in actions]
        if (q_lo, q_hi) != prev_clip:
            if prev_clip != (None, None) or q_lo is not None:
                label = f"[{q_lo:.3f}, {q_hi:.3f}]" if q_lo is not None else "unrestricted"
                print(f"  [curriculum @ step {step}: action range → {label}]", flush=True)
            prev_clip = (q_lo, q_hi)

        # --- environment step ---
        next_obs_list, rewards, dones, _ = env.step(actions)
        last_rewards = rewards

        # --- update each agent independently ---
        for i in range(N):
            tr = Transition(
                obs=obs_list[i],
                action=actions[i],
                reward=rewards[i],
                next_obs=next_obs_list[i],
                done=dones[i],
            )
            states[i], _log = mods[i].update(states[i], tr, cfgs[i])
            if _log:
                _td = _log.get("td_error")
                if _td is not None:
                    win_td[i].append(abs(float(_td)))
                _a = _log.get("alpha")
                if _a is not None:
                    _last_alpha[i] = float(_a)
            win_rewards[i].append(rewards[i])
            win_actions[i].append(actions[i])

        # --- episode reset (if episode_length configured) ---
        # ⚠️ `eval_obs` é a observação PRODUZIDA pelo passo que acabou de rodar. O `_eval`
        # compara os preços correntes com `last_rewards`, que também vem desse passo; usar
        # `obs_list` pós-reset compararia preços de reset — aleatórios com
        # `init_mode: random` — contra recompensas do episódio anterior. Corrompia todo eval
        # em que `eval_interval % episode_length == 0`, que é o caso do `calvano_careful`
        # do p1 (os dois valem 10000).
        eval_obs = next_obs_list
        if episode_length and step % episode_length == 0:
            obs_list = env.reset(seed=step)
        else:
            obs_list = next_obs_list

        # --- snapshot por evento (c4c): 1ª mudança do argmax do estado sentinela ---
        if _ev_rows is not None and output_dir is not None and not all(_ev_fired.values()):
            for _k in _learner_slots:
                if _ev_fired[_k]:
                    continue
                _am = int(np.argmax(states[_k].q_table[_ev_rows[_k]]))
                if _am != _ev_prev[_k]:
                    _ev_fired[_k] = True
                    _ev_dir = Path(output_dir) / "agents_snap" / f"event_a{_k}_step_{step:09d}"
                    _ev_dir.mkdir(parents=True, exist_ok=True)
                    for _m2 in _learner_slots:
                        agent.save_state(states[_m2], _ev_dir / f"agent_{_m2}.npz")

        # --- série de snapshots de Q-table (c4) ---
        if step in _q_snap_steps and output_dir is not None and hasattr(agent, "save_state"):
            _snap_dir = Path(output_dir) / "agents_snap" / f"step_{step:09d}"
            _snap_dir.mkdir(parents=True, exist_ok=True)
            for _k in _learner_slots:
                agent.save_state(states[_k], _snap_dir / f"agent_{_k}.npz")

        # --- visit count tracking (N=2 Q-learning only) ---
        if _track_visits:
            _span = _q_max - _act_min
            _a0 = int(np.clip(round((actions[0] - _act_min) / _span * (_n_act - 1)), 0, _n_act - 1))
            _a1 = int(np.clip(round((actions[1] - _act_min) / _span * (_n_act - 1)), 0, _n_act - 1))
            _visit_counts[_a0, _a1] += 1
            if step in _snap_steps:
                _visit_snapshots[f"step_{step:07d}"] = _visit_counts.copy()

        if step % eval_interval == 0:
            record = _eval(
                step, env, mods, states, eval_obs, cfgs,
                win_actions, win_rewards, last_rewards,
                cf_grid, N, bm, env_type,
            )
            # curvas de treino: TD-error janelado, ε e α correntes (p3, etapa 3)
            for i in range(N):
                record[f"td_error_mean_{i}"] = (
                    round(float(np.mean(win_td[i])), 6) if win_td[i] else float("nan")
                )
                record[f"td_error_p95_{i}"] = (
                    round(float(np.percentile(win_td[i], 95)), 6) if win_td[i] else float("nan")
                )
                record[f"alpha_{i}"] = round(_last_alpha[i], 6)
            record["epsilon"] = round(float(getattr(states[0], "epsilon", float("nan"))), 6)
            history.append(record)
            _print_row(record, N)

            if _mlflow_on:
                try:
                    _mlflow.log_metrics({
                        "collusion_index": record["collusion_index"],
                        "nash_dist":       record["nash_dist"],
                        "ce_gap":          record["ce_gap"],
                        "sym_q":           record["sym_q"],
                        "epsilon":         states[0].epsilon,
                        **{f"greedy_q_{i}":    record[f"greedy_q_{i}"]    for i in range(N)},
                        **{f"mean_reward_{i}": record[f"mean_reward_{i}"] for i in range(N)},
                    }, step=step)
                except Exception as _e:
                    print(f"  [MLflow log_metrics failed at step {step}: {_e} — disabling tracking]", flush=True)
                    _mlflow_on = False

            last_win_actions = [list(win_actions[i]) for i in range(N)]
            last_win_rewards = [list(win_rewards[i]) for i in range(N)]
            for i in range(N):
                win_rewards[i].clear()
                win_actions[i].clear()
                win_td[i].clear()

            greedy_qs = [record[f"greedy_q_{i}"] for i in range(N)]
            ce_ok = record["ce_gap"] < es_ce_thr
            if es_prev_greedy_qs is None:
                stable = False
            else:
                stable = max(
                    abs(g - p) for g, p in zip(greedy_qs, es_prev_greedy_qs)
                ) < es_stability_thr
            es_prev_greedy_qs = greedy_qs

            if stable:
                es_stable_count += 1
            else:
                es_stable_count = 0

            if ce_ok and stable:
                es_count += 1
            else:
                es_count = 0

            if es_enabled and step >= es_min_steps and es_count >= es_patience:
                print(
                    f"  [early stop @ step {step}: "
                    f"CE-gap<{es_ce_thr} and stable for {es_patience} evals]",
                    flush=True,
                )
                stopped_early = True
                break
            if es_enabled and step >= es_min_steps and es_stable_count >= es_patience * 3:
                print(
                    f"  [early stop @ step {step}: "
                    f"greedy policy converged (stable for {es_patience * 3} evals)]",
                    flush=True,
                )
                stopped_early = True
                break

    if stopped_early:
        history[-1]["stopped_early"] = True
    _print_footer(N)

    # Save visit count snapshots (N=2 Q-learning)
    if _track_visits and output_dir is not None and history:
        _visit_snapshots[f"step_{step:07d}"] = _visit_counts.copy()
        np.savez(Path(output_dir) / "visit_counts.npz", **_visit_snapshots)

    if full_config is not None and history:
        cce_data = None
        if last_win_actions is not None:
            cce_data = _build_cce_data(env, last_win_actions, last_win_rewards, cf_grid, N)
        folder = multi_agent_output_writer.write(
            full_config, history, bm, output_dir=output_dir, cce_data=cce_data
        )
        print(f"Results written to: {folder}\n")

        # Salvamento genérico: qualquer agente que cumpra o contrato (`save_state`)
        # serializa. Sem isto, um algoritmo novo treina mas não pode ser avaliado —
        # nem strategy test, nem cross-deploy, nem análise de política.
        if hasattr(agent, "save_state"):
            _save_fn = agent.save_state
            agents_dir = folder / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            for i in _learner_slots:
                _save_fn(states[i], agents_dir / f"agent_{i}.npz")
            print(f"Agent states saved to: {agents_dir}\n")

        if _mlflow_on and _mlflow_run:
            _mlflow.log_artifact(str(folder / "config.yaml"))
            if (folder / "metrics_history.csv").exists():
                _mlflow.log_artifact(str(folder / "metrics_history.csv"))
            _mlflow.end_run()

    if history:
        fm = compute_final_metrics(
            history,
            eval_window=int(training_config.get("eval_window", eval_interval * 20)),
            eval_interval=eval_interval,
            benchmarks=bm,
            N=N,
        )
        return {
            "ci":             fm["ci"],
            "ce_gap":         fm["ce_gap"],
            "nash_dist":      fm["nash_dist"],
            "dynamics_label": fm["dynamics_label"],
            "final_step":     history[-1].get("step", 0),
        }
    return {
        "ci": float("nan"), "ce_gap": float("nan"),
        "nash_dist": float("nan"), "dynamics_label": "unknown", "final_step": 0,
    }


# ---------------------------------------------------------------------------
# CCE data builder
# ---------------------------------------------------------------------------

def eval_run(
    env_config: dict,
    agent_config: dict,
    states: list,
    eval_steps: int,
    eval_interval: int = 1_000,
    seed: int = 99,
    output_dir: Path | None = None,
) -> dict:
    """
    Run pre-trained agents in evaluation mode (no Q-table updates, greedy policy).

    Used by cross_deploy_experiment.py to measure CI when agents trained
    independently are deployed together for the first time.

    Parameters
    ----------
    env_config   : environment config (same as used during training)
    agent_config : agent config (same as used during training)
    states       : list of N pre-loaded agent states (e.g. QLearningState)
    eval_steps   : number of environment steps to evaluate
    eval_interval: steps between metric recordings
    seed         : env reset seed (fixed across all eval episodes for comparability)
    output_dir   : if given, writes metrics_history.csv and summary.md

    Returns
    -------
    dict with keys: ci, nash_dist, ce_gap, final_step, history (list of records)
    """
    import importlib

    agent_type = agent_config.get("type", "q_learning")
    agent      = importlib.import_module(f"src.agents.{agent_type}")
    N          = int(env_config["market"]["n_firms"])

    env_type = env_config.get("type", "cournot")
    if env_type == "bertrand":
        env     = MultiAgentBertrandEnv(env_config)
        cf_grid = np.linspace(env.p_min, env.p_max, _N_CF_ACTIONS)
    else:
        env     = MultiAgentCournotEnv(env_config)
        cf_grid = np.linspace(0.0, float(env._q_max), _N_CF_ACTIONS)

    bm = env.benchmarks()

    # Sync agent grid bounds with env
    agent_config = dict(agent_config)
    if env_type == "bertrand":
        agent_config["action_min"] = env.p_min
        agent_config["q_max"]      = env.p_max
    elif env_type == "matrix":
        agent_config["action_min"] = 0.0
        agent_config["q_max"]      = float(env.n_actions - 1)

    rng = jax.random.PRNGKey(seed)
    obs_list = env.reset(seed=seed)

    win_rewards = [[] for _ in range(N)]
    win_actions = [[] for _ in range(N)]
    history: list[dict] = []
    last_rewards = [0.0] * N

    _print_header(N)

    for step in range(1, eval_steps + 1):
        actions: list[float] = []
        for i in range(N):
            rng, action_rng = jax.random.split(rng)
            # training=False → greedy (argmax Q), no epsilon
            a, _ = agent.select_action(
                states[i], obs_list[i], action_rng, agent_config, training=False
            )
            actions.append(a)

        next_obs_list, rewards, dones, _ = env.step(actions)
        last_rewards = rewards

        for i in range(N):
            win_rewards[i].append(rewards[i])
            win_actions[i].append(actions[i])

        obs_list = next_obs_list

        if step % eval_interval == 0:
            record = _eval(
                step, env, agent, states, obs_list, agent_config,
                win_actions, win_rewards, last_rewards,
                cf_grid, N, bm, env_type,
            )
            history.append(record)
            _print_row(record, N)
            for i in range(N):
                win_rewards[i].clear()
                win_actions[i].clear()

    _print_footer(N)

    if output_dir is not None and history:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        csv_path = output_dir / "metrics_history.csv"
        with open(csv_path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        from src.metrics.multi_agent import final_metrics as _fm
        fm = _fm(history, eval_window=eval_interval * 20, eval_interval=eval_interval,
                 benchmarks=bm, N=N)
        summary_path = output_dir / "summary.md"
        summary_path.write_text(
            f"# Eval Run Summary\n\n"
            f"- **eval_steps:** {eval_steps}\n"
            f"- **seed:** {seed}\n"
            f"- **CI:** {fm['ci']:.4f}\n"
            f"- **Nash dist:** {fm['nash_dist']:.4f}\n"
            f"- **CE-gap:** {fm['ce_gap']:.4f}\n"
            f"- **Dynamics:** {fm['dynamics_label']}\n"
        )
        print(f"Eval results written to: {output_dir}\n")

    if history:
        from src.metrics.multi_agent import final_metrics as _fm
        fm = _fm(history, eval_window=eval_interval * 20, eval_interval=eval_interval,
                 benchmarks=bm, N=N)
        return {
            "ci":             fm["ci"],
            "nash_dist":      fm["nash_dist"],
            "ce_gap":         fm["ce_gap"],
            "dynamics_label": fm["dynamics_label"],
            "final_step":     history[-1]["step"],
            "history":        history,
        }
    return {"ci": float("nan"), "nash_dist": float("nan"), "ce_gap": float("nan"),
            "dynamics_label": "unknown", "final_step": 0, "history": []}


def _build_cce_data(
    env,
    win_actions: list[list[float]],
    win_rewards: list[list[float]],
    cf_grid: np.ndarray,
    N: int,
) -> dict:
    """
    Build CCE characterisation data from the last eval window.

    Parameters
    ----------
    win_actions : list[list[float]]  shape (N, T) — stochastic actions played
    win_rewards : list[list[float]]  shape (N, T) — rewards received
    cf_grid     : np.ndarray         coarse grid used during training
    N           : int

    Returns
    -------
    dict with keys: actions, rewards, cf_grid (fine), expected_cf, mean_profits
    """
    actions_arr = np.array(win_actions).T            # (T, N)
    fine_grid   = np.linspace(cf_grid[0], cf_grid[-1], 50)
    expected_cf = [
        env.expected_counterfactual_profits(i, fine_grid, actions_arr)
        for i in range(N)
    ]
    mean_profits = [float(np.mean(win_rewards[i])) for i in range(N)]
    return {
        "actions":      win_actions,    # list[list[float]], (N, T)
        "rewards":      win_rewards,    # list[list[float]], (N, T)
        "cf_grid":      fine_grid,      # np.ndarray, (50,)
        "expected_cf":  expected_cf,    # list[np.ndarray], (N, 50)
        "mean_profits": mean_profits,   # list[float], len N
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _eval(
    step: int,
    env,
    agent,
    states: list,
    obs_list: list,
    agent_config: dict | list[dict],
    win_actions: list[list],
    win_rewards: list[list],
    last_rewards: list[float],
    cf_grid: np.ndarray,
    N: int,
    bm: dict,
    env_type: str,
) -> dict:
    mean_actions = [float(np.mean(win_actions[i])) for i in range(N)]
    mean_rewards = [float(np.mean(win_rewards[i])) for i in range(N)]

    # p7 (`matrix`) reusa o caminho do Bertrand: `benchmarks()` entrega `p_NE` = a ação
    # não-cooperativa e `p_monopoly` = a cooperativa, então o `collusion_index` impresso
    # é **exatamente a taxa de cooperação CR** — a métrica primária do projeto.
    if env_type in ("bertrand", "matrix"):
        p_rl      = float(np.mean(mean_actions))
        ci        = compute_ci_bertrand(p_rl, bm["p_NE"], bm["p_monopoly"])
        nash_dist = nash_distance_joint(mean_actions, bm["p_NE"])
        action_total = p_rl * N
    else:
        Q_rl      = float(sum(mean_actions))
        ci        = compute_ci_cournot(Q_rl, bm["Q_NE"], bm["Q_monopoly"])
        nash_dist = nash_distance_joint(mean_actions, bm["q_NE"])
        action_total = Q_rl

    # CE-gap: single-sample estimate using last step's rival actions
    ce_gap = 0.0
    for i in range(N):
        cf   = env.counterfactual_profits(i, cf_grid)
        gain = float(np.max(cf) - last_rewards[i])
        if gain > ce_gap:
            ce_gap = gain

    # Greedy actions (deterministic policy). `agent` pode ser um módulo único ou a
    # lista `mods` por slot (oponente fixo da c5 no último slot).
    _cfg_of = agent_config if isinstance(agent_config, list) else [agent_config] * N
    _mod_of = agent if isinstance(agent, list) else [agent] * N
    greedy_actions = [
        float(_mod_of[i].greedy_action(states[i], obs_list[i], _cfg_of[i]))
        for i in range(N)
    ]
    sym_q = inter_agent_symmetry(greedy_actions)

    cross_mean = float(np.mean(mean_actions))
    cross_std  = float(np.std(mean_actions))

    record: dict = {
        "step":            step,
        "collusion_index": round(ci, 6),
        "nash_dist":       round(nash_dist, 6),
        "ce_gap":          round(max(ce_gap, 0.0), 6),
        "sym_q":           round(sym_q, 6),
        "Q_rl_total":      round(action_total, 6),
        "action_mean":     round(cross_mean, 6),
        "action_std":      round(cross_std, 6),
    }
    for i in range(N):
        record[f"mean_q_{i}"]      = round(mean_actions[i], 6)
        record[f"mean_reward_{i}"] = round(mean_rewards[i], 6)
        record[f"greedy_q_{i}"]    = round(greedy_actions[i], 6)

    return record


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def _get_curriculum_clip(step: int, cfg: dict, q_max_env: float) -> tuple:
    """
    Compute action clip bounds for the current step from an expanding curriculum.

    The interval starts at [q_min_init, q_max_init] and remains fixed until
    expand_start_step. After that, it expands by expand_delta on each side
    every expand_interval steps, until it covers [0, q_max_env].

    Returns (None, None) when no clipping is active (curriculum absent or
    interval has reached the full action space).

    Config keys (all required when curriculum is present):
      q_min_init        : float — initial lower bound
      q_max_init        : float — initial upper bound
      expand_start_step : int   — step at which expansion begins
      expand_interval   : int   — steps between each expansion
      expand_delta      : float — amount added/subtracted per expansion (each side)
    """
    if not cfg:
        return None, None

    q_min_init   = float(cfg["q_min_init"])
    q_max_init   = float(cfg["q_max_init"])
    start        = int(cfg["expand_start_step"])
    interval     = int(cfg["expand_interval"])
    delta        = float(cfg["expand_delta"])

    if step <= start:
        return q_min_init, q_max_init

    n = (step - start) // interval
    q_lo = max(0.0,        q_min_init - n * delta)
    q_hi = min(q_max_env,  q_max_init + n * delta)

    if q_lo <= 0.0 and q_hi >= q_max_env:
        return None, None

    return q_lo, q_hi


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _log_mlflow_params(mlflow, config: dict, prefix: str = "") -> None:
    """Recursively log all config leaf values as MLflow params."""
    for k, v in config.items():
        key = f"{prefix}.{k}" if prefix else k
        if key.startswith("_"):
            continue
        if isinstance(v, dict):
            _log_mlflow_params(mlflow, v, prefix=key)
        else:
            mlflow.log_param(key[:250], str(v)[:250])


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print_header(N: int) -> None:
    agent_cols = "  ".join(f"{'p'+str(i):>8}" for i in range(N))
    print(f"\n{'step':>8}  {'CI':>8}  {'nash_dist':>10}  {'CE-gap':>10}  {'sym_q':>8}  {'mean':>8}  {'std':>8}  {agent_cols}")
    print("-" * (80 + 11 * N))


def _print_row(record: dict, N: int) -> None:
    agent_vals = "  ".join(f"{record[f'mean_q_{i}']:>8.4f}" for i in range(N))
    print(
        f"{record['step']:>8}  "
        f"{record['collusion_index']:>8.4f}  "
        f"{record['nash_dist']:>10.4f}  "
        f"{record['ce_gap']:>10.4f}  "
        f"{record['sym_q']:>8.4f}  "
        f"{record['action_mean']:>8.4f}  "
        f"{record['action_std']:>8.4f}  "
        f"{agent_vals}"
    )


def _print_footer(N: int) -> None:
    print("-" * (80 + 11 * N))
    print("Training complete.\n")
