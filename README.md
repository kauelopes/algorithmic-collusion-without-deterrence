# Algorithmic Collusion Without Deterrence

Replication package for

> Kauê Lopes de Moraes and Rodrigo De Losso. 2026. **Algorithmic Collusion Without
> Deterrence: Auditing the Punishment Mechanism in Q-Learning Pricing Agents.** In
> *7th ACM International Conference on AI in Finance (ICAIF '26)*, November 14–17,
> 2026, Milan, Italy. ACM. https://doi.org/10.1145/3858063.3858865

The paper replicates the Bertrand–logit duopoly of Calvano, Calzolari, Denicolò and
Pastorello (AER, 2020), confirms the impulse-response pattern within co-trained pairs
of tabular Q-learning agents, and then audits the converged policies with learning
off: directional shocks, scripted opponents, a random-policy null model, a
representational-capacity control, a frozen-policy rematching ("stranger") test, and
an exact dynamic-programming best response against each converged partner.

Everything the figures and tables rest on is in this repository: the environment and
agents, the experiment runners, the converged Q-tables, the analysis scripts and the
camera-ready source of the paper.

## Layout

```
src/                 environment (Bertrand–logit), tabular Q-learning, trainer, metrics,
                     scripted opponents, Q-table utilities
experiments/         runners: replication sweep, co-training + cross-deployment,
                     impulse response, fixed-opponent battery, grid-granularity sweep
configs/             YAML configurations (agents, environment, training, experiments)
analysis/scripts/    one script per figure or table of the paper
analysis/data/       small CSVs produced by the analysis scripts
results/             converged Q-tables (agents/*.npz) and run summaries used by the paper
figures/             the figures as they appear in the paper
paper/               camera-ready LaTeX source (ACM sigconf), bibliography and PDF
tests/               unit tests for the environment, the agent and the trainer
```

## Setup

Python 3.10+ with the packages in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests        # 318 tests
```

JAX runs on CPU by default; a GPU is not required for anything in this package.

## From data to figures

The converged policies are already in `results/`, so every figure can be regenerated
without retraining. Run the scripts from the repository root.

| Paper | Figure / result | Script | Inputs |
|---|---|---|---|
| §4.1 | replication, CI = 0.6245 (50 seeds); reference run at 10M steps | `experiments/sweep_calvano_replication.py`; `experiments/run_experiment_phase2.py --config configs/experiments/calvano_careful.yaml` | `results/calvano_replication__20260505_220647/`, `results/calvano_careful__42__20260624_213325/` |
| §4.2, Fig. 3 | impulse responses, within vs cross pair | `analysis/scripts/plot_impulse_ce_gap.py` | `results/impulse_response__20260807/` |
| §4.3(ii), Fig. 4 | grim / Nash / monopoly opponents from a cooperative start | `analysis/scripts/overlay_cooperative_init.py`, `plot_grim_vs_nash_coop.py` | `results/cross_deploy__20260621_203040/training/*/agents/` |
| §4.3(iii) | reaction functions and policy heatmaps | `analysis/scripts/plot_policy_functions.py` | same Q-tables |
| §4.3(iv), Fig. 5 | random-policy null: basins and return rates | `analysis/scripts/null_model_impulse_response.py` | same Q-tables |
| §4.3(v), Fig. 6 | capacity control: hand-built punishment policies | `analysis/scripts/punishment_representational_capacity.py` | none (constructive) |
| §4.4, Fig. 7 | within-pair vs cross-pair collusion index | `analysis/scripts/plot_cross_deploy_analysis.py` | `results/cross_deploy__20260621_203040/eval/` |
| §4.5 | one-period deviation is unprofitable (18/18) | `analysis/scripts/deviator_net_profit.py` | `results/impulse_response__20260807/` |
| §4.5, Fig. 8 | exact best-response gain (14.8%) | `analysis/scripts/interpretable_ce_gap.py` | Q-tables + `results/calvano_careful__42__20260624_213325/` |
| §4.6, Fig. 9 | grid-granularity sweep | `analysis/scripts/plot_grid_sensitivity.py` | `results/grid_sensitivity__20260807_144718/` |
| §4.4 | cross-pairing at scale: 1,024 pairs, CI 0.669 within vs 0.134 across | `analysis/scripts/attractor_distribution.py`, `attractor_cross_prediction.py` | `results/attractor_pilot__*/` (regenerate below, ~12 min on 24 CPU cores); derived data in `analysis/data/attractor_*` |

Figures 1 and 2 are diagrams and have no script.

## Retraining from scratch

The runs in `results/` were produced by

```bash
python experiments/cross_deploy_experiment.py --config configs/experiments/cross_deploy_bertrand.yaml
python experiments/impulse_response.py --pairs_dir results/cross_deploy__<timestamp>/training
python experiments/strategy_test.py --sweep_dir results/cross_deploy__<timestamp>/training
python experiments/sweep_grid_sensitivity_vectorized.py --levels 5,15,50,100,200
python experiments/sweep_calvano_replication.py --n_seeds 50
python experiments/attractor_pilot.py --n_pairs 1024 --k 128      # the 1,024-pair check (seeds fixed)
```

Each runner prints its usage with `-h`. Seeds are fixed in the configuration files and
on the command line; runs are written to `results/<experiment>__<timestamp>/`.

## Citation

```bibtex
@inproceedings{moraes2026deterrence,
  author    = {Lopes de Moraes, Kau{\^e} and De Losso, Rodrigo},
  title     = {Algorithmic Collusion Without Deterrence: Auditing the Punishment
               Mechanism in {Q}-Learning Pricing Agents},
  booktitle = {7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Milan, Italy},
  doi       = {10.1145/3858063.3858865}
}
```

## License

Code and data are released under the MIT License (see `LICENSE`). The paper in
`paper/` is published open access under CC BY 4.0.
