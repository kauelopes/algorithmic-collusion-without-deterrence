# Experiment: cross_deploy_calvano

## Configuration
- Agent: q_learning | Seed: 23 | Steps: 2000000
- Market: N=2, demand=logit(a=2.00, μ=0.250), c=1.00
- Info structure: rival_prices

## Analytical Benchmarks
- Nash: p/firm=1.4729, q/firm=0.4714, π/firm=0.2229
- Monopoly: p/firm=1.9250, q/firm=0.3649
- Competitive: p/firm=1.0000

## Equilibrium Results (final window)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Collusion Index | 0.6723 | strong collusion |
| Nash Distance (mean) | 0.3039 | — |
| CE-gap | 0.0458 | deviation incentive present |
| Symmetry (std greedy) | 0.0145 | — |
| Dynamics | converging | — |

## Per-Agent Greedy Prices

| Agent | greedy_price | Nash dist |
|-------|----------|-----------|
| 0 | 1.7958 | 0.3229 |
| 1 | 1.8055 | 0.3326 |

## Interpretation

With **q_learning** agents under **rival_prices** information and **N=2** firms, the system is still converging. The collusion index is **0.672** (strong collusion), and the outcome is not consistent with equilibrium — agents have incentive to deviate (CE-gap = 0.0458). Positive CE-gap means at least one agent would benefit from deviating unilaterally — the current distribution is not an equilibrium.

## Training Curves

![Training Curves](plots/training_curves.png)

See [tables/equilibrium.md](tables/equilibrium.md) for full classification detail.
See [tables/per_agent.md](tables/per_agent.md) for per-agent breakdown.
