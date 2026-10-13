# Experiment: cross_deploy_calvano

## Configuration
- Agent: q_learning | Seed: 19 | Steps: 2000000
- Market: N=2, demand=logit(a=2.00, μ=0.250), c=1.00
- Info structure: rival_prices

## Analytical Benchmarks
- Nash: p/firm=1.4729, q/firm=0.4714, π/firm=0.2229
- Monopoly: p/firm=1.9250, q/firm=0.3649
- Competitive: p/firm=1.0000

## Equilibrium Results (final window)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Collusion Index | 0.6884 | strong collusion |
| Nash Distance (mean) | 0.3112 | — |
| CE-gap | 0.0340 | deviation incentive present |
| Symmetry (std greedy) | 0.0000 | — |
| Dynamics | collusive_fixed_point | — |

## Per-Agent Greedy Prices

| Agent | greedy_price | Nash dist |
|-------|----------|-----------|
| 0 | 1.7958 | 0.3229 |
| 1 | 1.7958 | 0.3229 |

## Interpretation

With **q_learning** agents under **rival_prices** information and **N=2** firms, the system is converged to a stationary point. The collusion index is **0.688** (strong collusion), and the outcome is not consistent with equilibrium — agents have incentive to deviate (CE-gap = 0.0340). Positive CE-gap means at least one agent would benefit from deviating unilaterally — the current distribution is not an equilibrium.

## Training Curves

![Training Curves](plots/training_curves.png)

See [tables/equilibrium.md](tables/equilibrium.md) for full classification detail.
See [tables/per_agent.md](tables/per_agent.md) for per-agent breakdown.
