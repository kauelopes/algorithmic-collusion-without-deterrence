# Equilibrium Identification

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Collusion Index | 0.4989 | strong collusion |
| Nash Distance (mean) | 0.2255 | distance from q_NE=0.4714 |
| CE-gap | 0.0165 | deviation incentive present |
| Inter-agent symmetry (std) | 0.0000 | 0 = symmetric play |
| Dynamics label | collusive_fixed_point | — |

## Classification Logic

| Condition | Label |
|-----------|-------|
| autocorr(Q_greedy) > 0.90 AND CI < 0.10 | nash_fixed_point |
| autocorr(Q_greedy) > 0.90 AND CI > 0.20 | collusive_fixed_point |
| 0.30 < autocorr(Q_greedy) ≤ 0.90 | cycling |
| std(Q_greedy) > 0.20 × Q_NE | unstable |
| otherwise | converging |
