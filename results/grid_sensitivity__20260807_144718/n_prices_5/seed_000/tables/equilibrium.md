# Equilibrium Identification

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Collusion Index | 1.1626 | full collusion (near monopoly) |
| Nash Distance (mean) | 0.5256 | distance from q_NE=0.4714 |
| CE-gap | 0.1105 | deviation incentive present |
| Inter-agent symmetry (std) | 0.0000 | 0 = symmetric play |
| Dynamics label | converging | — |

## Classification Logic

| Condition | Label |
|-----------|-------|
| autocorr(Q_greedy) > 0.90 AND CI < 0.10 | nash_fixed_point |
| autocorr(Q_greedy) > 0.90 AND CI > 0.20 | collusive_fixed_point |
| 0.30 < autocorr(Q_greedy) ≤ 0.90 | cycling |
| std(Q_greedy) > 0.20 × Q_NE | unstable |
| otherwise | converging |
