# EXP-01: Cross-Deploy Experiment — Results

## Hypothesis
- **H1a:** CI within-pair ≈ 0.62 — collusion persists with original training partner
- **H1b:** CI cross-pair << 0.62 — collusion collapses with new partner
- **Evidence for SCE:** if H1a holds and H1b holds, collusion is dyadic (training-path artifact)

## Summary

| Condition | Mean CI | std | n |
|-----------|---------|-----|---|
| Training (Env A) | 0.6337 ±0.1082 (n=12) |
| Training (Env B) | 0.6112 ±0.1188 (n=12) |
| **Within-pair deploy** | 0.6741 ±0.1259 (n=12) |
| **Cross-pair deploy**  | 0.1260 ±0.1895 (n=12)  |

## Per-pair Results

| Pair | CI train A | CI train B | CI within | CI cross | Nash within | Nash cross |
|------|-----------|-----------|----------|---------|------------|-----------|
| 000 | 0.6282 | 0.8725 | **0.6607** | **0.0714** | 0.2987 | 0.0969 |
| 001 | 0.6764 | 0.4908 | **0.7143** | **0.2** | 0.3229 | 0.0904 |
| 002 | 0.6803 | 0.4909 | **0.7143** | **0.2643** | 0.3229 | 0.1195 |
| 003 | 0.6827 | 0.6723 | **0.7143** | **0.2857** | 0.3229 | 0.1292 |
| 004 | 0.8439 | 0.6716 | **0.9286** | **-0.1429** | 0.4198 | 0.0969 |
| 005 | 0.4891 | 0.4819 | **0.5** | **-0.0357** | 0.226 | 0.0484 |
| 006 | 0.4853 | 0.4852 | **0.5** | **-0.0625** | 0.226 | 0.0363 |
| 007 | 0.7708 | 0.6884 | **0.8214** | **0.1786** | 0.3713 | 0.0969 |
| 008 | 0.5892 | 0.6499 | **0.7143** | **0.3776** | 0.3229 | 0.1707 |
| 009 | 0.4865 | 0.6705 | **0.5** | **0.0714** | 0.226 | 0.0969 |
| 010 | 0.5879 | 0.6781 | **0.6071** | **0.4464** | 0.2745 | 0.2018 |
| 011 | 0.684 | 0.4821 | **0.7143** | **-0.1429** | 0.3229 | 0.0646 |

## Interpretation

If CI_cross << CI_within, the co-trained strategy is **not** Nash — it is a Self-Confirming Equilibrium calibrated to the specific training trajectory.
The high price behavior does not transfer to a new partner, which would be expected if it were a genuinely learned market strategy.
