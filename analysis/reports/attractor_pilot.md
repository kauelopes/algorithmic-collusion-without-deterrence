# Piloto: distribuição dos atratores conjuntos

Run: `attractor_pilot__20261013_180049` · 1024 pares · grid_extend 0.1 · 2,000,000 passos · script `attractor_distribution.py`

## Forma do atrator (a partir do canto cooperativo)

- ponto fixo: **60.4%** · ciclo de período 2: 21.3% · período ≥ 3: 18.4%
- pontos fixos simétricos (i = j): **287 de 618** (46.4% dos pontos fixos)
- massa na diagonal do heatmap: 33.9%
- CI do atrator: média **0.668**, mediana 0.671, sd 0.184, min 0.029, max 1.100

## Estrutura de bacia

- atratores por par: média 1.13 · um único atrator em **89.3%** dos pares
- bacia dominante: média **99.3%** · ≥ 99% em 95.2% dos pares
- o atrator do canto cooperativo é o dominante em 99.5% dos pares

## Células mais frequentes (canto cooperativo)

| firma 1 | firma 2 | share |
|---|---|---|
| 1.7377 (CI 0.59) | 1.7377 (CI 0.59) | 6.4% |
| 1.7765 (CI 0.67) | 1.7765 (CI 0.67) | 5.7% |
| 1.8540 (CI 0.84) | 1.8540 (CI 0.84) | 5.6% |
| 1.8152 (CI 0.76) | 1.8152 (CI 0.76) | 5.0% |
| 1.8152 (CI 0.76) | 1.7765 (CI 0.67) | 4.5% |

## Convergência (variação total entre duas metades disjuntas)

| N | TV média | sd |
|---|---|---|
| 64 | 0.631 | 0.073 |
| 128 | 0.510 | 0.047 |
| 256 | 0.416 | 0.030 |
| 512 | 0.321 | 0.024 |
| 1024 | 0.246 | 0.022 |

Figuras: `figures/fig_attractor_heatmap.png`, `figures/fig_attractor_convergence.png`. Dados: `data/attractor_pilot_pairs.csv`, `data/attractor_pilot_heatmaps.npz`.
