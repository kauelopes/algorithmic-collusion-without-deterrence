# Previsão do cross-pair a partir da distribuição de atratores

Run: `attractor_pilot__20261013_180049` · 1024 pares · 2000 pareamentos sorteados (slot 0 de p × slot 1 de q) · medida: CI do preço médio no ciclo terminal, ponderado pela bacia (225 inícios)

| | média | mediana | sd | < Nash (CI<0) |
|---|---|---|---|---|
| dentro do par (mesma medida) | **0.669** | 0.671 | 0.182 | 0.0% |
| entre pares sorteados | **0.134** | 0.127 | 0.097 | 7.4% |

- Queda prevista: **80.0%** (paper, 12 pareamentos: 0.674 → 0.126, −81%; 40 pareamentos: 0.109, 30% abaixo de Nash).
- Pareamentos cruzados com um único atrator: 21.1% (atratores por pareamento: média 2.43).
- CI cruzado vs distância entre os atratores próprios: Spearman ρ = 0.003, p = 8.8e-01.
- CI cruzado por quartil da distância |CI_p − CI_q| (do mais parecido ao mais diferente): 0.133 (n=470) · 0.136 (n=500) · 0.129 (n=505) · 0.136 (n=525)

## Heatmap cruzado (massa ponderada pela bacia)

- massa na diagonal: dentro do par 34.0% · cruzado 14.2%
- células mais frequentes no cruzado: (1.428, 1.466) 4.6% · (1.466, 1.466) 4.3% · (1.428, 1.505) 4.1% · (1.428, 1.428) 4.0% · (1.466, 1.428) 3.9%

Figuras: `figures/fig_attractor_cross_prediction.png`, `figures/fig_attractor_heatmap_cross.png`. Dados: `data/attractor_cross_prediction.csv`, `data/attractor_cross_heatmaps.npz`.
