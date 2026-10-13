# Auditorias em escala: 1024 pares

Run: `attractor_pilot__20261013_180049` · grid_extend 0.1 · 2,000,000 passos · script `audits_at_scale.py` · δ = 0.95 · horizonte 100

⚠️ População diferente da do paper: os 18 pares de §4.2–§4.5 foram treinados na grade larga; estes estão na grade de Calvano. Mesmo teste, outra população — reportar lado a lado, não substituir.

Atratores: 618 pontos fixos (60.4%), 406 ciclos. CI pré-choque médio 0.668, preço pré-choque médio 1.775.

## §4.2 — choque para baixo (melhor resposta estática por um período)

- retornam ao atrator em ≤ 100 períodos: **1006 / 1024** (98.2%) · em pontos fixos 615 / 618 · em ciclos 391 / 406
- retornam em ≤ 5 períodos: **618 / 1024** (60.4%) · em pontos fixos 419 / 618
- os 18 que não retornam têm mais de um atrator em 18 casos: o choque os leva a OUTRO atrator
- passos até o retorno (entre os que retornam): média 4.72, mediana 4.7
- queda do rival em t = 1: média 0.229 (preço pré-choque 1.775); o rival cai em 968 / 1024 pares

## Auditoria (i) — choque para cima (p_M por um período)

- retornam: **1006 / 1024** (98.2%) · em ≤ 5: 660 / 1024
- passos até o retorno: média 4.34
- o rival CAI após a generosidade em 822 / 1024 pares (queda média 0.170)

## Auditoria (iv) — um único atrator

- um único atrator no espaço de estados: **914 / 1024** (89.3%)
- bacia do atrator do canto: média 99.0%, ≥ 99% em 95.1% dos pares

## §4.5 — o desvio ingênuo compensa?

- compensa (net > 0) em **78 / 1024** pares; não compensa em 946 / 1024 (92.4%)
- dos 78 em que compensa: 23 são pontos fixos e 55 ciclos (em ciclo a baseline é a média do ciclo, e a fase pós-choque pode cair numa fase melhor); 21 são pontos fixos que RETORNAM e ainda assim compensa — a resposta do rival existe, mas é branda demais
- ganho em t = 0: média +0.035 · perda descontada: média -0.208 · net: média -0.173

## §4.5 — melhor resposta exata (δ = 0.95, no estado do atrator)

- ganho sobre o lucro convergido: média **17.5%**, mediana 16.2%, máximo 93.7%
- agentes já em melhor resposta exata: **124 / 2048** (6.1%)
- pares com os DOIS agentes em melhor resposta: 1 / 1024

Dados: `data/audits_at_scale.csv`. Os 18 pares do paper: 16/18 retornam, 17/18 um único atrator, 0/18 desvio compensa, +14.8% (10 de 38 em BR).
