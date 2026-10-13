"""
A8a — "O numero que falta" (devil's advocate, painel 2026-08-07): o lucro
liquido do DESVIANTE no down-shock, o teste de deterrence-in-effect que o
proprio Calvano et al. (2020, secao V) reporta.

Pergunta: dado o que o rival de fato faz depois do desvio forcado, o desvio
compensa? Se o ganho de um periodo for mais que compensado pelas perdas na
fase de "punicao", ha deterrence-in-effect (mesmo sem esquema representado);
se o desvio for lucrativo, nem a leitura comportamental de punicao se sustenta.

Metrica, por par (agente 0 = desviante, forcado em t=0):
  baseline  = media de profit_0 na janela pre-choque (t in [-25,-1])
  gain_t0   = profit_0(0) - baseline           (ganho do periodo do desvio)
  loss_ph   = sum_{t>=1} delta^t (profit_0(t) - baseline)   (fase pos-desvio)
  net       = gain_t0 + loss_ph                (>0: desvio compensa)
Reporta tambem a versao nao descontada truncada no retorno exato.

Input : results/impulse_response__20260807/pair_*/impulse.csv (18 pares)
Output: analysis/data/deviator_net_profit.csv + resumo no stdout
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
# Raiz do repo por marcador, não por profundidade — ver src/general_utils/paths.py
def _repo_root(p=None):
    q = Path(p or __file__).resolve()
    q = q if q.is_dir() else q.parent
    for c in (q, *q.parents):
        if (c / "README.md").exists() or (c / ".git").exists():
            return c
    raise RuntimeError("raiz do repositório não encontrada")


ROOT = _repo_root()
IR_DIR = ROOT / "results" / "impulse_response__20260807"
OUT = ROOT / "projects" / "p1_deterrence" / "analysis" / "data" / "deviator_net_profit.csv"

DELTA = 0.95  # desconto de treino (agent_cfg["gamma"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir_dir", type=str, default=str(IR_DIR))
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    ir_dir = Path(args.ir_dir) if Path(args.ir_dir).is_absolute() \
        else ROOT / args.ir_dir
    out_path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    rows = []
    for d in sorted(ir_dir.glob("pair_*")):
        f = d / "impulse.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        pre = df[df.t < 0]
        post = df[df.t >= 0].reset_index(drop=True)
        baseline = pre.profit_0.mean()
        base_price = pre.price_0.mean()

        dev = post.profit_0.values - baseline
        disc = DELTA ** np.arange(len(dev))
        gain_t0 = dev[0]
        loss_ph = float((disc[1:] * dev[1:]).sum())
        net_disc = gain_t0 + loss_ph

        # versao nao descontada, truncada no primeiro retorno exato ao preco
        # pre-choque do proprio desviante (se nunca retorna, usa o horizonte)
        prices = post.price_0.values
        ret_idx = None
        for t in range(1, len(prices)):
            if abs(prices[t] - post.price_0.values[0]) > 1e-9 and \
               abs(prices[t] - base_price) < 1e-6:
                ret_idx = t
                break
        horizon = ret_idx if ret_idx is not None else len(dev) - 1
        net_undisc_ep = float(dev[:horizon + 1].sum())

        rows.append({
            "pair": d.name,
            "baseline_profit": round(baseline, 5),
            "gain_t0": round(gain_t0, 5),
            "loss_punishment_disc": round(loss_ph, 5),
            "net_discounted": round(net_disc, 5),
            "return_step": ret_idx if ret_idx is not None else -1,
            "net_undiscounted_episode": round(net_undisc_ep, 5),
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n = len(out)
    neg = int((out.net_discounted < 0).sum())
    print(f"n = {n} pares (impulse down-shock, agente 0 desviante)")
    print(f"gain_t0:            média {out.gain_t0.mean():+.4f}  "
          f"(min {out.gain_t0.min():+.4f}, max {out.gain_t0.max():+.4f})")
    print(f"perda fase pós (δ): média {out.loss_punishment_disc.mean():+.4f}")
    print(f"NET descontado:     média {out.net_discounted.mean():+.4f}  "
          f"mediana {out.net_discounted.median():+.4f}")
    print(f"  desvio NÃO compensa (net<0) em {neg}/{n} pares")
    neg_u = int((out.net_undiscounted_episode < 0).sum())
    print(f"NET não-descontado até o retorno: média "
          f"{out.net_undiscounted_episode.mean():+.4f} — net<0 em {neg_u}/{n}")
    print(f"\n→ {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
