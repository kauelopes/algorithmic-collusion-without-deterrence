"""
As auditorias do paper (§4.2, §4.3(iv), §4.5) recomputadas sobre os 1.024 pares
do piloto de atratores — tudo determinístico sobre as políticas congeladas.

Entrada: results/attractor_pilot__<ts>/batch_*/q_tables.npy
         (K, 2, 225, 15), produzido por experiments/p1_attractor_pilot.py.
         Grade de Calvano (grid_extend 0.1), 2M passos — NÃO é a população dos
         18 pares do paper (grade larga); é o mesmo teste noutra população.

Convenção de estado (q_commons.make_greedy_fn): s = idx(p0)·n + idx(p1) para
os dois agentes; mapa conjunto F(s) = pol_a[s]·n + pol_b[s].

Por par, a partir do atrator alcançado do canto cooperativo (p_M, p_M), o mesmo
do piloto. Se o atrator é um ciclo, cada teste de choque é aplicado em CADA fase
e o par conta como "retorna" só se retorna de todas; os passos são a média.

  1. Choque para baixo (§4.2): a firma 0 é forçada por um período à melhor
     resposta estática ao último preço do rival (experiments/impulse_response.py);
     o rival joga a própria política. Depois, política gulosa dos dois.
     - retorna: o estado volta ao CONJUNTO atrator em ≤ 100 períodos
       (em ponto fixo isso é "retorno exato ao preço pré-choque");
     - passos até o retorno; retorno em ≤ 5 períodos;
     - queda do rival: p1_pre − p1(t = 1), a reação imediata.
  2. Choque para cima (auditoria i): a firma 0 é forçada a p_M. Mesmas medidas.
  3. Um único atrator (auditoria iv): bacia do atrator do canto == 225 estados.
  4. Lucro líquido do desviante (§4.5, deviator_net_profit.py): baseline = lucro
     médio da firma 0 no atrator; gain_t0 = π0(0) − baseline;
     loss = Σ_{t=1..100} δ^t (π0(t) − baseline); net = gain + loss; δ = 0.95.
  5. Melhor resposta exata (§4.5, interpretable_ce_gap.py): V_BR − V_path
     sobre V_path no estado do atrator (primeira fase), por agente; "em melhor
     resposta" se o ganho arredondado a 2 casas é 0.00.

Saídas (analysis/):
  data/audits_at_scale.csv       uma linha por par
  reports/audits_at_scale.md     contagens, com as definições acima

Uso: python analysis/scripts/audits_at_scale.py [run_dir]
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.qtable_analysis.q_commons import P_NE, P_M, ci as ci_of  # noqa: E402
from attractor_distribution import load_run, terminal_cycles  # noqa: E402
from interpretable_ce_gap import _payoff_matrix, _policy_value, _best_response_value  # noqa: E402

OUT = ROOT / "analysis"
HORIZON = 100          # como --post 100 do runner original
DELTA = 0.95           # como deviator_net_profit.py
WITHIN = 5             # "within about five periods"


def shock_path(F, pol1, s, forced_a, n, horizon):
    """Estados visitados após forçar a firma 0 a `forced_a` no estado s (t = 0)."""
    path = [forced_a * n + pol1[s]]          # t = 0: firma 0 forçada, rival joga pol1[s]
    for _ in range(horizon):
        path.append(int(F[path[-1]]))
    return path


def shock_stats(F, pol1, attr, forced_of_state, PI, grid, n):
    """Agrega um choque sobre todas as fases do atrator."""
    attr_set = set(attr)
    rets, steps, drops, nets, gains, losses = [], [], [], [], [], []
    base = float(np.mean([PI[a // n, a % n] for a in attr]))  # lucro médio da firma 0 no atrator
    for s in attr:
        path = shock_path(F, pol1, s, forced_of_state(s), n, HORIZON)
        ret = next((t for t in range(1, len(path)) if path[t] in attr_set), None)
        rets.append(ret is not None); steps.append(ret if ret is not None else HORIZON + 1)
        drops.append(grid[s % n] - grid[path[1] % n])
        pi0 = np.array([PI[x // n, x % n] for x in path]) - base
        disc = DELTA ** np.arange(len(pi0))
        gains.append(pi0[0]); losses.append(float((disc[1:] * pi0[1:]).sum()))
        nets.append(gains[-1] + losses[-1])
    return {
        "returns": int(all(rets)), "steps": float(np.mean(steps)),
        "within5": int(all(rets) and max(steps) <= WITHIN), "rival_drop": float(np.mean(drops)),
        "gain_t0": float(np.mean(gains)), "loss_disc": float(np.mean(losses)), "net": float(np.mean(nets)),
    }


def main(run_dir: Path):
    cfg, grid, Q = load_run(run_dir)
    P, _, S, n = Q.shape
    d, c = cfg["environment"]["demand"], float(cfg["environment"]["cost"]["c"])
    gamma = float(cfg["agent"]["gamma"])
    PI = _payoff_matrix(grid, float(d["a"]), float(d["mu"]), c, float(d.get("a0", 0.0)))
    i_m = int(np.argmin(np.abs(grid - P_M))); s_c = i_m * n + i_m
    br_static = PI.argmax(axis=0)            # br_static[j]: melhor resposta a grid[j]

    rows = []
    for p in range(P):
        pol0 = Q[p, 0].argmax(axis=1); pol1 = Q[p, 1].argmax(axis=1)
        F = pol0 * n + pol1
        cyc_id, _, cycles = terminal_cycles(F)
        attr = list(cycles[cyc_id[s_c]])
        single = int(len(cycles) == 1)
        basin = float((cyc_id == cyc_id[s_c]).mean())
        down = shock_stats(F, pol1, attr, lambda s: int(br_static[s % n]), PI, grid, n)
        up = shock_stats(F, pol1, attr, lambda s: i_m, PI, grid, n)
        p_pre = float(np.mean([grid[a // n] for a in attr] + [grid[a % n] for a in attr]))
        # melhor resposta exata, por agente, no estado do atrator (primeira fase)
        s_star = attr[0]; br = []
        for i in (0, 1):
            V_path = _policy_value(pol0, pol1, PI, gamma, n, agent=i)
            V_br, _ = _best_response_value((pol1, pol0)[i], PI, gamma, n, agent=i)
            vp, vb = float(V_path[s_star]), float(V_br[s_star])
            br.append(round(100 * (vb - vp) / vp, 2))
        rows.append({
            "pair": p, "cycle_len": len(attr), "ci_pre": round(ci_of(p_pre), 4), "p_pre": round(p_pre, 4),
            "single_attractor": single, "corner_basin": round(basin, 4),
            "down_returns": down["returns"], "down_steps": round(down["steps"], 2), "down_within5": down["within5"],
            "down_rival_drop": round(down["rival_drop"], 4),
            "up_returns": up["returns"], "up_steps": round(up["steps"], 2), "up_within5": up["within5"],
            "up_rival_drop": round(up["rival_drop"], 4),
            "dev_gain_t0": round(down["gain_t0"], 5), "dev_loss_disc": round(down["loss_disc"], 5),
            "dev_net": round(down["net"], 5), "dev_pays": int(down["net"] > 0),
            "br_gain_pct_0": br[0], "br_gain_pct_1": br[1],
            "at_br_0": int(br[0] == 0.0), "at_br_1": int(br[1] == 0.0),
        })

    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / "audits_at_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    fp = R["cycle_len"] == 1
    gains = np.concatenate([R["br_gain_pct_0"], R["br_gain_pct_1"]])
    at_br = int(R["at_br_0"].sum() + R["at_br_1"].sum())
    pays = int(R["dev_pays"].sum())

    def pct(x): return f"{100 * x:.1f}%"
    ts = run_dir.name.split("__")[-1]
    L = [f"# Auditorias em escala: {P} pares", "",
         f"Run: `{run_dir.name}` · grid_extend {cfg['environment']['market']['grid_extend']} · "
         f"{cfg['training']['total_steps']:,} passos · script `audits_at_scale.py` · δ = {DELTA} · horizonte {HORIZON}", "",
         "⚠️ População diferente da do paper: os 18 pares de §4.2–§4.5 foram treinados na grade larga; "
         "estes estão na grade de Calvano. Mesmo teste, outra população — reportar lado a lado, não substituir.", "",
         f"Atratores: {int(fp.sum())} pontos fixos ({pct(fp.mean())}), {int((~fp).sum())} ciclos. "
         f"CI pré-choque médio {R['ci_pre'].mean():.3f}, preço pré-choque médio {R['p_pre'].mean():.3f}.", "",
         "## §4.2 — choque para baixo (melhor resposta estática por um período)", "",
         f"- retornam ao atrator em ≤ {HORIZON} períodos: **{int(R['down_returns'].sum())} / {P}** ({pct(R['down_returns'].mean())})"
         f" · em pontos fixos {int(R['down_returns'][fp].sum())} / {int(fp.sum())} · em ciclos {int(R['down_returns'][~fp].sum())} / {int((~fp).sum())}",
         f"- retornam em ≤ {WITHIN} períodos: **{int(R['down_within5'].sum())} / {P}** ({pct(R['down_within5'].mean())})"
         f" · em pontos fixos {int(R['down_within5'][fp].sum())} / {int(fp.sum())}",
         f"- os {int((R['down_returns'] == 0).sum())} que não retornam têm mais de um atrator em "
         f"{int(((R['down_returns'] == 0) & (R['single_attractor'] == 0)).sum())} casos: o choque os leva a OUTRO atrator",
         f"- passos até o retorno (entre os que retornam): média {R['down_steps'][R['down_returns'] == 1].mean():.2f}, "
         f"mediana {np.median(R['down_steps'][R['down_returns'] == 1]):.1f}",
         f"- queda do rival em t = 1: média {R['down_rival_drop'].mean():.3f} (preço pré-choque {R['p_pre'].mean():.3f}); "
         f"o rival cai em {int((R['down_rival_drop'] > 1e-9).sum())} / {P} pares",
         "", "## Auditoria (i) — choque para cima (p_M por um período)", "",
         f"- retornam: **{int(R['up_returns'].sum())} / {P}** ({pct(R['up_returns'].mean())}) · em ≤ {WITHIN}: {int(R['up_within5'].sum())} / {P}",
         f"- passos até o retorno: média {R['up_steps'][R['up_returns'] == 1].mean():.2f}",
         f"- o rival CAI após a generosidade em {int((R['up_rival_drop'] > 1e-9).sum())} / {P} pares (queda média {R['up_rival_drop'].mean():.3f})",
         "", "## Auditoria (iv) — um único atrator", "",
         f"- um único atrator no espaço de estados: **{int(R['single_attractor'].sum())} / {P}** ({pct(R['single_attractor'].mean())})",
         f"- bacia do atrator do canto: média {pct(R['corner_basin'].mean())}, ≥ 99% em {pct((R['corner_basin'] >= 0.99).mean())} dos pares",
         "", "## §4.5 — o desvio ingênuo compensa?", "",
         f"- compensa (net > 0) em **{pays} / {P}** pares; não compensa em {P - pays} / {P} ({pct((P - pays) / P)})",
         f"- dos {pays} em que compensa: {int((R['dev_pays'] * fp).sum())} são pontos fixos e {int((R['dev_pays'] * ~fp).sum())} ciclos "
         f"(em ciclo a baseline é a média do ciclo, e a fase pós-choque pode cair numa fase melhor); "
         f"{int((R['dev_pays'] * fp * R['down_returns']).sum())} são pontos fixos que RETORNAM e ainda assim compensa — "
         f"a resposta do rival existe, mas é branda demais",
         f"- ganho em t = 0: média {R['dev_gain_t0'].mean():+.3f} · perda descontada: média {R['dev_loss_disc'].mean():+.3f} · net: média {R['dev_net'].mean():+.3f}",
         "", "## §4.5 — melhor resposta exata (δ = 0.95, no estado do atrator)", "",
         f"- ganho sobre o lucro convergido: média **{gains.mean():.1f}%**, mediana {np.median(gains):.1f}%, máximo {gains.max():.1f}%",
         f"- agentes já em melhor resposta exata: **{at_br} / {2 * P}** ({pct(at_br / (2 * P))})",
         f"- pares com os DOIS agentes em melhor resposta: {int((R['at_br_0'] * R['at_br_1']).sum())} / {P}",
         "", f"Dados: `data/audits_at_scale.csv`. Os 18 pares do paper: 16/18 retornam, 17/18 um único atrator, "
         "0/18 desvio compensa, +14.8% (10 de 38 em BR)."]
    (OUT / "reports" / "audits_at_scale.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = sorted((ROOT / "results").glob("attractor_pilot__*"))
        run = runs[-1]
    main(run if run.is_absolute() else ROOT / run)
