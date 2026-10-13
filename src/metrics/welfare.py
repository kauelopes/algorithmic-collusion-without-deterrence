"""
Welfare metrics for linear Cournot markets.

All computations use jax.numpy.
"""

import jax.numpy as jnp


def consumer_surplus(Q: float, a: float, b: float) -> float:
    """CS = 0.5 * (a - p(Q)) * Q  under linear demand."""
    Q   = jnp.array(Q)
    a_  = jnp.array(a)
    b_  = jnp.array(b)
    price       = jnp.maximum(0.0, a_ - b_ * Q)
    choke_Q     = a_ / b_
    eff_Q       = jnp.minimum(Q, choke_Q)
    return float(0.5 * (a_ - price) * eff_Q)


def producer_surplus(q_firms: list[float], price: float, c: float) -> float:
    """PS = sum_i (price - c) * q_i"""
    qs    = jnp.array(q_firms)
    price = jnp.array(price)
    c_    = jnp.array(c)
    return float(jnp.sum((price - c_) * qs))


def total_welfare(Q: float, q_firms: list[float], a: float, b: float, c: float) -> float:
    price = float(jnp.maximum(0.0, jnp.array(a) - jnp.array(b) * jnp.array(Q)))
    return consumer_surplus(Q, a, b) + producer_surplus(q_firms, price, c)


def deadweight_loss(Q_actual: float, Q_competitive: float, a: float, b: float, c: float) -> float:
    """DWL = W(Q_competitive) - W(Q_actual)."""
    w_comp   = total_welfare(Q_competitive, [Q_competitive], a, b, c)
    w_actual = total_welfare(Q_actual,      [Q_actual],      a, b, c)
    return float(jnp.maximum(0.0, jnp.array(w_comp - w_actual)))


def welfare_table(Q_rl: float, n: int, benchmarks: dict, a: float, b: float, c: float) -> dict:
    """Welfare for RL outcome and all benchmarks. Returns dict of dicts."""
    def row(Q, price):
        qs = [Q / n] * n
        cs = consumer_surplus(Q, a, b)
        ps = producer_surplus(qs, price, c)
        return {"CS": round(cs, 6), "PS": round(ps, 6), "W": round(cs + ps, 6)}

    p_rl = float(jnp.maximum(0.0, jnp.array(a) - jnp.array(b) * jnp.array(Q_rl)))

    return {
        "RL":          row(Q_rl,                    p_rl),
        "Nash":        row(benchmarks["Q_NE"],       benchmarks["p_NE"]),
        "Monopoly":    row(benchmarks["Q_monopoly"], benchmarks["p_monopoly"]),
        "Competitive": row(benchmarks["Q_competitive"], benchmarks["p_competitive"]),
    }


# ---------------------------------------------------------------------------
# Bem-estar para o ambiente BERTRAND-LOGIT (p3/c4) — a lacuna que o README.md
# registrava ("welfare só Cournot linear"). Forma fechada padrão do logit.
# ---------------------------------------------------------------------------

def logit_consumer_surplus(prices, a: float, mu: float, a0: float = 0.0) -> float:
    """Excedente do consumidor esperado do logit (log-sum, por consumidor/período):

        CS(p) = μ · ln( exp(a0/μ) + Σ_i exp((a − p_i)/μ) )

    Decrescente em cada preço; com todos os preços → ∞ colapsa em a0 (só a opção
    externa). É a moeda da leitura antitruste: CS(p_NE) − CS(p) é o custo, por
    período, do desvio de preços em relação ao equilíbrio estático.
    """
    import numpy as _np
    p = _np.asarray(prices, dtype=float)
    return float(mu * _np.log(_np.exp(a0 / mu)
                              + _np.sum(_np.exp((a - p) / mu))))


def logit_welfare_row(prices, a: float, mu: float, c: float,
                      a0: float = 0.0) -> dict:
    """CS, lucro total e bem-estar total num vetor de preços (por período)."""
    import numpy as _np
    p = _np.asarray(prices, dtype=float)
    logits = _np.exp((a - p) / mu)
    shares = logits / (_np.exp(a0 / mu) + logits.sum())
    ps = float(_np.sum((p - c) * shares))
    cs = logit_consumer_surplus(p, a, mu, a0)
    return {"cs": cs, "ps": ps, "welfare": cs + ps}
