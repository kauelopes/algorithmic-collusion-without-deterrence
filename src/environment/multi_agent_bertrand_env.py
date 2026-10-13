"""
Multi-agent Bertrand environment with logit demand — JAX-optimised.

Follows the Calvano et al. (2020) setup for algorithmic collusion research.
N firms simultaneously choose prices; demand is modelled with a logit system
(differentiated products with an outside option).

Demand model:
    q_i(p) = exp((a - p_i)/mu) / (exp(a0/mu) + sum_j exp((a - p_j)/mu))
    pi_i   = (p_i - c) * q_i(p)

Benchmarks (Bertrand-Nash and joint-monopoly prices) are solved numerically
at construction time via fixed-point iteration in src/econ_utils/benchmarks.py.

Observation space per agent depends on info_structure:

  bandit       : own profit this period              → shape (1,)
  own_price    : own price chosen this period        → shape (1,)
  rival_prices : all prices this period (Calvano)    → shape (N,)

The Calvano baseline uses rival_prices with memory_length=1 for Q-learning:
the Q-table state is the full price vector from the previous period.

Config dict expected keys:
  demand.a, demand.mu, demand.a0   (a0 defaults to 0.0)
  cost.c
  market.n_firms, market.p_min, market.p_max
  state.info_structure ∈ {"bandit", "own_price", "rival_prices"}
  market.grid_extend    (opcional) grade dinâmica: p_min = max(c, p_NE - ext·(p_M-p_NE)),
                        p_max = p_M + ext·(p_M-p_NE). Convenção do projeto desde
                        2026-08-12: **0.1**, a de Calvano (11 dos 15 preços na faixa
                        colusiva). Com 1.0 só 5 ficam na faixa — 5 abaixo de Nash e 5
                        ACIMA do monopólio, onde subir preço destrói lucro.
  market.n_actions      (opcional) K da grade discreta. O env normalmente não precisa
                        saber disso — só o monitoramento imperfeito precisa, porque a
                        corrupção de sinal é um sorteio uniforme SOBRE a grade.
  state.monitoring      (opcional, default perfeito) **P6 — monitoramento imperfeito**:
                        {type: noisy_price, eta: 0.05, scope: public|private}. Com prob.
                        `eta` o preço observado de uma firma é substituído por um sorteio
                        uniforme na grade ⇒ suporte comum, e "o rival desviou" fica
                        indistinguível de "houve ruído". ⚠️ O LUCRO usa sempre o preço
                        VERDADEIRO: o ruído está na observação, não no mercado. Ver
                        docs/theory/modelo_formal_bertrand_logit.md §4.5.
  market.fast_step      (opcional, default False) caminho numpy float32 no `step`:
                        **2,0× end-to-end**, 18× na chamada isolada, porque o array tem 2
                        elementos e o despacho do JAX domina. ⚠️ NÃO é bit-exato (meio
                        ULP em `exp`); equivale a trocar a seed. Uma campanha não mistura
                        os dois caminhos. Ver `_step_numpy` e
                        `docs/dev-notes/todo_optimizations.md`.

⚠️ `bandit` sob Bertrand é DEGENERADO: o obs é um LUCRO (~0.2–0.35) discretizado com o
   discretizador de PREÇO, e tudo satura em s=0. Não use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.econ_utils.benchmarks import logit_benchmarks


# ---------------------------------------------------------------------------
# JIT-compiled kernel cache
# ---------------------------------------------------------------------------

_step_cache: dict = {}
_cf_cache:   dict = {}


def _get_step_fn(n: int, a: float, mu: float, a0: float, c: float):
    """
    Return a JIT-compiled market-clearing function for N Bertrand firms.

    Compiled once per (n, a, mu, a0, c).
    Takes float32 prices array [n], returns:
      prices   : clipped prices              [n]
      shares   : logit demand shares         [n]
      profits  : per-firm profits            [n]
    """
    key = (n, a, mu, a0, c)
    if key not in _step_cache:
        a_  = jnp.float32(a)
        mu_ = jnp.float32(mu)
        a0_ = jnp.float32(a0)
        c_  = jnp.float32(c)

        @jax.jit
        def _step(prices_f32: jnp.ndarray):
            logits  = jnp.exp((a_ - prices_f32) / mu_)
            denom   = jnp.exp(a0_ / mu_) + logits.sum()
            shares  = logits / denom
            profits = (prices_f32 - c_) * shares
            return prices_f32, shares, profits

        _step_cache[key] = _step
    return _step_cache[key]


def _get_cf_fn(a: float, mu: float, a0: float, c: float):
    """
    Return a JIT+vmap counterfactual profits function for Bertrand.

    Compiled once per (a, mu, a0, c).
    Signature: (price_grid [K], D_rivals scalar) → profits [K]
    where D_rivals = exp(a0/mu) + sum_{j≠i} exp((a-p_j)/mu).
    """
    key = (a, mu, a0, c)
    if key not in _cf_cache:
        a_  = jnp.float32(a)
        mu_ = jnp.float32(mu)
        c_  = jnp.float32(c)

        def _single(p_prime: jnp.ndarray, D_rivals: jnp.ndarray) -> jnp.ndarray:
            logit_i = jnp.exp((a_ - p_prime) / mu_)
            share_i = logit_i / (D_rivals + logit_i)
            return (p_prime - c_) * share_i

        _cf_cache[key] = jax.jit(jax.vmap(_single, in_axes=(0, None)))
    return _cf_cache[key]


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class MultiAgentBertrandEnv:
    def __init__(self, config: dict):
        d = config["demand"]
        self.a:    float = float(d["a"])
        self.mu:   float = float(d["mu"])
        self.a0:   float = float(d.get("a0", 0.0))
        self.c:    float = float(config["cost"]["c"])
        self.n:    int   = int(config["market"]["n_firms"])
        self.info_structure: str = config["state"]["info_structure"]
        self.init_mode: str = config["market"].get("init_mode", "p_min")

        if self.info_structure not in ("bandit", "own_price", "rival_prices"):
            raise ValueError(f"Unknown info_structure: {self.info_structure!r}")
        if self.n < 2:
            raise ValueError(f"n_firms must be >= 2, got {self.n}")

        # Features categóricas do rival (p3, etapa 8) — anexadas ao obs DEPOIS dos
        # preços, como inteiros em float32. Cardinalidades em FEATURE_CARDINALITY;
        # o agente as decodifica via `agent.obs_radices`. São computadas AQUI (e não
        # no agente) para que strategy players, eval e treino vejam o mesmo obs.
        self.features: list[str] = list(config["state"].get("features") or [])
        # "features puras" (p3, c1b): obs EGOCÊNTRICO [p_rival, features...] — sem o
        # próprio preço (que o agente conhece por ter jogado). Elimina a convenção de
        # slot: as tabelas ficam simétricas e snapshots não permutam nada.
        self.drop_own_price: bool = bool(config["state"].get("drop_own_price", False))
        if self.drop_own_price and not self.features:
            raise ValueError("state.drop_own_price exige state.features não-vazio")
        if self.features:
            if self.info_structure != "rival_prices":
                raise ValueError("state.features exige info_structure: rival_prices")
            if self.n != 2:
                raise ValueError("state.features só está definido para n_firms = 2")
            unknown = set(self.features) - set(self.FEATURE_CARDINALITY)
            if unknown:
                raise ValueError(
                    f"features desconhecidas: {sorted(unknown)}; "
                    f"disponíveis: {sorted(self.FEATURE_CARDINALITY)}"
                )

        self._bm = logit_benchmarks(self.a, self.mu, self.c, self.n, self.a0)

        # Price grid — static from config or dynamic (Calvano-style) via grid_extend
        grid_extend = config["market"].get("grid_extend")
        if grid_extend is not None:
            extend     = float(grid_extend)
            p_ne       = float(self._bm["p_NE"])
            p_m        = float(self._bm["p_monopoly"])
            _span      = p_m - p_ne
            _p_min_raw = p_ne - extend * _span
            self.p_min = max(self.c, _p_min_raw)
            self.p_max = p_m + extend * _span
            # ⚠️ ARMADILHA ADORMECIDA, agora barulhenta. O `max(c, ·)` acima é
            # economicamente necessário (preço abaixo do custo marginal não faz sentido),
            # mas quando ele MORDE a grade deixa de ser simétrica em torno de
            # (p_NE + p_M)/2 — e com isso morre a única propriedade que dá significado a
            # qualquer índice normalizado por [p_NE, p_M]:
            #
            #     E[CI de preço] = 1/2  sob jogo uniforme, para QUALQUER extend e n_actions
            #
            # (Proposição 8 de docs/theory/modelo_formal_bertrand_logit.md, §3.7). Se o
            # clamp morde, o acaso deixa de valer 0.500 e passa a valer o que a âncora
            # ditar — silenciosamente. Ler CI contra 0.500 numa grade descentrada foi o
            # tipo de erro mais caro do projeto (armadilha 16), então isto falha alto.
            #
            # O clamp morde exatamente quando extend > (p_NE − c)/(p_M − p_NE); nos
            # parâmetros de Calvano isso é 1.0462, e as duas convenções em uso (0.1 e 1.0)
            # estão abaixo. Quem precisar de uma grade assim tem de passar `p_min`/`p_max`
            # explícitos e assumir a responsabilidade pela leitura.
            if _p_min_raw < self.c - 1e-12:
                _xi_max = (p_ne - self.c) / _span if _span > 0 else 0.0
                raise ValueError(
                    f"grid_extend={extend} faz p_min={_p_min_raw:.6f} cair abaixo do custo "
                    f"marginal c={self.c}, e o clamp em c DESCENTRA a grade: "
                    f"E[CI] deixa de ser 0.500 (Prop. 8). Máximo seguro para estes "
                    f"parâmetros: grid_extend <= {_xi_max:.4f}. Para uma grade assimétrica "
                    f"deliberada, use `market.p_min`/`market.p_max` explícitos."
                )
        else:
            self.p_min = float(config["market"]["p_min"])
            self.p_max = float(config["market"]["p_max"])

        if self.p_min >= self.p_max:
            raise ValueError(f"p_min={self.p_min} must be < p_max={self.p_max}")

        # Warm up JIT compilation at construction time
        self._step_fn = _get_step_fn(self.n, self.a, self.mu, self.a0, self.c)
        self._cf_fn   = _get_cf_fn(self.a, self.mu, self.a0, self.c)

        # Caminho rápido em numpy — opt-in, ver _step_numpy
        self._fast_step = bool(config["market"].get("fast_step", False))
        _d = config["demand"]
        self._a32   = np.float32(_d["a"])
        self._mu32  = np.float32(_d["mu"])
        self._c32   = np.float32(config["cost"]["c"])
        self._out32 = np.float32(np.exp(float(_d.get("a0", 0.0)) / float(_d["mu"])))

        # Last-step state (updated by step, read by _obs_for)
        self._last_prices:  np.ndarray = np.full(self.n, self.p_min, dtype=np.float32)
        self._last_profits: np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._last_shares:  np.ndarray = np.zeros(self.n, dtype=np.float32)
        # penúltimo período — só as features precisam (direção do rival, Δ-lucro)
        self._prev_prices:  np.ndarray = self._last_prices.copy()
        self._prev_profits: np.ndarray = self._last_profits.copy()

        # ------------------------------------------------------------------
        # P6 — MONITORAMENTO IMPERFEITO (opt-in; default = perfeito e bit-exato)
        # ------------------------------------------------------------------
        # `state.monitoring: {type: noisy_price, eta: 0.05, scope: public}`
        #
        # Implementa hip.p6.3 de docs/theory/modelo_formal_bertrand_logit.md §4.5: um
        # sinal público ruidoso de **suporte comum**. Com probabilidade `eta` o preço
        # observado de uma firma é substituído por um sorteio uniforme na grade; com
        # `1 − eta` é o preço verdadeiro. Logo
        #
        #     P(observar j | verdadeiro j') = (1 − eta)·1[j = j'] + eta/K  >  0   ∀ j, j'
        #
        # — todo sinal tem probabilidade positiva sob toda ação, que é exatamente a
        # condição de suporte comum. É ela que torna "o rival desviou" e "houve ruído"
        # **indistinguíveis**, e é dela que a teoria (Green–Porter; APS) deriva que a
        # punição passa a ser executada NO CAMINHO de equilíbrio.
        #
        # ⚠️ **O lucro é sempre calculado sobre o preço VERDADEIRO.** O ruído está na
        # observação, não no mercado: as firmas ganham o que de fato venderam e enxergam
        # com erro o que o rival fez. Trocar isso transformaria o experimento em outra
        # coisa (ruído de demanda, que é hip.p6.1 e mora em `demand_shock`), então há
        # teste pinando a separação.
        #
        # `scope`:
        #   public  — as duas firmas veem o MESMO vetor corrompido (hip.p6.3, o caso que
        #             a teoria de APS resolve; a história continua comum)
        #   private — cada firma sorteia sua própria corrupção (monitoramento privado, em
        #             que a teoria diz que a coordenação fica ESTRITAMENTE mais difícil)
        # K da grade discreta. O env normalmente não precisa saber disso (a grade vive no
        # agente), mas a corrupção de sinal é um sorteio uniforme SOBRE a grade, então
        # aqui ele é obrigatório — e só aqui.
        _na = config["market"].get("n_actions")
        self.n_actions: int | None = int(_na) if _na is not None else None

        _mon = (config.get("state") or {}).get("monitoring") or {}
        self.monitoring_type  = str(_mon.get("type", "perfect"))
        self.monitoring_eta   = float(_mon.get("eta", 0.0))
        self.monitoring_scope = str(_mon.get("scope", "public"))
        if self.monitoring_type not in ("perfect", "noisy_price"):
            raise ValueError(
                f"state.monitoring.type desconhecido: {self.monitoring_type!r} "
                f"(esperado 'perfect' ou 'noisy_price')"
            )
        if self.monitoring_scope not in ("public", "private"):
            raise ValueError(
                f"state.monitoring.scope desconhecido: {self.monitoring_scope!r} "
                f"(esperado 'public' ou 'private')"
            )
        if not (0.0 <= self.monitoring_eta <= 1.0):
            raise ValueError(f"state.monitoring.eta fora de [0,1]: {self.monitoring_eta}")
        self._noisy_mon = (self.monitoring_type == "noisy_price"
                           and self.monitoring_eta > 0.0)
        if self._noisy_mon and self.n_actions is None:
            raise ValueError(
                "state.monitoring.type=noisy_price exige `market.n_actions` (o K da "
                "grade): a corrupção é um sorteio uniforme SOBRE A GRADE, e sem K não há "
                "grade discreta definida no env"
            )
        # RNG dedicado, semeado em `reset` — o ruído de monitoramento é reprodutível e
        # NÃO consome a aleatoriedade de nenhum outro componente
        self._mon_rng: np.random.Generator = np.random.default_rng(0)
        # preços OBSERVADOS (o que `_obs_for` entrega). Com monitoramento perfeito é
        # sempre o mesmo objeto que `_last_prices`, então nada muda no caminho quente.
        self._obs_prices: np.ndarray = self._last_prices

    # ------------------------------------------------------------------
    # P6 — a corrupção do sinal
    # ------------------------------------------------------------------

    def _corrupt(self, prices: np.ndarray) -> np.ndarray:
        """Aplica a corrupção de monitoramento. Devolve `prices` INTOCADO se perfeito.

        Com `scope: public` há um único vetor corrompido, visto pelas duas firmas. Com
        `scope: private` cada firma sorteia o seu — e nesse caso o retorno é uma matriz
        `(n, n)` em que a linha `i` é o que a firma `i` observa.
        """
        if not self._noisy_mon:
            return prices
        K = int(self.n_actions)
        grid = np.linspace(self.p_min, self.p_max, K)
        if self.monitoring_scope == "public":
            hit = self._mon_rng.random(self.n) < self.monitoring_eta
            if not hit.any():
                return prices
            out = prices.copy()
            draw = grid[self._mon_rng.integers(0, K, size=int(hit.sum()))]
            out[hit] = draw.astype(prices.dtype)
            return out
        # private: (n, n) — linha i = visão da firma i
        hit = self._mon_rng.random((self.n, self.n)) < self.monitoring_eta
        out = np.tile(prices, (self.n, 1))
        n_hit = int(hit.sum())
        if n_hit:
            out[hit] = grid[self._mon_rng.integers(0, K, size=n_hit)].astype(prices.dtype)
        return out

    def _observed(self) -> np.ndarray:
        """O vetor de preços que a firma `agent_idx` deve ver — resolvido em `_obs_for`."""
        return self._obs_prices

    # cardinalidades das features categóricas (p3) — a fonte para `agent.obs_radices`
    FEATURE_CARDINALITY = {
        "rival_direction": 3,   # 0 desceu · 1 manteve · 2 subiu (vs o penúltimo dele)
        "rival_relative":  3,   # 0 cortou · 1 igualou · 2 superou (vs MEU último preço)
        "own_profit_sign": 2,   # 0 lucro não subiu · 1 subiu (vs o penúltimo meu)
    }
    _FEATURE_EPS = 1e-6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0) -> list[np.ndarray]:
        """
        Reset environment. Returns initial observations for all N agents.

        init_mode (set in config["market"]["init_mode"]):
          "p_min"  — deterministic reset to p_min (default, Calvano baseline)
          "random" — uniform random prices in [p_min, p_max] using seed
          "p_nash" — reset to Nash equilibrium price

        Returns
        -------
        obs_list : list[np.ndarray], len=N
        """
        if self.init_mode == "random":
            rng = np.random.default_rng(seed)
            self._last_prices = rng.uniform(self.p_min, self.p_max, size=self.n).astype(np.float32)
        elif self.init_mode == "p_nash":
            p_ne = float(self._bm["p_NE"])
            self._last_prices = np.full(self.n, p_ne, dtype=np.float32)
        elif self.init_mode == "p_monopoly":
            # início ON-PATH (c4_estabilidade): partir de p_min dispararia o gatilho
            # do GRIM no t=0 e o run inteiro mediria outra coisa (risco 3 do pré-registro)
            p_m = float(self._bm["p_monopoly"])
            self._last_prices = np.full(self.n, p_m, dtype=np.float32)
        else:  # "p_min" — retrocompatível
            self._last_prices = np.full(self.n, self.p_min, dtype=np.float32)
        self._last_profits = np.zeros(self.n, dtype=np.float32)
        self._last_shares  = np.zeros(self.n, dtype=np.float32)
        self._prev_prices  = self._last_prices.copy()
        self._prev_profits = self._last_profits.copy()
        # P6: RNG do monitoramento semeado a partir do seed do episódio, em um stream
        # SEPARADO do `init_mode: random` (offset) para que ligar o ruído não mexa nos
        # preços iniciais de um run já rodado
        self._mon_rng = np.random.default_rng((int(seed) + 1) * 1_000_003)
        self._obs_prices = self._corrupt(self._last_prices)
        return [self._obs_for(i) for i in range(self.n)]

    def _step_numpy(self, actions):
        """Caminho rápido em numpy float32 para o passo logit.

        **18× mais rápido que o JAX jitado neste tamanho** (92.7 µs → 5.1 µs): o array
        tem 2 elementos, e o overhead de despacho do JAX domina completamente o custo de
        calcular. Ver docs/dev-notes/todo_optimizations.md.

        ⚠️ **NÃO é bit-exato.** Contra o caminho JAX: 260/525 pontos idênticos, diferença
        máxima 5.96e-08 — meio ULP de float32, vindo de implementações diferentes de
        `exp`. Ao longo de 5M passos isso muda algumas decisões de `argmax`, então um run
        individual NÃO é reproduzível entre os dois caminhos. A perturbação é equivalente
        a trocar a seed: sem viés sistemático, e a distribuição sobre 20 seeds é a mesma.
        Por isso é **opt-in** (`market.fast_step: true`) e uma campanha não deve misturar
        os dois caminhos.
        """
        f32 = np.float32
        p = np.clip(np.asarray(actions, dtype=np.float32), f32(self.p_min), f32(self.p_max))
        e = np.exp((self._a32 - p) / self._mu32, dtype=np.float32)
        d = self._out32 + e.sum(dtype=np.float32)
        shares = e / d
        return p, shares, (p - self._c32) * shares

    def step(
        self, actions: list[float]
    ) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
        """
        Execute one market period with all N agents choosing prices simultaneously.

        Parameters
        ----------
        actions : list[float], len=N
            Price chosen by each agent. Clipped to [p_min, p_max].

        Returns
        -------
        obs_list : list[np.ndarray]   next observations, one per agent
        rewards  : list[float]        profits, one per agent
        dones    : list[bool]         always [False]*N (infinite horizon)
        info     : dict               full market outcome
        """
        # o "último" vira "penúltimo" ANTES da reatribuição — as features leem os dois
        self._prev_prices, self._prev_profits = self._last_prices, self._last_profits
        if self._fast_step:
            prices, shares, profits = self._step_numpy(actions)
            self._last_prices, self._last_profits, self._last_shares = prices, profits, shares
        else:
            prices_f32 = jnp.clip(
                jnp.array(actions, dtype=jnp.float32),
                jnp.float32(self.p_min),
                jnp.float32(self.p_max),
            )
            prices, shares, profits = self._step_fn(prices_f32)
            self._last_prices  = np.array(prices)
            self._last_profits = np.array(profits)
            self._last_shares  = np.array(shares)

        # ⚠️ ordem obrigatória: `rewards` sai do preço VERDADEIRO; só depois o sinal é
        # corrompido. Inverter isso faria as firmas serem pagas pelo que o rival PARECEU
        # cobrar — outro modelo. Pinado em test_lucro_usa_preco_verdadeiro_nao_o_observado.
        rewards  = self._last_profits.tolist()
        self._obs_prices = self._corrupt(self._last_prices)
        obs_list = [self._obs_for(i) for i in range(self.n)]
        dones    = [False] * self.n

        p_NE = self._bm["p_NE"]
        info = {
            "prices":          self._last_prices.tolist(),
            "shares":          self._last_shares.tolist(),
            "rewards":         rewards,
            "p_NE":            p_NE,
            "nash_dist_mean":  float(np.mean(np.abs(self._last_prices - p_NE))),
        }
        return obs_list, rewards, dones, info

    def counterfactual_profits(
        self, agent_idx: int, price_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute agent i's profit for each price in price_grid, holding
        all other agents' last prices fixed.

        Used to compute the CE-gap metric.

        Parameters
        ----------
        agent_idx  : int            index of the agent to evaluate
        price_grid : np.ndarray     shape [K], candidate prices for agent i

        Returns
        -------
        cf_profits : np.ndarray     shape [K], pi_i(p', p_{-i}) for each p'
        """
        # D_rivals = exp(a0/mu) + sum_{j≠i} exp((a-p_j)/mu)
        a0_exp   = float(np.exp(self.a0 / self.mu))
        rival_ps = np.delete(self._last_prices, agent_idx)
        D_rivals = jnp.float32(
            a0_exp + float(np.sum(np.exp((self.a - rival_ps) / self.mu)))
        )
        grid_f32 = jnp.array(price_grid, dtype=jnp.float32)
        cf = self._cf_fn(grid_f32, D_rivals)
        return np.array(cf)

    def expected_counterfactual_profits(
        self, agent_idx: int, cf_grid: np.ndarray, actions_history: np.ndarray
    ) -> np.ndarray:
        """
        Expected counterfactual profit over the empirical action distribution.

        For each candidate price p' in cf_grid, computes
        E_t[π_i(p', a_{-i,t})] averaged over all T steps in actions_history.
        Used for CCE verification: if max_p'(result - mean_own_profit) ≤ 0,
        agent i has no incentive to deviate unilaterally.

        Parameters
        ----------
        agent_idx       : int
        cf_grid         : np.ndarray  shape (K,)
        actions_history : np.ndarray  shape (T, N) — prices at each step

        Returns
        -------
        np.ndarray  shape (K,)
        """
        a0_exp   = float(np.exp(self.a0 / self.mu))
        grid_f32 = jnp.array(cf_grid, dtype=jnp.float32)
        total    = np.zeros(len(cf_grid))
        for prices_t in actions_history:
            rival_ps = np.delete(prices_t, agent_idx)
            D_rivals = jnp.float32(
                a0_exp + float(np.sum(np.exp((self.a - rival_ps) / self.mu)))
            )
            total += np.array(self._cf_fn(grid_f32, D_rivals))
        return total / len(actions_history)

    def benchmarks(self) -> dict:
        """Return all analytical benchmarks as a flat dict."""
        return dict(self._bm)

    def obs_size(self) -> int:
        """Dimension of the observation vector for each agent."""
        if self.info_structure in ("bandit", "own_price"):
            return 1
        n_prices = 1 if self.drop_own_price else self.n
        return n_prices + len(self.features)   # rival_prices: preços (+ features do p3)

    def obs_radices(self, n_actions: int) -> list[int]:
        """Radices para `agent.obs_radices` correspondentes ao obs deste env:
        um dígito `n_actions` por preço + a cardinalidade de cada feature."""
        n_prices = 1 if self.drop_own_price else self.n
        return [int(n_actions)] * n_prices + [self.FEATURE_CARDINALITY[f]
                                              for f in self.features]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _feature_values(self, agent_idx: int) -> list[float]:
        """Valores inteiros (em float32) das features, na ordem de `self.features`."""
        j = 1 - agent_idx
        eps = self._FEATURE_EPS
        out = []
        for f in self.features:
            if f == "rival_direction":
                d = float(self._last_prices[j]) - float(self._prev_prices[j])
                out.append(0.0 if d < -eps else (2.0 if d > eps else 1.0))
            elif f == "rival_relative":
                d = float(self._last_prices[j]) - float(self._last_prices[agent_idx])
                out.append(0.0 if d < -eps else (2.0 if d > eps else 1.0))
            else:  # own_profit_sign
                d = float(self._last_profits[agent_idx]) - float(self._prev_profits[agent_idx])
                out.append(1.0 if d > eps else 0.0)
        return out

    def _obs_for(self, agent_idx: int) -> np.ndarray:
        """Build obs for agent i from cached last-step values."""
        if self.info_structure == "bandit":
            return np.array([self._last_profits[agent_idx]], dtype=np.float32)
        elif self.info_structure == "own_price":
            return np.array([self._last_prices[agent_idx]], dtype=np.float32)
        # P6: com monitoramento perfeito `_obs_prices is _last_prices` e nada muda.
        # Com `scope: private` a corrupção é uma matriz (n, n) — a linha `agent_idx` é o
        # que ESTA firma observa.
        _p = self._obs_prices
        seen = _p[agent_idx] if (self._noisy_mon
                                 and self.monitoring_scope == "private") else _p
        if self.features:  # rival_prices + features categóricas (p3)
            prices = (seen[1 - agent_idx:2 - agent_idx]
                      if self.drop_own_price else seen)
            return np.concatenate([
                prices,
                np.asarray(self._feature_values(agent_idx), dtype=np.float32),
            ])
        # rival_prices — Calvano: all N prices including own
        return np.asarray(seen, dtype=np.float32).copy()
