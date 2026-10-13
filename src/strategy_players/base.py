"""
Shared state type and observation helpers for all strategy players.
"""

from typing import NamedTuple


class StrategyState(NamedTuple):
    own_last_price: float = 0.0
    last_profit: float = 0.0
    triggered: bool = False
    step: int = 0
    prev_profit: float = 0.0   # lucro do período ANTERIOR — o wsls precisa comparar os dois


def extract_rival_price(obs, own_idx: int, n_agents: int, fallback: float) -> float:
    """Extract mean rival price from rival_prices obs (shape N). Returns fallback if obs too small."""
    if obs is None or len(obs) < n_agents:
        return fallback
    prices = [float(obs[j]) for j in range(n_agents) if j != own_idx]
    if not prices:
        return fallback
    return sum(prices) / len(prices)
