"""
Strategy players for Bertrand competition analysis.

Each module exposes the same interface as src/agents/q_learning.py so any
strategy can be dropped into the training loop or strategy_analysis runner:

  init(rng, config)                               -> StrategyState
  select_action(state, obs, rng, config, training) -> (float, StrategyState)
  greedy_action(state, obs, config)                -> float
  update(state, transition, config)                -> (StrategyState, dict)
"""
