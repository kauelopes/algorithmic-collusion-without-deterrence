# CCE Verification

Deviation gain = E[π(p′, a_{-i})] − E[π(aᵢ, a_{-i})] averaged over final eval window.
CCE holds if max deviation gain ≤ 0 for all agents.

| Agent | Mean price | Mean profit | Max deviation gain | Best deviation price | CCE? |
|-------|------------|-------------|-------------------|----------------------|------|
| 0 | 1.9900 | 0.3178 | +0.1082 | 1.6851 | ✗ |
| 1 | 1.9860 | 0.3197 | +0.1079 | 1.6851 | ✗ |

**Overall CCE: ✗ not satisfied**

Nash reference price: 1.4729
