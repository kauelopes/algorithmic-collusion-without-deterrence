# CCE Verification

Deviation gain = E[π(p′, a_{-i})] − E[π(aᵢ, a_{-i})] averaged over final eval window.
CCE holds if max deviation gain ≤ 0 for all agents.

| Agent | Mean price | Mean profit | Max deviation gain | Best deviation price | CCE? |
|-------|------------|-------------|-------------------|----------------------|------|
| 0 | 1.6988 | 0.3019 | +0.0184 | 1.5744 | ✗ |
| 1 | 1.6980 | 0.3022 | +0.0183 | 1.5744 | ✗ |

**Overall CCE: ✗ not satisfied**

Nash reference price: 1.4729
