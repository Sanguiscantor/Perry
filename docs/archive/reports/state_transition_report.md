# State Transition Report (Phase 4)

**Method:** KMeans k=5 on volatility state features (ATR, return std, range, Hurst, entropy, trend, volume).  
**Data:** BTCUSDT 15m, aligned with move labels.

## Transition Matrix P(State_{t+1} | State_t)

Rows = current state, columns = next state. Values are empirical probabilities from consecutive 15m bars.

| From \ To | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0** | **0.963** | 0.013 | 0.020 | 0.003 | 0.000 |
| **1** | 0.002 | **0.937** | 0.018 | 0.022 | 0.022 |
| **2** | 0.004 | 0.017 | **0.921** | 0.035 | 0.023 |
| **3** | 0.015 | 0.070 | 0.220 | **0.405** | 0.290 |
| **4** | 0.000 | 0.003 | 0.006 | 0.040 | **0.951** |

## Persistence (diagonal)

States **0, 1, 2, 4** are highly persistent (>92% stay). State **3** is transitional (41% self-loop).

## Stationary distribution

State **4** dominates (~50% of bars) — a long-run “compressed vol” regime. State **0** is rare (~5.6%) but **spike-prone**.

## Move rate by state P(move | state)

| State | Move rate (\|r\|>0.5% in 12 bars) |
| ---: | ---: |
| 0 | **66.4%** |
| 1 | 43.3% |
| 2 | 48.7% |
| 3 | 52.6% |
| 4 | **28.8%** |

**Interpretation:** State **0** is elevated move risk; state **4** is low move risk. This aligns with Perry’s vol-expansion thesis but states are **not temporally stable** across years (Phase 3 ARI ≈ 0), so they should be used as **descriptive** buckets, not fixed regime IDs for trading rules.

Full matrices: `research_phases/artifacts/phase4_transitions.json`
