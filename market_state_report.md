# Market State Report (Phase 3)

## Methods tested

| Method | Silhouette | Temporal stability (ARI early vs late) |
| --- | ---: | ---: |
| KMeans k=4 | **0.236** | ~0.000 |
| KMeans k=3 | 0.230 | ~−0.001 |
| KMeans k=5 | 0.232 | ~−0.002 |
| GMM k=4 | 0.112 | ~0.004 |
| HMM | skipped | hmmlearn not installed |
| HDBSCAN | skipped | hdbscan not installed |

## Findings

- **Geometric structure exists:** Volatility features separate into clusters with silhouette ~0.23.
- **Temporal identity does not:** Adjusted Rand Index between early (first 60%) and late (last 30%) windows is **≈ 0** for KMeans — cluster labels **do not persist** as stable “market regimes” over calendar time.
- **Best method by silhouette:** `kmeans_4` (used for Phase 4 transitions).

## Decision Point C

**Stable states for regime trading:** **No** — pivot to using continuous vol features (as Perry already does) rather than discrete regime HMM products.

**Continue transition analysis:** Completed descriptively in Phase 4; not recommended as primary strategy layer.

Artifacts: `research_phases/artifacts/phase3_market_states.json`
