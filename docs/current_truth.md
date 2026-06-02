# Current Truth — Perry Research Memory

**Last updated:** 2026-06-02T17:04:22+00:00

Auto-maintained from `research_phases/memory_registry.json`. Do not edit by hand; use `Model/research_memory.py` or frontier pipeline.

## CONFIRMED

### Volatility-expansion (move) signal exists on 15m crypto OHLCV.

- **Confidence:** High
- **Evidence:** Macro F1 ~0.60, bootstrap CI above chance
- **Source:** research_report.md


### Move signal survives on BTC, ETH, SOL, BNB, XRP individually.

- **Confidence:** High
- **Evidence:** 5/5 assets pass falsification thresholds
- **Source:** falsification_report.md


### Best target is move/no-move at 0.5% over 12 candles.

- **Confidence:** High
- **Evidence:** Dominates direction/multiclass in target screen
- **Source:** research_report.md


### ATR, range, and return-variance features dominate predictors.

- **Confidence:** High
- **Evidence:** CatBoost importance + MI analysis
- **Source:** information_theory_report.md


### Information bottleneck exceeds model bottleneck on OHLCV.

- **Confidence:** High
- **Evidence:** Optuna plateau; MI flat; direction fails with same model class
- **Source:** grand_research_report.md


### BTC move labels are contemporaneously coupled with alt move labels (corr 0.34-0.46 at lag 0).

- **Confidence:** High
- **Evidence:** See cross_asset_flow_report.md
- **Source:** cross_asset_flow_report.md


## REJECTED

### TSFresh materially improves move detection.

- **Confidence:** High
- **Evidence:** Fold-local TSFresh below compact causal set
- **Source:** research_report.md


### Additional CatBoost tuning yields meaningful gains.

- **Confidence:** High
- **Evidence:** Optuna +0.00001 macro F1
- **Source:** research_log.md


### Current directional strategy is profitable after 5 bps.

- **Confidence:** High
- **Evidence:** Mean net −5.19 bps
- **Source:** research_report.md


### Current hierarchical move→direction strategy is profitable.

- **Confidence:** High
- **Evidence:** Mean net −1.63 bps; Phase 5 all thresholds negative
- **Source:** conditional_direction_report.md


### Stable market regime labels persist over calendar time.

- **Confidence:** High
- **Evidence:** KMeans ARI ≈ 0 early vs late
- **Source:** market_state_report.md


### Funding/OI (15m ffilled) do not materially improve move OOS vs OHLCV baseline.

- **Confidence:** Medium
- **Evidence:** Derivatives lift -0.0020
- **Source:** enhanced_research_report.md


### Explicit cross-asset engineered features improve BTC move OOS beyond single-asset OHLCV.

- **Confidence:** High
- **Evidence:** Ablation lift -0.0039 balanced acc
- **Source:** cross_asset_flow_report.md


### No alternative free data source produced economically viable directional edge after fees.

- **Confidence:** High
- **Evidence:** Best config move_only_baseline; all ci_low_bps <= 0
- **Source:** directional_edge_report.md


### Taker buy/sell flow from futures klines improves directional economics.

- **Confidence:** High
- **Evidence:** Taker-only net -4.19 bps; OHLCV+taker net -5.33 bps
- **Source:** directional_edge_report.md


### Binance global/top trader L/S ratios improve full-sample direction OOS.

- **Confidence:** High
- **Evidence:** Sentiment merge net -6.54 bps on full walk-forward
- **Source:** directional_edge_report.md


## PARTIAL EVIDENCE

### Universal pooled model transfers to all assets including BTC.

- **Confidence:** Medium
- **Evidence:** BTC LOO fails (0.545) without BTC in train
- **Source:** cross_asset_report.md


### State transition structure contains tradeable information.

- **Confidence:** Low
- **Evidence:** Descriptive move rates by state; ARI unstable
- **Source:** state_transition_report.md


### BTC move/return structure correlates with alt moves at short lags; cross-asset causal features exist.

- **Confidence:** Medium
- **Evidence:** Best BTC→ETHUSDT move corr lag 0: 0.4615; OOS lift -0.0039 balanced acc
- **Source:** cross_asset_flow_report.md


### May 2026 subset shows positive direction mean net (+1.76 bps) but CI spans zero.

- **Confidence:** Low
- **Evidence:** Short sentiment history; not deployable
- **Source:** directional_edge_report.md


## UNTESTED

### Funding rates improve move or direction prediction.

- **Confidence:** N/A
- **Evidence:** Not yet integrated in walk-forward
- **Source:** data_expansion_report.md


### Open interest improves move or direction prediction.

- **Confidence:** N/A
- **Evidence:** Not yet integrated
- **Source:** data_expansion_report.md


### Liquidations improve prediction.

- **Confidence:** N/A
- **Evidence:** Not yet integrated
- **Source:** data_expansion_report.md


### Order book imbalance improves prediction.

- **Confidence:** N/A
- **Evidence:** No historical book data
- **Source:** data_expansion_report.md

