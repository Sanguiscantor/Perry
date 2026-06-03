# Research Journal — Perry

**Last updated:** 2026-06-02T18:38:50+00:00

## Chronological log

### Research log
# Research Log
- 2026-06-02: Created `snapshot/pre-research-20260602-184454` at `be801da` and copied the two pre-existing dirty research scripts to `backups/pre-research-20260602-184454`.
- 2026-06-02: Audited existing pipelines. The saved TSFresh-selected column list was selected against the complete 40k labeled sample and is not valid as clean out-of-sample evidence.
- 2026-06-02: Audited `Model/features_data/market_structure.py`. Existing swing and rejection routines inspect candles after the evaluated timestamp, so they are excluded from predictive research.
- 2026-06-02: Added a separate causal research engine with purged chronological walk-forward evaluation and per-experiment artifacts.
- 2026-06-02: Used a local pickle feature cache to avoid adding an unnecessary Parquet runtime dependency.
- 2026-06-02: Completed 126 broad target-screen experiments with four purged chronological folds. Short-horizon move detection is the strongest baseline family; direction is weakly above chance and multiclass direction is weaker.
- 2026-06-02: Compared eight model families across four causal feature sets for the best move target. CatBoost with all or fold-local top-100 features is strongest; XGBoost and histogram boosting collapse toward one class and are rejected.
- 2026-06-02: Normalized multiclass prediction shape handling after CatBoost returned column-shaped predictions.
- 2026-06-02: Added randomized CatBoost tuning and a chronological hierarchical move-then-direction evaluation mode.
- 2026-06-02: Added SVM, soft voting, chronological inner-holdout stacking, and fold-local TSFresh subset screening modes.
- 2026-06-02: Filtered all-null TSFresh columns inside each training fold before feature selection; TSFresh emits columns that cannot be populated by a 96-candle window.
- 2026-06-02: Completed TSFresh subset screen. TSFresh top-1000 is weaker than the compact causal feature set. Disabled CatBoost's auxiliary file output and ignored its prior generated directory.
- 2026-06-02: Added a reproducible report generator with top-50 ranking, daily block-bootstrap significance, economics, and feature-importance summaries.
- 2026-06-02: Removed the report generator's optional `tabulate` dependency by rendering Markdown tables directly.
- 2026-06-02: Installed Optuna, added Bayesian CatBoost tuning, and added calibration PNG persistence plus retrospective plot generation.
- 2026-06-02: Completed 15 Optuna trials. Best macro F1 improved only from 0.600918 to 0.600929, confirming a tuning plateau. Generated the final report, 181 calibration plots, and a research dependency manifest.
- 2026-06-02T13:22:08+00:00: Built causal feature cache with 84740 rows and 135 features.
- 2026-06-02T13:22:17+00:00: Starting broad target screen with 126 configurations and 4 folds.
- 2026-06-02T13:31:02+00:00: Completed broad target screen.
- 2026-06-02T13:31:46+00:00: Starting model comparison with 32 configurations for move h=12 threshold=0.005.
- 2026-06-02T13:34:17+00:00: Completed model comparison.
- 2026-06-02T13:34:30+00:00: Starting model comparison with 32 configurations for direction h=12 threshold=None.
- 2026-06-02T13:37:05+00:00: Completed model comparison.
- 2026-06-02T13:37:15+00:00: Starting model comparison with 32 configurations for multiclass h=12 threshold=0.004.
- 2026-06-02T13:38:25+00:00: Starting model comparison with 32 configurations for multiclass h=12 threshold=0.004.
- 2026-06-02T13:42:00+00:00: Completed model comparison.
- 2026-06-02T13:43:34+00:00: Starting randomized CatBoost move-target search with 20 trials.
- 2026-06-02T13:49:04+00:00: Completed randomized CatBoost move-target search.
- 2026-06-02T13:49:53+00:00: Completed hierarchical move-then-direction evaluation.
- 2026-06-02T13:51:26+00:00: Starting advanced-model comparison with 9 configurations.
- 2026-06-02T14:18:42+00:00: Completed advanced-model comparison.
- 2026-06-02T14:19:24+00:00: Loading causal rolling-window TSFresh matrix for fold-local subset screen.
- 2026-06-02T14:20:00+00:00: Loading causal rolling-window TSFresh matrix for fold-local subset screen.
- 2026-06-02T14:23:02+00:00: Completed fold-local TSFresh feature screen.
- 2026-06-02T14:30:24+00:00: Starting Optuna Bayesian CatBoost move-target search with 15 trials.
- 2026-06-02T14:33:12+00:00: Completed Optuna search. Best macro F1: 0.600929; params: {'iterations': 420, 'depth': 4, 'learning_rate': 0.023404527272255594, 'l2_leaf_reg': 5.475344508142733, 'random_strength': 0.660228740609402}.
- 2026-06-02T15:23:49+00:00: Research program started
- 2026-06-02T15:23:49+00:00: Phase 1: falsification — cross-asset, holdout, regimes
- 2026-06-02T15:24:26+00:00:   BTCUSDT: balanced=0.6069 pass=True
- 2026-06-02T15:25:03+00:00:   ETHUSDT: balanced=0.6038 pass=True
- 2026-06-02T15:25:41+00:00:   SOLUSDT: balanced=0.5788 pass=True
- 2026-06-02T15:26:18+00:00:   BNBUSDT: balanced=0.6101 pass=True
- 2026-06-02T15:26:56+00:00:   XRPUSDT: balanced=0.6073 pass=True
- 2026-06-02T15:27:39+00:00: Phase 2: universality — pooled train / held-out asset
- 2026-06-02T15:29:51+00:00:   train=4 majors test=XRP: balanced=0.5768
- 2026-06-02T15:32:05+00:00:   LOO test=BTCUSDT: balanced=0.5452
- 2026-06-02T15:34:17+00:00:   LOO test=ETHUSDT: balanced=0.6020
- 2026-06-02T15:36:30+00:00:   LOO test=SOLUSDT: balanced=0.5786
- 2026-06-02T15:38:42+00:00:   LOO test=BNBUSDT: balanced=0.5957
- 2026-06-02T15:40:53+00:00:   LOO test=XRPUSDT: balanced=0.5768
- 2026-06-02T15:41:17+00:00: Phase 3: market state discovery
- 2026-06-02T15:41:55+00:00: Research program started (from_phase=3)
- 2026-06-02T15:42:19+00:00: Phase 3: market state discovery
- 2026-06-02T15:42:24+00:00: Research program started (from_phase=3)
- 2026-06-02T15:42:48+00:00: Phase 3: market state discovery
- 2026-06-02T15:42:58+00:00: Phase 4: state transition matrices
- 2026-06-02T15:42:58+00:00: Phase 5: conditional direction given move probability
- 2026-06-02T15:44:51+00:00: Phase 6: data expansion ranking
- 2026-06-02T15:44:51+00:00: Phase 7: advanced information-theoretic analysis
- 2026-06-02T15:44:54+00:00: Research program completed
- 2026-06-02T16:33:57+00:00: Information frontier Phase 1: cross-asset flow
- 2026-06-02T16:35:50+00:00: Information frontier Phase 2: data integration
- 2026-06-02T16:35:50+00:00: Information frontier Phase 3: enhanced re-evaluation
- 2026-06-02T16:38:41+00:00: Information frontier program completed
- 2026-06-02T17:01:44+00:00: Directional edge: evaluation only
- 2026-06-02T17:02:10+00:00:   evaluate direction: baseline_core
- 2026-06-02T17:02:13+00:00:   evaluate direction: ohlcv_all
- 2026-06-02T17:02:18+00:00:   evaluate direction: taker_flow
- 2026-06-02T17:02:26+00:00:   evaluate direction: sentiment_only
- 2026-06-02T17:02:31+00:00:   evaluate direction: funding_oi
- 2026-06-02T17:02:37+00:00:   evaluate direction: cross_asset
- 2026-06-02T17:02:39+00:00:   evaluate direction: all_alternative
- 2026-06-03T09:12:00+00:00: Implemented microstructure acquisition script `Data/acquire_microstructure.py` to fetch Deribit options instruments/trades, attempt Binance liquidation pagination, and compute futures-spot basis from local klines.
- 2026-06-03T09:12:30+00:00: Ran acquisition script: Deribit instrument fetch returned 400 (parameter format); Binance liquidation endpoint returned maintenance error; basis computation succeeded and wrote `artifacts/data/microstructure/basis_15m.csv` (424,215 rows).
- 2026-06-03T09:13:00+00:00: Wrote microstructure feasibility report `docs/microstructure_data_report.md` and updated `docs/information_catalog.md`.
- 2026-06-03T09:20:00+00:00: Implemented `Data/acquire_deribit_iv.py` to probe Deribit for IV-like fields and saved two live snapshots to `artifacts/data/microstructure/`.
- 2026-06-03T09:30:00+00:00: Implemented `Data/validate_iv.py` and ran leakage-safe validation against baseline; result: no uplift (IV-augmented ≈ baseline). Saved `artifacts/validation/iv_vs_baseline.json`.
- 2026-06-03T09:40:00+00:00: Implemented `Data/backfill_deribit_iv.py` (best-effort). Attempted a 6-month backfill via public API probes — no historical IV snapshots available; backfill blocked by data availability. Recommendation: purchase vendor IV/L2/liquidation feeds or reconstruct IV from historical option price feeds.
- 2026-06-03T09:45:00+00:00: Created `docs/remaining_frontiers.md` with a ranked inventory of remaining data sources and cheapest test paths.
- 2026-06-02T18:32:59+00:00: Directional edge program started
- 2026-06-02T18:32:59+00:00: Directional edge: acquiring datasets
- 2026-06-02T18:32:59+00:00: Running download_derivatives.py
- 2026-06-02T18:33:10+00:00: Running download_extended_klines.py
- 2026-06-02T18:35:38+00:00: Running download_binance_sentiment.py
- 2026-06-02T18:37:07+00:00:   evaluate direction: baseline_core
- 2026-06-02T18:37:09+00:00:   evaluate direction: ohlcv_all
- 2026-06-02T18:37:15+00:00:   evaluate direction: taker_flow
- 2026-06-02T18:37:24+00:00:   evaluate direction: sentiment_only
- 2026-06-02T18:37:30+00:00:   evaluate direction: funding_oi
- 2026-06-02T18:37:35+00:00:   evaluate direction: cross_asset
- 2026-06-02T18:37:37+00:00:   evaluate direction: all_alternative
- 2026-06-02T18:38:50+00:00: Directional edge program completed

### Research program log
# Perry Research Program Log
- 2026-06-02T15:23:49+00:00: Research program started
- 2026-06-02T15:23:49+00:00: Phase 1: falsification — cross-asset, holdout, regimes
- 2026-06-02T15:24:26+00:00:   BTCUSDT: balanced=0.6069 pass=True
- 2026-06-02T15:25:03+00:00:   ETHUSDT: balanced=0.6038 pass=True
- 2026-06-02T15:25:41+00:00:   SOLUSDT: balanced=0.5788 pass=True
- 2026-06-02T15:26:18+00:00:   BNBUSDT: balanced=0.6101 pass=True
- 2026-06-02T15:26:56+00:00:   XRPUSDT: balanced=0.6073 pass=True
- 2026-06-02T15:27:39+00:00: Phase 2: universality — pooled train / held-out asset
- 2026-06-02T15:29:51+00:00:   train=4 majors test=XRP: balanced=0.5768
- 2026-06-02T15:32:05+00:00:   LOO test=BTCUSDT: balanced=0.5452
- 2026-06-02T15:34:17+00:00:   LOO test=ETHUSDT: balanced=0.6020
- 2026-06-02T15:36:30+00:00:   LOO test=SOLUSDT: balanced=0.5786
- 2026-06-02T15:38:42+00:00:   LOO test=BNBUSDT: balanced=0.5957
- 2026-06-02T15:40:53+00:00:   LOO test=XRPUSDT: balanced=0.5768
- 2026-06-02T15:41:17+00:00: Phase 3: market state discovery
- 2026-06-02T15:41:55+00:00: Research program started (from_phase=3)
- 2026-06-02T15:42:19+00:00: Phase 3: market state discovery
- 2026-06-02T15:42:24+00:00: Research program started (from_phase=3)
- 2026-06-02T15:42:48+00:00: Phase 3: market state discovery
- 2026-06-02T15:42:58+00:00: Phase 4: state transition matrices
- 2026-06-02T15:42:58+00:00: Phase 5: conditional direction given move probability
- 2026-06-02T15:44:51+00:00: Phase 6: data expansion ranking
- 2026-06-02T15:44:51+00:00: Phase 7: advanced information-theoretic analysis
- 2026-06-02T15:44:54+00:00: Research program completed
- 2026-06-02T16:33:57+00:00: Information frontier Phase 1: cross-asset flow
- 2026-06-02T16:35:50+00:00: Information frontier Phase 2: data integration
- 2026-06-02T16:35:50+00:00: Information frontier Phase 3: enhanced re-evaluation
- 2026-06-02T16:38:41+00:00: Information frontier program completed
- 2026-06-02T17:01:44+00:00: Directional edge: evaluation only
- 2026-06-02T17:02:10+00:00:   evaluate direction: baseline_core
- 2026-06-02T17:02:13+00:00:   evaluate direction: ohlcv_all
- 2026-06-02T17:02:18+00:00:   evaluate direction: taker_flow
- 2026-06-02T17:02:26+00:00:   evaluate direction: sentiment_only
- 2026-06-02T17:02:31+00:00:   evaluate direction: funding_oi
- 2026-06-02T17:02:37+00:00:   evaluate direction: cross_asset
- 2026-06-02T17:02:39+00:00:   evaluate direction: all_alternative
- 2026-06-02T18:32:59+00:00: Directional edge program started
- 2026-06-02T18:32:59+00:00: Directional edge: acquiring datasets
- 2026-06-02T18:32:59+00:00: Running download_derivatives.py
- 2026-06-02T18:33:10+00:00: Running download_extended_klines.py
- 2026-06-02T18:35:38+00:00: Running download_binance_sentiment.py
- 2026-06-02T18:37:07+00:00:   evaluate direction: baseline_core
- 2026-06-02T18:37:09+00:00:   evaluate direction: ohlcv_all
- 2026-06-02T18:37:15+00:00:   evaluate direction: taker_flow
- 2026-06-02T18:37:24+00:00:   evaluate direction: sentiment_only
- 2026-06-02T18:37:30+00:00:   evaluate direction: funding_oi
- 2026-06-02T18:37:35+00:00:   evaluate direction: cross_asset
- 2026-06-02T18:37:37+00:00:   evaluate direction: all_alternative
- 2026-06-02T18:38:50+00:00: Directional edge program completed

## Hypotheses

| ID | Status | Summary | Evidence | Updated |
| --- | --- | --- | --- | --- |
| H001 | CONFIRMED | Move/vol-expansion is predictable from causal OHLCV. |  | None |
| H002 | REJECTED | TSFresh adds OOS move signal. |  | None |
| H003 | REJECTED | Direction is profitable after fees with current features. |  | None |
| H004 | CONFIRMED | BTC contemporaneously co-moves with alts on 15m. | ETH 1-bar lead ~0 | 2026-06-02T16:39:36+00:00 |
| H005 | REJECTED | Discrete regime labels are stable over time. |  | None |
| H006 | REJECTED | Funding alone insufficient at 15m merge. | Lift -0.0020 | 2026-06-02T16:38:41+00:00 |
| H007 | REJECTED | OI alone insufficient at 15m merge. | Lift -0.0020 | 2026-06-02T16:38:41+00:00 |
| H008 | REJECTED | Cross-asset features redundant with single-asset OHLCV for move OOS. | Lift -0.0039 | 2026-06-02T16:39:36+00:00 |
| H009 | UNTESTED | NSE/macro internals transfer to crypto. |  | None |
| H010 | UNTESTED | Options IV/skew improves move timing. |  | None |
| H011 | REJECTED | Taker flow features enable profitable direction. | Net -4.19 bps taker-only | 2026-06-02T17:04:22+00:00 |
| H012 | REJECTED | Crowding (global L/S) enables profitable direction on full history. | Net -6.54 bps | 2026-06-02T17:04:22+00:00 |

### H001 — CONFIRMED

Move/vol-expansion is predictable from causal OHLCV.

- **Evidence:** 
- **Updated:** None

### H002 — REJECTED

TSFresh adds OOS move signal.

- **Evidence:** 
- **Updated:** None

### H003 — REJECTED

Direction is profitable after fees with current features.

- **Evidence:** 
- **Updated:** None

### H004 — CONFIRMED

BTC contemporaneously co-moves with alts on 15m.

- **Evidence:** ETH 1-bar lead ~0
- **Updated:** 2026-06-02T16:39:36+00:00

### H005 — REJECTED

Discrete regime labels are stable over time.

- **Evidence:** 
- **Updated:** None

### H006 — REJECTED

Funding alone insufficient at 15m merge.

- **Evidence:** Lift -0.0020
- **Updated:** 2026-06-02T16:38:41+00:00

### H007 — REJECTED

OI alone insufficient at 15m merge.

- **Evidence:** Lift -0.0020
- **Updated:** 2026-06-02T16:38:41+00:00

### H008 — REJECTED

Cross-asset features redundant with single-asset OHLCV for move OOS.

- **Evidence:** Lift -0.0039
- **Updated:** 2026-06-02T16:39:36+00:00

### H009 — UNTESTED

NSE/macro internals transfer to crypto.

- **Evidence:** 
- **Updated:** None

### H010 — UNTESTED

Options IV/skew improves move timing.

- **Evidence:** 
- **Updated:** None

### H011 — REJECTED

Taker flow features enable profitable direction.

- **Evidence:** Net -4.19 bps taker-only
- **Updated:** 2026-06-02T17:04:22+00:00

### H012 — REJECTED

Crowding (global L/S) enables profitable direction on full history.

- **Evidence:** Net -6.54 bps
- **Updated:** 2026-06-02T17:04:22+00:00

## Findings

### CONFIRMED
- Volatility-expansion (move) signal exists on 15m crypto OHLCV. (High) — Macro F1 ~0.60, bootstrap CI above chance [research_report.md]
- Move signal survives on BTC, ETH, SOL, BNB, XRP individually. (High) — 5/5 assets pass falsification thresholds [falsification_report.md]
- Best target is move/no-move at 0.5% over 12 candles. (High) — Dominates direction/multiclass in target screen [research_report.md]
- ATR, range, and return-variance features dominate predictors. (High) — CatBoost importance + MI analysis [information_theory_report.md]
- Information bottleneck exceeds model bottleneck on OHLCV. (High) — Optuna plateau; MI flat; direction fails with same model class [grand_research_report.md]
- BTC move labels are contemporaneously coupled with alt move labels (corr 0.34-0.46 at lag 0). (High) — See cross_asset_flow_report.md [cross_asset_flow_report.md]

### REJECTED
- TSFresh materially improves move detection. (High) — Fold-local TSFresh below compact causal set [research_report.md]
- Additional CatBoost tuning yields meaningful gains. (High) — Optuna +0.00001 macro F1 [research_log.md]
- Current directional strategy is profitable after 5 bps. (High) — Mean net −5.19 bps [research_report.md]
- Current hierarchical move→direction strategy is profitable. (High) — Mean net −1.63 bps; Phase 5 all thresholds negative [conditional_direction_report.md]
- Stable market regime labels persist over calendar time. (High) — KMeans ARI ≈ 0 early vs late [market_state_report.md]
- Funding/OI (15m ffilled) do not materially improve move OOS vs OHLCV baseline. (Medium) — Derivatives lift -0.0020 [enhanced_research_report.md]
- Explicit cross-asset engineered features improve BTC move OOS beyond single-asset OHLCV. (High) — Ablation lift -0.0039 balanced acc [cross_asset_flow_report.md]
- No alternative free data source produced economically viable directional edge after fees. (High) — Best config move_only_baseline; all ci_low_bps <= 0 [directional_edge_report.md]
- Taker buy/sell flow from futures klines improves directional economics. (High) — Taker-only net -4.19 bps; OHLCV+taker net -5.33 bps [directional_edge_report.md]
- Binance global/top trader L/S ratios improve full-sample direction OOS. (High) — Sentiment merge net -6.54 bps on full walk-forward [directional_edge_report.md]
- No alternative free data source produced economically viable directional edge after fees. (High) — Best config move_only_baseline; all ci_low_bps <= 0 [directional_edge_report.md]

### PARTIAL EVIDENCE
- Universal pooled model transfers to all assets including BTC. (Medium) — BTC LOO fails (0.545) without BTC in train [cross_asset_report.md]
- State transition structure contains tradeable information. (Low) — Descriptive move rates by state; ARI unstable [state_transition_report.md]
- BTC move/return structure correlates with alt moves at short lags; cross-asset causal features exist. (Medium) — Best BTC→ETHUSDT move corr lag 0: 0.4615; OOS lift -0.0039 balanced acc [cross_asset_flow_report.md]
- May 2026 subset shows positive direction mean net (+1.76 bps) but CI spans zero. (Low) — Short sentiment history; not deployable [directional_edge_report.md]

### UNTESTED
- Funding rates improve move or direction prediction. (N/A) — Not yet integrated in walk-forward [data_expansion_report.md]
- Open interest improves move or direction prediction. (N/A) — Not yet integrated [data_expansion_report.md]
- Liquidations improve prediction. (N/A) — Not yet integrated [data_expansion_report.md]
- Order book imbalance improves prediction. (N/A) — No historical book data [data_expansion_report.md]

## Current roadmap

**Current frontier:** Acquire liquidation/L2 history or deploy move-score vol product.

### Highest-value unknowns

- Paid liquidation feed directional lift
- L2 imbalance at 15m aggregation
- Perp-spot basis intraday

### Recommended experiments

- Coinglass liquidation CSV merge
- Deribit options skew panel
- Move-score vol backtest (non-directional monetization)
