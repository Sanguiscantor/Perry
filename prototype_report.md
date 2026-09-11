# Perry Prototype Report

## Current State

- Asset: BTCUSDT
- Data Source: Local Cache
- Last Market Timestamp: 2026-07-19 21:00:00 India Standard Time
- Data Refresh Status: UpToDate
- Market State: Directional Expansion
- Position: Equilibrium
- Confluence: High
- Move Probability: 30%
- Directional Bias: Bearish
- Bullish Score: 45%
- Bearish Score: 55%
- Expected Move: -0.2% to -0.4%
- Directional Confidence: Low
- Model: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\models\20260602T143131-b37a08e286.joblib
- Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_causal_features.pkl
- State Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_state_space_features.csv

## Top Drivers

1. Above equilibrium
2. Structure support intact
3. Resistance overhead
4. Negative trend geometry

## Interpretation

Large move possible.
Current state favors downside.
Confidence remains low.

## Runtime Diagnostics

- latest_datetime: 2026-07-19 15:30:00
- latest_close: 64483.2
- compression_ratio_50: 0.9999999999976088
- position_score_50: 0.5349115255845618
- confluence_score: 1.0
- state_entropy: 0.2492508681256154
- trend_persistence_50: 0.88
- causal_feature_status: cached
- state_feature_status: cached
- model_status: trained_model
- refresh_occurred: False
- new_candles_downloaded: 0
- refresh_status: UpToDate
- refresh_warning: 
- raw_dataset_path: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\raw\futures_klines_15m.csv
- report_timestamp_source: raw_dataset_last_datetime
- report_timestamp_source_file: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\raw\futures_klines_15m.csv
- market_timestamp_freshness: fresh
- market_timestamp_stale: false
- market_timestamp_timezone: India Standard Time

## Method

`python app.py` loads the latest available BTCUSDT data, generates base features,
state-space features, anchored structure features, market-state features,
geometry features, confluence features, current market state, move probability,
directional bias, and this markdown report. The prototype estimates directional
bias only; it is not a hard BUY/SELL predictor and does not claim certainty.
