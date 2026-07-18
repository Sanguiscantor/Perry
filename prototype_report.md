# Perry Prototype Report

## Current State

- Asset: BTCUSDT
- Data Source: Binance
- Last Market Timestamp: 2026-07-18 23:30:00 India Standard Time
- Data Refresh Status: Refreshed
- Market State: Directional Expansion
- Position: Upper Structure
- Confluence: High
- Move Probability: 16%
- Directional Bias: Bullish
- Bullish Score: 62%
- Bearish Score: 38%
- Expected Move: +0.3% to +0.6%
- Directional Confidence: Moderate
- Model: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\models\20260602T143131-b37a08e286.joblib
- Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_causal_features.pkl
- State Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_state_space_features.csv

## Top Drivers

1. Above equilibrium
2. Structure support intact
3. Resistance overhead
4. Positive trend geometry

## Interpretation

Large move possible.
Current state favors upside.
Confidence remains moderate.

## Runtime Diagnostics

- latest_datetime: 2026-07-18 18:00:00
- latest_close: 64437.9
- compression_ratio_50: 0.999999999998195
- position_score_50: 0.9492779783376393
- confluence_score: 0.5
- state_entropy: 0.5231341184310511
- trend_persistence_50: 1.0
- causal_feature_status: built
- state_feature_status: built
- model_status: trained_model
- refresh_occurred: True
- new_candles_downloaded: 1357
- refresh_status: Refreshed
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
