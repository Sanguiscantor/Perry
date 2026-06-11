# Perry Prototype Report

## Current State

- Asset: BTCUSDT
- Data Source: Binance
- Last Market Timestamp: 2026-06-11 15:00:00
- Data Refresh Status: Success
- Market State: Compression
- Position: Equilibrium
- Confluence: Low
- Move Probability: 64%
- Directional Bias: Bullish
- Bullish Score: 66%
- Bearish Score: 34%
- Expected Move: +0.8% to +1.4%
- Directional Confidence: Moderate
- Model: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\models\20260602T143131-b37a08e286.joblib
- Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_causal_features.pkl
- State Feature Dataset: C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\datasets\processed\btcusdt_state_space_features.csv

## Top Drivers

1. Above equilibrium
2. Compression release
3. Structure support intact
4. Positive trend geometry

## Interpretation

Large move possible.
Current state favors upside.
Confidence remains moderate.

## Runtime Diagnostics

- latest_datetime: 2026-06-11 15:00:00
- latest_close: 62771.0
- compression_ratio_50: 0.6554966749788793
- position_score_50: 0.6315288895929053
- confluence_score: 0.0
- state_entropy: 0.5750160397873006
- trend_persistence_50: 1.0
- causal_feature_status: built
- state_feature_status: built
- model_status: trained_model

## Method

`python app.py` loads the latest available BTCUSDT data, generates base features,
state-space features, anchored structure features, market-state features,
geometry features, confluence features, current market state, move probability,
directional bias, and this markdown report. The prototype estimates directional
bias only; it is not a hard BUY/SELL predictor and does not claim certainty.
