# Perry Prototype Report

## Current State

- Asset: BTCUSDT
- Data Source: Binance
- Last Market Timestamp: 2026-06-06 18:45:00
- Data Refresh Status: Success
- Market State: Compression
- Position: Lower Structure
- Confluence: Low
- Move Probability: 73%
- Directional Bias: Bearish
- Bullish Score: 42%
- Bearish Score: 58%
- Directional Confidence: Moderate

## Top Drivers

1. Trend Geometry
2. Compression Energy
3. Structure Position
4. Equilibrium Distance

## Interpretation

Large move likely.
Current state favors downside.
Confidence remains moderate.

## Runtime Diagnostics

- latest_datetime: 2026-06-06 18:45:00
- latest_close: 60522.6
- compression_ratio_50: 0.45747134349920593
- position_score_50: 0.2833709556055043
- confluence_score: 0.125
- state_entropy: 0.4265125753289974
- trend_persistence_50: 0.92

## Method

`python app.py` loads the latest available BTCUSDT data, generates base features,
state-space features, anchored structure features, market-state features,
geometry features, confluence features, current market state, move probability,
directional bias, and this markdown report. The prototype estimates directional
bias only; it is not a hard BUY/SELL predictor and does not claim certainty.
