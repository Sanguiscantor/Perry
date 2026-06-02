# Falsification Report (Phase 1)

**Generated:** 2026-06-02T15:27:39+00:00

## Executive Summary

The move-detection signal **survived** falsification. Cross-asset replication, BTC future holdout, and regime splits support a real volatility-expansion effect.

## Cross-Asset Validation

| symbol | macro_f1 | balanced_acc | ci_low | pass |
| --- | --- | --- | --- | --- |
| BTCUSDT | 0.5992 | 0.6069 | 0.6181 | True |
| ETHUSDT | 0.6037 | 0.6038 | 0.5907 | True |
| SOLUSDT | 0.5441 | 0.5788 | 0.5652 | True |
| BNBUSDT | 0.6060 | 0.6101 | 0.6242 | True |
| XRPUSDT | 0.5977 | 0.6073 | 0.5954 | True |


### Answers

- **Does the signal survive?** Yes
- **Strongest asset:** BNBUSDT
- **Weakest asset:** SOLUSDT
- **Assets passing:** 5 / 5

## BTC Future Holdout (last 15%)

- Balanced accuracy: **0.6275** (bootstrap CI low: **0.6054**)
- Passes survival criteria: **True**

## Market Regimes (BTC OOS)

| regime | rows | balanced_acc | pass |
| --- | --- | --- | --- |
| vol_tercile:low | 10176 | 0.6195 | True |
| vol_tercile:mid | 10175 | 0.6294 | True |
| vol_tercile:high | 10175 | 0.6514 | True |
| trend_regime:downtrend | 14140 | 0.6406 | True |
| trend_regime:uptrend | 16392 | 0.6273 | True |


## Decision Point A

**Continue advanced research:** True