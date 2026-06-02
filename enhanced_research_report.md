# Enhanced Research Report

**Generated:** 2026-06-02T16:38:41+00:00

Frozen CatBoost move spec; no hyperparameter search.

| Feature set | Macro F1 | Balanced acc | Lift vs baseline |
| --- | ---: | ---: | ---: |
| baseline_ohlcv | 0.5992 | 0.6069 | +0.0000 |
| ohlcv_plus_cross | 0.5951 | 0.6029 | -0.0039 |
| ohlcv_plus_derivatives | 0.5979 | 0.6048 | -0.0020 |
| ohlcv_plus_cross_plus_derivatives | 0.5973 | 0.6044 | -0.0025 |
| direction_baseline | 0.5207 | 0.5250 | +0.0000 |
| direction_enhanced | 0.5210 | 0.5247 | +0.0000 |

## Direction (logistic, no tuning)

- Baseline balanced acc: **0.5250**
- Enhanced balanced acc: **0.5247**
- Lift: **-0.0003**
