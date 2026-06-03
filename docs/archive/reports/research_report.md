# Predictive Signal Research Report

## Executive Summary

The repository contains a genuine, temporally stable **move-detection** signal for BTCUSDT 15-minute candles. The strongest target is whether absolute forward return exceeds **0.5%** over **12 candles (3 hours)**. The best model is CatBoost with 100 fold-local selected causal features.

There is **not** enough evidence for a standalone directional trading strategy. Short-horizon direction is weakly predictable, but the effect is economically negative after a 5 bps fee. The hierarchy can detect that a move is coming but cannot reliably assign bullish versus bearish direction.

## Methodology

- Source: `Data\master_raw_dataset.csv`, 84,740 BTCUSDT 15-minute candles.
- Evaluation: expanding chronological walk-forward folds only. No random train/test splits.
- Purging: each training fold ends at least one forecast horizon before its test fold.
- Ranking: macro F1, then balanced accuracy, then accuracy.
- Rejection: experiments are flagged when any class precision or recall is below 10%, or when one predicted class exceeds 90%.
- Feature selection: fold-local only. Saved TSFresh-selected columns from the earlier pipeline were not used as clean evidence because they were selected against the full labeled 40k matrix.
- Economics: directional returns subtract 5 bps per trade and are also checked on non-overlapping 12-candle observations.

## Experiments Performed

Recorded experiments: **283**. Clean non-collapsed experiments: **215**.

- 126 target-screen experiments: multiclass, binary direction, move/no-move, and volatility regime across all requested horizons and thresholds.
- Eight standard model families: logistic regression, SGD linear classifier, ExtraTrees, RandomForest, HistGradientBoosting, XGBoost, LightGBM, and CatBoost.
- Advanced models: SVM, soft voting, and chronological inner-holdout stacking.
- 20 randomized CatBoost tuning trials around the strongest move target.
- 15 Optuna Bayesian CatBoost tuning trials around the strongest move target.
- Four hierarchical move-then-direction probability thresholds.
- Six TSFresh fold-local subset sizes: 50, 100, 250, 500, 1000, and all available features.
- Causal technical, statistical, momentum, volatility, entropy, Hurst, fractal, breakout, and market-structure-derived features.

## Best Configuration

```json
{
  "target": "move",
  "horizon": 12,
  "threshold": 0.005,
  "feature_set": "top_100",
  "model": "catboost",
  "folds": 4,
  "train_fraction": 0.55,
  "test_fraction": 0.09,
  "fee_bps": 5.0,
  "max_train_rows": 45000,
  "model_params": {
    "iterations": 420,
    "depth": 4,
    "learning_rate": 0.023404527272255594,
    "l2_leaf_reg": 5.475344508142733,
    "random_strength": 0.660228740609402
  },
  "notes": "Optuna Bayesian CatBoost move-target search"
}
```

| Metric | Value |
| --- | ---: |
| Mean macro F1 | 0.6009 |
| Macro F1 standard deviation | 0.0229 |
| Worst fold macro F1 | 0.5706 |
| Best fold macro F1 | 0.6308 |
| Mean balanced accuracy | 0.6084 |
| Mean accuracy | 0.6479 |

Daily block bootstrap on the combined OOS predictions gives balanced accuracy **0.6346**, 95% CI **[0.6189, 0.6501]**, with empirical probability at or below chance **0.0000**.

## Target Comparison

| target | horizon | threshold | feature_set | model | macro_f1_mean | macro_f1_std | macro_f1_worst | macro_f1_best | balanced_accuracy_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| move | 12.0000 | 0.0050 | top_100 | catboost | 0.6009 | 0.0229 | 0.5706 | 0.6308 | 0.6084 | 0.6479 |
| direction | 12.0000 |  | core | logistic | 0.5221 | 0.0114 | 0.5117 | 0.5412 | 0.5260 | 0.5244 |
| multiclass | 12.0000 | 0.0050 | top_100+core | catboost+logistic | 0.4046 | 0.0146 | 0.3899 | 0.4289 | 0.4116 | 0.5465 |
| volatility | 12.0000 |  | core | logistic | 0.3900 | 0.0373 | 0.3516 | 0.4273 | 0.4112 | 0.4411 |

The strongest target is move/no-move at 12 candles and 0.5%. Direction is strongest at 12 candles but weak. Multiclass and volatility-regime prediction are substantially worse.

The best hierarchy uses a 0.50 move-probability threshold: macro F1 **0.4046**, balanced accuracy **0.4116**.

## Model Comparison

| model | feature_set | macro_f1_mean | balanced_accuracy_mean | accuracy_mean |
| --- | --- | --- | --- | --- |
| voting | all | 0.5956 | 0.6000 | 0.6512 |
| svm | all | 0.5788 | 0.5884 | 0.6330 |
| temporal_stacking | top_100 | 0.5668 | 0.5960 | 0.6126 |

Tuned CatBoost remains best. Soft voting approaches it but does not improve it. SVM and temporal stacking are weaker. Several XGBoost and HistGradientBoosting configurations were rejected for class collapse.

## Feature Findings

| feature | importance |
| --- | --- |
| range_mean_4 | 7.5340 |
| range_mean_96 | 5.7316 |
| range_mean_48 | 4.6027 |
| atr_14 | 4.3270 |
| range_mean_24 | 4.2455 |
| volume_ratio_192 | 3.9892 |
| range_mean_12 | 3.7535 |
| range_pct | 2.9603 |
| hurst_96 | 2.8269 |
| range_mean_192 | 2.7772 |
| return_entropy_48 | 2.7618 |
| return_std_192 | 2.6501 |
| return_var_192 | 2.5018 |
| range_mean_8 | 2.3211 |
| return_autocorr_48 | 2.2333 |
| return_var_24 | 2.2098 |
| return_var_96 | 2.1202 |
| return_std_24 | 1.8926 |
| return_skew_192 | 1.8596 |
| return_kurt_48 | 1.7774 |
| return_std_96 | 1.7692 |
| volume_z_192 | 1.6899 |
| return_std_48 | 1.5207 |
| return_var_48 | 1.3289 |
| return_skew_48 | 1.2358 |

The signal is dominated by realized range, ATR, rolling return variance, long-window volume ratio, entropy, Hurst exponent, and autocorrelation. This is coherent with volatility-expansion prediction. Directional models instead emphasize ROC and momentum, but their edge is too small to trade after fees.

TSFresh does not improve the result. Its best fold-local setting is `top_1000` with macro F1 **0.5741** and balanced accuracy **0.5771**, below the compact engineered feature set.

## Stability And Economics

- Move signal daily bootstrap: balanced accuracy **0.6346**, 95% CI **[0.6189, 0.6501]**.
- Direction daily bootstrap: balanced accuracy **0.5249**, 95% CI **[0.5129, 0.5364]**.
- Direction net return after 5 bps on non-overlapping forecasts: **-5.19 bps**, 95% CI **[-8.27, -2.16] bps**.
- Hierarchical net return after 5 bps on non-overlapping forecasts: **-1.63 bps**, 95% CI **[-4.04, 0.84] bps**.

The move signal is stable enough to justify downstream use as a risk filter, execution-timing input, or volatility-strategy feature. It is not directly monetizable without a separate direction, options, or market-making component.

## Leakage Audit

- Existing TSFresh windows end at the prediction timestamp and are causal.
- Existing saved TSFresh-selected columns were selected against the full labeled matrix; they were excluded from clean evidence.
- Existing `Model/features_data/market_structure.py` swing and rejection routines inspect future candles; they were excluded from predictive experiments.
- New research features use only current and prior candle information.

## Recommendations

1. Treat move detection as the validated research product. Do not deploy the current directional strategy.
2. Test the move detector on untouched later data and additional symbols before production use.
3. Evaluate monetization through volatility-aware execution, position sizing, options, or market-making logic.
4. Add order-book, spread, funding, open-interest, and cross-asset features if available; these are more plausible sources of incremental direction signal.
5. Keep TSFresh optional. Its compute cost is not justified by the observed result.

## Top 50 Configurations

The complete machine-readable table is also stored at `reports/top_50_configurations.csv`.

| experiment_id | target | horizon | threshold | feature_set | model | macro_f1_mean | balanced_accuracy_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260602T143131-b37a08e286 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.6009 | 0.6084 | 0.6479 |
| 20260602T134816-be766b7b20 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.6009 | 0.6085 | 0.6468 |
| 20260602T143054-4921f1ee3e | move | 12.0000 | 0.0050 | top_100 | catboost | 0.6004 | 0.6083 | 0.6477 |
| 20260602T134409-11105086e8 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5993 | 0.6073 | 0.6459 |
| 20260602T143101-0086c24364 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5991 | 0.6069 | 0.6472 |
| 20260602T134335-5022f8467e | move | 12.0000 | 0.0050 | all | catboost | 0.5990 | 0.6064 | 0.6433 |
| 20260602T134724-2b0c195bb5 | move | 12.0000 | 0.0050 | all | catboost | 0.5989 | 0.6068 | 0.6446 |
| 20260602T143203-d36903cf27 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5988 | 0.6061 | 0.6426 |
| 20260602T143110-50dc5c8067 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5988 | 0.6064 | 0.6445 |
| 20260602T134854-c37c9eace8 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5987 | 0.6065 | 0.6449 |
| 20260602T143248-d1d0acc80d | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5987 | 0.6063 | 0.6406 |
| 20260602T143122-cc33dab6d6 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5985 | 0.6064 | 0.6495 |
| 20260602T143234-d29495672f | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5984 | 0.6060 | 0.6411 |
| 20260602T134612-3d5c8b703c | move | 12.0000 | 0.0050 | all | catboost | 0.5980 | 0.6057 | 0.6442 |
| 20260602T143151-fd6fb6d3ee | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5977 | 0.6050 | 0.6429 |
| 20260602T133409-f9e1a0daca | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5975 | 0.6051 | 0.6458 |
| 20260602T143222-8907ded519 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5972 | 0.6050 | 0.6453 |
| 20260602T133250-e1689d0e77 | move | 12.0000 | 0.0050 | all | catboost | 0.5970 | 0.6051 | 0.6441 |
| 20260602T134627-e84ba0b3c8 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5967 | 0.6044 | 0.6503 |
| 20260602T143037-2a24c2b894 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5967 | 0.6044 | 0.6418 |
| 20260602T134423-0e55e61435 | move | 12.0000 | 0.0050 | all | catboost | 0.5967 | 0.6045 | 0.6462 |
| 20260602T134639-b50ae7bdaa | move | 12.0000 | 0.0050 | all | catboost | 0.5966 | 0.6039 | 0.6404 |
| 20260602T143258-e0dd33b741 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5964 | 0.6041 | 0.6408 |
| 20260602T143144-5e9afa028a | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5961 | 0.6038 | 0.6478 |
| 20260602T134506-046a2d2f7c | move | 12.0000 | 0.0050 | all | catboost | 0.5959 | 0.6034 | 0.6467 |
| 20260602T140858-eac4b31d1a | move | 12.0000 | 0.0050 | all | voting | 0.5956 | 0.6000 | 0.6512 |
| 20260602T134343-0319651dc3 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5956 | 0.6026 | 0.6393 |
| 20260602T134739-55044ce189 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5954 | 0.6031 | 0.6456 |
| 20260602T133306-9599ac6aab | move | 12.0000 | 0.0050 | top_50 | random_forest | 0.5954 | 0.5971 | 0.6582 |
| 20260602T141756-72be7b3c0c | move | 12.0000 | 0.0050 | top_100 | voting | 0.5954 | 0.6001 | 0.6561 |
| 20260602T134753-474befbedd | move | 12.0000 | 0.0050 | all | catboost | 0.5949 | 0.6028 | 0.6415 |
| 20260602T133341-a786f947f8 | move | 12.0000 | 0.0050 | top_100 | extra_trees | 0.5948 | 0.6000 | 0.6528 |
| 20260602T134707-28d53c3802 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5947 | 0.6023 | 0.6419 |
| 20260602T133324-42f4036713 | move | 12.0000 | 0.0050 | top_50 | catboost | 0.5945 | 0.6034 | 0.6434 |
| 20260602T134520-b931309d7b | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5945 | 0.6012 | 0.6384 |
| 20260602T134355-07beeed6b8 | move | 12.0000 | 0.0050 | all | catboost | 0.5942 | 0.6010 | 0.6389 |
| 20260602T133303-94f87c3d04 | move | 12.0000 | 0.0050 | top_50 | extra_trees | 0.5935 | 0.5998 | 0.6510 |
| 20260602T134448-aad9e3e2d1 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5934 | 0.6009 | 0.6338 |
| 20260602T134550-2183e43da1 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5931 | 0.5999 | 0.6328 |
| 20260602T134535-41c2bb3238 | move | 12.0000 | 0.0050 | all | catboost | 0.5930 | 0.5997 | 0.6390 |
| 20260602T133225-6ccc914f51 | move | 12.0000 | 0.0050 | all | extra_trees | 0.5928 | 0.5982 | 0.6527 |
| 20260602T133346-9fb93b5f2e | move | 12.0000 | 0.0050 | top_100 | random_forest | 0.5926 | 0.5945 | 0.6593 |
| 20260602T133406-b14447f675 | move | 12.0000 | 0.0050 | top_100 | lightgbm | 0.5921 | 0.5997 | 0.6361 |
| 20260602T133229-155d78d0f1 | move | 12.0000 | 0.0050 | all | random_forest | 0.5920 | 0.5943 | 0.6620 |
| 20260602T133153-419d2afce2 | move | 12.0000 | 0.0050 | core | random_forest | 0.5919 | 0.5936 | 0.6545 |
| 20260602T133247-2be6a1e43d | move | 12.0000 | 0.0050 | all | lightgbm | 0.5912 | 0.5990 | 0.6364 |
| 20260602T143024-4ef2b047ca | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5911 | 0.5980 | 0.6317 |
| 20260602T134833-4e21c826e5 | move | 12.0000 | 0.0050 | all | catboost | 0.5908 | 0.5965 | 0.6327 |
| 20260602T143044-f5a44e7401 | move | 12.0000 | 0.0050 | top_100 | catboost | 0.5905 | 0.5969 | 0.6352 |
| 20260602T133150-1fd0078dca | move | 12.0000 | 0.0050 | core | extra_trees | 0.5903 | 0.5962 | 0.6466 |
