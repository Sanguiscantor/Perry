# Information Theory Report (Phase 7)

**Method:** Mutual information (sklearn `mutual_info_classif`) between causal features and move label on 15k BTC sample.

## Top features by mutual information

| Feature | MI |
| --- | ---: |
| atr_14 | 0.0509 |
| range_mean_4 | 0.0451 |
| return_std_8 | 0.0449 |
| return_var_8 | 0.0449 |
| range_mean_12 | 0.0442 |
| return_std_24 | 0.0397 |
| range_mean_8 | 0.0394 |

## Conclusion

MI rankings **confirm** CatBoost importance from prior research: short-horizon range and volatility dominate; no novel feature family discovered. **No justification** for representation learning or dynamical-systems complexity on OHLCV alone — bottleneck is **data**, not feature transform (consistent with Phase 5).

Artifact: `research_phases/artifacts/phase7_advanced.json`
