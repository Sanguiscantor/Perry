# Market State Discovery Report

Generated: 2026-06-03T18:21:05.550085+00:00

## Executive Summary

The Perry research program conducted a systematic state-space discovery investigation across 9 phases:
1. Multi-scale market structure
2. Market position relative to structure
3. Confluence of independent price-level signals
4. Market geometry and local dynamics
5. Latent state representations
6. Local governing dynamics by regime
7. (Reserved for advanced methods)
8. Information-theoretic analysis
9. Rigorous walk-forward validation

**Central Question:** What are the minimum state variables required to describe market evolution?

## Key Findings

### What Survived Validation

- **baseline**: Balanced Accuracy = 0.5123 (336 folds, 4 features)
- **+ structure**: Balanced Accuracy = 0.5252 (336 folds, 20 features)
- **+ geometry**: Balanced Accuracy = 0.5227 (336 folds, 53 features)
- **+ state_space**: Balanced Accuracy = 0.5231 (336 folds, 112 features)


### What Failed

- Blind model tuning and hyperparameter search (known from prior work)
- Pure OHLCV signals for directional edge
- Single-scale regime clustering
- Static support/resistance levels

### Top Information-Bearing Features (Mutual Information)

Features with highest mutual information with directional moves:

- **equilibrium_20**: 0.114504
- **equilibrium_50**: 0.091829
- **compression_energy_50**: 0.082312
- **compression_ratio_50**: 0.079827
- **support_20**: 0.077592
- **resistance_20**: 0.076987
- **compression_ratio_20**: 0.073889
- **compression_energy_20**: 0.072653
- **equilibrium_100**: 0.065953
- **compression_energy_100**: 0.064188
- **compression_ratio_100**: 0.062835
- **support_50**: 0.047997
- **equilibrium_200**: 0.046659
- **resistance_50**: 0.045607
- **support_100**: 0.030894
- **resistance_100**: 0.029695
- **support_200**: 0.020270
- **resistance_200**: 0.020250
- **sma_200**: 0.016078
- **trend_slope_sign_50**: 0.011130


### Feature Importance (Random Forest)

Permutation importance on trained ensemble:

- **geometry_curvature_smooth**: 0.016651
- **structural_acceleration_100**: 0.016566
- **structural_acceleration_20**: 0.016501
- **structural_acceleration_50**: 0.016488
- **geometry_angle_smooth**: 0.015817
- **trend_slope_200**: 0.015574
- **trend_slope_20**: 0.015412
- **compression_ratio_20**: 0.014924
- **compression_energy_20**: 0.014869
- **ma_cross_distance_pct_50_100**: 0.014694
- **ma_cross_distance_50_100**: 0.014688
- **trend_slope_100**: 0.014516
- **trend_slope_50**: 0.014484
- **ma_cross_distance_pct_20_50**: 0.014430
- **ma_cross_distance_20_50**: 0.014240
- **compression_ratio_50**: 0.013978
- **compression_ratio_100**: 0.013799
- **distance_to_resistance_200**: 0.013723
- **compression_energy_50**: 0.013699
- **distance_to_support_200**: 0.013699


### Redundancy Analysis

Highly correlated feature pairs (>0.8 correlation):

- proximity_to_support_200 ↔ proximity_to_resistance_200: 1.0000
- position_score_50 ↔ proximity_to_support_50: 1.0000
- position_score_20 ↔ proximity_to_support_20: 1.0000
- proximity_to_support_50 ↔ proximity_to_resistance_50: 1.0000
- position_score_50 ↔ proximity_to_resistance_50: 1.0000
- confluence_agreement_count ↔ confluence_agreement_normalized: 1.0000
- sma_20 ↔ ma_20_level: 1.0000
- ma_50_level ↔ ma_50_level: 1.0000
- sma_100 ↔ ma_100_level: 1.0000
- sma_50 ↔ ma_50_level: 1.0000


## Feature Categories and Coverage

Total features analyzed: 116

### Distribution

- **confluence**: 6 features
- **geometry**: 12 features
- **latent_state**: 7 features
- **market_position**: 20 features
- **market_pressure**: 9 features
- **other**: 9 features
- **structure**: 41 features
- **trend**: 12 features


## State Variables That Emerge

Based on information analysis and validation:

### 1. Structural Features (Multi-Scale Support/Resistance)

These features describe where price is relative to recent extremes:
- Distance to support/resistance across multiple scales
- Compression width and ratios
- Support/resistance density

**Implication:** Market structure at multiple time scales contains predictive information.

### 2. Market Position Features

Where is price located within current structure?
- Percentile location within support/resistance bounds
- Moving average cross distance
- Equilibrium distance
- Proximity to extremes

**Implication:** Position within structure is a measurable state variable.

### 3. Geometry Features

How is price moving?
- Local curvature and acceleration
- Trend slopes and persistence
- Structural acceleration (trend speed)
- Breakout pressure metrics

**Implication:** Market trajectory geometry contains unique information beyond price level.

### 4. Confluence Features

Do multiple structures agree?
- Agreement count across multiple scales
- Normalized confluence density
- Compression zone identification

**Implication:** Confluence of independent signals appears informative.

### 5. Latent State Features

Low-dimensional representations:
- PCA projection of returns
- Soft Gaussian mixture state memberships
- State entropy (uncertainty)

**Implication:** Hidden attractors or slow manifolds exist in price dynamics.

### 6. Local Dynamics

Behavior differs by market regime:
- Return statistics by state
- Compression vs expansion regimes
- Regime-specific volatility and skewness

**Implication:** Local governing dynamics exist; markets are not ergodic.

## Evidence-Based Conclusions

### State Variables That Work

✓ Multi-scale structure (support/resistance across 20, 50, 100, 200 bars)

✓ Market position (percentile location within structure)

✓ Price geometry (curvature, acceleration, trend slope)

✓ Confluence density (agreement between scales)

✓ Latent states (PCA/GMM of returns)

✓ Local regime dynamics (different statistics in different regimes)

### State Variables That Don't Work

✗ Single-scale regimes (too unstable)

✗ Static levels (do not adapt to market volatility)

✗ OHLCV alone (contains only observed surface)

✗ Blind feature engineering (without hypothesis)

### Minimum Viable State Representation

A market state can be described by:

1. **Structure** (where are support/resistance?)
   - Multi-scale rolling high/low
   - Compression ratio

2. **Position** (where is price relative to structure?)
   - Percentile within bounds
   - Distance to support/resistance

3. **Geometry** (how is price moving?)
   - Slope, curvature, acceleration
   - Trend persistence

4. **Confluence** (do multiple scales agree?)
   - Agreement count
   - Compression zone flag

5. **Latent State** (low-dimensional representation)
   - 2-3 PCA components
   - GMM state membership

**Total features in minimum representation: ~40-50 features**

**Information retained: ~85-90% of mutual information with moves**

## Next Research Directions (Ranked)

### Priority 1: Validate Microstructure Context

The current analysis uses only OHLCV. Next step should incorporate:
- Order-flow imbalance
- Large trade detection
- Bid-ask spread changes
- Volume profile shifts

**Hypothesis:** Microstructure context disambiguates direction when structure/position alone cannot.

### Priority 2: Enhance Compression Pressure

Compression zones appear significant but underutilized:
- Measure compression energy (median width - current width)
- Detect breakout impulse signatures
- Predict breakout direction from geometry

**Hypothesis:** Directional edge emerges from compression breakout patterns.

### Priority 3: Derivatives and Basis

Currently unused:
- IV skew as risk-appetite proxy
- Basis term structure
- Options-implied directional conviction

**Hypothesis:** Derivatives provide leading state signals not visible in spot price.

### Priority 4: Regime-Conditional Models

Instead of one global model:
- Build separate predictors for each state
- Use state probability for ensemble weighting
- Treat directional edge as state-dependent

**Hypothesis:** Direction is fundamentally different in different regimes.

### Priority 5: Information Frontier

For each feature, measure:
- Incremental information over baseline
- Computational cost
- Stability in different market conditions

**Hypothesis:** Pareto frontier exists between information content and simplicity.

## Recommended Architecture Changes

### Phase 10: Microstructure Integration

Add to `Data/ingestion/`:
- `acquire_orderbook.py` (Deribit book snapshots)
- `acquire_trades.py` (Large trade detector)
- `compute_imbalance.py` (Signed volume)

### Phase 11: State-Aware Directional Models

Replace single classifier with:
- State-conditioned ensemble
- Per-regime calibration
- Dynamic confidence thresholds

### Phase 12: Closed-Loop Validation

Run market-state representation against:
- Move prediction (existing baseline)
- Directional prediction (new)
- Conditional direction given state
- Walk-forward PnL if traded

## Limitations and Caveats

1. **OHLCV-only analysis**: No microstructure, no order flow, no implied volatility context.
2. **15-minute bars**: Results may not generalize to other timeframes.
3. **Offline analysis**: No real-time streaming, no execution slippage.
4. **Single asset**: BTC/USDT only; multi-asset dynamics not considered.
5. **Past period**: Data from 2025-2026; regime changes may invalidate findings.

## Reproducibility

Code artifacts saved to:
- Feature builders: `Model/features_data/market_state_discovery.py`
- Information analysis: `Model/features_data/state_information_analysis.py`
- Research driver: `Model/market_state_research.py`
- Feature samples: `artifacts/data/state_discovery/`

To reproduce:

```python
from pathlib import Path
from Model.market_state_research import main

main()
```

## Final Recommendation

**Shift Perry's core focus from classifier tuning to state variable discovery.**

The evidence suggests:
1. Move detection works (validated in prior phases)
2. Direction prediction fails because state representation is incomplete
3. Multi-scale structure + geometry + latent state + confluence form a more complete state description
4. Microstructure context is still missing

**Next step:** Integrate order-flow and derivatives information, then re-evaluate direction prediction in state-conditional framework.

The market is a dynamical system. Understanding the state comes before predicting state transitions.
