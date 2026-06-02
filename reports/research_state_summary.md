# Research State Summary

## Confirmed findings

- **Move detection is real.** Perry’s strongest validated signal is a 12-bar move/no-move target on BTCUSDT 15m, with macro F1 ≈ 0.60 and balanced accuracy ≈ 0.61 in purged chronological walk-forward.
- **OHLCV volatility features dominate.** The strongest predictive inputs are range, ATR, rolling return variance/std, volume ratios, entropy, Hurst, and autocorrelation.
- **Move score is stable.** Cross-asset and future holdout falsification tests pass for the move model: 5/5 major assets when BTC is included in training, and BTC future holdout also passes.
- **Direction remains weak.** Directional prediction on the same feature space is only slightly above chance (~0.52 balanced accuracy) and is economically negative after 5 bps fees.
- **Existing alternate features do not add direction.** Funding, taker flow, cross-asset features, global/top-trader long-short ratios, and aggregated sentiment all failed to produce a viable directional edge on the available history.

## Rejected hypotheses

- **Standalone directional spot/perp trade is deployable.** Negative net returns after fees for all tested full-history configurations.
- **TSFresh is a reliable improvement.** Fold-local TSFresh results are worse than compact engineered causal features and more expensive to compute.
- **Stable regime labels are sufficient for trading.** Market-state clusters exist descriptively, but adjusted Rand index shows state labels do not persist over time.
- **Cross-asset features materially improve BTC move detection.** Explicit cross-asset augmentation hurts or has no lift for the move model.
- **Funding/OI sufficient for direction.** Existing free Binance derivatives data did not produce a profitable or robust directional signal.

## Strongest signals

- **Primary validated signal:** BTC move/no-move over 12 × 15m bars with 0.5% absolute return threshold.
- **Most predictive feature family:** short-to-medium horizon realized range/volatility and volume-based measures.
- **Highest-value research path:** use the move score as a volatility/risk filter or execution timing input rather than as a directional signal.

## Weakest assumptions

- **Assumption:** OHLCV‑only features can unlock direction. This is weak; evidence points to data limitations, not model architecture.
- **Assumption:** more model tuning will solve direction. Evidence shows tuning plateaued and the bottleneck is missing information sources.
- **Assumption:** cross-asset or funding derivatives alone will rescue direction. They do not on current free datasets.
- **Assumption:** current 15m regime clusters are tradable states. They are descriptive, not stable trade regimes.

## Current bottlenecks

- **Missing high-fidelity data.** The top untested sources are microstructure and option market information: L2 order book, historical liquidations, options IV/skew/term structure.
- **Incomplete derivative coverage.** Open interest history is partial and limited to recent days by API pagination.
- **Direction data gap.** All tested alternative free sources show no economic lift for direction, suggesting the problem is data, not model complexity.
- **Generalization gap.** The signal has not been validated on an untouched later time period beyond the last walk-forward test window.

## Unanswered questions

- Can the move score be monetized through volatility products, execution timing, or risk scaling without relying on direction?
- Will full historical open interest or liquidation history produce a viable directional edge?
- Can options implied volatility/skew or futures basis deliver direction when combined with the move score?
- Is the strongest directional edge now in microstructure data rather than in OHLCV or derivatives features?
- What is the best form of the trading architecture if only the move signal is deployable?

## Recommended next priorities

1. Treat the move detector as the validated research product and build around it.
2. Acquire and integrate the missing high-value datasets before pursuing new live directional models.
3. Avoid further OHLCV-only model optimization or TSFresh sweeps until new directional data arrives.
4. Perform a true untouched tail holdout and multi-symbol generalization test for the frozen move model.
