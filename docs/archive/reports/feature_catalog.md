# Feature Catalog

This document records the causal feature families built from newly acquired datasets and the existing OHLCV baseline. All features are computed at candle close with no lookahead.

## 1. Baseline causal features (existing OHLCV)

The current validated move model uses engineered candle-close features including:

- Range / ATR / volatility: `range_mean`, `atr_14`, `return_std`, `return_var`, `return_skew`, `return_kurt`.
- Volume and flow: `volume_ratio`, `volume_z`, `quote_volume` measures.
- Persistence and memory: rolling autocorrelation, Hurst exponent, entropy, fractal metrics.
- Short- and long-window volatility ratios: 4, 8, 12, 24, 48, 96, 192 bar windows.
- Price momentum and dispersion: lagged returns, percentiles, and normalized range measures.

These features form the validated causal basis for the move detection signal.

## 2. Cross-asset features

Implemented in `Model/enhanced_features.py` via `build_cross_asset_features`.

### Feature groups

- **Lagged BTC returns:** `btc_ret_lag_{1,2,4,8,12}` and absolute lagged returns `btc_abs_ret_lag_{...}`.
- **Alt instrument lead/lag:** `eth_ret_lag_1`, `sol_ret_lag_1`, `bnb_ret_lag_1`, `xrp_ret_lag_1`.
- **Relative strength:** `{alt}_rel_strength_24` computed as a short-term deviation of alt vs BTC price level.
- **Volatility ratio:** `{alt}_vol_ratio_48` as alt return std / BTC return std over 48 bars.
- **Correlation:** `{alt}_corr_btc_48` as the 48-bar rolling correlation to BTC returns.
- **Alt index measures:** `alt_index_ret_1`, `alt_dispersion_48`, `alt_breadth_up_12`.
- **BTC range history:** `btc_range_pct_lag_1`, `btc_range_pct_lag_4`.

### Purpose

These features are designed to assess whether cross-asset price and volatility dynamics add incremental information to BTC move detection or directional inference.

## 3. Derivatives / funding / open interest features

Implemented in `Model/enhanced_features.py` via `build_derivatives_features` and exposed through `build_funding_oi_features`.

### Funding features

- `funding_rate` resampled to 15m by forward-fill.
- `funding_rate_z_96`: z-score relative to a 96-bar rolling mean/std.
- `funding_rate_change_8h`: funding rate change over 32 bars (8h).

### Open interest features

- `oi_pct_change_12`: percent change in open interest over 12 bars.
- `oi_pct_change_48`: percent change in open interest over 48 bars.
- `oi_z_96`: z-score of open interest relative to the prior 96 bars.

### Purpose

These features quantify derivatives positioning and leverage flow around BTC and are intended to capture directional pressure from futures market activity.

## 4. Taker flow features

Implemented in `Model/alternative_features.py` via `build_taker_flow_features`.

### Feature groups

- `taker_buy_ratio_mean_{4,12,24,48,96}`: rolling mean of aggressive buyer participation.
- `taker_imbalance_mean_{4,12,24,48,96}`: rolling mean of signed taker imbalance.
- `trades_z_{4,12,24,48,96}`: z-score of trade count over the same windows.
- `quote_vol_z_{4,12,24,48,96}`: z-score of quote volume.
- Lagged directional signals: `taker_buy_ratio_lag_1`, `taker_imbalance_lag_1`, `taker_imbalance_lag_4`.

### Purpose

These features encode aggressive futures trading flow, which is a proxy for directional pressure and liquidity exhaustion.

## 5. Sentiment and positioning features

Implemented in `Model/alternative_features.py` via `build_sentiment_features`.

### Data sources

- `global_long_short_15m.csv`
- `top_trader_long_short_15m.csv`
- `taker_buy_sell_15m.csv`

### Feature groups

- Global and top-trader long/short ratios: `gls_long_short_ratio`, `top_long_short_ratio`.
- Long-account percentage: `gls_long_account_pct`, `top_long_account_pct`.
- Aggregated taker ratios and volumes: `agg_taker_buy_sell_ratio`, `agg_taker_buy_vol`, `agg_taker_sell_vol`.
- Normalized derivatives sentiment: `{feature}_z_{12,48,96}`.
- Change rates: `{feature}_chg_{12,48,96}`.

### Purpose

These features capture crowd positioning and sentiment biases from derivatives market participants.

## 6. Combined alternative feature set

Implemented via `build_all_alternative` in `Model/alternative_features.py`.

- Merges taker flow, sentiment, funding, and open interest into a single aligned dataset.
- Has `Datetime` alignment and NaN-safe merging.
- Designed for experiments that test all non-OHLCV alternate signals together.

## 7. Causality and anti-leakage

- All engineered features are shifted or rolling-backed to ensure they are available at candle close.
- Funding is forward-filled from its native 8h event time, which is conservative but may introduce staleness.
- Cross-asset and derivatives features merge on `Datetime` only; no future returns or labels are used.

## 8. Feature deployment notes

- The strongest validated move model still relies on the OHLCV causal feature set.
- Alternative feature groups are useful for directional research and diagnostics but have not produced a deployable edge yet.
- Future feature engineering should focus on higher-fidelity directional sources (liquidations, L2, options, complete OI) rather than additional OHLCV transforms.
