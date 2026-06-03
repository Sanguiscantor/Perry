# Data Acquisition Report

## Acquired datasets

### BTCUSDT OHLCV master dataset
- **Path:** `Data/master_raw_dataset.csv`
- **Coverage:** 84,740 BTCUSDT 15m candles used by the main move detection research.
- **Source:** Existing repository dataset; likely Binance USDT-M or reconstructed from exchange klines.
- **Limitations:** Single-symbol 15m OHLCV only; no microstructure, funding, or alternative flow data inside the file.
- **Update process:** Not automated in current repo; refresh by re-running source collection if available.

### Futures klines with taker flow
- **Path:** `Data/futures_klines_15m.csv`
- **Coverage:** 2024-01-01 → 2026-06-02 BTCUSDT futures 15m klines.
- **Source:** `Data/download_extended_klines.py` from Binance USDT-M REST.
 **Fields included:** `Open`, `High`, `Low`, `Close`, `Volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`, plus computed taker ratios and imbalance.
- **Limitations:** Taker flow is a coarse proxy for aggressive trade direction and is available only at 15m frequency.
- **Update process:** `python Data/download_extended_klines.py`

### Derivatives funding rates
- **Path:** `Data/derivatives/funding_rates.csv`
- **Coverage:** 2024-01-01 → 2026-06-02 for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT.
- **Source:** `Data/download_derivatives.py` from Binance USDT-M funding history.
- **Limitations:** Native 8h frequency; merged to 15m with forward fill, which introduces event-time staleness.
- **Update process:** `python Data/download_derivatives.py`

### Derivatives open interest
- **Path:** `Data/derivatives/open_interest_15m.csv`
- **Coverage:** 2026-05-28 → 2026-06-02 for major symbols; short history due to API pagination.
- **Source:** `Data/download_derivatives.py` from Binance USDT-M open interest endpoint.
- **Limitations:** Partial coverage only; likely 30-day window limitations in free endpoint.
- **Update process:** `python Data/download_derivatives.py`

### Long/short and sentiment proxies
- **Paths:**
  - `Data/derivatives/global_long_short_15m.csv`
  - `Data/derivatives/top_trader_long_short_15m.csv`
  - `Data/derivatives/taker_buy_sell_15m.csv`
- **Coverage:** ~2026-05-02 → 2026-06-02 (approximate one-month span).
- **Source:** Binance derivatives positioning endpoints via `Data/download_binance_sentiment.py`.
- **Limitations:** Short time window; not enough history for robust directional testing.
- **Update process:** `python Data/download_binance_sentiment.py`

### Cross-asset spot dataset
- **Path:** `Data/multi_asset_dataset.csv`
- **Coverage:** Multi-symbol aligned spot or futures OHLCV data for BTCUSDT and major crypto assets.
- **Source:** `Data/download_multi_asset.py`.
- **Limitations:** Not explicitly documented; likely aligned 15m candles only. Useful for cross-asset feature engineering.
- **Update process:** `python Data/download_multi_asset.py`

## Pipeline status

- The repository already contains ingestion code for the acquired datasets.
- Feature builders are implemented in `Model/enhanced_features.py` and `Model/alternative_features.py`.
- The directional edge study is implemented in `Model/directional_edge_program.py` and uses the acquired data.

## Not acquired / blocked datasets

- **Liquidations:** No free full-history REST or archive was acquired. This is a major remaining gap.
- **Order book imbalance / L2 depth:** No historical level 2 archive available in the current repository.
- **Options IV / skew / term structure:** Not acquired or integrated; Deribit or similar history remains untested.
- **ETF flows / macro flows:** Not acquired; these are outside the current crypto-native data scope.
- **Stablecoin flows:** Not acquired as a discrete dataset, though implied by exchange inflow/outflow possibility.
- **Exchange flows:** Not acquired in historical form; only partial derivatives flow proxies are present.

## Practical update commands

```bash
python Data/download_extended_klines.py
python Data/download_derivatives.py
python Data/download_binance_sentiment.py
python Data/download_multi_asset.py
```

## Recommendations

- Preserve the current data acquisition artifacts; do not overwrite them without versioning.
- Target the missing high-value datasets in this order: liquidations, order-book archive, options IV/skew, full-history open interest.
- Treat the existing derivatives and sentiment files as experimental context that can be extended once longer histories are available.
