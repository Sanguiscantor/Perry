# Directional Edge Report

**Generated:** 2026-06-02T18:38:50+00:00

## 1. Which information sources helped?

_None improved economics materially._

## 2. Which failed?

- **baseline_core:** net **-4.31** bps, balanced **0.5250**
- **ohlcv_all:** net **-5.26** bps, balanced **0.5104**
- **taker_flow:** net **-5.03** bps, balanced **0.5082**
- **sentiment_only:** net **-5.26** bps, balanced **0.5104**
- **funding_oi:** net **-5.09** bps, balanced **0.5108**
- **cross_asset:** net **-4.19** bps, balanced **0.5247**
- **all_alternative:** net **-6.93** bps, balanced **0.5035**
- **move_gated_all_features:** net **-7.60** bps, balanced **0.4951**

## Data quality

| File | BTC rows | Start | End |
| --- | ---: | --- | --- |
| funding_rates.csv | 2652 | 2024-01-01 00:00:00 | 2026-06-02 16:00:00 |
| open_interest_15m.csv | 2976 | 2026-05-02 18:45:00 | 2026-06-02 18:30:00 |
| global_long_short_15m.csv | 2976 | 2026-05-02 18:45:00 | 2026-06-02 18:30:00 |
| top_trader_long_short_15m.csv | 2976 | 2026-05-02 18:45:00 | 2026-06-02 18:30:00 |
| taker_buy_sell_15m.csv | 2971 | 2026-05-02 18:30:00 | 2026-06-02 18:15:00 |
| futures_klines_15m.csv | 84843 | 2024-01-01 00:00:00 | 2026-06-02 18:30:00 |

## 3. Strongest directional signal?

**cross_asset** — balanced acc **0.5247**, non-overlap net **-4.19** bps, CI **[-7.24, -1.12]**

## 4. Best trading architecture?

**Move-gated direction** (CatBoost move → logistic direction on move-conditioned train) or **taker-flow-augmented logistic** if lift positive; otherwise **no directional deployment** — use move score as vol filter only.

## 5. What to research next?

1. Paid liquidation + L2 archives. 2. Deribit IV skew. 3. Vol-product backtest on move score.

## Source ranking (pre-acquisition)

| Rank | Source | Composite | Direction prob |
| ---: | --- | ---: | ---: |
| 1 | Taker buy/sell flow (futures klines) | 8.95 | 8 |
| 2 | Aggregated taker buy/sell ratio (Binance) | 7.3 | 7 |
| 3 | Top trader position L/S ratio | 7.3 | 7 |
| 4 | Funding rate / carry | 7.3 | 5 |
| 5 | Cross-asset lead-lag | 7.0 | 4 |
| 6 | Open interest changes | 6.85 | 6 |
| 7 | Global long/short account ratio | 6.7 | 6 |
| 8 | Options IV / skew (Deribit) | 6.4 | 7 |
| 9 | Crypto Fear & Greed Index | 6.25 | 4 |
| 10 | L2 order book imbalance | 6.2 | 9 |
