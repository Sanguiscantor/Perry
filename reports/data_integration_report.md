# Data Integration Report

**Generated:** 2026-06-02T16:35:50+00:00

## Binance USDT-M funding rate

- **file:** C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\derivatives\funding_rates.csv
- **rows:** 13260
- **symbols:** ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
- **start:** 2024-01-01 00:00:00
- **end:** 2026-06-02 16:00:00
- **native_frequency:** 8h
- **merged_frequency:** 15m forward-fill
- **limitations:** Event-time 8h; ffilled to candles introduces staleness
- **quality:** Official exchange; complete for majors since 2024

## Binance open interest history

- **file:** C:\Users\Admin\Downloads\Experiment\Perry The Platypus\Data\derivatives\open_interest_15m.csv
- **rows:** 2500
- **symbols:** ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
- **start:** 2026-05-28 11:45:00
- **end:** 2026-06-02 16:30:00
- **native_frequency:** 15m
- **limitations:** Endpoint max ~500 rows per call; paginated
- **quality:** Good alignment with Perry bars

## Not acquired

- **Liquidations:** No free long-history REST; requires paid/third-party
- **Order book imbalance:** No historical L2 archive on free tier
- **Options IV/skew:** Deribit history not integrated this run
- **NSE internals:** Out of scope for crypto-native Perry