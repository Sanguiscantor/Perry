# Data Expansion Report (Phase 6)

| name | predictive | cost | availability | integration | composite | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Cross-asset returns (ETH, SOL leads) | 7 | 1 | 10 | 9 | 8.3 | Already have OHLCV multi-asset; needs feature engineering |
| Funding rates | 7 | 2 | 9 | 9 | 7.949999999999999 | Free from Binance futures API; regime and squeeze signal |
| Open interest | 8 | 3 | 8 | 8 | 7.85 | Confirms positioning; complements move detection |
| Liquidations | 8 | 4 | 7 | 7 | 7.300000000000001 | Event-driven vol expansion; aligns with Perry move thesis |
| Market breadth (alt index) | 6 | 2 | 9 | 8 | 7.300000000000001 | Risk-on/off filter for move regimes |
| Long/short ratios | 6 | 2 | 8 | 9 | 7.3 | Crowding proxy; weaker than OI for direction |
| Order book imbalance / depth | 9 | 7 | 6 | 5 | 6.7 | Strong microstructure alpha; needs co-located feed or exchange WS |
| Options implied vol / skew | 9 | 8 | 5 | 4 | 6.1499999999999995 | Best forward vol ground truth; expensive data |
| NSE market internals | 3 | 6 | 4 | 3 | 3.35 | Low relevance for crypto-native Perry unless macro overlay desired |

