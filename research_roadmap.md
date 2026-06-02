# Research Roadmap — Perry

**Last updated:** 2026-06-02T18:38:50+00:00

## Current frontier

Acquire liquidation/L2 history or deploy move-score vol product.

## Highest-value unknowns

- Paid liquidation feed directional lift
- L2 imbalance at 15m aggregation
- Perp-spot basis intraday

## Recommended next experiments

- Coinglass liquidation CSV merge
- Deribit options skew panel
- Move-score vol backtest (non-directional monetization)
 - Compute and test futures-spot basis feature (done: basis_15m.csv)
 - Retry Deribit options acquisition and IV/skew panel (fix parameter and sample expiries)
 - Trial paid L2/sample snapshot from vendor (Kaiko/CryptoCompare) for 1-week replay
 - Compute orderbook imbalance proxies from tick/trade histories and compare to taker-flow
