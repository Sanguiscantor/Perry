# Research Roadmap — Perry

**Last updated:** 2026-06-02T18:38:50+00:00

## Current frontier

Acquire liquidation/L2 history or deploy move-score vol product.

**Note (2026-06-03):** Historical IV acquisition via public Deribit APIs was attempted and is currently blocked — Deribit does not expose a 6-month IV history via public endpoints. Live IV snapshots are being collected, but a full backfill requires vendor data or reconstruction from archived option prices/order-books.

## Highest-value unknowns

- Paid liquidation feed directional lift
- L2 imbalance at 15m aggregation
- Perp-spot basis intraday

## Recommended next experiments

- Coinglass liquidation CSV merge
- Deribit options skew panel
- Move-score vol backtest (non-directional monetization)
 - Deribit options skew panel (live snapshots available; historical backfill blocked by public API)
 - Vendor L2 / liquidation acquisition (recommended immediate next step)
