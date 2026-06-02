# Microstructure Data Report — Acquisition Feasibility

**Generated:** 2026-06-03

Scope: evaluate microstructure and options-derived datasets for directional information potential. This report summarizes availability, cost, historical depth, API access, legal accessibility, integration difficulty, and expected directional value.

Sources evaluated

1) Historical Liquidation Data (Binance futures)
- Availability: Partial — Binance exposes force/liquidation orders via public FAPI endpoints for recent history and limited ranges. Full historical archives are not provided for long time ranges.
- Cost: Free for public endpoints; full historical vendor archives (Kaiko/CryptoCompare) are paid.
- Historical depth: Moderate (depends on retention and pagination); retrieving multi-year data requires paginated requests and time-window stitching; may be incomplete for early dates.
- API availability: REST endpoints accessible via `https://fapi.binance.com` (paginated). WebSocket provides live notifications.
- Legal accessibility: Public API — permissible for research. Respect Binance TOS and rate limits.
- Integration difficulty: Low–moderate (pagination, rate-limits, occasional chunk failures).
- Expected directional value: Medium — liquidation cascades can produce short-term directional moves; highest value around low-liquidity events and leverage extremes.

2) Historical L2 Order Book Snapshots
- Availability: Low (exchange APIs provide current snapshots and websockets, but historical L2 archives are typically not free). Some community projects share limited snapshots; major vendors (Kaiko, CryptoCompare, Amberdata, CoinAPI) provide paid archives.
- Cost: Usually paid for multi-symbol, long-duration archives. Small samples may be free.
- Historical depth: Vendor-dependent; exchanges rarely provide deep historical L2 via public APIs.
- API availability: Not available for bulk historical retrieval via Binance public API; live snapshots via REST (`/depth`) and websockets for streaming updates only.
- Legal accessibility: Vendor license terms apply for paid datasets.
- Integration difficulty: High (large storage, replay mechanics, sequence reconstruction, gap filling).
- Expected directional value: High for ultra-short horizons; moderate for 15m horizons when aggregated.

3) Order Book Imbalance Archives (precomputed)
- Availability: Mixed — some vendors publish imbalance aggregates (e.g., 1s/1m imbalance) or indices. Exchanges do not provide precomputed imbalance historically in bulk for free.
- Cost: Often part of paid data packages; custom compute feasible from L2 or trade streams.
- Historical depth: Depends on whether snapshots/trades were preserved.
- API availability: Vendor-dependent.
- Legal accessibility: Vendor license terms apply.
- Integration difficulty: Moderate–high (compute from raw L2 or trades if L2 unavailable).
- Expected directional value: Medium–high at short horizons.

4) Options Implied Volatility, Skew, Term Structure (Deribit, Binance Options)
- Availability: Good — Deribit exposes full options chain via public API; Binance Options also exposes data for listed instruments. Historical trades and order book snapshots are accessible via public APIs (Deribit provides trade history and instrument lists; Deribit also publishes implied volatility in instrument metadata).
- Cost: Free for Deribit public API; some derived historical archives may be paid.
- Historical depth: Deribit has extensive historical records; depth varies by instrument.
- API availability: REST and websocket endpoints (Deribit: `https://www.deribit.com/api/v2`), instrument metadata returns IV and greeks for instruments.
- Legal accessibility: Public API — permissible; observe terms.
- Integration difficulty: Moderate (option instrument naming and expiries handling, strike grids).
- Expected directional value: Medium — skew and short-term IV surface changes can indicate directional bias (e.g., put-heavy skew before downside moves).

5) Basis / Futures Spreads
- Availability: High — futures and spot prices are public via exchanges (Binance). Basis is computable from futures vs spot price or futures vs perpetual basis (funding).
- Cost: Free via exchange APIs.
- Historical depth: Good if historical futures/spot klines are available (we already have futures_klines_15m.csv and master_raw_dataset.csv).
- API availability: REST endpoints and historical klines.
- Legal accessibility: Public API.
- Integration difficulty: Low (merge kline datasets and compute spreads).
- Expected directional value: Low–medium (basis signals funding and carry; extreme basis can precede mean reversion or risk-on/off moves).

6) Market-Maker Positioning Proxies (top trader ratios, open interest, funding)
- Availability: High — Binance provides globalLongShortAccountRatio, topLongShortPositionRatio, open interest, funding rates.
- Cost: Free for public API access.
- Historical depth: Moderate (depends on retention), but our repo already downloads these metrics.
- API availability: REST endpoints; available in `Data/derivatives` via existing scripts.
- Legal accessibility: Public.
- Integration difficulty: Low (parsing CSV outputs already generated).
- Expected directional value: Medium — changes in top trader positions and OI often precede directional move or liquidity shifts.

7) Exchange Trade Timestamps / Tick-by-tick Trades
- Availability: High (trade history via REST and websockets). Historical trade ticks are accessible but may be rate-limited for bulk download.
- Cost: Free via public API for limited windows; bulk archives are paid.
- Historical depth: Depends on API limits; reconstructing long history requires pagination and time windows.
- API availability: REST and websocket.
- Legal accessibility: Public API.
- Integration difficulty: Moderate (volume of data, deduping, aligning to intervals).
- Expected directional value: Medium — tick imbalances, aggression, and trade footprints can be predictive at short horizons.

8) Exchange Funding and Index Data
- Availability: High — funding rates and index prices are public.
- Cost: Free.
- Historical depth: Good to moderate.
- API availability: REST.
- Integration difficulty: Low.
- Expected directional value: Low–medium (funding spikes can indicate directional pressure).

Summary recommendations

- Immediate free acquisitions to implement automatically:
  1. Deribit options chain and trade snapshots (public API) — high utility for IV/skew term-structure experiments.
  2. Binance liquidation orders (public FAPI endpoint) — moderate utility; implement paginated retrieval.
  3. Compute basis/futures spreads from existing `Data/futures_klines_15m.csv` and `Data/master_raw_dataset.csv` — immediate low-difficulty feature set.
  4. Pull trade tick history for targeted assets via Binance trade endpoints (limited windows) for imbalance proxies.

- Near-term paid vendors to consider (if needed for L2 archives): Kaiko, CryptoCompare, Amberdata, CoinAPI — they provide historical L2 order book snapshots and imbalance indices.

Next actions performed automatically

- Implemented acquisition script `Data/acquire_microstructure.py` to fetch Deribit options metadata/trades, attempt Binance liquidation pagination, compute basis from local klines, and write artifacts to `artifacts/data/microstructure/` with a manifest and simple validation checks.

Run results and validation

- 2026-06-03: Computed and saved `artifacts/data/microstructure/basis_15m.csv` (424,215 rows).
- 2026-06-03: Retried Deribit instruments/trades and saved `artifacts/data/microstructure/deribit_instruments_BTC.csv` and sampled trade CSVs.
- 2026-06-03: Binance liquidation endpoint returned maintenance error during best-effort fetch; no reliable full-history liquidations were obtained via FAPI in this run.

Validation summary

- Performed a leakage-safe walk-forward comparison: baseline move model vs baseline+`basis` features. Outcome: `basis` did not improve performance (baseline balanced acc 0.6066 vs basis-augmented 0.6053). Bootstrap CI overlaps. See `artifacts/validation/basis_vs_baseline.json` for details.

Artifacts

- `artifacts/data/microstructure/basis_15m.csv`
- `artifacts/data/microstructure/deribit_instruments_BTC.csv`
- `artifacts/data/microstructure/deribit_trades_*.csv`
- `artifacts/data/microstructure/manifest.json`
- `artifacts/validation/basis_vs_baseline.json`

Next recommended experiments

- Ingest Deribit IV and compute skew/term-structure at 15m cadence for a 6-month panel and run the identical walk-forward evaluation (no tuning).
- Acquire a paid L2 snapshot sample (1 week) to compute book imbalance features; compare to taker-flow proxies before broader purchase.


