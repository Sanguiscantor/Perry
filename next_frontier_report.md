# Next Frontier Report — Microstructure Acquisition & Findings

**Generated:** 2026-06-03

Summary: executed an autonomous acquisition sweep of free microstructure datasets, computed a futures-spot basis from local klines, attempted Deribit options and Binance liquidation retrievals, and produced a prioritized plan for next experiments.

1) What information sources remain unexplored?
- Historical L2 order book snapshot archives (long-duration, multi-symbol). Vendor datasets (Kaiko, CryptoCompare, Amberdata, CoinAPI) remain unexplored and are likely paid.
- Full historical liquidation archives beyond what exchange public endpoints provide. Coinglass and other aggregator CSVs may be available.
- Comprehensive Deribit historical option trades and IV surface for instrument panels (attempted but needs corrected params).
- Exchange-provided market-maker inventory datasets (rare / vendor-specific).
- Aggregated imbalance indices from third-party providers.

2) Which can actually be obtained?
- Deribit options chain and trade history: public API (requires correct parameters); feasible to obtain programmatically.
- Trade ticks and klines from exchanges (Binance) via REST websockets; bulk historical retrieval feasible by paginated requests for limited windows.
- Basis/futures spreads: already computable from existing klines (done).
- Exchange-derived metrics (funding, OI, long/short ratios): already available and integrated.
- Liquidation records: partial via Binance FAPI (subject to endpoint availability); third-party aggregators provide CSV exports (may be free or paid).
- Paid L2 archives: obtainable from vendors for a fee.

3) Which were successfully acquired?
- Futures-spot basis (15m): `artifacts/data/microstructure/basis_15m.csv` (424,215 rows) — computed from `Data/futures_klines_15m.csv` and `Data/master_raw_dataset.csv`.
- Existing derivatives metrics and taker flow were already present in the repo and reaffirmed as available.
- Attempts to fetch Deribit instruments and Binance liquidations were executed but did not return usable archives in this run (Deribit returned 400, Binance liquidation endpoint reported out-of-maintenance).

4) Which appear most promising?
- Deribit options IV / skew / term-structure: strong candidate for medium-horizon directional signals via changes in skew and short-term IV moves.
- Historical L2 order book imbalance: highest predictive potential for ultra-short horizons and aggregated 15m imbalance may help direction.
- Liquidation imbalances: high signal potential around stressed leverage events.
- Trade-tick aggression / taker imbalance (already present): pragmatic and already integrated; further refinement may yield incremental value.

5) What is Perry's highest-value next experiment?
- Acquire a Deribit options panel (BTC & ETH) covering 6–12 months of expiries, compute IV surface and skew features at 15m cadence, and run a leakage-safe walk-forward test comparing baseline move/direction models vs baseline+options features.
- Run a paid 1-week L2 orderbook snapshot trial (vendor) to test whether aggregated book imbalance features at 15m improve direction beyond taker-flow.

6) What is currently preventing Perry from becoming a deployable trading system?
- Lack of robust, replicable directional improvements after accounting for fees and slippage (current best nets are negative).
- Missing high-resolution microstructure archives (historical L2 snapshots, reliable liquidation histories) that are most likely to materially shift short-horizon direction performance.
- Execution & transaction costs, market impact models, and realistic execution simulation are not yet fully integrated.
- Risk of data leakage or lookahead when integrating raw L2/option features without careful synchronization and granularity control.

Actions taken autonomously
- Wrote `docs/microstructure_data_report.md` summarizing feasibility.
- Implemented `Data/acquire_microstructure.py` which:
  - attempted Deribit instruments & trades
  - attempted Binance liquidation pagination
  - computed futures-spot basis and wrote `artifacts/data/microstructure/basis_15m.csv`
  - wrote `artifacts/data/microstructure/manifest.json`
- Updated canonical memory files: `current_truth.md`, `research_journal.md`, `research_roadmap.md` with acquisition outcomes and next steps.

Recommended next steps (autonomous)
1. Retry Deribit acquisition with corrected query parameters; fetch instrument lists and a 6-month trade history for selected strikes; persist to artifacts and validate.
2. If Deribit public retrieval insufficient, purchase or request a sample from Deribit or third-party dumps.
3. Trial a paid L2 snapshot provider for a 1-week sample and compute book imbalance features.
4. Implement leakage-safe walk-forward tests for options IV, basis, and liquidation features (no model tuning; compare baseline architecture).
5. Integrate execution cost model and slippage simulation in validation.


