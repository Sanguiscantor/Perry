# Microstructure Hypotheses & Test Designs

**Generated:** 2026-06-03

This document lists hypotheses derived from acquired microstructure artifacts and outlines leakage-safe causal tests for each.

A. Basis (futures - spot, 15m)

Hypotheses
- H1: Large positive basis (future > spot) predicts short-term downward pressure on spot returns over next 1–4 candles due to mean reversion of basis.
- H2: Rapid expansions in basis magnitude predict elevated volatility and higher move probability (move target) in the next 12 candles.

Test design
- Features: `basis`, `basis_delta_1`, `basis_zscore_rolling(24)`, `basis_volatility`.
- Evaluation: Leakage-safe purged chronological walk-forward using existing move/direction labels and identical training/evaluation folds as baseline.
- Metrics: balanced accuracy, macro F1, calibration (Brier), profitability (net bps), CI via block bootstrap.
- Controls: include baseline OHLCV features and taker-flow; enforce same sample selection and purging as main research program.

B. Liquidation imbalance (if archive acquired)

Hypotheses
- H3: Net liquidation volume imbalance (long vs short) aggregated to 1m and rolled to 15m predicts direction within short horizons (1–6 candles).
- H4: Clusters of large liquidations concentrated within a short window increase move probability and magnitude.

Test design
- Features: per-symbol `liq_long_vol`, `liq_short_vol`, `liq_net`, `liq_count`, `liq_max_single` aggregated to 15m.
- Evaluation: Purged walk-forward with event-time exclusion windows to avoid label leakage around large liquidation events.
- Metrics: same as A.

C. Options IV / Skew / Term-structure (Deribit)

Hypotheses
- H5: Steepening put skew (IV_put - IV_call) predicts downside directional risk over 1–12 candles.
- H6: Short-term ATM IV spikes predict increased move probability but not necessarily direction.
- H7: Cross-sectional skew term-structure slope (short-dated vs long-dated) predicts directional bias when short IV > long IV abruptly.

Test design
- Features: ATM IV, put-call skew at multiple deltas (10d, 25d, ATM), slope between tenors (7d vs 30d), delta-weighted skew.
- Evaluation: Align options snapshots to 15m klines (use last-known IV before interval end), purged walk-forward.
- Metrics: same as A.

D. Orderbook Imbalance (L2 snapshots or trade-derived proxies)

Hypotheses
- H8: L2 book imbalance (weighted depth on bid vs ask) aggregated to 15m predicts immediate short-term direction; stronger signal at 1–3m.
- H9: Trade aggression imbalance (taker_buy_vol - taker_sell_vol) at 1m aggregated to 15m predicts direction similar to L2 but more robust when L2 missing.

Test design
- Features: L2 imbalance metrics (top N levels weight), cumulative signed volume from trades, normalized imbalance z-scores.
- Evaluation: Run parallel tests: (1) with vendor L2 snapshots, (2) with trade-derived proxies; compare lifts and cost of acquisition.
- Metrics: same as A.

General validation protocol
- Use the identical purged chronological folds, label definitions, and performance metrics used by the main research program to ensure comparability.
- Do not tune model hyperparameters beyond the established baseline training regime; report results with identical model architecture and training settings.
- Report both statistical significance and economics (after fees/slippage) and bootstrap confidence intervals.

Data & artifact linkage
- Basis: `artifacts/data/microstructure/basis_15m.csv`
- Liquidations: intended `artifacts/data/microstructure/binance_liquidations.csv` (if obtained)
- Options: intended `artifacts/data/microstructure/deribit_instruments_*.csv` and `deribit_trades_*.csv`

