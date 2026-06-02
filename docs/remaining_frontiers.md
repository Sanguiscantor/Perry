# Remaining Frontiers — Prioritized Data Sources (directional information)

Generated: 2026-06-03 — concise prioritized inventory and cheapest test paths. Focus: highest expected directional value with minimal cost to smoke-test.

Rank key candidates below (score = directional value / acquisition difficulty). Short notes include acquisition difficulty, rough cost estimate, historical availability, expected directional value, and cheapest path to test.

1) Liquidation history (exchange-level + aggregated)
 - Acquisition difficulty: Low–Medium (CSV downloads from aggregators; some scraping)
 - Cost: Free → $0–$500 (free sources: Coinglass, CCXT scraping; paid feeds $500+/month)
 - Historical availability: Good (aggregators keep weeks→years), but reliability varies
 - Expected directional value: High (liquidations cluster with volatility spikes and may signal directional moves)
 - Cheapest path to test: Merge Coinglass/Coinalyze CSV exports or scrape public liquidation pages for 6 months; run simple 15m alignment vs move labels.

2) Per-exchange L2 order-book archives (snapshots / depth)
 - Acquisition difficulty: High (requires vendor; heavy storage & compute)
 - Cost: High ($1k–$10k+ depending on vendor & history length)
 - Historical availability: Good from vendors (Kaiko, Amberdata, LedgerPrime); limited via exchanges
 - Expected directional value: Very High (imbalance, hidden liquidity, execution pressure)
 - Cheapest path to test: Purchase 1–2 weeks of L2 snapshots from a vendor for BTC; compute simple 15m imbalance and test uplift.

3) Options surfaces & skew (historical IV per expiry/strike)
 - Acquisition difficulty: Medium (Deribit live snapshots easy; historical full panel requires vendor)
 - Cost: Medium–High (vendor $500–3k+/month for historical surfaces)
 - Historical availability: Limited on public API; vendors have multi-month archives
 - Expected directional value: High (skew/term-structure often precedes directional moves)
 - Cheapest path to test: Use existing live snapshots to compute short-term skew features; if promising, buy 1–3 months of historical surfaces or reconstruct IV from historical option mids (requires trade/book history).

4) Basis / futures spreads (perp–spot and calendar spreads)
 - Acquisition difficulty: Low (futures/perp klines available from exchanges; compute spreads)
 - Cost: Low (free via exchange APIs or existing klines)
 - Historical availability: Excellent (exchanges provide full history)
 - Expected directional value: Medium–High (funding and basis shifts correlate with flows)
 - Cheapest path to test: Compute perp–spot basis at 15m from existing futures & spot klines (already implemented basis_15m.csv); run small ablation vs baseline.

5) Cross-exchange flow (top-of-book and trade imbalances across exchanges)
 - Acquisition difficulty: Medium (need multi-exchange klines/trades + alignment)
 - Cost: Low–Medium (free API data; engineering cost for alignment)
 - Historical availability: Good for trades/klines; L2 more limited
 - Expected directional value: Medium (cross-exchange arbitrage and flows can forecast short moves)
 - Cheapest path to test: Align trade-volume imbalances across 2–3 major exchanges (Binance, Bybit, Deribit futures) at 15m and test lift.

6) Market-maker positioning proxies (OI, concentrated OI by expiry, skewed open interest)
 - Acquisition difficulty: Low–Medium (OI per instrument on Deribit; requires aggregation)
 - Cost: Low (free API for current; historical OI may be partial)
 - Historical availability: Partial (exchanges publish OI, but full historical snapshots may be limited)
 - Expected directional value: Medium (shifts in OI near expiries signal positioning)
 - Cheapest path to test: Aggregate per-instrument OI from existing instrument list and compute expiry-weighted OI skew.

7) Market-maker / top-trader public metrics (Binance top traders, global L/S)
 - Acquisition difficulty: Low (public endpoints and scraped CSVs exist)
 - Cost: Low (free or small fees for CSV access)
 - Historical availability: Moderate (some sites keep rolling history)
 - Expected directional value: Medium (crowding signals sometimes predictive)
 - Cheapest path to test: Merge Binance L/S ratios and run small walk-forward ablation.

8) On-chain flows (exchange inflows/outflows, large transfers)
 - Acquisition difficulty: Medium (on-chain tooling; data APIs exist)
 - Cost: Low–Medium (free APIs vs paid enrichment)
 - Historical availability: Excellent
 - Expected directional value: Medium (exchange inflows often precede selling pressure)
 - Cheapest path to test: Use free on-chain aggregated exchange flow CSVs for BTC at daily→hourly cadence; align to 15m where possible.

9) Funding/funding slope and OI by exchange (high-frequency funding shifts)
 - Acquisition difficulty: Low (public APIs provide funding history)
 - Cost: Low
 - Historical availability: Good
 - Expected directional value: Low–Medium (useful as context, rarely standalone)
 - Cheapest path to test: Merge funding and OI timeseries into existing base features (fast test).

10) Additional datasets to consider (lower priority)
 - Block/OTC trade feeds (low availability; high cost)
 - Sentiment at high granularity (Twitter/GitHub streams) — noisy, moderate cost
 - Derivatives liquidation waterfall (exchange-provided detailed margin calls) — rare, high value if obtainable

Summary recommendations (cheapest-first to rule in/out):
 - Immediate cheap tests (0–$500): basis/futures spreads (already implemented), liquidation CSV merges (Coinglass), funding/OI merges, cross-exchange klines.
 - Medium effort ( $500–3k ): options live-skew analysis (use existing snapshots), market-maker positioning proxies (aggregate OI by expiry), short historical surfaces from vendors (1–3 months).
 - High effort / high value ($1k–10k+): L2 order-book archives (1–4 weeks) and vendor liquidation feeds; purchase narrow samples to validate signal before committing to long windows.

If you want, I can next produce exact vendor contacts/pricing options (Kaiko/Amberdata/Kaiko/CCXT vendors) and a one-week purchase plan to cheaply test L2 and options surfaces.
