# Directional Information Catalog

This catalog ranks candidate directional information sources for Perry, with mechanism, predicted value, availability, difficulty, complexity, and directional edge probability.

| Rank | Candidate | Mechanism | Expected predictive value | Historical availability | Acquisition difficulty | Implementation complexity | Probability of directional edge | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Liquidations | Large forced exits can push price directionally and signal crowding unwinds. | High | Low (free long history rare) | High | Medium | High | Aligns with move/vol-expansion thesis and may add direction beyond OHLCV. |
| 2 | Order book imbalance / depth | Asymmetric liquidity or iceberg orders can reveal directional conviction and impending price pressure. | High | Low | High | High | High | Requires L2 or aggregated depth archive; best microstructure signal candidate. |
| 3 | Options implied volatility / skew | Changes in implied vol and skew reflect directional bias and demand for protection. | High | Medium | Medium–High | Medium | High | Very promising for crypto if Deribit or similar history can be acquired. |
| 4 | Options term structure | Shape of the volatility curve and roll yield reveal directional sentiment and hedging pressure. | Medium–High | Medium | Medium–High | Medium | Medium-High | Useful for bridging move signal to directional convexity and regime awareness. |
| 5 | Basis and futures spreads | Perpetual basis and futures curve changes summarize funding-driven directional pressure. | Medium | Medium | Medium | Medium | Medium | Free to acquire from futures markets; already partially tested via funding/OI but not full curve. |
| 6 | Open interest history | Rising OI with directionally biased funding or flow can confirm trend intent. | Medium | Medium | Medium | Medium | Medium | Partial free history available; strong candidate if extended. |
| 7 | Stablecoin flows | Large inflows/outflows to/from exchanges often precede directional moves. | Medium | Medium | Medium | Medium | Medium | Valuable for crypto/spot; historical exchange balance data may be available. |
| 8 | Exchange flows | Net deposits/withdrawals shift liquidity and can precede price pressure. | Medium | Medium | Medium–High | Medium | Medium | Works best with exchange-specific flow data. |
| 9 | Market maker positioning | Inventory imbalance and delta hedging can create directional pressure, especially in options/perps. | Medium | Low | High | High | Medium | Hard to observe directly without proprietary feeds, but high payoff if available. |
| 10 | On-chain metrics | Exchange reserves, transfer volumes, and chain flows can indicate supply-side directional risk. | Medium | Medium | Medium | Medium | Medium | Good for crypto; may be noisy at 15m resolution. |
| 11 | ETF flows / macro flows | Equity and crypto ETF flows convey broad demand shifts that can bias direction. | Low–Medium | Medium | Medium | Medium | Low–Medium | Better for macro overlay than short-term directional edge. |
| 12 | Cross-asset leads | Returns or vol moves in correlated instruments can give directional hints. | Low–Medium | High | Low | Low | Low | Already tested; incremental value for direction is weak on BTC move model. |
| 13 | Funding rates / carry | Changes in funding rates reflect long/short demand and leverage pressure. | Low | High | Low | Low | Low | Already tested; not sufficient alone for direction, though still useful as context. |
| 14 | Long/short ratios | Crowd positioning metrics can signal directional bias, especially when extreme. | Low | Medium | Medium | Low | Low | Useful as sentiment context, but current free history is short and not decisive. |
| 15 | Taker flow / trade imbalance | Aggressive buyer/seller volume may precede short-term moves. | Low | High | Low | Low | Low | Already tested; did not produce a profitable directional edge on full history. |

## Summary

- **Most promising high-payoff sources:** liquidations, order book imbalance, options IV/skew, and options term structure.
- **Next-best practical sources:** full open interest history, funding and carry curve, stablecoin/ exchange flows, and market maker positioning.
- **Lower-value but useful context:** cross-asset leads, taker flow, and long/short ratios—these are already present or partially tested and are not sufficient by themselves.
- **Primary barrier:** free long-history acquisition for microstructure and options data. Perry’s current directional bottleneck is missing external directional observables, not model architecture.

## Implementation guidance

- Prioritize **causal, candle-close features** on any acquired source.
- Keep new features aligned with Perry’s existing purged chronological evaluation and avoid lookahead.
- Use ordering: `microstructure → options → derivatives positioning → on-chain / flow → macro overlay`.
- Do not invest more in OHLCV-only or TSFresh-driven direction research until one or more high-value directional sources are acquired.
