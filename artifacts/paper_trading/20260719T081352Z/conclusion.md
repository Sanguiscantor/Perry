# Perry Paper Trading Laboratory Report

## Experiment Summary

- Experiment Duration: 12 cycle(s)
- Asset: BTCUSDT
- Starting Capital: 100000.00
- Ending Capital: 100000.00
- Net Return: 0.00%
- Prediction Count: 12
- Trade Count: 0
- Trades Skipped: 12
- Win Rate: 0.00%
- Profit Factor: 0.00
- Average Trade: 0.00
- Largest Win: 0.00
- Largest Loss: 0.00
- Maximum Drawdown: 0.00%

## Confidence Calibration

- 50-60%: predictions=11, trades=0, hit_rate=0.00%
- 60-70%: predictions=1, trades=0, hit_rate=0.00%

## Performance By State

- Balance: predictions=11, trades=0, win_rate=0.00%, profit_factor=0.00
- Directional Expansion: predictions=1, trades=0, win_rate=0.00%, profit_factor=0.00

## Driver Observations

- Structure support intact: 12
- Resistance overhead: 12
- State-space regime active: 11
- Below equilibrium: 6
- Above equilibrium: 6
- Positive trend geometry: 1

## Common Entry Reasons

- Balance state|low confidence|high confluence|46% move probability|upper structure|evidence below execution threshold: 6
- Balance state|low confidence|high confluence|46% move probability|lower structure|evidence below execution threshold: 5
- Directional Expansion state|moderate confidence|high confluence|22% move probability|equilibrium|evidence below execution threshold: 1

## Common Exit Reasons

- None recorded.

## Objective Observations

- Perry generated observations, but the decision layer found no setup with enough independent evidence.
- State-level statistics remain sample-limited for: Balance, Directional Expansion.

## Evidence-Based Recommendations

- Continue collecting observations before changing predictive logic.
- Compare confidence buckets only after each bucket has a meaningful sample size.
- Investigate states with frequent no-trade outcomes separately from losing traded states.
