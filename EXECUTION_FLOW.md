# Execution Flow

## Command

```bash
python app.py
```

## Flow

1. Load latest data
   - Preferred source: `Data/datasets/raw/futures_klines_15m.csv`
   - Fallbacks: `Data/datasets/raw/master_raw_dataset.csv`,
     `Data/master_raw_dataset.csv`
   - Filters to `BTCUSDT`
   - Uses the latest recent rows for a fast current-state prototype

2. Generate base features
   - Returns
   - Range percent
   - Volume z-score

3. Generate state-space features
   - PCA latent representation
   - GMM soft state representation
   - Local price-path geometry

4. Generate structure features
   - Anchored support
   - Anchored resistance
   - Distance to anchored structure
   - Breakout and breakdown flags

5. Generate market-state features
   - Multi-scale support/resistance
   - Compression
   - Equilibrium distance
   - Position inside structure
   - Confluence density
   - Trend geometry
   - Structural acceleration
   - Latent state membership

6. Determine current market state
   - Labels compression, directional expansion, transition, or balance
   - Labels lower, upper, or equilibrium structure position
   - Labels confluence as low, moderate, or high

7. Run move prediction
   - Estimates move probability from compression energy, confluence, position
     extremity, recent range, and volatility

8. Estimate directional bias
   - Produces bullish score, bearish score, directional bias, and confidence
   - Does not produce a hard trade signal

9. Generate reports
   - Prints terminal report
   - Writes `prototype_report.md`

## Single Execution Path

The root `app.py` is the Proton application entry point. The former
`Model/app.py` training runner has been archived as
`archive/legacy/Model_app.py` to avoid two competing app commands.
