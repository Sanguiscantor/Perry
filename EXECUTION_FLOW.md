# Execution Flow

## Command

```bash
python app.py
```

Phase 2 paper laboratory mode is available through the same entry point:

```bash
python app.py --paper-lab --cycles 1
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

## Phase 2 Paper Trading Laboratory

When `--paper-lab` is supplied, Perry runs the same live refresh and feature
pipeline, then records the observation as a scientific paper-trading experiment.

Each cycle:

1. Refreshes market data and validates freshness diagnostics.
2. Builds causal, state-space, market-structure, and market-state features.
3. Produces Perry's current market state, bias, confidence, expected move, top
   drivers, and human-readable reasoning.
4. Updates any open virtual position using the latest observed price.
5. Uses a multi-evidence decision layer to choose `BUY`, `SELL`, or `NO TRADE`.
6. Records the prediction, portfolio snapshot, trade history, state transition,
   dashboard payload, runtime metadata, and laboratory conclusion.

Artifacts are written to a new immutable experiment directory:

```text
artifacts/paper_trading/<timestamp>/
    predictions.csv
    trades.csv
    portfolio.csv
    state_transitions.csv
    runtime.json
    dashboard.json
    conclusion.md
```

The laboratory is fully simulated. It does not use broker APIs, does not perform
reinforcement learning, and does not feed PnL back into Perry's model logic.

## Single Execution Path

The root `app.py` is the Proton application entry point. The former
`Model/app.py` training runner has been archived as
`archive/legacy/Model_app.py` to avoid two competing app commands.
