# Proton Specification

## Objective

Proton packages the Perry research code into a runnable prototype. The only
required command is:

```bash
python app.py
```

The command loads the latest BTCUSDT data, generates validated Perry state
features, estimates current market state, estimates move probability and
directional bias, prints a terminal report, and writes `prototype_report.md`.

## Scope

Proton is an application branch. It does not perform new research, model tuning,
hyperparameter optimization, or experiment search. It reuses existing feature
builders and validated state-discovery concepts.

## Runtime Components

- `app.py`: Root application entry point and report orchestration.
- `Model/features_data/market_state_discovery.py`: Multi-scale support,
  resistance, compression, equilibrium, confluence, geometry, latent state, and
  market position features.
- `Model/features_data/market_structure.py`: Anchored support/resistance and
  structure distance features.
- `Model/features_data/state_space.py`: PCA, GMM, and local geometry
  state-space features.
- `Data/datasets/raw/futures_klines_15m.csv`: Preferred latest BTCUSDT runtime
  data source.
- `prototype_report.md`: Generated current-state markdown report.

## Output Contract

`python app.py` prints:

- Asset
- Market State
- Position
- Confluence
- Move Probability
- Directional Bias
- Bullish Score
- Bearish Score
- Directional Confidence
- Top Drivers
- Interpretation

It also writes the same current-state result with diagnostics to
`prototype_report.md`.

## Directional Bias Policy

Proton does not emit BUY/SELL recommendations. Direction is represented as:

- Bullish Score
- Bearish Score
- Directional Bias
- Directional Confidence

The scores are heuristic prototype estimates based on market position, support
and resistance structure, equilibrium distance, trend geometry, confluence
density, compression state, and market state.

## Assumptions

- BTCUSDT is the prototype asset.
- The latest local 15-minute futures dataset is the authoritative runtime input.
- The prototype uses a recent lookback window for demonstration speed and
  current-state clarity.
- Existing research outputs remain recoverable in `docs`, `reports`,
  `artifacts`, `experiments`, `models`, and `archive`.

## Dependencies

The runtime uses the repository's existing Python stack:

- pandas
- numpy
- scikit-learn

No new dependency family was introduced for Proton.
