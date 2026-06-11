# Proton Bootstrap Flow

`python app.py` is the only required runtime entry point.

## Execution Order

1. `app.py`
   - Starts `main()`.
   - Calls `ensure_raw_dataset()`.

2. Raw dataset discovery and ingestion
   - Searches `Data/datasets/raw/**/*.csv`.
   - Accepts the newest CSV with `Datetime`, `Open`, `High`, `Low`, `Close`, and `Volume`.
   - Also checks legacy paths:
     - `Data/datasets/raw/master_raw_dataset.csv`
     - `Data/datasets/raw/futures_klines_15m.csv`
     - `Data/master_raw_dataset.csv`
   - If no usable raw CSV exists, calls the ingestion path through `refresh_data()`, which uses `Data/download_extended_klines.py`.
   - If ingestion fails and no raw dataset can be discovered, startup fails with a direct missing-data error.

3. Raw data loading
   - `load_latest_data()` loads the discovered or ingested raw dataset.
   - Filters to `BTCUSDT` by default.
   - Normalizes the runtime frame to `Datetime`, `Open`, `High`, `Low`, `Close`, `Volume`, and `symbol`.

4. Processed causal feature generation
   - `ensure_feature_dataset()` checks `Data/datasets/processed/btcusdt_causal_features.pkl`.
   - If missing, builds causal model features with `Model.research_engine.build_causal_features()`.
   - Writes:
     - `Data/datasets/processed/btcusdt_causal_features.pkl`
     - `Data/datasets/processed/btcusdt_causal_features.csv`

5. State, structure, geometry, and market-state feature generation
   - `build_runtime_feature_frame()` checks `Data/datasets/processed/btcusdt_state_space_features.csv`.
   - If missing, builds from the latest raw rows:
     - Base return/range/volume features from `app.py`
     - State-space features from `Model/features_data/state_space.py`
     - Anchored structure features from `Model/features_data/market_structure.py`
     - Market state, geometry, confluence, compression, and latent-state features from `Model/features_data/market_state_discovery.py`
   - Writes `Data/datasets/processed/btcusdt_state_space_features.csv`.

6. Trained model loading
   - `discover_model()` reads `results/experiments.csv`.
   - Selects the best non-collapsed saved model that exists under `models/`.
   - Loads the matching `.joblib` model.
   - Reads the matching `experiments/<experiment_id>/config.json` when present.
   - Falls back to the newest model file if no summary metadata is usable.

7. Prediction
   - `predict_move_with_model()` aligns the latest causal feature row to the loaded model's `feature_names_in_`.
   - Produces move probability from the trained move classifier when possible.
   - Falls back to the deterministic prototype move heuristic if the model is missing or incompatible.
   - `estimate_directional_bias()` produces bullish, bearish, and confidence scores from structure and geometry features.
   - `estimate_expected_move()` produces a simple volatility-scaled move range using the model horizon when available.
   - `determine_market_state()` labels the current market state from compression, trend persistence, and entropy.

8. Output
   - Prints a prototype terminal summary:
     - Market State
     - Direction Bias
     - Expected Move
     - Confidence
     - Top Drivers
   - Writes `prototype_report.md`.

## Artifact Reuse

The bootstrap flow reuses existing research outputs instead of rerunning research:

- Saved models from `models/`
- Experiment metadata from `results/experiments.csv`
- Model configs from `experiments/<experiment_id>/config.json`
- Market-state discovery artifacts from `artifacts/`
- Existing structure, geometry, and state-space feature builders under `Model/features_data/`

The runtime path does not call broad research commands such as target screening, model comparison, Optuna tuning, or hierarchy experiments.
