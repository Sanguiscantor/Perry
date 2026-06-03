# Current Architecture

## Overview
Perry now uses a hybrid architecture:
- `Model/` remains the research and evaluation core.
- `Data/` is the new ingestion and dataset surface.
- `docs/` is the consolidated documentation system.
- `artifacts/` stores generated validation and microstructure outputs.

## Top-level directories
- `Data/`: ingestion scripts, validation utilities, manifests, and dataset storage.
- `Data/ingestion/`: source acquisition scripts and live/refresh helpers.
- `Data/validation/`: evaluation scripts that validate new data additions and feature augmentations.
- `Data/manifests/`: data inventory and artifact manifest generation utilities.
- `Data/datasets/raw/`: canonical raw input datasets and downloaded derivatives.
- `Data/datasets/processed/`: reserved for processed feature datasets and pipeline outputs.
- `Model/`: research engine, experiment orchestration, feature builders, and evaluation routines.
- `docs/`: canonical documentation, current truth, roadmap, journal, and archive.
- `docs/archive/`: historical reports and logs moved out of the active documentation surface.
- `artifacts/`: generated data and validation artifacts.
- `experiments/`: stored experiment output folders.
- `results/`: analysis and experiment result tables.

## Data flow
1. Raw market data is downloaded by ingestion scripts in `Data/ingestion/`.
2. Raw CSVs are stored in `Data/datasets/raw/`.
3. Derivative outputs are stored under `Data/datasets/raw/derivatives/`.
4. Validation scripts in `Data/validation/` verify data quality and run feature augmentation comparisons.
5. Feature assembly is expected to output into `Data/datasets/processed/` and `Model/features_data/`.

## Ingestion flow
- `Data/ingestion/download_multi_asset.py`: downloads cross-asset Binance klines.
- `Data/ingestion/download_extended_klines.py`: downloads futures 15m klines with taker flow.
- `Data/ingestion/download_derivatives.py`: downloads Binance funding and open interest history.
- `Data/ingestion/download_binance_sentiment.py`: downloads futures sentiment metrics.
- `Data/ingestion/acquire_deribit_iv.py`: captures live Deribit implied volatility snapshots.
- `Data/ingestion/acquire_microstructure.py`: fetches microstructure sources and computes basis features.
- `Data/ingestion/backfill_deribit_iv.py`: attempts historical Deribit IV backfill.
- `Data/ingestion/data.py`: builds the BTC master dataset.

## Storage flow
- Raw CSVs are centralized under `Data/datasets/raw/`.
- Artifact outputs remain under `artifacts/`, including microstructure snapshots and validation results.
- `Data/datasets/processed/` is the intended destination for feature pipeline outputs.

## Feature flow
- `Model/features_data/` is the existing feature-engineering output layer.
- `Model/enhanced_features.py`, `Model/alternative_features.py`, and `Model/research_engine.py` compose the feature and evaluation pipeline.
- New ingestion outputs are expected to feed into those modules through raw data paths.

## Model flow
- `Model/research_program.py` and `Model/research_engine.py` drive experiments and walk-forward evaluation.
- `Model/data_loader.py` loads feature datasets for model training and analysis.
- `Model/pipeline_hierarchy.py`, `Model/pipeline_multiclass.py`, and related scripts are research-facing modeling components.

## Research flow
- Canonical research guidance lives in `docs/CURRENT_TRUTH.md`, `docs/ROADMAP.md`, and `docs/JOURNAL.md`.
- Historical reports are archived in `docs/archive/reports/`.
- The repository now distinguishes active research artifacts from archived historical work.

## Current status
- The repository has been reorganized for clarity and maintainability.
- Legacy script clutter has been grouped into modular data architecture layers.
- The research core remains intact and ready for future feature engineering and validation work.
