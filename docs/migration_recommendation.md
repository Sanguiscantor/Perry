# Migration Recommendation for Perry The Platypus

## Objective
Create a maintainable, scalable architecture by retaining Perry's research strengths while adopting a cleaner data ingestion and storage layer from the cloned `Perry-Nikhil` architecture.

## Recommended target architecture

1. `ingestion/`
   - Source modules for each external feed (Deribit, Binance, etc.)
   - Snapshot and streaming collectors
   - Validation and gap detection
2. `storage/`
   - Unified output abstraction (Parquet + metadata manifest)
   - Configurable destination paths
3. `core/`
   - Sequence validator, orderbook state, source normalization
   - Common utilities for batching and error handling
4. `features/` or `feature_pipeline/`
   - Convert raw storage artifacts into feature-ready datasets
   - Generate `features_data/master_feature_dataset.csv`
5. `Model/`
   - Keep existing research engine, evaluation, and `research_program.py`
   - Refactor to consume the new feature pipeline outputs instead of raw ad hoc CSVs

## Why this migration
- `Perry The Platypus` currently mixes one-off data scripts with research logic.
- `Perry-Nikhil` shows a small but well-factored ingestion architecture.
- A hybrid architecture avoids throwing away Perry's research investment while removing pipeline brittleness.

## Key implementation steps

### Step 1: Scaffold the new pipeline
- Add `config/config.py` for source settings and directories.
- Add `ingestion/snapshot.py` and `ingestion/collector.py` modules.
- Add `storage/parquet_store.py` or `storage/artifact_store.py`.
- Add `core/validator.py` and `core/orderbook.py` if orderbook support is needed.

### Step 2: Migrate existing scripts
- Refactor `Data/acquire_microstructure.py` and `Data/acquire_deribit_iv.py` into source modules.
- Preserve CSV output during transition; write both CSV and Parquet if necessary.
- Replace `Data/update_micro_manifest.py` with pipeline-managed metadata.

### Step 3: Connect to research
- Add a feature assembly layer that reads stored artifacts and writes `Model/features_data/master_feature_dataset.csv`.
- Update `Model/data_loader.py` and `Model/research_program.py` to use the new pipeline outputs.

### Step 4: Validate and stabilize
- Keep current validation output (`artifacts/validation/iv_vs_baseline.json`) as a regression benchmark.
- Add unit tests for ingestion modules and manifest generation.
- Maintain existing `Data/` scripts as compatibility wrappers until the pipeline is stable.

## Recommended outcome
- A single ingestion interface for all real-time and snapshot sources.
- Clean separation between acquisition, storage, and model input generation.
- A stable, maintainable architecture that supports new data sources without rewriting the research core.

## Immediate next action
Implement the pipeline scaffold in the Perry repo and migrate one source first (best candidate: Deribit IV / microstructure acquisition).