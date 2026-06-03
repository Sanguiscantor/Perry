# Perry The Platypus Repository Audit

## Audit scope
This audit reviews the repository structure, documentation, data ingestion surface, experimental outputs, and duplicate/obsolete content in the current `L2` branch.

## Key findings

### 1. Documentation fragmentation
- Multiple duplicate or overlapping documentation sources exist:
  - `current_truth.md` in both root and `docs/`
  - `research_roadmap.md` in root and `docs/`
  - `research_journal.md` in root, plus many ad hoc research notes in `docs/`
- The `reports/` folder contains at least 18 historical research reports.
- The current docs layout is not centralized: active memory, roadmap, journal, and architecture notes are scattered across root and `docs/`.

### 2. Data structure is ad hoc
- `Data/` contains a mix of ingestion scripts, raw dataset CSVs, derivative output folders, validation scripts, and manifest utilities.
- Raw datasets live at the top level of `Data/`:
  - `master_raw_dataset.csv`
  - `multi_asset_dataset.csv`
  - `futures_klines_15m.csv`
- Scripts also write into `Data/derivatives/` and `artifacts/data/microstructure/` directly, without a shared storage contract.
- There is no clear `raw` vs `processed` dataset separation.

### 3. Duplicate and legacy code
- `Data/update_micro_manifest.py` is a one-off manifest generator.
- `backfill_deribit_iv.py` is a best-effort historical backfill tool and belongs in archive or ingestion support.
- `retry_deribit.py` appears to be a temporary retry helper.
- `Data/data.py`, `download_multi_asset.py`, `download_extended_klines.py`, and `download_derivatives.py` each implement separate historical data pipelines.
- The `reports/` folder and root report markdown files are historical outputs, not active code.

### 4. Experimental outputs and datasets
- `experiments/` contains a large number of timestamped folders; these are historical experiment artifacts.
- `artifacts/` contains model, validation, and data artifacts; this is a useful active artifact store, but the current hierarchy is not fully documented.
- `backups/` and `catboost_info/` are archive/support artifacts that should remain but be clearly labeled.

### 5. Active research engine and model code
- The core research engine is in `Model/` and is the primary active asset.
- `Model/data_loader.py`, `Model/research_engine.py`, `Model/research_program.py`, and `Model/enhanced_features.py` are active and should be preserved.
- `Model/features_data/` is the existing feature pipeline and should be retained as the processing target.

## Duplicate files identified
- `current_truth.md` (root vs `docs/`)
- `research_roadmap.md` (root vs `docs/`)
- `next_frontier_report.md` exists in root while similar outputs exist in `reports/`

## Obsolete or archive-worthy artifacts
- All `reports/*.md` are historical research reports and should be moved into `docs/archive/`.
- `Data/` one-off scripts for ingestion and backfill should be reorganized into `Data/ingestion/`, `Data/validation/`, `Data/manifests/`.
- Old experiment folders in `experiments/` should be treated as stored outputs, not active source.

## Recommended cleanup actions
1. Consolidate all active documentation into `docs/` with canonical files:
   - `docs/CURRENT_TRUTH.md`
   - `docs/ROADMAP.md`
   - `docs/JOURNAL.md`
2. Move historical research reports and archival markdown into `docs/archive/`.
3. Restructure `Data/` into:
   - `Data/ingestion/`
   - `Data/validation/`
   - `Data/manifests/`
   - `Data/datasets/raw/`
   - `Data/datasets/processed/`
4. Update code references to use the new raw dataset path layout.
5. Preserve existing model and feature pipeline code in `Model/`.

## Summary
The repository is research-rich but cluttered by duplicated docs, unsupported ingestion scripts, and a flat `Data/` directory. A clean `L2` architecture should preserve Perry's research engine while adopting a modular ingestion and storage layout, centralized docs, and an archive for historical outputs.
