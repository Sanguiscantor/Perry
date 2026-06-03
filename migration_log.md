# Migration Log

## Goal
Create the cleanest possible future architecture for Perry while preserving research assets.

## Actions completed

1. Created `repository_audit.md` to inventory repository clutter, duplicate docs, and legacy scripts.
2. Established the new docs consolidation system under `docs/`.
3. Created canonical documentation files:
   - `docs/CURRENT_TRUTH.md`
   - `docs/ROADMAP.md`
   - `docs/JOURNAL.md`
4. Archived historical research report outputs to `docs/archive/reports/`.
5. Archived legacy logs to `docs/archive/logs/`.
6. Reorganized `Data/` into modular subdirectories:
   - `Data/ingestion/`
   - `Data/validation/`
   - `Data/manifests/`
   - `Data/datasets/raw/`
   - `Data/datasets/processed/`
7. Moved raw dataset CSVs and derivatives output into `Data/datasets/raw/`.
8. Updated ingestion and model code to use the new raw data paths and ingestion script locations.
9. Verified syntax of moved Python files.

## Next steps
- Add a formal `docs/README.md` or landing page for the consolidated documentation system.
- Review `Model/` workflows for any remaining references to old flat Data paths in helper scripts or notebooks.
- Add feature assembly and manifest generation to the new `Data/` pipeline structure.
