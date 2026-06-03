# Cleanup Report

## Summary
This cleanup move consolidates the repository by grouping active ingestion scripts and raw datasets, archiving historical reports, and centralizing documentation.

## What changed
- Created a new data architecture under `Data/`:
  - `Data/ingestion/`
  - `Data/validation/`
  - `Data/manifests/`
  - `Data/datasets/raw/`
  - `Data/datasets/processed/`
- Moved legacy ingestion scripts into `Data/ingestion/`.
- Moved validation scripts into `Data/validation/`.
- Moved manifest utilities into `Data/manifests/`.
- Moved raw dataset CSVs into `Data/datasets/raw/`.
- Moved derivative outputs into `Data/datasets/raw/derivatives/`.
- Consolidated documentation into `docs/` and archived historical reports in `docs/archive/reports/`.
- Moved old research logs into `docs/archive/logs/`.

## Files archived
- `reports/` → `docs/archive/reports/`
- `next_frontier_report.md` → `docs/archive/reports/`
- `research_report.md` → `docs/archive/reports/`
- `research_log.md` → `docs/archive/logs/`
- `research_program_log.md` → `docs/archive/logs/`

## Files consolidated
- `docs/current_truth.md` → `docs/CURRENT_TRUTH.md`
- `docs/research_roadmap.md` → `docs/ROADMAP.md`
- `research_journal.md` → `docs/JOURNAL.md`

## Validation
- Syntax validation completed on moved ingestion scripts and core model modules.

## Notes
- No information was deleted; all historical content was archived.
- The core research engine in `Model/` remains intact.
