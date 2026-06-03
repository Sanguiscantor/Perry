# Perry The Platypus

This branch is an architecture-focused reorganization of Perry for long-term research and data engineering.

## Key directories
- `Data/` — ingestion, validation, manifests, and raw/processed datasets.
- `Model/` — research engine, feature builders, experiment orchestration.
- `docs/` — canonical documentation and archive.
- `artifacts/` — generated data and validation outputs.

## Canonical docs
- `docs/CURRENT_TRUTH.md`
- `docs/ROADMAP.md`
- `docs/JOURNAL.md`
- `docs/archive/` — historical reports and logs.

## Migration artifacts
- `repository_audit.md`
- `migration_log.md`
- `cleanup_report.md`
- `current_architecture.md`

## How to use
1. Read `docs/CURRENT_TRUTH.md` for what Perry currently knows.
2. Read `docs/ROADMAP.md` for the next research direction.
3. Use `Data/ingestion/` to acquire and refresh raw datasets.
4. Use `Data/validation/` to validate added feature sources.
5. Use `Model/` to run research experiments and evaluations.
