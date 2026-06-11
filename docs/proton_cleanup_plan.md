# Proton Cleanup Plan

No files have been deleted. This plan separates runtime-critical assets from research artifacts that can be archived or ignored after review.

## Keep For Prototype Runtime

- `app.py`
- `Data/datasets/raw/`
- `Data/download_extended_klines.py`
- `Data/datasets/processed/`
- `Model/features_data/state_space.py`
- `Model/features_data/market_structure.py`
- `Model/features_data/market_state_discovery.py`
- `Model/research_engine.py`
- `models/`
- `results/experiments.csv`
- `artifacts/phase3_market_states.json`
- `artifacts/btc_state_labels.csv`
- `artifacts/data/state_discovery/feature_importance.csv`
- `requirements-research.txt`
- `docs/proton_bootstrap_flow.md`
- `docs/proton_cleanup_plan.md`

## Archive Candidates

These directories are valuable as research history but are not needed by `python app.py` once the selected model, config, and summary metadata are retained.

- `experiments/`
  - Contains 285 experiment run directories.
  - Keep configs and metrics for models retained under `models/`.
  - Archive old run folders after selecting a smaller blessed model set.

- `reports/`
  - Contains 20 research reports.
  - Useful for audit trail and interpretation.
  - Not required for the bootstrap runtime.

- `research_phases/`
  - Contains `memory_registry.json`.
  - Useful as research memory.
  - Not required by `app.py`.

- Root research notes:
  - `research_journal.md`
  - `research_log.md`
  - `research_program_log.md`
  - `research_roadmap.md`
  - `next_frontier_report.md`
  - `current_truth.md`
  - These can move under `docs/archive/` after review.

## Ignore Candidates

These are generated outputs or local caches and should not be required in source control unless there is a deliberate reproducibility reason.

- `catboost_info/`
- `__pycache__/`
- `Data/.yfinance_cache/`
- `Data/datasets/processed/*.pkl`
- `Data/datasets/processed/*.csv`
- `prototype_report.md`
- `results/causal_features.pkl`

## Remove Candidates After Archive

Only remove these after the archive is confirmed and a blessed runtime model is selected.

- Superseded experiment directories under `experiments/`
- Superseded model files under `models/`
- Duplicated reports already copied into `docs/archive/`
- One-off backup snapshots under `backups/`

## Runtime Dependency Notes

`python app.py` currently needs:

- At least one usable raw OHLCV CSV under `Data/datasets/raw/`, or network access for Binance ingestion.
- At least one saved `.joblib` model under `models/` for trained move probability. If missing, the app still runs with the heuristic fallback.
- `results/experiments.csv` to choose the best saved model. If missing, the app falls back to the newest `.joblib` file.

## Recommended Cleanup Sequence

1. Choose a blessed model ID and keep its `models/<id>.joblib`, `experiments/<id>/config.json`, and metrics.
2. Move unneeded `experiments/` runs into an archive bundle.
3. Move research-only root Markdown files into `docs/archive/`.
4. Add generated caches and runtime reports to `.gitignore`.
5. Re-run `python app.py` from a clean checkout with only the retained runtime assets.
