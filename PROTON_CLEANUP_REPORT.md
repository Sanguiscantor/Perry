# Proton Cleanup Report

## Summary

The repository has been consolidated around a root `app.py` application command.
Validated research remains recoverable. One duplicate execution path was
archived: the old experiment-oriented `Model/app.py` runner now lives at
`archive/legacy/Model_app.py`.

No research programs were rerun. No model tuning or new experiments were added.

## Required For Proton

### `app.py`

Purpose: Proton runtime orchestrator. Loads data, builds features, estimates
market state, estimates move probability and directional bias, prints terminal
output, and writes `prototype_report.md`.

Necessity: Required. This is the single command entry point.

Duplicates: Replaces the old `Model/app.py` execution path.

### `Model/features_data/market_state_discovery.py`

Purpose: Validated state-discovery concepts: support distance, resistance
distance, compression, equilibrium distance, market position, confluence density,
trend geometry, local geometry, latent state features, and soft state labels.

Necessity: Required for Proton state generation.

Duplicates: Some local geometry overlaps with `state_space.py`; Proton keeps both
because one is part of market-state composition and one is part of state-space
representation.

### `Model/features_data/market_structure.py`

Purpose: Anchored support and resistance trendline detection plus distance,
breakout, and breakdown features.

Necessity: Required for structure and directional-bias context.

Duplicates: Simple rolling support/resistance also exists in
`market_state_discovery.py`; both are retained because anchored structure and
rolling structure serve different prototype explanations.

### `Model/features_data/state_space.py`

Purpose: State-space and latent representation helpers using PCA, GMM soft
states, and local geometry.

Necessity: Required for explicit state-space integration in Proton.

Duplicates: Geometry naming overlaps with market-state discovery; runtime
prefixes these columns to avoid collisions.

### `Data/datasets/raw`

Purpose: Local market data sources. Proton prefers `futures_klines_15m.csv`.

Necessity: Required. `python app.py` is offline and depends on local data.

Duplicates: `master_raw_dataset.csv` is a fallback. It should remain available.

### `PROTON_SPEC.md`, `EXECUTION_FLOW.md`, `prototype_report.md`

Purpose: Application documentation and generated output.

Necessity: Required for usability and demonstration value.

## Research Only

### `Model/research_engine.py`, `Model/research_program.py`,
`Model/directional_edge_program.py`, `Model/information_frontier.py`,
`Model/market_state_research.py`, `Model/state_space_research.py`

Purpose: Research execution, frontier exploration, and validation workflows.

Necessity: Not required for Proton runtime. Preserve for recoverability.

Cleanup Decision: Keep in place for now because imports and historical reports
may reference these paths. Treat as research-only.

### `experiments`

Purpose: Historical experiment outputs, metrics, predictions, and plots.

Necessity: Not required for Proton runtime.

Cleanup Decision: Research archive candidate. Not moved in this pass because it
is already clearly separated and large; preserving paths avoids breaking report
references.

### `models`

Purpose: Historical trained model artifacts.

Necessity: Not required for the current Proton heuristic prototype.

Cleanup Decision: Research archive candidate. Preserve intact.

### `catboost_info`

Purpose: CatBoost training logs.

Necessity: Not required for Proton runtime.

Cleanup Decision: Research/legacy generated output. Archive candidate.

### `research_phases`

Purpose: Research memory and phase registry.

Necessity: Not required for Proton runtime.

Cleanup Decision: Preserve as research memory.

## Legacy

### `archive/legacy/Model_app.py`

Purpose: Archived old `Model/app.py` training runner.

Necessity: Not required for Proton runtime.

Cleanup Decision: Archived to eliminate duplicate application entry points while
preserving recoverability.

### `Model/pipeline_hierarchy.py`, `Model/pipeline_multiclass.py`,
`Model/data_loader.py`, `Model/model.py`

Purpose: Legacy model training and evaluation pipelines.

Necessity: Not required by `python app.py`.

Cleanup Decision: Preserve as legacy/research support. Do not call from Proton.

### `Model/features_data/logic.py`

Purpose: Older tsfresh feature compilation script with GUI completion side
effects.

Necessity: Not required by Proton runtime.

Cleanup Decision: Legacy. Avoid in application execution because it performs
manual feature extraction and UI side effects.

## Documentation And Historical Truth

### `current_truth.md` and `docs/CURRENT_TRUTH.md`

Purpose: Historical truth records and validated findings.

Necessity: Not required for runtime, required for research preservation.

Cleanup Decision: Preserve intact. These files must remain recoverable.

### `docs`, `reports`, root research logs

Purpose: Research documentation, findings, reports, and logs.

Necessity: Not required for runtime, required for historical context.

Cleanup Decision: Preserve. Future cleanup can move root research logs under
`docs/archive/logs` after validating references.

## Duplicate Or Obsolete Code

- Duplicate app command: `Model/app.py` was archived to
  `archive/legacy/Model_app.py`.
- Duplicate support/resistance concepts exist across anchored and rolling
  implementations. They are intentionally retained because they represent
  different validated concepts.
- Old training pipelines remain available but are no longer Proton execution
  paths.

## Target Structure

- Application: `app.py`, `PROTON_SPEC.md`, `EXECUTION_FLOW.md`,
  `prototype_report.md`
- Research: `Model/*research*.py`, `experiments`, `models`,
  `research_phases`, `reports`, `artifacts`
- Documentation: `docs`, root Proton docs
- Archives: `archive`, including `archive/legacy`

## Remaining Cleanup Candidates

These are safe candidates for a future dedicated archival pass, but were not
moved automatically because preserving validated research paths is higher value
than cosmetic structure changes:

- Move root research logs into `docs/archive/logs`
- Move `catboost_info` into `archive/research_outputs`
- Move completed `experiments` into `archive/research_outputs/experiments`
- Move historical model artifacts into `archive/research_outputs/models`
