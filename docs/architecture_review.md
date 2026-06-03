# Architecture Review: Perry The Platypus vs Perry-Nikhil

## 1. High-level comparison

### Perry The Platypus
- Current architecture is research-first:
  - `Model/` contains the core research engine, evaluation pipeline, and feature dataset loader.
  - `Data/` contains standalone acquisition and validation scripts for different sources.
- Data ingestion is ad hoc and file-based:
  - Scripts like `Data/acquire_deribit_iv.py`, `Data/acquire_microstructure.py`, and `Data/backfill_deribit_iv.py` each implement their own fetching, CSV writing, and manifest logic.
  - Artifact storage is centralized under `artifacts/data/microstructure/`, but there is no shared ingestion abstraction or common pipeline orchestration.
- Feature generation is partly centralized in `Model/` but not integrated with raw ingestion:
  - `Model/data_loader.py` loads a flattened `features_data/master_feature_dataset.csv`.
  - `Model/research_program.py` drives research evaluation from raw market CSVs and does not depend on a common data ingestion service.
- Strengths:
  - Research and evaluation logic is mature and self-contained.
  - Existing one-off scripts make quick experiments easy.
- Weaknesses:
  - Pipeline fragility due to separate scripts and repeated file handling.
  - Lack of shared storage API, config, or metadata for data sources.
  - Manual orchestration makes scaling and live refresh harder.

### Perry-Nikhil
- Data-ingestion-oriented architecture focused on stream ingestion and validated storage.
- `layer2-orderbook/` demonstrates a clean modular structure:
  - `config/config.py` centralizes environment and endpoint settings.
  - `ingestion/snapshot.py` and `ingestion/collector.py` separate initial snapshot loading from websocket streaming.
  - `core/validator.py` and `core/orderbook.py` isolate validation and state maintenance.
  - `storage/parquet_store.py` centralizes output format and destination.
- Strengths:
  - Modular ingestion pipeline with explicit layers.
  - Parquet output and batched writes support efficient downstream processing.
  - Simple but robust gap detection and sequence validation.
- Weaknesses:
  - Focus is narrow: only orderbook ingestion, no research pipeline or feature generation.
  - It is not yet integrated with Perry's existing evaluation and model artifacts.

## 2. Architectural gap analysis

| Dimension | Perry The Platypus | Perry-Nikhil | Gap / Opportunity |
|---|---|---|---|
| Ingestion modularity | Low | High | Build a shared ingestion package for Perry.
| Storage abstraction | Low (CSV ad hoc) | Medium (Parquet store) | Adopt a storage layer and metadata manifest.
| Research integration | High | Low | Keep Perry research core; integrate new pipeline outputs.
| Live/update support | Manual scripts | Streaming batch write | Use Nikhil-style streaming and validation for live microstructure.
| Data schema management | Weak | Minimal | Add schema/contract definitions and feature pipeline.
| Feature pipeline | Fragmented | Not present | Add a conversion layer from storage artifacts to feature sets.

## 3. Recommended decision

### Recommended approach: Hybridize
- **Keep** Perry The Platypus as the canonical research and validation engine.
- **Replace/upgrade** Perry's current ingestion layer with a **Nikhil-inspired modular pipeline**.
- **Hybridize** by preserving existing research logic and gradually migrating data acquisition into a new package structure.

This gives the best balance:
- preserve the existing research investment and validation flow;
- eliminate the current brittle `Data/` script collection;
- add clear ingestion, storage, and validation contracts;
- enable future live updates, multi-source acquisition, and scalable feature assembly.

## 4. Practical migration direction

### Short-term
1. Keep all current `Data/` scripts as compatibility fallbacks.
2. Create a new `data_pipeline/` or `ingestion/` package.
3. Add `config/config.py`, `ingestion/`, `core/`, `storage/`, and `utils/` modules.
4. Mirror Nikhil's clean separation of concerns for new sources:
   - Deribit IV snapshots
   - Binance/Deribit microstructure and orderbook data
   - exchange metadata

### Medium-term
1. Migrate `Data/acquire_deribit_iv.py` into the new pipeline as a source module.
2. Switch artifact storage to a configurable, schema-aware format (Parquet preferred).
3. Add a manifest/metadata writer like `Data/update_micro_manifest.py` but built into the pipeline.
4. Expose a single `main.py` entrypoint for ingestion and batch refresh.

### Long-term
1. Add downstream feature assembly modules that convert raw artifacts to `features_data/master_feature_dataset.csv`.
2. Introduce versioned data contracts and replayable ingest states.
3. Enable incremental refresh and live sampling in research experiments.

## 5. Recommendation summary
- Keep Perry's research engine and evaluation stack.
- Replace the current ad hoc data scripts with a Nikhil-inspired modular ingestion architecture.
- Hybridize by connecting new ingestion/storage layers to the existing feature generation and research workflows.

> Verdict: **Hybrid architecture** — preserve the research core, replace the ingestion pipeline, and integrate the two with a new, clean pipeline layer.
