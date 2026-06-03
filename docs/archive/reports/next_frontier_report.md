# Next Frontier Report

**Generated:** 2026-06-02  
**Program:** `Model/information_frontier.py`  
**Memory:** `docs/current_truth.md`, `docs/hypotheses.md`, `docs/research_roadmap.md`

---

## 1. What information was missing?

Perry’s single-asset OHLCV stack could not **see**:

| Missing layer | Status after this program |
| --- | --- |
| Cross-asset lead-lag / relative strength | **Measured** — redundant for OOS move model |
| Funding rate positioning | **Integrated** (2024–present, 8h→15m ff) — **no OOS lift** |
| Open interest changes | **Partial** (~5 days only; API window limits) |
| Liquidations | **Not acquired** (no free long history) |
| Order book imbalance | **Not acquired** |
| Options IV/skew | **Not acquired** |
| NSE / macro internals | **Not tested** (out of scope) |

The largest **remaining** blind spots are **microstructure** (L2, liquidations) and **full-history OI**.

---

## 2. What was learned?

### Cross-asset information flow (new)

- **Contemporaneous coupling:** BTC move label correlates with alt move labels at lag 0 (**0.34–0.46**). This is real market structure Perry “felt” indirectly via BTC range features.
- **Vol propagation:** \|BTC return\| at lag 1 correlates with \|alt return\| at **~0.24–0.27** — shocks propagate within 15 minutes.
- **ETH does not lead alts** at 1 bar (corr ≈ 0).
- **OOS ablation:** Adding explicit cross-asset features **hurts** move detection slightly (**−0.0039** balanced acc) — information is **already encoded** in BTC OHLCV vol features.

### Derivatives (new)

- **Funding rates** downloaded for 5 symbols (**13,260** events, Jan 2024 → Jun 2026).
- **Funding features do not improve** frozen CatBoost move OOS (**−0.0020** balanced acc vs baseline).
- **Direction** unchanged (**−0.0003** balanced acc with all new features).

### Research memory (new)

- Living docs under `docs/` auto-sync from `research_phases/memory_registry.json` on each discovery.

---

## 3. What hypotheses were confirmed?

| ID | Summary |
| --- | --- |
| **H001** | Move/vol-expansion predictable from causal OHLCV (prior) |
| **H004** | BTC contemporaneously co-moves with alts on 15m |
| *(prior)* | Move signal real on all five majors; falsification survived |

---

## 4. What hypotheses were rejected?

| ID | Summary | Evidence |
| --- | --- | --- |
| **H002** | TSFresh helps (prior) | — |
| **H003** | Direction profitable (prior) | — |
| **H005** | Stable regime labels (prior) | — |
| **H006** | Funding improves move OOS | Lift **−0.0020** |
| **H008** | Cross-asset features improve BTC move OOS | Lift **−0.0039** |

**H007** (OI) remains **UNTESTED** until full history is paginated.

---

## 5. Strongest edge currently known?

Unchanged: **move/no-move**, \|r\| > **0.5%** over **12×15m**, CatBoost on **causal single-asset OHLCV** (macro F1 ~**0.60**, bootstrap CI well above chance).  

**New nuance:** Cross-sectional structure exists but is **not incremental** for prediction — do not add complexity for flow features on BTC move.

---

## 6. Highest-value next experiment?

1. **Paginate Binance OI** backward to 2024 (30-day windows) and re-run Phase 3 ablation — only untested free derivative with positioning semantics.  
2. **Liquidation event feed** (Coinglass / exchange agg) — aligns with vol-expansion mechanism.  
3. **Vol-product backtest:** scale exposure when move probability > 0.6 (monetize existing edge without direction).

---

## 7. Shortest path toward deployable trading intelligence?

```
OHLCV move score (validated)
    → risk / sizing / execution filter (deploy now)
    → + liquidation + L2 (direction research)
    → options / MM only if IV + book available
```

**Do not deploy** directional hierarchy. **Do deploy** a vol-awareness layer driven by the existing move score.

---

## Artifacts

| Report | Path |
| --- | --- |
| Cross-asset flow | `cross_asset_flow_report.md` |
| Data integration | `data_integration_report.md` |
| Enhanced evaluation | `enhanced_research_report.md` |
| JSON | `research_phases/artifacts/phase_*.json` |

**Reproduce:**

```bash
python Model/research_memory.py          # seed/sync docs
python Data/download_derivatives.py      # funding (+ partial OI)
python Model/information_frontier.py     # phases 1–3 + memory updates
```
