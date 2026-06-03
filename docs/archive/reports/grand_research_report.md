# Grand Research Report — Perry

**Generated:** 2026-06-02  
**Program:** `Model/research_program.py`  
**Artifacts:** `research_phases/artifacts/*.json`  
**Prior work preserved:** `experiments/`, `results/experiments.csv`, `research_report.md` (unchanged)

---

## Executive Conclusion

Perry’s **volatility-expansion (move-detection) signal is real, replicable, and largely universal** across major USDT perpetual markets on 15-minute OHLCV data. It **survived** deliberate falsification (cross-asset, future holdout, regime splits). It is **not** a deployable directional trading system: conditional direction after move gating remains **economically negative** after 5 bps fees.

**Verdict:** Perry is a legitimate **research discovery** suitable as a **vol/risk filter**; it is not yet a **directional alpha product**.

---

## 1. What signal exists?

| Property | Value |
| --- | --- |
| **Target** | Binary move: \|forward return\| > **0.5%** over **12×15m** (3 hours) |
| **Model** | CatBoost, fold-local **top_100** causal features (frozen from prior research) |
| **Mechanism** | Anticipating volatility expansion from range, ATR, return variance, volume, entropy, Hurst |
| **BTC walk-forward** | Macro F1 **0.599**, balanced acc **0.607**, bootstrap CI **[0.618, 0.649]** |
| **BTC future holdout** (last 15%, purged) | Balanced acc **0.628**, CI low **0.605** |

The signal is **statistically stable** (all five assets pass pre-registered survival thresholds: balanced acc ≥ 0.55, bootstrap CI low ≥ 0.52).

---

## 2. What signal does not exist?

- **Fee-viable directional trading** when conditioning on move probability (Phase 5): best gate at P(move)≥0.70 still yields **−1.3 bps** mean net return; all thresholds fail viability.
- **Reliable latent market regimes** that persist over time (Phase 3): clustering shows geometry (silhouette ~0.24) but **ARI ≈ 0** early vs late — states are not stable identities.
- **BTC transfer from pooled alt training** (Phase 2 LOO): training on ETH+SOL+BNB+XRP and testing BTC gives balanced acc **0.545** (fails survival) — BTC has idiosyncratic structure not fully captured by alts-only training.
- **Standalone multiclass / hierarchy trading** (prior `research_report.md`): already falsified; confirmed unchanged.

---

## 3. Is Perry real?

**Yes.** Falsification (Phase 1) was designed to disprove the discovery:

| Test | Result |
| --- | --- |
| Cross-asset (BTC, ETH, SOL, BNB, XRP) | **5/5 pass** |
| BTC untouched future holdout | **Pass** |
| Regime splits (vol tercile, trend) | **5/5 pass**; strongest in **high-vol** OOS (balanced acc **0.651**) |

**Decision A:** Continue — signal survived.

Weakest per-asset replication: **SOL** (macro F1 **0.544**, still passes). Strongest: **BNB** (macro F1 **0.606**).

---

## 4. Is Perry universal?

**Mostly yes, with caveats.**

| Test | Pass? | Balanced acc | Notes |
| --- | --- | --- | --- |
| ETH LOO | Yes | 0.602 | |
| SOL LOO | Yes | 0.579 | Weakest in-asset WF |
| BNB LOO | Yes | 0.596 | |
| XRP LOO | Yes | 0.577 | |
| Train 4 / Test XRP | Yes | 0.577 | |
| **BTC LOO** (train alts only) | **No** | 0.545 | BTC-specific features matter |

**Decision B:** Signal **generalizes across assets** when BTC is in the training pool (4/5 LOO pass). It is **not fully universal** for zero-shot BTC from alt-only models. Future work should **include BTC in training** or use per-asset calibration.

---

## 5. Strongest edge discovered?

**Move detection on BNBUSDT** (in-asset walk-forward, macro F1 **0.606**, CI low **0.624**). On BTC, the **high-volatility regime** OOS slice is strongest (balanced acc **0.651**).

No configuration produced **positive net directional returns** after fees.

---

## 6. What data would most improve Perry?

Ranked by composite score (predictive value, cost, availability, integration) — see `data_expansion_report.md`:

1. **Cross-asset returns** (already downloaded; needs engineered lead/lag features)
2. **Funding rates** (free, high integration)
3. **Open interest**
4. **Liquidations**
5. **Order book imbalance** (highest raw predictive potential, higher cost)

For **direction** specifically, order book + funding + OI are the highest-leverage additions (Phase 5 confirms OHLCV direction is data-limited).

---

## 7. What should be researched next?

1. **Per-asset move thresholds** (0.5% fixed may underfit SOL, overfit others).
2. **Funding + OI + liquidation features** on the same purged walk-forward framework.
3. **Vol-product monetization backtests** (straddle timing, position sizing when P(move) high).
4. **Causal market-structure features** (re-engineer without lookahead).
5. **BTC-inclusive pooled models** for production (do not deploy alt-only → BTC transfer).

Do **not** prioritize more CatBoost tuning on OHLCV alone (prior Optuna plateau + this program’s frozen config).

---

## 8. Deployable product path?

| Path | Viability | Rationale |
| --- | --- | --- |
| **Vol / risk filter** | **High** | Move signal survives falsification; use to scale exposure, widen stops, or pause entries |
| **Execution timing** | **Medium** | Same signal; needs execution simulation |
| **Options / vol selling** | **Medium** | Theoretically aligned; needs IV data and book constraints |
| **Directional spot/perp** | **Low** | Conditional direction fails economics |
| **Market-making** | **Low–medium** | Requires book data not yet integrated |

Most likely near-term product: **“Perry Move Score”** API/filter over 15m bars, not an auto-trader.

---

## Phase Summaries

| Phase | Report | Outcome |
| --- | --- | --- |
| 1 Falsification | `falsification_report.md` | **SURVIVED** |
| 2 Universality | `cross_asset_report.md`, `universal_asset_report.md` | **GENERALIZED** (4/5 LOO; BTC LOO fails) |
| 3 Market states | `market_state_report.md` | Clusters exist; **temporal stability weak** (ARI ≈ 0) |
| 4 Transitions | `state_transition_report.md` | High-persistence low-move state (4); spike state (0) has **66%** move rate |
| 5 Conditional direction | `conditional_direction_report.md` | **Not viable** |
| 6 Data expansion | `data_expansion_report.md` | Ranked sources |
| 7 Information theory | `research_phases/artifacts/phase7_advanced.json` | MI confirms ATR/range dominance |

---

## Reproducibility

```bash
# Download multi-asset data (once)
python Data/download_multi_asset.py

# Full program
python Model/research_program.py

# Resume from phase N (uses cached artifacts)
python Model/research_program.py 3
```

Configuration: frozen `BEST_SPEC` in `Model/research_program.py` matching Optuna winner from `research_report.md`.

---

## Decision Log

| Point | Decision | Action |
| --- | --- | --- |
| **A** | Signal survived | Continued to Phase 2–7 |
| **B** | Generalizes (with BTC in train) | Do not rely on alt-only → BTC |
| **C** | Weak temporal state stability | Document; pivot away from HMM trading regimes |
| **D** | Direction weak | Data expansion (Phase 6), not model tuning |
