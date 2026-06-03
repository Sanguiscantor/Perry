# Perry — Current State Report

**Generated:** 2026-06-02  
**Sources:** `research_report.md`, `research_log.md`, `Model/research_engine.py`, repository layout  
**Scope:** Predictive-signal research on BTCUSDT 15-minute candles (~84,740 rows in `Data/master_raw_dataset.csv`). Experiment artifacts live under `experiments/`, `results/`, and `reports/` (gitignored locally).

---

## What Perry Is Today

Perry is a **research codebase**, not a deployed trading system. It has two layers:

| Layer | Role | Evidence status |
| --- | --- | --- |
| **Research engine** (`Model/research_engine.py`) | Leakage-safe chronological walk-forward evaluation, 135 causal engineered features, 283 logged experiments | **Authoritative** for current conclusions |
| **Legacy pipelines** (`pipeline_hierarchy.py`, `pipeline_multiclass.py`, `inspect_tsfresh_importance.py`, visualization app) | Earlier hierarchical / TSFresh workflows, often with random splits or global feature selection | **Not** clean OOS evidence; some paths are explicitly excluded or superseded |

The validated research product is **move detection** (whether absolute forward return exceeds a threshold), not directional alpha.

---

## 1. What Has Been Validated?

### Move / volatility-expansion signal

- **Target:** Binary move — `|future_return| > 0.5%` over **12 candles (3 hours)**.
- **Model:** CatBoost with **fold-local** `top_100` features (SelectKBest inside each training fold only).
- **Evaluation:** 4 purged expanding chronological folds (train ~55%, test ~9% per fold, horizon purge, max 45k train rows).
- **Performance (OOS aggregates):**
  - Mean macro F1: **0.6009** (σ 0.0229; worst fold 0.5706)
  - Mean balanced accuracy: **0.6084**
  - Daily block bootstrap on combined OOS predictions: balanced accuracy **0.6346**, 95% CI **[0.6189, 0.6501]**, P(≤ chance) **0.0000**
- **Stability:** Low fold-to-fold variance; Optuna tuning moved macro F1 only from 0.600918 → 0.600929 (**plateau**), suggesting the configuration is near a local optimum rather than a lucky single split.

### Methodology and leakage controls

- Purged walk-forward only; no random train/test splits in the research engine.
- Collapse rejection rules applied (per-class precision/recall &lt; 10%, or &gt;90% predicted in one class).
- **135 causal features** at candle close (range, ATR, rolling variance, volume, entropy, Hurst, autocorrelation, etc.).
- **Fold-local feature selection** — saved TSFresh columns selected on the full 40k labeled matrix were **excluded** from clean evidence.
- **Causal-only** swing flags in `build_causal_features`; lookahead routines in `market_structure.py` were **excluded** from predictive experiments.

### Secondary but real effects

- **Direction** at horizon 12 is **weakly above chance** (best: logistic + `core`, macro F1 ~0.522, balanced accuracy ~0.526; bootstrap CI above 0.5).
- **Hierarchy** (move CatBoost → direction logistic) achieves macro F1 ~0.405 at move probability threshold 0.50 — better than raw multiclass logistic screening, but still poor as a trading system (see falsified section).

### Infrastructure validated for reproducibility

- Experiment registry (`results/experiments.csv`), per-run artifacts (metrics, confusion matrices, predictions, calibration plots).
- Report generator (`Model/generate_research_report.py`) with bootstrap significance and economics checks.
- Research dependency manifest (`requirements-research.txt`).

---

## 2. What Has Been Falsified?

| Claim / approach | Result |
| --- | --- |
| **Standalone directional trading** (12-candle horizon, 5 bps fee, non-overlapping forecasts) | Mean net return **−5.19 bps**, 95% CI **[−8.27, −2.16]** — economically negative |
| **Hierarchical spot strategy** (move then direction, same fee assumptions) | Mean net **−1.63 bps**, 95% CI **[−4.04, 0.84]** — not robustly profitable |
| **Deploy current directional or hierarchical strategy as-is** | Explicitly rejected in `research_report.md` |
| **TSFresh improves move detection** | Best fold-local TSFresh (`top_1000`, ExtraTrees): macro F1 **0.5741** vs compact causal **~0.60** — **worse**; compute not justified |
| **Global TSFresh feature list** as clean evidence | Falsified by audit — selected on full labeled matrix |
| **Lookahead market-structure features** | Excluded; swing/rejection logic uses future candles |
| **Multiclass (up / flat / down) as primary target** | Macro F1 ~0.39–0.40 — substantially weaker than move-only |
| **Volatility-regime (tertile) prediction** | Macro F1 ~0.39 — weak |
| **XGBoost / HistGradientBoosting** on best move setup | Frequent **class collapse** → rejected under engine rules |
| **Ensembles beating tuned CatBoost** | Soft voting (~0.5956), SVM (~0.5788), temporal stacking (~0.5668) all **below** best CatBoost (~0.6009) |
| **Heavy hyperparameter search unlocking large gains** | 20 random + 15 Optuna trials — **negligible** lift |

### Not falsified, but not yet validated

- **Out-of-sample extension:** No reported holdout on data strictly after the last walk-forward test window, and **only BTCUSDT** in the clean engine run.
- **Production monetization paths** (options, market-making, execution timing) — recommended, not executed.
- **Legacy `pipeline_hierarchy.py` / `inspect_tsfresh_importance.py`** — still use random `train_test_split` and pre-selected TSFresh columns; they should be treated as **exploratory**, not as confirmation of the research engine results.

---

## 3. Strongest Current Signal

**Move / no-move at 12 × 15m bars (3 hours) with a 0.5% absolute return threshold**, predicted by **CatBoost** on **100 fold-local-selected causal features**.

```text
Target:     move  (|r_{t→t+12}| > 0.005)
Horizon:    12 candles
Features:   top_100 (fold-local SelectKBest on 135 causal columns)
Model:      CatBoost (Optuna-tuned: 420 trees, depth 4, lr ≈ 0.023)
Signal type: Volatility expansion / large-move anticipation
```

**Why this is the strongest signal:** It dominates every other target family screened (direction, multiclass, volatility regime), every other model family that did not collapse, and every TSFresh variant. Importance is dominated by **realized range, ATR, rolling return variance, long-window volume ratio, entropy, Hurst, and autocorrelation** — a coherent “something big is about to happen” story, not reliable “which way.”

**What it is not:** A fee-viable directional edge. Direction remains ~2.6% above chance in balanced accuracy but **loses money** after 5 bps per trade.

---

## 4. Next Five Highest-Value Research Directions

Ranked by expected information gain given what is already known:

### 1. True holdout + cross-symbol generalization

Run the **frozen** best move configuration on:
- A **chronological tail** never used in any fold or tuning decision.
- **Additional symbols** (ETH, majors) with the same feature pipeline.

This is the highest-value falsification test: the current signal is strong **in-sample of the walk-forward design** but has not been reported on untouched later periods or other instruments.

### 2. Economic use of move detection (non-directional monetization)

Simulate how the move probability improves:
- **Volatility strategies** (straddle/strangle timing, vol targeting).
- **Execution** (scale in/out when elevated move risk).
- **Risk filters** (reduce size or widen stops when P(move) is high).

The report already argues this path; it has not been quantified end-to-end in Perry.

### 3. External / microstructure features for direction

The largest **unfilled** gap is **direction after fees**. Prioritized additions if data can be sourced:
- Order book imbalance, spread, depth.
- Funding rate, open interest, liquidations.
- Cross-asset leads (ETH, indices, DXY proxy).

These are more plausible sources of **incremental directional** signal than more tuning on OHLCV alone.

### 4. Calibration and decision thresholds on the move model

181 calibration plots were generated; systematic work remains:
- Choose **operating points** on move probability (precision/recall vs downstream utility).
- Test whether **regime conditioning** (e.g. high vs low baseline vol) improves stability without refitting on test labels.

Low model-tuning upside was shown; **decision-layer** optimization may still matter for deployment as a filter.

### 5. Causal market-structure features (re-engineered)

Original `market_structure.py` was rejected for **lookahead**. Rebuilding **strictly causal** structure features (confirmed swings only, no future bars) and adding them to the causal cache is a bounded experiment that could either add signal or confirm OHLCV volatility features already capture the edge.

**Explicitly deprioritized** unless requirements change: more TSFresh sweeps, more CatBoost Optuna trials, and redeploying legacy random-split hierarchical scripts without porting them to the research engine.

---

## Repository Snapshot

| Component | Status |
| --- | --- |
| `Model/research_engine.py` | Complete research platform (build-features, screen, compare, tune, optuna, hierarchy, tsfresh-screen, summarize) |
| `research_report.md` | Final synthesized report (283 experiments; 215 non-collapsed) |
| `research_log.md` | Chronological audit and run log |
| `Model/generate_research_report.py` | Regenerates report from `results/` + `experiments/` |
| Legacy ML + viz (`Model/app.py`, `pipeline_*.py`, frontend) | Operational for exploration; **not** the evidentiary basis for deployment |
| Git-modified `inspect_tsfresh_importance.py` | Legacy TSFresh inspection; separate from clean engine evidence |

---

## Bottom Line

Perry has **validated a stable, economically interpretable move-detection signal** on BTCUSDT 15m data under rigorous chronological evaluation. It has **falsified** treating direction, hierarchy, or TSFresh as ready trading products on the same data and fee assumptions. The project’s honest next phase is **generalization testing** and **downstream use of move probability**, not more OHLCV model tuning on the same sample.
