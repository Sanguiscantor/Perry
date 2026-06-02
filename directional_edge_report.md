# Directional Edge Report

**Generated:** 2026-06-02  
**Program:** `Model/directional_edge_program.py`  
**Artifacts:** `research_phases/artifacts/directional_edge_evaluation.json`, `information_source_ranking.json`

---

## Executive conclusion

**No statistically significant, economically viable directional edge** was discovered among all **obtainable free** information sources tested on BTCUSDT 15m with purged walk-forward evaluation and **5 bps** fees.

The **move-detection** signal remains valid (macro F1 ~0.61). Direction stays near chance (~0.52 balanced acc) with **negative** net returns on every full-history configuration.

---

## 1. Which information sources helped?

### For move prediction (reference)

| Source | Move balanced acc lift vs OHLCV |
| --- | ---: |
| OHLCV baseline | 0.6069 (reference) |
| + taker flow | ~unchanged / slight hurt in prior frontier |
| + funding | −0.0020 |

### For direction (this study)

**None improved economics on full 2024→2026 walk-forward.**

| Config | Balanced acc | Net bps (non-overlap) | 95% CI low | Viable? |
| --- | ---: | ---: | ---: | --- |
| baseline_core (OHLCV) | 0.5250 | **−4.69** | < 0 | No |
| + taker flow (full klines) | 0.5073 | −5.33 | < 0 | No |
| taker-only (no OHLCV core) | 0.5124 | **−4.19** | −7.33 | No |
| + funding/OI | 0.5097 | −6.93 | < 0 | No |
| + sentiment (L/S, OI, agg taker) | 0.5116 | −6.54 | < 0 | No |
| + cross-asset | 0.5247 | −5.06 | < 0 | No |
| + all alternative | 0.5033 | −6.71 | < 0 | No |
| move-gated + all features | 0.4986 | −6.27 | < 0 | No |

**Partial / misleading:** `recent_window_may2026` (data only where sentiment exists) shows **+1.76 bps** mean net but **CI [−9.03, +12.44]** — **not significant**, ~30 days of history, **not deployable**.

---

## 2. Which failed?

| Source | Status | Why |
| --- | --- | --- |
| **Taker flow (futures klines)** | **Evaluated — failed** | Full history acquired; direction net **−4.2 to −5.3 bps** |
| **Funding rate** | **Evaluated — failed** | Full history; net **−6.9 bps** with OHLCV |
| **Open interest** | **Evaluated — failed** | ~31 days history only; merged with funding |
| **Global / top-trader L/S** | **Evaluated — failed** | ~31 days paginated; full WF net **−6.5 bps** |
| **Agg. taker buy/sell ratio** | **Evaluated — failed** | Same window as L/S |
| **Cross-asset flow** | **Evaluated — failed** | net **−5.1 bps** |
| **Liquidations** | **Not acquired** | No free long-history REST |
| **L2 order book** | **Not acquired** | No free historical archive |
| **Options IV/skew** | **Not acquired** | Deribit history not integrated |
| **ETF / on-chain / NSE** | **Not acquired** | Out of scope or paid |

---

## 3. Strongest directional signal discovered?

**On full out-of-sample history:** `baseline_core` — balanced acc **0.5250**, net **−4.69 bps** (least negative, still unprofitable).

**On May 2026 subset only:** balanced acc **0.5614**, net **+1.76 bps** — **not robust** (wide CI, short window, multiple-testing risk).

**Strongest validated signal in Perry overall remains move detection**, not direction.

---

## 4. Current best trading architecture?

| Layer | Recommendation |
| --- | --- |
| **Deploy** | **Move score** (CatBoost, \|r\|>0.5%/12 bars) as **vol/risk filter** |
| **Do not deploy** | Directional spot/perp from logistic/CatBoost on OHLCV ± free derivatives |
| **Do not deploy** | Move-gated direction (worse than baseline direction) |
| **Research-only** | Hierarchical move→direction until L2/liquidations/options data |

```
[Move model] → scale risk / timing
[Direction model] → NOT PRODUCTION (negative expectancy)
```

---

## 5. What should be researched next?

1. **Paid or scraped liquidation + L2 history** — only remaining high-priority sources not testable free.
2. **Deribit options skew / IV term structure** — forward vol ground truth.
3. **Monetize move score** via vol products or execution rules (proven edge path).
4. **Do not** run more OHLCV/derivative logistic tuning without new data.

---

## Data acquired (Priority 2)

| Dataset | Path | BTC coverage |
| --- | --- | --- |
| Futures klines + taker flow | `Data/futures_klines_15m.csv` | 2024-01-01 → 2026-06-02 |
| Funding | `Data/derivatives/funding_rates.csv` | 2024-01-01 → 2026-06-02 |
| OI / L/S / agg taker | `Data/derivatives/*.csv` | ~2026-05-02 → 2026-06-02 |

Pipelines: `Data/download_extended_klines.py`, `Data/download_derivatives.py`, `Data/download_binance_sentiment.py`

---

## Source ranking (Priority 1 — pre-test prior)

See `research_phases/artifacts/information_source_ranking.json`. Top prior: taker flow, top-trader L/S, funding. **Empirical result:** none delivered viable direction OOS on available history.

---

## Reproduce

```bash
python Data/download_extended_klines.py
python Data/download_derivatives.py
python Data/download_binance_sentiment.py
python Model/directional_edge_program.py
```

Memory docs auto-update via `Model/research_memory.py`.

---

## Stop condition

All **viable free** sources have been **evaluated** or **documented as unobtainable**. Program stops pending **paid microstructure/options** data or pivot to **move-score productization**.
