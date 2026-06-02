"""Generate the final predictive-signal research report from recorded artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiments.csv"
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "research_report.md"
RNG = np.random.default_rng(42)


def clean_ranked(results: pd.DataFrame) -> pd.DataFrame:
    clean = results[~results["rejected_for_collapse"].astype(bool)].copy()
    return clean.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
    )


def daily_block_bootstrap(experiment_id: str) -> dict[str, float]:
    predictions = pd.read_csv(
        EXPERIMENTS / experiment_id / "predictions.csv",
        parse_dates=["Datetime"],
    )
    predictions["day"] = predictions["Datetime"].dt.floor("D")
    labels = sorted(predictions["actual"].unique())
    days = predictions["day"].unique()
    counts = []
    for day in days:
        block = predictions[predictions["day"] == day]
        counts.append([
            [
                ((block["actual"] == label) & (block["predicted"] == label)).sum(),
                (block["actual"] == label).sum(),
            ]
            for label in labels
        ])
    counts = np.asarray(counts, dtype=float)
    observed = counts.sum(axis=0)
    balanced_accuracy = np.mean(observed[:, 0] / observed[:, 1])
    draws = RNG.integers(0, len(days), size=(5000, len(days)))
    sampled = counts[draws].sum(axis=1)
    estimates = np.mean(sampled[:, :, 0] / sampled[:, :, 1], axis=1)
    return {
        "balanced_accuracy": balanced_accuracy,
        "ci_low": np.quantile(estimates, 0.025),
        "ci_high": np.quantile(estimates, 0.975),
        "p_at_or_below_chance": np.mean(estimates <= 0.5),
        "days": len(days),
    }


def economic_check(experiment_id: str, target: str, horizon: int = 12) -> dict[str, float]:
    predictions = pd.read_csv(EXPERIMENTS / experiment_id / "predictions.csv")
    if target == "direction":
        position = np.where(predictions["predicted"] == 1, 1.0, -1.0)
    else:
        position = np.select(
            [predictions["predicted"] == 0, predictions["predicted"] == 2],
            [-1.0, 1.0],
            default=0.0,
        )
    net = position * predictions["future_return"].to_numpy() - (position != 0) * 0.0005
    independent = net[::horizon]
    estimates = RNG.choice(independent, size=(5000, len(independent)), replace=True).mean(axis=1)
    return {
        "all_rows_mean": net.mean(),
        "nonoverlap_mean": independent.mean(),
        "ci_low": np.quantile(estimates, 0.025),
        "ci_high": np.quantile(estimates, 0.975),
        "p_at_or_below_zero": np.mean(estimates <= 0),
    }


def backfill_calibration_plots() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = 0
    for calibration_path in EXPERIMENTS.glob("*/calibration.csv"):
        plot_path = calibration_path.with_name("calibration_plot.png")
        if plot_path.exists():
            continue
        calibration = pd.read_csv(calibration_path)
        figure, axis = plt.subplots(figsize=(4.8, 4.0))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
        for fold, frame in calibration.groupby("fold"):
            axis.plot(frame["mean_predicted"], frame["fraction_positive"], marker="o", label=f"fold {fold}")
        axis.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Calibration")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=140)
        plt.close(figure)
        count += 1
    return count


def compact_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headings = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    backfill_calibration_plots()
    results = pd.read_csv(RESULTS)
    ranked = clean_ranked(results)
    top_50 = ranked.head(50).copy()
    top_50.to_csv(REPORTS / "top_50_configurations.csv", index=False)
    best = ranked.iloc[0]
    direction = ranked[ranked["target"] == "direction"].iloc[0]
    multiclass = ranked[ranked["target"] == "multiclass"].iloc[0]
    volatility = ranked[ranked["target"] == "volatility"].iloc[0]
    hierarchy = ranked[ranked["model"] == "catboost+logistic"].iloc[0]
    tsfresh = ranked[ranked["notes"].fillna("").str.contains("TSFresh")].iloc[0]
    advanced = ranked[ranked["model"].isin(["svm", "voting", "temporal_stacking"])].groupby("model").head(1)
    best_config = json.loads((EXPERIMENTS / best["experiment_id"] / "config.json").read_text())
    importance = (
        pd.read_csv(EXPERIMENTS / best["experiment_id"] / "feature_importance.csv")
        .groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
    )
    importance.head(50).to_csv(REPORTS / "feature_importance_top_50.csv", index=False)
    move_significance = daily_block_bootstrap(best["experiment_id"])
    direction_significance = daily_block_bootstrap(direction["experiment_id"])
    direction_economics = economic_check(direction["experiment_id"], "direction")
    hierarchy_economics = economic_check(hierarchy["experiment_id"], "multiclass")
    significance = pd.DataFrame([
        {"signal": "move", **move_significance},
        {"signal": "direction", **direction_significance},
    ])
    significance.to_csv(REPORTS / "significance.csv", index=False)

    top_table = compact_table(top_50, [
        "experiment_id", "target", "horizon", "threshold", "feature_set", "model",
        "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean",
    ])
    family_table = compact_table(pd.DataFrame([best, direction, multiclass, volatility]), [
        "target", "horizon", "threshold", "feature_set", "model", "macro_f1_mean",
        "macro_f1_std", "macro_f1_worst", "macro_f1_best", "balanced_accuracy_mean", "accuracy_mean",
    ])
    advanced_table = compact_table(advanced, [
        "model", "feature_set", "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean",
    ])
    importance_table = compact_table(importance.head(25), ["feature", "importance"])

    report = f"""# Predictive Signal Research Report

## Executive Summary

The repository contains a genuine, temporally stable **move-detection** signal for BTCUSDT 15-minute candles. The strongest target is whether absolute forward return exceeds **0.5%** over **12 candles (3 hours)**. The best model is CatBoost with 100 fold-local selected causal features.

There is **not** enough evidence for a standalone directional trading strategy. Short-horizon direction is weakly predictable, but the effect is economically negative after a 5 bps fee. The hierarchy can detect that a move is coming but cannot reliably assign bullish versus bearish direction.

## Methodology

- Source: `{(ROOT / "Data" / "master_raw_dataset.csv").relative_to(ROOT)}`, {len(pd.read_csv(ROOT / "Data" / "master_raw_dataset.csv")):,} BTCUSDT 15-minute candles.
- Evaluation: expanding chronological walk-forward folds only. No random train/test splits.
- Purging: each training fold ends at least one forecast horizon before its test fold.
- Ranking: macro F1, then balanced accuracy, then accuracy.
- Rejection: experiments are flagged when any class precision or recall is below 10%, or when one predicted class exceeds 90%.
- Feature selection: fold-local only. Saved TSFresh-selected columns from the earlier pipeline were not used as clean evidence because they were selected against the full labeled 40k matrix.
- Economics: directional returns subtract 5 bps per trade and are also checked on non-overlapping 12-candle observations.

## Experiments Performed

Recorded experiments: **{len(results)}**. Clean non-collapsed experiments: **{len(ranked)}**.

- 126 target-screen experiments: multiclass, binary direction, move/no-move, and volatility regime across all requested horizons and thresholds.
- Eight standard model families: logistic regression, SGD linear classifier, ExtraTrees, RandomForest, HistGradientBoosting, XGBoost, LightGBM, and CatBoost.
- Advanced models: SVM, soft voting, and chronological inner-holdout stacking.
- 20 randomized CatBoost tuning trials around the strongest move target.
- 15 Optuna Bayesian CatBoost tuning trials around the strongest move target.
- Four hierarchical move-then-direction probability thresholds.
- Six TSFresh fold-local subset sizes: 50, 100, 250, 500, 1000, and all available features.
- Causal technical, statistical, momentum, volatility, entropy, Hurst, fractal, breakout, and market-structure-derived features.

## Best Configuration

```json
{json.dumps(best_config, indent=2)}
```

| Metric | Value |
| --- | ---: |
| Mean macro F1 | {best["macro_f1_mean"]:.4f} |
| Macro F1 standard deviation | {best["macro_f1_std"]:.4f} |
| Worst fold macro F1 | {best["macro_f1_worst"]:.4f} |
| Best fold macro F1 | {best["macro_f1_best"]:.4f} |
| Mean balanced accuracy | {best["balanced_accuracy_mean"]:.4f} |
| Mean accuracy | {best["accuracy_mean"]:.4f} |

Daily block bootstrap on the combined OOS predictions gives balanced accuracy **{move_significance["balanced_accuracy"]:.4f}**, 95% CI **[{move_significance["ci_low"]:.4f}, {move_significance["ci_high"]:.4f}]**, with empirical probability at or below chance **{move_significance["p_at_or_below_chance"]:.4f}**.

## Target Comparison

{family_table}

The strongest target is move/no-move at 12 candles and 0.5%. Direction is strongest at 12 candles but weak. Multiclass and volatility-regime prediction are substantially worse.

The best hierarchy uses a 0.50 move-probability threshold: macro F1 **{hierarchy["macro_f1_mean"]:.4f}**, balanced accuracy **{hierarchy["balanced_accuracy_mean"]:.4f}**.

## Model Comparison

{advanced_table}

Tuned CatBoost remains best. Soft voting approaches it but does not improve it. SVM and temporal stacking are weaker. Several XGBoost and HistGradientBoosting configurations were rejected for class collapse.

## Feature Findings

{importance_table}

The signal is dominated by realized range, ATR, rolling return variance, long-window volume ratio, entropy, Hurst exponent, and autocorrelation. This is coherent with volatility-expansion prediction. Directional models instead emphasize ROC and momentum, but their edge is too small to trade after fees.

TSFresh does not improve the result. Its best fold-local setting is `{tsfresh["feature_set"]}` with macro F1 **{tsfresh["macro_f1_mean"]:.4f}** and balanced accuracy **{tsfresh["balanced_accuracy_mean"]:.4f}**, below the compact engineered feature set.

## Stability And Economics

- Move signal daily bootstrap: balanced accuracy **{move_significance["balanced_accuracy"]:.4f}**, 95% CI **[{move_significance["ci_low"]:.4f}, {move_significance["ci_high"]:.4f}]**.
- Direction daily bootstrap: balanced accuracy **{direction_significance["balanced_accuracy"]:.4f}**, 95% CI **[{direction_significance["ci_low"]:.4f}, {direction_significance["ci_high"]:.4f}]**.
- Direction net return after 5 bps on non-overlapping forecasts: **{direction_economics["nonoverlap_mean"] * 10000:.2f} bps**, 95% CI **[{direction_economics["ci_low"] * 10000:.2f}, {direction_economics["ci_high"] * 10000:.2f}] bps**.
- Hierarchical net return after 5 bps on non-overlapping forecasts: **{hierarchy_economics["nonoverlap_mean"] * 10000:.2f} bps**, 95% CI **[{hierarchy_economics["ci_low"] * 10000:.2f}, {hierarchy_economics["ci_high"] * 10000:.2f}] bps**.

The move signal is stable enough to justify downstream use as a risk filter, execution-timing input, or volatility-strategy feature. It is not directly monetizable without a separate direction, options, or market-making component.

## Leakage Audit

- Existing TSFresh windows end at the prediction timestamp and are causal.
- Existing saved TSFresh-selected columns were selected against the full labeled matrix; they were excluded from clean evidence.
- Existing `Model/features_data/market_structure.py` swing and rejection routines inspect future candles; they were excluded from predictive experiments.
- New research features use only current and prior candle information.

## Recommendations

1. Treat move detection as the validated research product. Do not deploy the current directional strategy.
2. Test the move detector on untouched later data and additional symbols before production use.
3. Evaluate monetization through volatility-aware execution, position sizing, options, or market-making logic.
4. Add order-book, spread, funding, open-interest, and cross-asset features if available; these are more plausible sources of incremental direction signal.
5. Keep TSFresh optional. Its compute cost is not justified by the observed result.

## Top 50 Configurations

The complete machine-readable table is also stored at `reports/top_50_configurations.csv`.

{top_table}
"""
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
