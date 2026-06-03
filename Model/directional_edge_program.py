"""
Perry directional-edge discovery program.

Priority: acquire free non-OHLCV data, build causal features, walk-forward direction + economics.
Updates research memory automatically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "Model") not in sys.path:
    sys.path.insert(0, str(ROOT / "Model"))

from alternative_features import build_all_alternative, build_funding_oi_features, build_sentiment_features, build_taker_flow_features
from enhanced_features import build_cross_asset_features, merge_features
from research_engine import ExperimentSpec, build_causal_features, make_target
from research_memory import record_finding, update_hypothesis, update_roadmap
from research_program import (
    ARTIFACTS,
    BEST_SPEC,
    RNG,
    evaluate_move_signal,
    load_multi_asset,
    log,
    save_json,
    slice_symbol,
)

DERIV_DIR = ROOT / "Data" / "datasets" / "raw" / "derivatives"
FEE_BPS = 5.0
HORIZON = 12
ARTIFACT = "directional_edge_evaluation"
RANKING_ARTIFACT = "information_source_ranking"

# Pre-registered viability
VIABLE_BALANCED = 0.52
VIABLE_NET_BPS = 0.0


SOURCE_CANDIDATES = [
    {
        "id": "taker_flow_klines",
        "name": "Taker buy/sell flow (futures klines)",
        "info_gain": 9,
        "history": 10,
        "cost": 10,
        "acquisition": 9,
        "direction_prob": 8,
        "notes": "Full history since 2024; true microstructure proxy",
    },
    {
        "id": "taker_agg_ratio",
        "name": "Aggregated taker buy/sell ratio (Binance)",
        "info_gain": 8,
        "history": 4,
        "cost": 10,
        "acquisition": 8,
        "direction_prob": 7,
        "notes": "~30-90d paginated 15m",
    },
    {
        "id": "global_long_short",
        "name": "Global long/short account ratio",
        "info_gain": 7,
        "history": 4,
        "cost": 10,
        "acquisition": 8,
        "direction_prob": 6,
        "notes": "Crowding; contrarian potential",
    },
    {
        "id": "top_trader_ls",
        "name": "Top trader position L/S ratio",
        "info_gain": 8,
        "history": 4,
        "cost": 10,
        "acquisition": 8,
        "direction_prob": 7,
        "notes": "Smart-money positioning proxy",
    },
    {
        "id": "open_interest",
        "name": "Open interest changes",
        "info_gain": 8,
        "history": 4,
        "cost": 10,
        "acquisition": 7,
        "direction_prob": 6,
        "notes": "Paginated; positioning not direction alone",
    },
    {
        "id": "funding_rate",
        "name": "Funding rate / carry",
        "info_gain": 7,
        "history": 9,
        "cost": 10,
        "acquisition": 9,
        "direction_prob": 5,
        "notes": "Full history; already tested weak for move",
    },
    {
        "id": "fear_greed",
        "name": "Crypto Fear & Greed Index",
        "info_gain": 5,
        "history": 7,
        "cost": 10,
        "acquisition": 10,
        "direction_prob": 4,
        "notes": "Daily only; alternative.me",
    },
    {
        "id": "cross_asset",
        "name": "Cross-asset lead-lag",
        "info_gain": 6,
        "history": 10,
        "cost": 10,
        "acquisition": 10,
        "direction_prob": 4,
        "notes": "Redundant for move; test direction",
    },
    {
        "id": "liquidations",
        "name": "Liquidation events",
        "info_gain": 9,
        "history": 2,
        "cost": 3,
        "acquisition": 2,
        "direction_prob": 8,
        "notes": "No free long history on Binance REST",
    },
    {
        "id": "l2_order_book",
        "name": "L2 order book imbalance",
        "info_gain": 10,
        "history": 1,
        "cost": 2,
        "acquisition": 1,
        "direction_prob": 9,
        "notes": "Requires paid/tick archive",
    },
    {
        "id": "options_iv",
        "name": "Options IV / skew (Deribit)",
        "info_gain": 9,
        "history": 5,
        "cost": 4,
        "acquisition": 3,
        "direction_prob": 7,
        "notes": "Not integrated this run",
    },
    {
        "id": "etf_flows",
        "name": "BTC ETF flows",
        "info_gain": 7,
        "history": 6,
        "cost": 5,
        "acquisition": 4,
        "direction_prob": 5,
        "notes": "Daily fund data; delayed",
    },
    {
        "id": "on_chain",
        "name": "On-chain exchange flows",
        "info_gain": 8,
        "history": 6,
        "cost": 4,
        "acquisition": 4,
        "direction_prob": 6,
        "notes": "Glassnode etc. paid",
    },
]


def rank_sources() -> list[dict]:
    ranked = []
    for s in SOURCE_CANDIDATES:
        composite = (
            s["info_gain"] * 0.30
            + s["direction_prob"] * 0.30
            + s["history"] * 0.15
            + s["cost"] * 0.10
            + s["acquisition"] * 0.15
        )
        ranked.append({**s, "composite": round(composite, 3)})
    return sorted(ranked, key=lambda x: x["composite"], reverse=True)


def economic_summary(predictions: pd.DataFrame, horizon: int = HORIZON) -> dict[str, float]:
    position = np.where(predictions["predicted"] == 1, 1.0, -1.0)
    fee = FEE_BPS / 10000
    net = position * predictions["future_return"].to_numpy() - fee
    independent = net[::horizon]
    boot = RNG.choice(independent, size=(5000, max(len(independent), 1)), replace=True).mean(axis=1)
    return {
        "mean_net_return_bps": float(net.mean() * 10000),
        "nonoverlap_mean_bps": float(independent.mean() * 10000),
        "ci_low_bps": float(np.quantile(boot, 0.025) * 10000),
        "ci_high_bps": float(np.quantile(boot, 0.975) * 10000),
        "p_positive": float(np.mean(net > 0)),
        "p_nonoverlap_positive": float(np.mean(independent > 0)),
        "trade_count": int(len(net)),
    }


def evaluate_direction(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    model: str = "logistic",
    feature_set: str = "core",
) -> dict[str, Any]:
    spec = ExperimentSpec("direction", HORIZON, None, feature_set, model, notes="directional edge program")
    result = evaluate_move_signal(raw, features, spec)
    if result.get("error"):
        return result
    pred = result["predictions"]
    econ = economic_summary(pred)
    viable = (
        econ["nonoverlap_mean_bps"] > VIABLE_NET_BPS
        and econ["ci_low_bps"] > VIABLE_NET_BPS
        and result["balanced_accuracy_mean"] >= VIABLE_BALANCED
    )
    return {**{k: v for k, v in result.items() if k != "predictions"}, **econ, "economically_viable": viable}


def evaluate_move_gated_direction(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    move_threshold: float = 0.55,
) -> dict[str, Any]:
    """Direction only when move model probability exceeds threshold."""
    from research_program import chronological_splits, model_factory, select_columns

    target_m, future_m = make_target(raw, "move", HORIZON, 0.005)
    target_d, _ = make_target(raw, "direction", HORIZON, None)
    valid = target_m.notna() & features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = features.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
    y_move = (target_m.loc[valid] == 1).astype(int).reset_index(drop=True)
    y_dir = target_d.loc[valid].astype(int).reset_index(drop=True)
    returns = future_m.loc[valid].reset_index(drop=True)
    ts = features.loc[valid, "Datetime"].reset_index(drop=True)
    trades = []

    for _fold, train_index, test_index in chronological_splits(len(X), BEST_SPEC):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Xm_tr, Xm_te, _ = select_columns(X_train, y_move.iloc[train_index], X_test, "top_100")
        move_model = model_factory("catboost", BEST_SPEC.model_params)
        move_model.fit(Xm_tr, y_move.iloc[train_index])
        mp = move_model.predict_proba(Xm_te)[:, 1]
        move_mask = y_move.iloc[train_index].values == 1
        if move_mask.sum() < 30:
            continue
        Xd_tr, Xd_te, _ = select_columns(
            X_train.loc[move_mask], y_dir.iloc[train_index].loc[move_mask], X_test, "all"
        )
        dir_model = model_factory("logistic")
        dir_model.fit(Xd_tr, y_dir.iloc[train_index].loc[move_mask])
        dp = np.asarray(dir_model.predict(Xd_te)).reshape(-1)
        for i, ti in enumerate(test_index):
            if mp[i] >= move_threshold:
                trades.append(
                    {
                        "Datetime": ts.iloc[ti],
                        "actual": int(y_dir.iloc[ti]),
                        "predicted": int(dp[i]),
                        "future_return": float(returns.iloc[ti]),
                    }
                )

    if len(trades) < 100:
        return {"error": "insufficient_trades", "n": len(trades)}
    pdf = pd.DataFrame(trades)
    from sklearn.metrics import balanced_accuracy_score, f1_score

    econ = economic_summary(pdf)
    return {
        "n_trades": len(pdf),
        "balanced_accuracy": float(balanced_accuracy_score(pdf["actual"], pdf["predicted"])),
        "macro_f1": float(f1_score(pdf["actual"], pdf["predicted"], average="macro")),
        **econ,
        "economically_viable": econ["nonoverlap_mean_bps"] > 0 and econ["ci_low_bps"] > 0,
    }


def data_quality_report() -> dict[str, Any]:
    report: dict[str, Any] = {"sources": [], "failures": []}
    checks = [
        ("funding_rates.csv", DERIV_DIR),
        ("open_interest_15m.csv", DERIV_DIR),
        ("global_long_short_15m.csv", DERIV_DIR),
        ("top_trader_long_short_15m.csv", DERIV_DIR),
        ("taker_buy_sell_15m.csv", DERIV_DIR),
        ("futures_klines_15m.csv", ROOT / "Data" / "datasets" / "raw"),
    ]
    for name, directory in checks:
        path = directory / name
        if not path.exists():
            report["failures"].append({"file": name, "reason": "not downloaded"})
            continue
        df = pd.read_csv(path, parse_dates=["Datetime"] if "Datetime" in pd.read_csv(path, nrows=0).columns else None)
        if "Datetime" not in df.columns:
            report["failures"].append({"file": name, "reason": "no Datetime column"})
            continue
        btc = df[df["symbol"] == "BTCUSDT"] if "symbol" in df.columns else df
        report["sources"].append(
            {
                "file": name,
                "rows": len(df),
                "btc_rows": len(btc),
                "start": str(btc["Datetime"].min()),
                "end": str(btc["Datetime"].max()),
                "null_pct": float(btc.isna().mean().mean()) if len(btc) else 1.0,
            }
        )
    return report


def run_acquisition() -> None:
    log("Directional edge: acquiring datasets")
    scripts = [
        ROOT / "Data" / "ingestion" / "download_derivatives.py",
        ROOT / "Data" / "ingestion" / "download_extended_klines.py",
        ROOT / "Data" / "ingestion" / "download_binance_sentiment.py",
    ]
    for script in scripts:
        log(f"Running {script.name}")
        subprocess.run([sys.executable, str(script)], check=False)


def run_evaluations(multi: pd.DataFrame) -> dict[str, Any]:
    btc_raw = slice_symbol(multi, "BTCUSDT")
    base = build_causal_features(btc_raw)
    cross = build_cross_asset_features(multi)
    taker = build_taker_flow_features("BTCUSDT")
    sentiment = build_sentiment_features("BTCUSDT")
    funding = build_funding_oi_features("BTCUSDT")
    alt_all = build_all_alternative("BTCUSDT")

    configs: dict[str, pd.DataFrame | None] = {
        "baseline_core": merge_features(base),
        "ohlcv_all": merge_features(base),  # feature_set=all in eval
        "taker_flow": merge_features(base, deriv=taker) if taker is not None else None,
        "sentiment_only": merge_features(base, deriv=sentiment) if sentiment is not None else None,
        "funding_oi": merge_features(base, deriv=funding) if funding is not None else None,
        "cross_asset": merge_features(base, cross=cross),
        "all_alternative": merge_features(base, cross=cross, deriv=alt_all) if alt_all is not None else None,
    }

    results: dict[str, Any] = {}
    for name, feat in configs.items():
        if feat is None:
            results[name] = {"skipped": True, "reason": "data unavailable"}
            log(f"  skip {name}")
            continue
        fset = "all" if name in {"ohlcv_all", "taker_flow", "sentiment_only", "funding_oi", "all_alternative"} else "core"
        log(f"  evaluate direction: {name}")
        results[name] = evaluate_direction(btc_raw, feat, feature_set=fset)

    results["move_gated_all_features"] = evaluate_move_gated_direction(
        btc_raw, merge_features(base, cross=cross, deriv=alt_all) if alt_all is not None else merge_features(base)
    )
    results["move_only_baseline"] = evaluate_move_signal(btc_raw, base)

    save_json(ARTIFACT, results)
    return results


def update_memory_from_results(results: dict[str, Any], ranking: list[dict], quality: dict) -> None:
    best_name, best_score = None, -999.0
    for name, m in results.items():
        if not isinstance(m, dict) or "balanced_accuracy_mean" not in m and "balanced_accuracy" not in m:
            continue
        ba = m.get("balanced_accuracy_mean", m.get("balanced_accuracy", 0))
        net = m.get("nonoverlap_mean_bps", -999)
        score = float(ba) + (0.001 if net and net > 0 else 0)
        if score > best_score:
            best_name, best_score = name, score

    any_viable = any(
        isinstance(m, dict) and m.get("economically_viable") for m in results.values()
    )

    if any_viable:
        record_finding(
            "CONFIRMED",
            "Economically viable directional edge discovered with alternative data.",
            "Medium",
            str({k: v.get("nonoverlap_mean_bps") for k, v in results.items() if isinstance(v, dict) and "nonoverlap_mean_bps" in v}),
            "directional_edge_report.md",
        )
        update_hypothesis("H003", "PARTIAL EVIDENCE", "Direction may be viable with new microstructure features.", str(best_name), "directional_edge_report.md")
    else:
        record_finding(
            "REJECTED",
            "No alternative free data source produced economically viable directional edge after fees.",
            "High",
            f"Best config {best_name}; all ci_low_bps <= 0",
            "directional_edge_report.md",
        )

    for src in ranking[:5]:
        if src["id"] in ("liquidations", "l2_order_book", "options_iv"):
            record_finding(
                "UNTESTED",
                f"{src['name']} remains unavailable on free tier.",
                "High",
                src["notes"],
                "directional_edge_report.md",
            )

    update_roadmap(
        current_frontier="Acquire liquidation/L2 history or deploy move-score vol product.",
        highest_value_unknowns=[
            "Paid liquidation feed directional lift",
            "L2 imbalance at 15m aggregation",
            "Perp-spot basis intraday",
        ],
        recommended_experiments=[
            "Coinglass liquidation CSV merge",
            "Deribit options skew panel",
            "Move-score vol backtest (non-directional monetization)",
        ],
    )


def generate_report(results: dict[str, Any], ranking: list[dict], quality: dict) -> None:
    lines = [
        "# Directional Edge Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 1. Which information sources helped?",
        "",
    ]
    helped = []
    for name, m in sorted(results.items()):
        if not isinstance(m, dict) or m.get("skipped"):
            continue
        net = m.get("nonoverlap_mean_bps", m.get("mean_net_return_bps"))
        ba = m.get("balanced_accuracy_mean", m.get("balanced_accuracy"))
        if net is None:
            continue
        lift_ba = ba - results.get("baseline_core", {}).get("balanced_accuracy_mean", 0.525) if ba else 0
        if lift_ba > 0.002 or (net and net > -3):
            helped.append((name, ba, net, m.get("economically_viable", False)))
    if helped:
        lines.append("| Source | Balanced acc | Net bps (non-overlap) | Viable |")
        lines.append("| --- | ---: | ---: | --- |")
        for name, ba, net, viable in sorted(helped, key=lambda x: x[2] or -999, reverse=True):
            lines.append(f"| {name} | {ba:.4f} | {net:.2f} | {viable} |")
    else:
        lines.append("_None improved economics materially._")
    lines.extend(["", "## 2. Which failed?", ""])
    for name, m in results.items():
        if m.get("skipped"):
            lines.append(f"- **{name}:** skipped — {m.get('reason', '')}")
        elif isinstance(m, dict) and m.get("nonoverlap_mean_bps", 0) < -2:
            lines.append(
                f"- **{name}:** net **{m.get('nonoverlap_mean_bps', 0):.2f}** bps, "
                f"balanced **{m.get('balanced_accuracy_mean', m.get('balanced_accuracy', 0)):.4f}**"
            )
    lines.extend(["", "## Data quality", "", "| File | BTC rows | Start | End |", "| --- | ---: | --- | --- |"])
    for s in quality.get("sources", []):
        lines.append(f"| {s['file']} | {s['btc_rows']} | {s['start']} | {s['end']} |")
    lines.extend(["", "## 3. Strongest directional signal?", ""])
    best_dir = max(
        ((n, m) for n, m in results.items() if isinstance(m, dict) and "balanced_accuracy_mean" in m),
        key=lambda x: (x[1].get("nonoverlap_mean_bps", -999), x[1]["balanced_accuracy_mean"]),
        default=(None, {}),
    )
    if best_dir[0]:
        m = best_dir[1]
        lines.append(
            f"**{best_dir[0]}** — balanced acc **{m['balanced_accuracy_mean']:.4f}**, "
            f"non-overlap net **{m.get('nonoverlap_mean_bps', 0):.2f}** bps, "
            f"CI **[{m.get('ci_low_bps', 0):.2f}, {m.get('ci_high_bps', 0):.2f}]**"
        )
    lines.extend(
        [
            "",
            "## 4. Best trading architecture?",
            "",
            "**Move-gated direction** (CatBoost move → logistic direction on move-conditioned train) "
            "or **taker-flow-augmented logistic** if lift positive; otherwise **no directional deployment** — use move score as vol filter only.",
            "",
            "## 5. What to research next?",
            "",
            "1. Paid liquidation + L2 archives. 2. Deribit IV skew. 3. Vol-product backtest on move score.",
            "",
            "## Source ranking (pre-acquisition)",
            "",
            "| Rank | Source | Composite | Direction prob |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for i, s in enumerate(ranking[:10], 1):
        lines.append(f"| {i} | {s['name']} | {s['composite']} | {s['direction_prob']} |")
    lines.append("")
    (ROOT / "directional_edge_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    log("Directional edge program started")
    ranking = rank_sources()
    save_json(RANKING_ARTIFACT, {"ranked": ranking})

    run_acquisition()
    quality = data_quality_report()
    save_json("directional_data_quality", quality)

    multi = load_multi_asset()
    results = run_evaluations(multi)
    update_memory_from_results(results, ranking, quality)
    generate_report(results, ranking, quality)
    log("Directional edge program completed")


if __name__ == "__main__":
    main()
