"""Experiment statistics for the paper trading laboratory."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any


CONFIDENCE_BUCKETS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


def bucket_confidence(value: int) -> str:
    for low, high in CONFIDENCE_BUCKETS:
        if low <= value < high:
            return f"{low}-{high if high <= 100 else 100}%"
    if value < 50:
        return "<50%"
    return "90-100%"


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [trade for trade in trades if float(trade.get("realized_pnl", 0.0)) > 0.0]
    losses = [trade for trade in trades if float(trade.get("realized_pnl", 0.0)) < 0.0]
    returns = [float(trade.get("return_pct", 0.0)) for trade in trades]
    gross_profit = sum(float(trade.get("realized_pnl", 0.0)) for trade in wins)
    gross_loss = abs(sum(float(trade.get("realized_pnl", 0.0)) for trade in losses))

    return {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "average_trade": mean([float(trade.get("realized_pnl", 0.0)) for trade in trades]) if trades else 0.0,
        "largest_win": max([float(trade.get("realized_pnl", 0.0)) for trade in trades], default=0.0),
        "largest_loss": min([float(trade.get("realized_pnl", 0.0)) for trade in trades], default=0.0),
        "sharpe_ratio": _sharpe(returns),
        "average_holding_seconds": mean([float(trade.get("holding_duration_seconds", 0.0)) for trade in trades]) if trades else 0.0,
    }


def summarize_by_state(predictions: list[dict[str, Any]], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prediction = {row["prediction_id"]: row for row in predictions}
    state_trades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_conf: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        state_conf[str(row.get("market_state", "Unknown"))].append(float(row.get("confidence_numeric", 0.0)))
    for trade in trades:
        prediction = by_prediction.get(trade.get("prediction_id"))
        state = str(prediction.get("market_state", "Unknown")) if prediction else "Unknown"
        state_trades[state].append(trade)

    states = sorted(set(state_conf) | set(state_trades))
    rows = []
    for state in states:
        summary = summarize_trades(state_trades[state])
        rows.append(
            {
                "market_state": state,
                "prediction_count": len(state_conf[state]),
                "trade_count": summary["trade_count"],
                "win_rate": summary["win_rate"],
                "average_return_pct": mean([float(t.get("return_pct", 0.0)) for t in state_trades[state]]) if state_trades[state] else 0.0,
                "average_holding_seconds": summary["average_holding_seconds"],
                "average_confidence": mean(state_conf[state]) if state_conf[state] else 0.0,
                "profit_factor": summary["profit_factor"],
                "maximum_drawdown": min([float(t.get("max_adverse_excursion", 0.0)) for t in state_trades[state]], default=0.0),
                "sharpe_ratio": summary["sharpe_ratio"],
            }
        )
    return rows


def summarize_confidence_calibration(predictions: list[dict[str, Any]], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traded_prediction_ids = {trade.get("prediction_id") for trade in trades}
    wins = {trade.get("prediction_id") for trade in trades if float(trade.get("realized_pnl", 0.0)) > 0.0}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        bucket = bucket_confidence(int(row.get("confidence_numeric", 0)))
        buckets[bucket].append(row)

    rows = []
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket]
        executed = [row for row in bucket_rows if row.get("prediction_id") in traded_prediction_ids]
        rows.append(
            {
                "confidence_bucket": bucket,
                "prediction_count": len(bucket_rows),
                "trade_count": len(executed),
                "hit_rate": len([row for row in executed if row.get("prediction_id") in wins]) / len(executed) if executed else 0.0,
                "average_evidence_score": mean([float(row.get("evidence_score", 0.0)) for row in bucket_rows]) if bucket_rows else 0.0,
            }
        )
    return rows


def driver_counts(predictions: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in predictions:
        for driver in str(row.get("top_drivers", "")).split("|"):
            driver = driver.strip()
            if driver:
                counter[driver] += 1
    return counter.most_common()


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sigma = pstdev(returns)
    if sigma == 0:
        return 0.0
    return mean(returns) / sigma
