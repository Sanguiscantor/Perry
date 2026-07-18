"""Scientific live paper trading laboratory for Perry."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from .decision import make_trade_decision
from .portfolio import VirtualPortfolio
from .statistics import driver_counts, summarize_by_state, summarize_confidence_calibration, summarize_trades


PerryResultBuilder = Callable[[], Any]


@dataclass(frozen=True)
class PaperTradingConfig:
    asset: str = "BTCUSDT"
    starting_capital: float = 100_000.0
    cycles: int = 1
    interval_seconds: float = 0.0
    minimum_evidence: float = 0.62
    artifacts_root: Path = Path("artifacts") / "paper_trading"


class PaperTradingLaboratory:
    """Run reproducible paper trading experiments from Perry observations."""

    def __init__(self, config: PaperTradingConfig, result_builder: PerryResultBuilder):
        self.config = config
        self.result_builder = result_builder
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_dir = config.artifacts_root / timestamp
        self.portfolio = VirtualPortfolio(starting_capital=config.starting_capital)
        self.predictions: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.portfolio_snapshots: list[dict[str, Any]] = []
        self.state_transitions: list[dict[str, Any]] = []
        self.previous_state: str | None = None
        self.previous_state_timestamp: str | None = None

    def run(self) -> Path:
        self.experiment_dir.mkdir(parents=True, exist_ok=False)
        started_at = datetime.now(timezone.utc).isoformat()
        runtime = {
            "experiment_id": self.experiment_dir.name,
            "started_at": started_at,
            "asset": self.config.asset,
            "starting_capital": self.config.starting_capital,
            "cycles_requested": self.config.cycles,
            "interval_seconds": self.config.interval_seconds,
            "minimum_evidence": self.config.minimum_evidence,
            "status": "running",
        }
        self._write_json("runtime.json", runtime)

        for cycle in range(self.config.cycles):
            result = self.result_builder()
            self.record_cycle(result)
            self.write_artifacts()
            if cycle < self.config.cycles - 1 and self.config.interval_seconds > 0:
                time.sleep(self.config.interval_seconds)

        runtime["ended_at"] = datetime.now(timezone.utc).isoformat()
        runtime["cycles_completed"] = len(self.predictions)
        runtime["status"] = "complete"
        self._write_json("runtime.json", runtime)
        self.write_artifacts()
        return self.experiment_dir

    def record_cycle(self, result: Any) -> None:
        prediction_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        price = float(result.diagnostics.get("latest_close", 0.0))
        confidence_numeric = _confidence_numeric(result.directional_confidence, result.move_probability)

        closed = self.portfolio.update(timestamp=timestamp, price=price)
        if closed:
            self.trades.extend([trade.__dict__.copy() for trade in closed])

        decision = make_trade_decision(
            market_state=result.market_state,
            directional_bias=result.directional_bias,
            bullish_score=result.bullish_score,
            bearish_score=result.bearish_score,
            confidence=result.directional_confidence,
            confluence=result.confluence,
            move_probability=result.move_probability,
            position=result.position,
            minimum_evidence=self.config.minimum_evidence,
        )

        trade_fields: dict[str, Any] = {
            "entry": "",
            "stop": "",
            "target": "",
            "position_size": "",
            "exit": "",
            "exit_reason": "",
            "holding_duration": "",
            "maximum_favorable_excursion": "",
            "maximum_adverse_excursion": "",
            "realized_pnl": "",
            "unrealized_pnl": "",
        }

        if decision.decision in {"BUY", "SELL"} and not self.portfolio.open_positions:
            stop, target = _risk_levels(price, decision.decision, result.expected_move_low, result.expected_move_high)
            position = self.portfolio.open_position(
                prediction_id=prediction_id,
                asset=result.asset,
                side=decision.decision,
                timestamp=timestamp,
                price=price,
                stop_price=stop,
                target_price=target,
                entry_reason="; ".join(decision.reasons),
            )
            if position is not None:
                trade_fields.update(
                    {
                        "entry": position.entry_price,
                        "stop": position.stop_price,
                        "target": position.target_price,
                        "position_size": position.quantity,
                        "unrealized_pnl": position.unrealized_pnl,
                    }
                )
            else:
                decision = decision.__class__("NO TRADE", decision.evidence_score, decision.reasons + ["portfolio rejected invalid risk levels"])

        prediction = {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "asset": result.asset,
            "current_price": price,
            "market_state": result.market_state,
            "previous_state": self.previous_state or "",
            "position_in_structure": result.position,
            "directional_bias": result.directional_bias,
            "bullish_score": result.bullish_score,
            "bearish_score": result.bearish_score,
            "confidence": result.directional_confidence,
            "confidence_numeric": confidence_numeric,
            "expected_move_low": result.expected_move_low,
            "expected_move_high": result.expected_move_high,
            "feature_confluence": result.confluence,
            "top_drivers": "|".join(result.top_drivers),
            "decision": decision.decision,
            "evidence_score": decision.evidence_score,
            "decision_reasons": "|".join(decision.reasons),
            "human_readable_reasoning": " ".join(result.interpretation),
            "data_freshness": result.diagnostics.get("market_timestamp_freshness", ""),
            "feature_freshness": result.diagnostics.get("state_feature_status", ""),
            "model_status": result.diagnostics.get("model_status", ""),
            **trade_fields,
        }
        self.predictions.append(prediction)
        self._record_state_transition(result.market_state, timestamp)
        self.portfolio_snapshots.append(self.portfolio.snapshot(timestamp))

    def write_artifacts(self) -> None:
        self._write_csv("predictions.csv", self.predictions)
        self._write_csv("trades.csv", self.trades)
        self._write_csv("portfolio.csv", self.portfolio_snapshots)
        self._write_csv("state_transitions.csv", self.state_transitions)
        self._write_json("dashboard.json", self.dashboard())
        (self.experiment_dir / "conclusion.md").write_text(self.conclusion(), encoding="utf-8")

    def dashboard(self) -> dict[str, Any]:
        latest = self.predictions[-1] if self.predictions else {}
        trade_summary = summarize_trades(self.trades)
        snapshot = self.portfolio_snapshots[-1] if self.portfolio_snapshots else self.portfolio.snapshot(datetime.now(timezone.utc).isoformat())
        return {
            "runtime": {
                "experiment_id": self.experiment_dir.name,
                "cycles_completed": len(self.predictions),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            "capital": snapshot,
            "current_pnl": snapshot.get("realized_pnl", 0.0),
            "open_trades": len(self.portfolio.open_positions),
            "closed_trades": len(self.trades),
            "win_rate": trade_summary["win_rate"],
            "drawdown": snapshot.get("drawdown", 0.0),
            "current_market_state": latest.get("market_state", ""),
            "confidence": latest.get("confidence", ""),
            "directional_bias": latest.get("directional_bias", ""),
            "expected_move": [latest.get("expected_move_low", ""), latest.get("expected_move_high", "")],
            "latest_reasoning": latest.get("human_readable_reasoning", ""),
            "data_freshness": latest.get("data_freshness", ""),
            "feature_freshness": latest.get("feature_freshness", ""),
            "model_status": latest.get("model_status", ""),
            "recent_predictions": self.predictions[-10:],
            "recent_trades": self.trades[-10:],
            "system_health": "ok",
        }

    def conclusion(self) -> str:
        trade_summary = summarize_trades(self.trades)
        state_stats = summarize_by_state(self.predictions, self.trades)
        calibration = summarize_confidence_calibration(self.predictions, self.trades)
        skipped = len([row for row in self.predictions if row.get("decision") == "NO TRADE"])
        latest_snapshot = self.portfolio_snapshots[-1] if self.portfolio_snapshots else self.portfolio.snapshot("")
        net_return = (float(latest_snapshot.get("equity", self.config.starting_capital)) - self.config.starting_capital) / self.config.starting_capital

        state_lines = "\n".join(
            f"- {row['market_state']}: predictions={row['prediction_count']}, trades={row['trade_count']}, "
            f"win_rate={row['win_rate']:.2%}, profit_factor={_format_number(row['profit_factor'])}"
            for row in state_stats
        ) or "- No state observations recorded."
        calibration_lines = "\n".join(
            f"- {row['confidence_bucket']}: predictions={row['prediction_count']}, trades={row['trade_count']}, hit_rate={row['hit_rate']:.2%}"
            for row in calibration
        ) or "- No confidence observations recorded."
        driver_lines = "\n".join(f"- {driver}: {count}" for driver, count in driver_counts(self.predictions)[:8]) or "- No drivers recorded."
        entry_reasons = _common_values(self.predictions, "decision_reasons")
        exit_reasons = _common_values(self.trades, "exit_reason")

        return f"""# Perry Paper Trading Laboratory Report

## Experiment Summary

- Experiment Duration: {len(self.predictions)} cycle(s)
- Asset: {self.config.asset}
- Starting Capital: {self.config.starting_capital:.2f}
- Ending Capital: {float(latest_snapshot.get("equity", self.config.starting_capital)):.2f}
- Net Return: {net_return:.2%}
- Prediction Count: {len(self.predictions)}
- Trade Count: {trade_summary["trade_count"]}
- Trades Skipped: {skipped}
- Win Rate: {trade_summary["win_rate"]:.2%}
- Profit Factor: {_format_number(trade_summary["profit_factor"])}
- Average Trade: {trade_summary["average_trade"]:.2f}
- Largest Win: {trade_summary["largest_win"]:.2f}
- Largest Loss: {trade_summary["largest_loss"]:.2f}
- Maximum Drawdown: {float(latest_snapshot.get("drawdown", 0.0)):.2%}

## Confidence Calibration

{calibration_lines}

## Performance By State

{state_lines}

## Driver Observations

{driver_lines}

## Common Entry Reasons

{entry_reasons}

## Common Exit Reasons

{exit_reasons}

## Objective Observations

{_objective_observations(len(self.predictions), skipped, trade_summary, state_stats)}

## Evidence-Based Recommendations

- Continue collecting observations before changing predictive logic.
- Compare confidence buckets only after each bucket has a meaningful sample size.
- Investigate states with frequent no-trade outcomes separately from losing traded states.
"""

    def _record_state_transition(self, current_state: str, timestamp: str) -> None:
        duration = 0
        if self.previous_state_timestamp:
            duration = _duration_seconds(self.previous_state_timestamp, timestamp)
        transition = {
            "timestamp": timestamp,
            "previous_state": self.previous_state or "",
            "current_state": current_state,
            "next_state": "",
            "transition_duration_seconds": duration,
            "transition_frequency": 1,
            "transition_success_rate": "",
        }
        if self.state_transitions:
            self.state_transitions[-1]["next_state"] = current_state
        self.state_transitions.append(transition)
        self.previous_state = current_state
        self.previous_state_timestamp = timestamp

    def _write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.experiment_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.experiment_dir / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _confidence_numeric(label: str, move_probability: int) -> int:
    if label == "High":
        return max(80, move_probability)
    if label == "Moderate":
        return max(60, min(79, move_probability))
    return max(50, min(59, move_probability))


def _risk_levels(price: float, side: str, low: float, high: float) -> tuple[float, float]:
    expected = max(abs(low), abs(high), 0.3) / 100.0
    target_distance = price * expected
    stop_distance = price * max(expected * 0.55, 0.002)
    if side == "BUY":
        return price - stop_distance, price + target_distance
    return price + stop_distance, price - target_distance


def _duration_seconds(start: str, end: str) -> int:
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(0, int((end_dt - start_dt).total_seconds()))


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == float("inf"):
        return "inf"
    return f"{number:.2f}"


def _common_values(rows: list[dict[str, Any]], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "- None recorded."
    return "\n".join(f"- {value}: {count}" for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8])


def _objective_observations(
    prediction_count: int,
    skipped: int,
    trade_summary: dict[str, Any],
    state_stats: list[dict[str, Any]],
) -> str:
    observations = []
    if prediction_count == 0:
        observations.append("- No observations were collected.")
    elif skipped == prediction_count:
        observations.append("- Perry generated observations, but the decision layer found no setup with enough independent evidence.")
    else:
        observations.append(f"- The laboratory executed {trade_summary['trade_count']} trade(s) from {prediction_count} prediction(s).")
    sparse_states = [row["market_state"] for row in state_stats if row["prediction_count"] < 20]
    if sparse_states:
        observations.append("- State-level statistics remain sample-limited for: " + ", ".join(map(str, sparse_states)) + ".")
    if not observations:
        observations.append("- No notable statistical pattern is available yet.")
    return "\n".join(observations)
