"""Evidence-based paper trading decision layer.

This module deliberately keeps decision logic separate from Perry's prediction
logic. Perry produces observations; the paper laboratory decides whether an
observation is experimentally actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DecisionSide = Literal["BUY", "SELL", "NO TRADE"]


@dataclass(frozen=True)
class TradeDecision:
    decision: DecisionSide
    evidence_score: float
    reasons: list[str]


def _confidence_points(confidence: str) -> float:
    normalized = confidence.lower()
    if normalized == "high":
        return 1.0
    if normalized == "moderate":
        return 0.55
    return 0.0


def _confluence_points(confluence: str) -> float:
    normalized = confluence.lower()
    if normalized == "high":
        return 1.0
    if normalized == "moderate":
        return 0.55
    return 0.15


def _state_points(market_state: str) -> float:
    if market_state in {"Compression", "Directional Expansion"}:
        return 1.0
    if market_state == "Transition":
        return 0.45
    return 0.25


def make_trade_decision(
    *,
    market_state: str,
    directional_bias: str,
    bullish_score: int,
    bearish_score: int,
    confidence: str,
    confluence: str,
    move_probability: int,
    position: str,
    minimum_evidence: float = 0.62,
) -> TradeDecision:
    """Convert Perry's multi-source evidence into BUY/SELL/NO TRADE."""
    bias_edge = abs(bullish_score - bearish_score) / 100.0
    probability_points = max(0.0, min(1.0, (move_probability - 50.0) / 35.0))
    evidence_score = (
        0.28 * probability_points
        + 0.24 * _confidence_points(confidence)
        + 0.20 * bias_edge
        + 0.16 * _confluence_points(confluence)
        + 0.12 * _state_points(market_state)
    )

    reasons: list[str] = [
        f"{market_state} state",
        f"{confidence.lower()} confidence",
        f"{confluence.lower()} confluence",
        f"{move_probability}% move probability",
        f"{position.lower()}",
    ]

    if evidence_score < minimum_evidence:
        return TradeDecision("NO TRADE", round(evidence_score, 4), reasons + ["evidence below execution threshold"])
    if directional_bias == "Bullish" and bullish_score >= bearish_score + 6:
        return TradeDecision("BUY", round(evidence_score, 4), reasons + ["bullish evidence dominates"])
    if directional_bias == "Bearish" and bearish_score >= bullish_score + 6:
        return TradeDecision("SELL", round(evidence_score, 4), reasons + ["bearish evidence dominates"])
    return TradeDecision("NO TRADE", round(evidence_score, 4), reasons + ["directional edge insufficient"])
