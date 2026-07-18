"""Simulated virtual portfolio for Perry paper experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4


Side = Literal["BUY", "SELL"]


@dataclass
class Position:
    position_id: str
    prediction_id: str
    asset: str
    side: Side
    entry_time: str
    entry_price: float
    quantity: float
    stop_price: float
    target_price: float
    entry_reason: str
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    latest_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    position_id: str
    prediction_id: str
    asset: str
    side: Side
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_price: float
    target_price: float
    exit_reason: str
    holding_duration_seconds: int
    max_favorable_excursion: float
    max_adverse_excursion: float
    realized_pnl: float
    return_pct: float
    entry_reason: str


@dataclass
class VirtualPortfolio:
    starting_capital: float = 100_000.0
    risk_fraction: float = 0.01
    max_exposure_fraction: float = 0.35
    cash: float | None = None
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[ClosedTrade] = field(default_factory=list)
    equity_high_watermark: float | None = None

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.starting_capital
        if self.equity_high_watermark is None:
            self.equity_high_watermark = self.starting_capital

    @property
    def realized_pnl(self) -> float:
        return sum(trade.realized_pnl for trade in self.closed_positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.open_positions)

    @property
    def equity(self) -> float:
        return float(self.cash or 0.0) + self.unrealized_pnl

    @property
    def exposure(self) -> float:
        return sum(abs(position.quantity * position.latest_price) for position in self.open_positions)

    @property
    def drawdown(self) -> float:
        self.equity_high_watermark = max(float(self.equity_high_watermark or 0.0), self.equity)
        if not self.equity_high_watermark:
            return 0.0
        return min(0.0, (self.equity - self.equity_high_watermark) / self.equity_high_watermark)

    def open_position(
        self,
        *,
        prediction_id: str,
        asset: str,
        side: Side,
        timestamp: str,
        price: float,
        stop_price: float,
        target_price: float,
        entry_reason: str,
    ) -> Position | None:
        stop_distance = abs(price - stop_price)
        if stop_distance <= 0:
            return None

        max_notional = self.equity * self.max_exposure_fraction
        risk_notional = self.equity * self.risk_fraction
        quantity = min(risk_notional / stop_distance, max_notional / max(price, 1e-9))
        if quantity <= 0:
            return None

        position = Position(
            position_id=str(uuid4()),
            prediction_id=prediction_id,
            asset=asset,
            side=side,
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            entry_reason=entry_reason,
            latest_price=price,
        )
        self.open_positions.append(position)
        return position

    def update(self, *, timestamp: str, price: float) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        survivors: list[Position] = []
        for position in self.open_positions:
            pnl = _position_pnl(position.side, position.entry_price, price, position.quantity)
            position.latest_price = price
            position.unrealized_pnl = pnl
            position.max_favorable_excursion = max(position.max_favorable_excursion, pnl)
            position.max_adverse_excursion = min(position.max_adverse_excursion, pnl)

            exit_reason = _exit_reason(position.side, price, position.stop_price, position.target_price)
            if exit_reason is None:
                survivors.append(position)
                continue

            trade = self._close_position(position, timestamp, price, exit_reason)
            closed.append(trade)

        self.open_positions = survivors
        return closed

    def snapshot(self, timestamp: str) -> dict[str, float | int | str]:
        return {
            "timestamp": timestamp,
            "starting_capital": self.starting_capital,
            "cash": float(self.cash or 0.0),
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "drawdown": self.drawdown,
            "exposure": self.exposure,
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
        }

    def trades_as_dicts(self) -> list[dict[str, object]]:
        return [asdict(trade) for trade in self.closed_positions]

    def _close_position(
        self,
        position: Position,
        timestamp: str,
        exit_price: float,
        exit_reason: str,
    ) -> ClosedTrade:
        pnl = _position_pnl(position.side, position.entry_price, exit_price, position.quantity)
        self.cash = float(self.cash or 0.0) + pnl
        duration = _duration_seconds(position.entry_time, timestamp)
        denominator = max(abs(position.entry_price), 1e-9)
        signed_move = (exit_price - position.entry_price) / denominator
        if position.side == "SELL":
            signed_move *= -1.0

        trade = ClosedTrade(
            position_id=position.position_id,
            prediction_id=position.prediction_id,
            asset=position.asset,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            stop_price=position.stop_price,
            target_price=position.target_price,
            exit_reason=exit_reason,
            holding_duration_seconds=duration,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            realized_pnl=pnl,
            return_pct=signed_move * 100.0,
            entry_reason=position.entry_reason,
        )
        self.closed_positions.append(trade)
        return trade


def _position_pnl(side: Side, entry_price: float, current_price: float, quantity: float) -> float:
    if side == "BUY":
        return (current_price - entry_price) * quantity
    return (entry_price - current_price) * quantity


def _exit_reason(side: Side, price: float, stop_price: float, target_price: float) -> str | None:
    if side == "BUY":
        if price <= stop_price:
            return "stop"
        if price >= target_price:
            return "target"
    else:
        if price >= stop_price:
            return "stop"
        if price <= target_price:
            return "target"
    return None


def _duration_seconds(start: str, end: str) -> int:
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return max(0, int((end_dt - start_dt).total_seconds()))
