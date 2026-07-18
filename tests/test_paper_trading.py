from types import SimpleNamespace

from Model.paper_trading.decision import make_trade_decision
from Model.paper_trading.laboratory import PaperTradingConfig, PaperTradingLaboratory
from Model.paper_trading.portfolio import VirtualPortfolio


def test_decision_requires_multiple_sources_of_evidence():
    decision = make_trade_decision(
        market_state="Directional Expansion",
        directional_bias="Bullish",
        bullish_score=72,
        bearish_score=28,
        confidence="High",
        confluence="High",
        move_probability=78,
        position="Lower Structure",
    )

    assert decision.decision == "BUY"
    assert decision.evidence_score >= 0.62


def test_decision_logs_no_trade_when_evidence_is_weak():
    decision = make_trade_decision(
        market_state="Balance",
        directional_bias="Neutral",
        bullish_score=51,
        bearish_score=49,
        confidence="Low",
        confluence="Low",
        move_probability=54,
        position="Equilibrium",
    )

    assert decision.decision == "NO TRADE"


def test_virtual_portfolio_closes_long_at_target():
    portfolio = VirtualPortfolio(starting_capital=10_000)
    position = portfolio.open_position(
        prediction_id="p1",
        asset="BTCUSDT",
        side="BUY",
        timestamp="2026-07-04T00:00:00+00:00",
        price=100.0,
        stop_price=95.0,
        target_price=110.0,
        entry_reason="test",
    )

    closed = portfolio.update(timestamp="2026-07-04T01:00:00+00:00", price=111.0)

    assert position is not None
    assert len(closed) == 1
    assert closed[0].exit_reason == "target"
    assert closed[0].realized_pnl > 0
    assert not portfolio.open_positions


def test_laboratory_writes_structured_artifacts(tmp_path):
    result = SimpleNamespace(
        asset="BTCUSDT",
        market_state="Directional Expansion",
        position="Lower Structure",
        confluence="High",
        move_probability=82,
        directional_bias="Bullish",
        bullish_score=75,
        bearish_score=25,
        directional_confidence="High",
        expected_move_low=1.0,
        expected_move_high=2.0,
        top_drivers=["Compression Energy", "Trend Geometry"],
        interpretation=["Large move likely.", "Current state favors upside."],
        diagnostics={
            "latest_close": 100.0,
            "market_timestamp_freshness": "fresh",
            "state_feature_status": "built",
            "model_status": "heuristic",
        },
    )
    config = PaperTradingConfig(cycles=1, artifacts_root=tmp_path)
    laboratory = PaperTradingLaboratory(config, result_builder=lambda: result)

    experiment_dir = laboratory.run()

    assert (experiment_dir / "predictions.csv").exists()
    assert (experiment_dir / "trades.csv").exists()
    assert (experiment_dir / "portfolio.csv").exists()
    assert (experiment_dir / "state_transitions.csv").exists()
    assert (experiment_dir / "runtime.json").exists()
    assert (experiment_dir / "dashboard.json").exists()
    assert (experiment_dir / "conclusion.md").exists()
