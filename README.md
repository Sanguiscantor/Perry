# Perry

Perry is a modular market intelligence laboratory. Phase 1 established live
market ingestion, runtime feature generation, market-state estimation,
directional bias, expected move estimation, and a prototype report from the
single entry point:

```bash
python app.py
```

## Phase 2: Paper Trading Laboratory

Phase 2 adds a live paper trading laboratory for scientific observation. It
does not train from PnL, alter historical results, call broker APIs, or perform
reinforcement learning. Every prediction is recorded as an experiment, including
`NO TRADE` decisions.

Run one paper laboratory cycle:

```bash
python app.py --paper-lab
```

Run multiple observation cycles:

```bash
python app.py --paper-lab --cycles 12 --interval-seconds 300
```

Useful options:

- `--asset BTCUSDT`: asset symbol to evaluate.
- `--starting-capital 100000`: virtual portfolio starting capital.
- `--minimum-evidence 0.62`: evidence threshold before a virtual trade is opened.

Each run writes a new timestamped experiment directory:

```text
artifacts/paper_trading/<timestamp>/
    predictions.csv
    trades.csv
    portfolio.csv
    state_transitions.csv
    runtime.json
    dashboard.json
    conclusion.md
```

## Architecture

- `app.py`: single application entry point, data refresh, feature generation,
  prediction assembly, prototype report, and Phase 2 command-line wiring.
- `Model/paper_trading/decision.py`: multi-evidence BUY/SELL/NO TRADE layer.
- `Model/paper_trading/portfolio.py`: fully simulated virtual portfolio.
- `Model/paper_trading/laboratory.py`: experiment orchestration and artifacts.
- `Model/paper_trading/statistics.py`: calibration, state performance, and trade
  summary helpers.

## Testing

```bash
python -m pytest
```
