# Information Catalog — Data Sources

**Generated:** 2026-06-03

## Local data acquisition scripts

- `Data/download_binance_sentiment.py`: paginates Binance FAPI metrics (open interest, global/top long-short ratios, taker buy/sell). Outputs to `Data/derivatives/`:
  - `open_interest_15m.csv`
  - `global_long_short_15m.csv`
  - `top_trader_long_short_15m.csv`
  - `taker_buy_sell_15m.csv`
  - Key config: `SYMBOLS = [BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT]`, `PERIOD=15m`, `START_MS=2024-01-01`.

- `Data/download_derivatives.py`: downloads funding rates and open interest history via Binance public API. Outputs to `Data/derivatives/`:
  - `funding_rates.csv`
  - `open_interest_15m.csv`
  - Notes: uses `https://fapi.binance.com` endpoints and a chunked pagination approach.

- `Data/download_extended_klines.py`: downloads futures klines (15m) with taker-flow and trade counts. Outputs `Data/futures_klines_15m.csv`.

- `Data/download_multi_asset.py`: uses `binance.client.Client` to fetch historical klines per symbol and writes `Data/multi_asset_dataset.csv`.

- `Data/data.py`: convenience runner that builds `Data/master_raw_dataset.csv` (BTC-focused by default). Also contains a small Tk popup on completion.

## Existing data files

- `Data/master_raw_dataset.csv` — BTC master dataset (created by `data.py`).
- `Data/multi_asset_dataset.csv` — multi-asset klines (created by `download_multi_asset.py`).
- `Data/futures_klines_15m.csv` — full futures klines with taker flow (created by `download_extended_klines.py`).
- `Data/derivatives/` — folder containing the various derivatives CSV outputs described above.
- `.yfinance_cache/` — cached yfinance artifacts (if used by older scripts).

## External sources / APIs

- Binance Futures API (FAPI): `https://fapi.binance.com` for funding, openInterestHist, klines, and other futures endpoints.
- `python-binance` client library: used by `download_multi_asset.py` and `data.py`.
- `requests` HTTP client: used in `download_binance_sentiment.py`, `download_derivatives.py`, and `download_extended_klines.py`.

## Run notes and cautions

- Many scripts assume UTC timestamps and a `START_MS` anchored at 2024-01-01.
- Binance pagination windows are rate/size-limited; scripts include sleep() backoffs and chunking but may hit transient failures — partial outputs are preserved.
- `download_multi_asset.py` and `data.py` rely on `python-binance` credentials/environment if private endpoints are used; currently they fetch public historical klines.

## Recommended next steps

- Automate scheduled downloads into `artifacts/data/` and include a small manifest (`artifacts/data/manifest.json`) recording run time, script name, and produced files.
- Add a lightweight ingestion script that validates expected CSV columns and records row counts.

---

