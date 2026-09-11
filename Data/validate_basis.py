"""Validate futures-spot basis as an added feature using leakage-safe walk-forward.

Produces `artifacts/validation/basis_vs_baseline.json` with evaluation summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "Model") not in sys.path:
    sys.path.insert(0, str(ROOT / "Model"))

from research_program import load_multi_asset, slice_symbol, save_json, evaluate_move_signal
from research_engine import build_causal_features
from enhanced_features import merge_features
from perry_config import filename_with_timeframe, get_primary_timeframe

ART_DIR = ROOT / "artifacts" / "validation"
ART_DIR.mkdir(parents=True, exist_ok=True)


def basis_path() -> Path:
    return ROOT / "artifacts" / "data" / "microstructure" / filename_with_timeframe("basis", get_primary_timeframe())


def load_basis_for(symbol: str = "BTCUSDT"):
    import pandas as pd

    basis_file = basis_path()
    if not basis_file.exists():
        return None

    df = pd.read_csv(basis_file, parse_dates=["Datetime"])
    df = df[df["symbol"] == symbol][["Datetime", "basis"]].sort_values("Datetime")
    return df.reset_index(drop=True)


def main():
    multi = load_multi_asset()
    btc_raw = slice_symbol(multi, "BTCUSDT")
    base = build_causal_features(btc_raw)

    # Baseline evaluation
    baseline_metrics = evaluate_move_signal(btc_raw, base)

    # Basis-augmented evaluation
    basis = load_basis_for()
    if basis is None or basis.empty:
        print("No basis data available; skipping basis-augmented run.")
        save_json("validation/basis_vs_baseline", {"baseline": baseline_metrics, "basis_augmented": "no_data"})
        return

    merged = merge_features(base, deriv=basis)
    basis_metrics = evaluate_move_signal(btc_raw, merged)

    payload = {"baseline": baseline_metrics, "basis_augmented": basis_metrics}
    save_json("validation/basis_vs_baseline", payload)
    print("Saved validation results to artifacts/validation/basis_vs_baseline.json")


if __name__ == "__main__":
    main()
