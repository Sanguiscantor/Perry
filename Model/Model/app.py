from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

try:
    from .data_loader import load_chronological_dataset
    from .evaluation import print_pipeline_report
    from .pipeline import run_walk_forward_pipeline
except ImportError:
    from data_loader import load_chronological_dataset
    from evaluation import print_pipeline_report
    from pipeline import run_walk_forward_pipeline

DEFAULT_FEATURE_COLUMNS = [
    "momentum",
    "return_1",
    "volume_spike",
    "volatility",
    "directional_volume",
]


def parse_feature_columns(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_dataset_path() -> str:
    package_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(package_dir, "..", "features_data", "master_feature_dataset.csv"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the financial walk-forward model pipeline on chronological feature data."
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Path to the chronological engineered feature CSV file. "
            "If omitted, the package default features_data/master_feature_dataset.csv is used."
        ),
    )
    parser.add_argument(
        "--train-window",
        default="30D",
        help="Training window length (e.g. 30D for 30 days).",
    )
    parser.add_argument(
        "--test-window",
        default="7D",
        help="Testing window length (e.g. 7D for 7 days).",
    )
    parser.add_argument(
        "--step-window",
        default="7D",
        help="Walk-forward step size; defaults to the test window.",
    )
    parser.add_argument(
        "--feature-cols",
        default="",
        help="Comma-separated feature columns to use for modeling.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the full evaluation report to a JSON file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    feature_columns = (
        parse_feature_columns(args.feature_cols)
        if args.feature_cols.strip()
        else DEFAULT_FEATURE_COLUMNS
    )

    dataset_path = os.path.abspath(args.dataset_path) if args.dataset_path else default_dataset_path()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run this from the Model/Model folder or pass --dataset-path explicitly."
        )

    print(f"Using dataset: {dataset_path}")
    df = load_chronological_dataset(dataset_path)

    report = run_walk_forward_pipeline(
        df,
        feature_cols=feature_columns,
        target_col="target",
        datetime_col="Datetime",
        train_window=args.train_window,
        test_window=args.test_window,
        step_window=args.step_window,
    )

    print_pipeline_report(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
