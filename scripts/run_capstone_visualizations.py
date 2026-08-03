from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.capstone_visualizations import (
    create_all_capstone_figures,
)


ANALYSIS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "capstone_daily_analysis.csv"
)

PREDICTION_DATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "regression"
    / "primary_prediction_residuals.csv"
)

KEY_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "capstone_key_metrics.csv"
)

FIGURE_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


def _require_file(
    file_path: Path,
    prerequisite_command: str,
) -> None:
    """Raise a clear error when a required input is unavailable."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}\n"
            f"Run this command first: {prerequisite_command}"
        )


def main() -> None:
    _require_file(
        ANALYSIS_DATA_PATH,
        "python .\\scripts\\run_capstone_etl.py",
    )

    _require_file(
        PREDICTION_DATA_PATH,
        "python .\\scripts\\run_capstone_statistics.py",
    )

    _require_file(
        KEY_METRICS_PATH,
        "python .\\scripts\\run_capstone_statistics.py",
    )

    analysis_data = pd.read_csv(
        ANALYSIS_DATA_PATH,
        parse_dates=["record_date"],
    )

    prediction_data = pd.read_csv(
        PREDICTION_DATA_PATH,
    )

    key_metrics = pd.read_csv(
        KEY_METRICS_PATH,
    )

    generated_files = create_all_capstone_figures(
        analysis_data=analysis_data,
        prediction_data=prediction_data,
        key_metrics=key_metrics,
        output_directory=FIGURE_OUTPUT_DIRECTORY,
    )

    print("Capstone visualizations completed successfully.")
    print()
    print(f"Analysis dataset: {ANALYSIS_DATA_PATH}")
    print(f"Prediction dataset: {PREDICTION_DATA_PATH}")
    print(f"Key metrics: {KEY_METRICS_PATH}")
    print()
    print("Generated figures:")

    for generated_file in generated_files:
        print(
            f"- {generated_file.relative_to(PROJECT_ROOT)}"
        )


if __name__ == "__main__":
    main()