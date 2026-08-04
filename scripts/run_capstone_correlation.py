from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.correlation_analysis import (
    run_correlation_analysis,
)


ANALYSIS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "capstone_daily_analysis.csv"
)

CORRELATION_MATRIX_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "capstone_correlation_matrix.csv"
)

CORRELATION_HEATMAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "capstone_correlation_heatmap.png"
)


def main() -> None:
    if not ANALYSIS_DATA_PATH.exists():
        raise FileNotFoundError(
            "The analysis-ready dataset was not found. "
            "Run scripts/run_capstone_etl.py first."
        )

    analysis_data = pd.read_csv(
        ANALYSIS_DATA_PATH,
        parse_dates=["record_date"],
    )

    correlation_matrix, generated_files = (
        run_correlation_analysis(
            dataframe=analysis_data,
            matrix_output_path=CORRELATION_MATRIX_PATH,
            heatmap_output_path=CORRELATION_HEATMAP_PATH,
        )
    )

    print("Capstone correlation analysis completed successfully.")
    print()
    print(f"Analysis observations: {len(analysis_data):,}")
    print(
        "Variables analyzed: "
        f"{len(correlation_matrix.columns)}"
    )
    print()
    print("Generated artifacts:")

    for generated_file in generated_files:
        print(
            f"- {generated_file.relative_to(PROJECT_ROOT)}"
        )

    print()
    print("Selected correlations with denied charges:")
    print(
        correlation_matrix["denied_charges"]
        .sort_values(
            ascending=False,
        )
        .to_string()
    )


if __name__ == "__main__":
    main()