from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.executive_dashboard import (
    create_executive_dashboard,
)


KEY_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "capstone_key_metrics.csv"
)

VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "capstone_data_validation.csv"
)

SCATTERPLOT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "capstone_primary_regression_scatterplot.png"
)

HEATMAP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "capstone_correlation_heatmap.png"
)

DASHBOARD_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "dashboard"
)

PNG_OUTPUT_PATH = (
    DASHBOARD_OUTPUT_DIRECTORY
    / "capstone_executive_dashboard.png"
)

PDF_OUTPUT_PATH = (
    DASHBOARD_OUTPUT_DIRECTORY
    / "capstone_executive_dashboard.pdf"
)


def _require_file(
    file_path: Path,
    prerequisite_command: str,
) -> None:
    """Raise a clear error when a required artifact is unavailable."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}\n"
            f"Run this command first: {prerequisite_command}"
        )


def main() -> None:
    _require_file(
        KEY_METRICS_PATH,
        "python .\\scripts\\run_capstone_statistics.py",
    )

    _require_file(
        VALIDATION_REPORT_PATH,
        "python .\\scripts\\run_capstone_etl.py",
    )

    _require_file(
        SCATTERPLOT_PATH,
        "python .\\scripts\\run_capstone_visualizations.py",
    )

    _require_file(
        HEATMAP_PATH,
        "python .\\scripts\\run_capstone_correlation.py",
    )

    key_metrics = pd.read_csv(
        KEY_METRICS_PATH,
    )

    validation_report = pd.read_csv(
        VALIDATION_REPORT_PATH,
    )

    png_path, pdf_path = create_executive_dashboard(
        key_metrics=key_metrics,
        validation_report=validation_report,
        scatterplot_path=SCATTERPLOT_PATH,
        heatmap_path=HEATMAP_PATH,
        png_output_path=PNG_OUTPUT_PATH,
        pdf_output_path=PDF_OUTPUT_PATH,
    )

    print("Capstone executive dashboard completed successfully.")
    print()
    print("Generated dashboard files:")
    print(f"- {png_path.relative_to(PROJECT_ROOT)}")
    print(f"- {pdf_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()