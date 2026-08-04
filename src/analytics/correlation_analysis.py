from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORRELATION_COLUMNS = [
    "average_turnaround_hours",
    "denied_charges",
    "denied_charge_rate",
    "denial_rate",
    "clearance_rate",
    "authorization_completion_rate",
    "work_queue_volume",
    "staff_fte",
    "workload_per_fte",
]


DISPLAY_LABELS = {
    "average_turnaround_hours": "Turnaround Hours",
    "denied_charges": "Denied Charges",
    "denied_charge_rate": "Denied Charge Rate",
    "denial_rate": "Denial Rate",
    "clearance_rate": "Clearance Rate",
    "authorization_completion_rate": "Auth Completion Rate",
    "work_queue_volume": "Work Queue Volume",
    "staff_fte": "Staff FTE",
    "workload_per_fte": "Workload per FTE",
}


def validate_correlation_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and prepare numeric fields for correlation analysis."""
    missing_columns = [
        column
        for column in CORRELATION_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The analysis dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    correlation_data = dataframe[
        CORRELATION_COLUMNS
    ].copy()

    for column in CORRELATION_COLUMNS:
        correlation_data[column] = pd.to_numeric(
            correlation_data[column],
            errors="coerce",
        )

    correlation_data = correlation_data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if correlation_data.empty:
        raise ValueError(
            "No complete observations are available for "
            "correlation analysis."
        )

    return correlation_data


def calculate_correlation_matrix(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate a Pearson correlation matrix."""
    correlation_data = validate_correlation_data(
        dataframe=dataframe,
    )

    correlation_matrix = correlation_data.corr(
        method="pearson"
    )

    return correlation_matrix.round(6)


def export_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Export the correlation matrix as a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlation_matrix.to_csv(
        output_path,
        index=True,
    )

    return output_path


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a labeled Pearson correlation heatmap."""
    display_matrix = correlation_matrix.rename(
        index=DISPLAY_LABELS,
        columns=DISPLAY_LABELS,
    )

    figure, axis = plt.subplots(
        figsize=(12, 9),
    )

    image = axis.imshow(
        display_matrix.values,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )

    axis.set_xticks(
        range(len(display_matrix.columns))
    )

    axis.set_yticks(
        range(len(display_matrix.index))
    )

    axis.set_xticklabels(
        display_matrix.columns,
        rotation=45,
        horizontalalignment="right",
    )

    axis.set_yticklabels(
        display_matrix.index,
    )

    axis.set_title(
        "Pearson Correlation Matrix for Revenue Cycle Metrics",
        pad=20,
    )

    for row_index in range(
        len(display_matrix.index)
    ):
        for column_index in range(
            len(display_matrix.columns)
        ):
            correlation_value = display_matrix.iloc[
                row_index,
                column_index,
            ]

            text_color = (
                "white"
                if abs(correlation_value) >= 0.60
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{correlation_value:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson Correlation Coefficient",
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def run_correlation_analysis(
    dataframe: pd.DataFrame,
    matrix_output_path: Path,
    heatmap_output_path: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    """Run the complete correlation analysis workflow."""
    correlation_matrix = calculate_correlation_matrix(
        dataframe=dataframe,
    )

    matrix_path = export_correlation_matrix(
        correlation_matrix=correlation_matrix,
        output_path=matrix_output_path,
    )

    heatmap_path = create_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        output_path=heatmap_output_path,
    )

    return correlation_matrix, [
        matrix_path,
        heatmap_path,
    ]