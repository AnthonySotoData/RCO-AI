from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEPENDENT_VARIABLE = "average_turnaround_hours"
DEPENDENT_VARIABLE = "denied_charges"


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing_columns)
        )


def create_regression_scatterplot(
    analysis_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    key_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create the primary scatterplot with fitted regression line."""
    _validate_columns(
        analysis_data,
        [
            INDEPENDENT_VARIABLE,
            DEPENDENT_VARIABLE,
        ],
        "Analysis dataset",
    )

    _validate_columns(
        prediction_data,
        [
            INDEPENDENT_VARIABLE,
            "predicted_value",
        ],
        "Prediction dataset",
    )

    primary_metrics = key_metrics.loc[
        key_metrics["analysis"] == "Primary"
    ]

    if primary_metrics.empty:
        raise ValueError(
            "Primary analysis metrics were not found."
        )

    pearson_correlation = float(
        primary_metrics.iloc[0]["pearson_correlation"]
    )

    r_squared = float(
        primary_metrics.iloc[0]["r_squared"]
    )

    coefficient_p_value = str(
        primary_metrics.iloc[0]["coefficient_p_value"]
    )

    sorted_predictions = prediction_data.sort_values(
        INDEPENDENT_VARIABLE
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.scatter(
        analysis_data[INDEPENDENT_VARIABLE],
        analysis_data[DEPENDENT_VARIABLE],
        alpha=0.35,
        s=18,
        label="Daily observations",
    )

    axis.plot(
        sorted_predictions[INDEPENDENT_VARIABLE],
        sorted_predictions["predicted_value"],
        linewidth=2,
        label="Fitted regression line",
    )

    axis.set_title(
        "Authorization Turnaround Time and Daily Denied Charges"
    )

    axis.set_xlabel(
        "Average Authorization Turnaround Time (Hours)"
    )

    axis.set_ylabel(
        "Daily Denied Charges ($)"
    )

    axis.ticklabel_format(
        style="plain",
        axis="y",
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend()

    metrics_text = (
        f"Pearson r = {pearson_correlation:.3f}\n"
        f"R² = {r_squared:.3f}\n"
        f"p {coefficient_p_value}"
    )

    axis.text(
        0.03,
        0.97,
        metrics_text,
        transform=axis.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.85,
        },
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


def create_residual_plot(
    prediction_data: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a residual-versus-predicted-values diagnostic plot."""
    _validate_columns(
        prediction_data,
        [
            "predicted_value",
            "residual",
        ],
        "Prediction dataset",
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.scatter(
        prediction_data["predicted_value"],
        prediction_data["residual"],
        alpha=0.35,
        s=18,
    )

    axis.axhline(
        y=0,
        linewidth=2,
        linestyle="--",
    )

    axis.set_title(
        "Primary Regression Residual Plot"
    )

    axis.set_xlabel(
        "Predicted Daily Denied Charges ($)"
    )

    axis.set_ylabel(
        "Residual ($)"
    )

    axis.ticklabel_format(
        style="plain",
        axis="both",
    )

    axis.grid(
        alpha=0.25,
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


def create_turnaround_histogram(
    analysis_data: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a histogram of daily authorization turnaround time."""
    _validate_columns(
        analysis_data,
        [INDEPENDENT_VARIABLE],
        "Analysis dataset",
    )

    turnaround_values = analysis_data[
        INDEPENDENT_VARIABLE
    ].dropna()

    mean_value = float(
        turnaround_values.mean()
    )

    median_value = float(
        turnaround_values.median()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.hist(
        turnaround_values,
        bins=30,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        mean_value,
        linewidth=2,
        linestyle="--",
        label=f"Mean: {mean_value:.2f} hours",
    )

    axis.axvline(
        median_value,
        linewidth=2,
        linestyle=":",
        label=f"Median: {median_value:.2f} hours",
    )

    axis.set_title(
        "Distribution of Daily Authorization Turnaround Time"
    )

    axis.set_xlabel(
        "Average Authorization Turnaround Time (Hours)"
    )

    axis.set_ylabel(
        "Number of Daily Observations"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

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


def create_denied_charges_histogram(
    analysis_data: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Create a histogram of daily denied charges."""
    _validate_columns(
        analysis_data,
        [DEPENDENT_VARIABLE],
        "Analysis dataset",
    )

    denied_charge_values = analysis_data[
        DEPENDENT_VARIABLE
    ].dropna()

    mean_value = float(
        denied_charge_values.mean()
    )

    median_value = float(
        denied_charge_values.median()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    axis.hist(
        denied_charge_values,
        bins=30,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        mean_value,
        linewidth=2,
        linestyle="--",
        label=f"Mean: ${mean_value:,.0f}",
    )

    axis.axvline(
        median_value,
        linewidth=2,
        linestyle=":",
        label=f"Median: ${median_value:,.0f}",
    )

    axis.set_title(
        "Distribution of Daily Denied Charges"
    )

    axis.set_xlabel(
        "Daily Denied Charges ($)"
    )

    axis.set_ylabel(
        "Number of Daily Observations"
    )

    axis.ticklabel_format(
        style="plain",
        axis="x",
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

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


def create_all_capstone_figures(
    analysis_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    key_metrics: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """Generate all Task 2 and Task 3 statistical figures."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = [
        create_regression_scatterplot(
            analysis_data=analysis_data,
            prediction_data=prediction_data,
            key_metrics=key_metrics,
            output_path=(
                output_directory
                / "capstone_primary_regression_scatterplot.png"
            ),
        ),
        create_residual_plot(
            prediction_data=prediction_data,
            output_path=(
                output_directory
                / "capstone_primary_residual_plot.png"
            ),
        ),
        create_turnaround_histogram(
            analysis_data=analysis_data,
            output_path=(
                output_directory
                / "capstone_turnaround_distribution.png"
            ),
        ),
        create_denied_charges_histogram(
            analysis_data=analysis_data,
            output_path=(
                output_directory
                / "capstone_denied_charges_distribution.png"
            ),
        ),
    ]

    return generated_files