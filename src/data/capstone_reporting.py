from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DICTIONARY: list[dict[str, str]] = [
    {
        "column_name": "record_date",
        "data_type": "date",
        "dataset_level": "raw, prepared, daily",
        "description": "Date associated with the operational record.",
        "business_meaning": (
            "Supports time-based aggregation, trending, and forecasting."
        ),
    },
    {
        "column_name": "specialty",
        "data_type": "categorical",
        "dataset_level": "raw, prepared",
        "description": "Clinical specialty associated with the record.",
        "business_meaning": (
            "Allows comparison of operational performance across service areas."
        ),
    },
    {
        "column_name": "payer",
        "data_type": "categorical",
        "dataset_level": "raw, prepared",
        "description": "Insurance payer associated with the record.",
        "business_meaning": (
            "Supports evaluation of payer-specific authorization and denial patterns."
        ),
    },
    {
        "column_name": "scheduled_accounts",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of accounts scheduled for service.",
        "business_meaning": (
            "Represents operational account volume and workload demand."
        ),
    },
    {
        "column_name": "cleared_accounts",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of accounts financially cleared.",
        "business_meaning": (
            "Measures completed financial clearance activity."
        ),
    },
    {
        "column_name": "authorization_required",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of accounts requiring payer authorization.",
        "business_meaning": (
            "Represents the volume of authorization-related work."
        ),
    },
    {
        "column_name": "authorizations_completed",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of required authorizations completed.",
        "business_meaning": (
            "Measures authorization workflow completion."
        ),
    },
    {
        "column_name": "denials",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of denied accounts.",
        "business_meaning": (
            "Represents adverse revenue cycle outcomes."
        ),
    },
    {
        "column_name": "total_charges",
        "data_type": "currency",
        "dataset_level": "raw, prepared, daily",
        "description": "Total charges associated with scheduled accounts.",
        "business_meaning": (
            "Provides the financial value of operational account volume."
        ),
    },
    {
        "column_name": "denied_charges",
        "data_type": "currency",
        "dataset_level": "raw, prepared, daily",
        "description": "Charges associated with denied accounts.",
        "business_meaning": (
            "Primary dependent variable and measure of financial risk."
        ),
    },
    {
        "column_name": "average_turnaround_hours",
        "data_type": "continuous numeric",
        "dataset_level": "raw, prepared, daily",
        "description": (
            "Average number of hours required to complete authorization activity."
        ),
        "business_meaning": (
            "Primary independent variable used to evaluate operational timeliness."
        ),
    },
    {
        "column_name": "work_queue_volume",
        "data_type": "integer",
        "dataset_level": "raw, prepared, daily",
        "description": "Number of items in the operational work queue.",
        "business_meaning": (
            "Measures outstanding workload and potential workflow congestion."
        ),
    },
    {
        "column_name": "staff_fte",
        "data_type": "continuous numeric",
        "dataset_level": "raw, prepared, daily",
        "description": "Available full-time-equivalent staffing level.",
        "business_meaning": (
            "Represents operational staffing capacity."
        ),
    },
    {
        "column_name": "clearance_rate",
        "data_type": "proportion",
        "dataset_level": "prepared, daily",
        "description": (
            "Cleared accounts divided by scheduled accounts."
        ),
        "business_meaning": (
            "Measures the proportion of scheduled accounts financially cleared."
        ),
    },
    {
        "column_name": "authorization_completion_rate",
        "data_type": "proportion",
        "dataset_level": "prepared, daily",
        "description": (
            "Completed authorizations divided by required authorizations."
        ),
        "business_meaning": (
            "Measures authorization workflow effectiveness."
        ),
    },
    {
        "column_name": "denial_rate",
        "data_type": "proportion",
        "dataset_level": "prepared, daily",
        "description": "Denials divided by scheduled accounts.",
        "business_meaning": (
            "Provides a volume-adjusted denial performance measure."
        ),
    },
    {
        "column_name": "denied_charge_rate",
        "data_type": "proportion",
        "dataset_level": "prepared, daily",
        "description": "Denied charges divided by total charges.",
        "business_meaning": (
            "Provides a volume-adjusted measure of denied financial value."
        ),
    },
    {
        "column_name": "workload_per_fte",
        "data_type": "continuous numeric",
        "dataset_level": "prepared, daily",
        "description": "Work queue volume divided by staff FTE.",
        "business_meaning": (
            "Measures operational workload relative to staffing capacity."
        ),
    },
]


DESCRIPTIVE_COLUMNS = [
    "scheduled_accounts",
    "cleared_accounts",
    "authorization_required",
    "authorizations_completed",
    "denials",
    "total_charges",
    "denied_charges",
    "average_turnaround_hours",
    "work_queue_volume",
    "staff_fte",
    "clearance_rate",
    "authorization_completion_rate",
    "denial_rate",
    "denied_charge_rate",
    "workload_per_fte",
]


def create_data_dictionary(
    output_path: Path,
) -> pd.DataFrame:
    """Export documentation for raw and engineered variables."""
    dictionary = pd.DataFrame(DATA_DICTIONARY)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictionary.to_csv(
        output_path,
        index=False,
    )

    return dictionary


def create_descriptive_statistics(
    daily_data: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Create a report-ready descriptive statistics table."""
    available_columns = [
        column
        for column in DESCRIPTIVE_COLUMNS
        if column in daily_data.columns
    ]

    descriptive = (
        daily_data[available_columns]
        .describe(
            percentiles=[0.25, 0.50, 0.75],
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "variable",
                "count": "observation_count",
                "mean": "mean",
                "std": "standard_deviation",
                "min": "minimum",
                "25%": "first_quartile",
                "50%": "median",
                "75%": "third_quartile",
                "max": "maximum",
            }
        )
    )

    numeric_columns = descriptive.select_dtypes(
        include="number"
    ).columns

    descriptive[numeric_columns] = (
        descriptive[numeric_columns].round(4)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive.to_csv(
        output_path,
        index=False,
    )

    return descriptive


def _validation_value(
    validation_report: pd.DataFrame,
    metric_name: str,
) -> Any:
    matching_rows = validation_report.loc[
        validation_report["validation_metric"]
        == metric_name,
        "value",
    ]

    if matching_rows.empty:
        return "Not available"

    return matching_rows.iloc[0]


def create_etl_summary(
    validation_report: pd.DataFrame,
    processed_data_path: Path,
    prepared_detail_path: Path,
    output_path: Path,
) -> str:
    """Create a human-readable summary of the completed ETL run."""
    run_timestamp = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    summary = "\n".join(
        [
            "=" * 72,
            "RCO AI DATA ENGINEERING CAPSTONE - ETL SUMMARY",
            "=" * 72,
            "",
            f"Run timestamp: {run_timestamp}",
            "Execution status: SUCCESS",
            "",
            "DATASET COUNTS",
            "-" * 72,
            (
                "Raw records: "
                f"{_validation_value(validation_report, 'Raw row count')}"
            ),
            (
                "Prepared records: "
                f"{_validation_value(validation_report, 'Prepared row count')}"
            ),
            (
                "Records removed during preparation: "
                f"{_validation_value(validation_report, 'Rows removed')}"
            ),
            (
                "Daily analysis records: "
                f"{_validation_value(validation_report, 'Daily analysis rows')}"
            ),
            "",
            "DATA QUALITY RESULTS",
            "-" * 72,
            (
                "Missing values detected: "
                f"{_validation_value(validation_report, 'Missing values detected')}"
            ),
            (
                "Duplicate rows detected: "
                f"{_validation_value(validation_report, 'Duplicate rows detected')}"
            ),
            (
                "Invalid dates detected: "
                f"{_validation_value(validation_report, 'Invalid dates detected')}"
            ),
            (
                "Negative numeric values detected: "
                f"{_validation_value(validation_report, 'Negative numeric values detected')}"
            ),
            "",
            "DATA SCOPE",
            "-" * 72,
            (
                "Minimum record date: "
                f"{_validation_value(validation_report, 'Minimum record date')}"
            ),
            (
                "Maximum record date: "
                f"{_validation_value(validation_report, 'Maximum record date')}"
            ),
            (
                "Unique specialties: "
                f"{_validation_value(validation_report, 'Unique specialties')}"
            ),
            (
                "Unique payers: "
                f"{_validation_value(validation_report, 'Unique payers')}"
            ),
            "",
            "OUTPUT FILES",
            "-" * 72,
            f"Prepared detail dataset: {prepared_detail_path}",
            f"Daily analysis dataset: {processed_data_path}",
            "",
            "The daily dataset contains one aggregated observation per date.",
            (
                "Authorization turnaround time is weighted by authorization-required "
                "volume."
            ),
            "",
            "=" * 72,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        summary,
        encoding="utf-8",
    )

    return summary