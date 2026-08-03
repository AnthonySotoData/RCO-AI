from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "record_date",
    "specialty",
    "payer",
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
]

NUMERIC_COLUMNS = [
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
]

NONNEGATIVE_COLUMNS = NUMERIC_COLUMNS.copy()


def extract_raw_data(source_path: Path) -> pd.DataFrame:
    """Extract the raw revenue-cycle dataset from a CSV file."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw dataset was not found: {source_path}"
        )

    dataframe = pd.read_csv(source_path)

    if dataframe.empty:
        raise ValueError("The raw dataset contains no records.")

    return dataframe


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Confirm that all required columns exist."""
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw operational data."""
    prepared = dataframe.copy()

    prepared["record_date"] = pd.to_datetime(
        prepared["record_date"],
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["specialty"] = (
        prepared["specialty"]
        .astype("string")
        .str.strip()
    )

    prepared["payer"] = (
        prepared["payer"]
        .astype("string")
        .str.strip()
    )

    prepared = prepared.drop_duplicates().copy()

    required_non_null_columns = [
        "record_date",
        "specialty",
        "payer",
        *NUMERIC_COLUMNS,
    ]

    prepared = prepared.dropna(
        subset=required_non_null_columns
    ).copy()

    for column in NONNEGATIVE_COLUMNS:
        prepared = prepared[
            prepared[column] >= 0
        ].copy()

    prepared = prepared[
        prepared["cleared_accounts"]
        <= prepared["scheduled_accounts"]
    ].copy()

    prepared = prepared[
        prepared["authorizations_completed"]
        <= prepared["authorization_required"]
    ].copy()

    prepared = prepared[
        prepared["denials"]
        <= prepared["scheduled_accounts"]
    ].copy()

    prepared = prepared.sort_values(
        by=["record_date", "specialty", "payer"]
    ).reset_index(drop=True)

    return prepared


def engineer_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create operational rates used for analysis and reporting."""
    engineered = dataframe.copy()

    scheduled_denominator = (
        engineered["scheduled_accounts"]
        .replace(0, pd.NA)
    )

    authorization_denominator = (
        engineered["authorization_required"]
        .replace(0, pd.NA)
    )

    charges_denominator = (
        engineered["total_charges"]
        .replace(0, pd.NA)
    )

    engineered["clearance_rate"] = (
        engineered["cleared_accounts"]
        / scheduled_denominator
    )

    engineered["authorization_completion_rate"] = (
        engineered["authorizations_completed"]
        / authorization_denominator
    )

    engineered["denial_rate"] = (
        engineered["denials"]
        / scheduled_denominator
    )

    engineered["denied_charge_rate"] = (
        engineered["denied_charges"]
        / charges_denominator
    )

    engineered["workload_per_fte"] = (
        engineered["work_queue_volume"]
        / engineered["staff_fte"].replace(0, pd.NA)
    )

    rate_columns = [
        "clearance_rate",
        "authorization_completion_rate",
        "denial_rate",
        "denied_charge_rate",
        "workload_per_fte",
    ]

    engineered[rate_columns] = (
        engineered[rate_columns]
        .fillna(0.0)
    )

    return engineered


def aggregate_daily_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate specialty-payer records into one observation per day.

    Turnaround time is weighted by authorization-required volume so
    higher-volume operational groups contribute proportionally.
    """
    working = dataframe.copy()

    working["weighted_turnaround"] = (
        working["average_turnaround_hours"]
        * working["authorization_required"]
    )

    daily = (
        working.groupby(
            "record_date",
            as_index=False,
        )
        .agg(
            scheduled_accounts=(
                "scheduled_accounts",
                "sum",
            ),
            cleared_accounts=(
                "cleared_accounts",
                "sum",
            ),
            authorization_required=(
                "authorization_required",
                "sum",
            ),
            authorizations_completed=(
                "authorizations_completed",
                "sum",
            ),
            denials=(
                "denials",
                "sum",
            ),
            total_charges=(
                "total_charges",
                "sum",
            ),
            denied_charges=(
                "denied_charges",
                "sum",
            ),
            weighted_turnaround=(
                "weighted_turnaround",
                "sum",
            ),
            work_queue_volume=(
                "work_queue_volume",
                "sum",
            ),
            staff_fte=(
                "staff_fte",
                "sum",
            ),
        )
    )

    authorization_denominator = (
        daily["authorization_required"]
        .replace(0, pd.NA)
    )

    scheduled_denominator = (
        daily["scheduled_accounts"]
        .replace(0, pd.NA)
    )

    charges_denominator = (
        daily["total_charges"]
        .replace(0, pd.NA)
    )

    daily["average_turnaround_hours"] = (
        daily["weighted_turnaround"]
        / authorization_denominator
    )

    daily["clearance_rate"] = (
        daily["cleared_accounts"]
        / scheduled_denominator
    )

    daily["authorization_completion_rate"] = (
        daily["authorizations_completed"]
        / authorization_denominator
    )

    daily["denial_rate"] = (
        daily["denials"]
        / scheduled_denominator
    )

    daily["denied_charge_rate"] = (
        daily["denied_charges"]
        / charges_denominator
    )

    daily["workload_per_fte"] = (
        daily["work_queue_volume"]
        / daily["staff_fte"].replace(0, pd.NA)
    )

    daily = daily.drop(
        columns=["weighted_turnaround"]
    )

    daily = daily.fillna(0.0)

    numeric_columns = daily.select_dtypes(
        include="number"
    ).columns

    daily[numeric_columns] = (
        daily[numeric_columns].round(4)
    )

    return daily.sort_values(
        "record_date"
    ).reset_index(drop=True)


def build_validation_report(
    raw_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a screenshot-ready data-quality summary."""
    invalid_date_count = pd.to_datetime(
        raw_data.get("record_date"),
        errors="coerce",
    ).isna().sum()

    missing_value_count = int(
        raw_data.isna().sum().sum()
    )

    duplicate_count = int(
        raw_data.duplicated().sum()
    )

    negative_value_count = 0

    for column in NONNEGATIVE_COLUMNS:
        if column in raw_data.columns:
            numeric_series = pd.to_numeric(
                raw_data[column],
                errors="coerce",
            )

            negative_value_count += int(
                (numeric_series < 0).sum()
            )

    metrics: list[dict[str, Any]] = [
        {
            "validation_metric": "Raw row count",
            "value": len(raw_data),
        },
        {
            "validation_metric": "Prepared row count",
            "value": len(prepared_data),
        },
        {
            "validation_metric": "Rows removed",
            "value": len(raw_data) - len(prepared_data),
        },
        {
            "validation_metric": "Raw column count",
            "value": len(raw_data.columns),
        },
        {
            "validation_metric": "Missing values detected",
            "value": missing_value_count,
        },
        {
            "validation_metric": "Duplicate rows detected",
            "value": duplicate_count,
        },
        {
            "validation_metric": "Invalid dates detected",
            "value": int(invalid_date_count),
        },
        {
            "validation_metric": "Negative numeric values detected",
            "value": negative_value_count,
        },
        {
            "validation_metric": "Daily analysis rows",
            "value": len(daily_data),
        },
        {
            "validation_metric": "Minimum record date",
            "value": prepared_data["record_date"].min().date(),
        },
        {
            "validation_metric": "Maximum record date",
            "value": prepared_data["record_date"].max().date(),
        },
        {
            "validation_metric": "Unique specialties",
            "value": prepared_data["specialty"].nunique(),
        },
        {
            "validation_metric": "Unique payers",
            "value": prepared_data["payer"].nunique(),
        },
    ]

    return pd.DataFrame(metrics)


def run_capstone_etl(
    source_path: Path,
    processed_path: Path,
    validation_path: Path,
    prepared_detail_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the complete capstone ETL workflow."""
    raw_data = extract_raw_data(source_path)
    validate_schema(raw_data)

    prepared_data = prepare_data(raw_data)
    engineered_data = engineer_features(prepared_data)
    daily_data = aggregate_daily_data(engineered_data)

    validation_report = build_validation_report(
        raw_data=raw_data,
        prepared_data=engineered_data,
        daily_data=daily_data,
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_data.to_csv(
        processed_path,
        index=False,
    )

    validation_report.to_csv(
        validation_path,
        index=False,
    )

    if prepared_detail_path is not None:
        prepared_detail_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        engineered_data.to_csv(
            prepared_detail_path,
            index=False,
        )

    return daily_data, validation_report