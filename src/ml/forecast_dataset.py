from __future__ import annotations

import pandas as pd


GROUP_COLUMNS = [
    "specialty",
    "payer",
]


def create_forecast_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a next-day operational forecasting dataset.

    Features represent information available on the current
    or previous days. The target is the following day's
    clearance rate for the same specialty and payer.
    """

    forecast_df = df.copy()

    forecast_df["record_date"] = pd.to_datetime(
        forecast_df["record_date"]
    )

    forecast_df["clearance_rate"] = (
        forecast_df["cleared_accounts"]
        / forecast_df["scheduled_accounts"]
    )

    forecast_df = forecast_df.sort_values(
        GROUP_COLUMNS + ["record_date"]
    ).reset_index(drop=True)

    grouped = forecast_df.groupby(
        GROUP_COLUMNS,
        group_keys=False,
    )

    forecast_df["clearance_rate_lag_1"] = grouped[
        "clearance_rate"
    ].shift(1)

    forecast_df["clearance_rate_lag_2"] = grouped[
        "clearance_rate"
    ].shift(2)

    forecast_df["clearance_rate_lag_7"] = grouped[
        "clearance_rate"
    ].shift(7)

    forecast_df["clearance_rate_rolling_7"] = grouped[
        "clearance_rate"
    ].transform(
        lambda series: (
            series.shift(1)
            .rolling(window=7, min_periods=3)
            .mean()
        )
    )

    forecast_df["queue_rolling_7"] = grouped[
        "work_queue_volume"
    ].transform(
        lambda series: (
            series.shift(1)
            .rolling(window=7, min_periods=3)
            .mean()
        )
    )

    forecast_df["turnaround_rolling_7"] = grouped[
        "average_turnaround_hours"
    ].transform(
        lambda series: (
            series.shift(1)
            .rolling(window=7, min_periods=3)
            .mean()
        )
    )

    forecast_df["target_clearance_rate"] = grouped[
        "clearance_rate"
    ].shift(-1)

    forecast_df["day_of_week"] = (
        forecast_df["record_date"].dt.dayofweek
    )

    forecast_df["month"] = (
        forecast_df["record_date"].dt.month
    )

    required_columns = [
        "clearance_rate_lag_1",
        "clearance_rate_lag_2",
        "clearance_rate_lag_7",
        "clearance_rate_rolling_7",
        "queue_rolling_7",
        "turnaround_rolling_7",
        "target_clearance_rate",
    ]

    forecast_df = forecast_df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    return forecast_df