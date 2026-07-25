import pandas as pd

from src.ml.feature_engineering import prepare_features
from src.ml.forecast_dataset import create_forecast_dataset


def prepare_training_data(
    df: pd.DataFrame,
):
    """
    Prepare a next-day clearance forecasting dataset using
    a chronological train/test split.

    Returns:
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test
    """

    forecast_df = create_forecast_dataset(df)
    forecast_df = prepare_features(forecast_df)

    forecast_df = forecast_df.sort_values(
        "record_date"
    ).reset_index(drop=True)

    dates = forecast_df["record_date"].copy()

    feature_columns = [
        "scheduled_accounts",
        "authorization_required",
        "authorizations_completed",
        "denials",
        "total_charges",
        "denied_charges",
        "average_turnaround_hours",
        "work_queue_volume",
        "staff_fte",
        "authorization_completion_rate",
        "denial_rate",
        "accounts_per_fte",
        "revenue_per_account",
        "clearance_rate",
        "clearance_rate_lag_1",
        "clearance_rate_lag_2",
        "clearance_rate_lag_7",
        "clearance_rate_rolling_7",
        "queue_rolling_7",
        "turnaround_rolling_7",
        "day_of_week",
        "month",
    ]

    categorical_features = pd.get_dummies(
        forecast_df[["specialty", "payer"]],
        drop_first=True,
        dtype="float32",
    )

    X = pd.concat(
        [
            forecast_df[feature_columns],
            categorical_features,
        ],
        axis=1,
    ).astype("float32")

    y = forecast_df[
        "target_clearance_rate"
    ].astype("float32")

    unique_dates = dates.sort_values().unique()

    date_split_index = int(
        len(unique_dates) * 0.80
    )

    cutoff_date = unique_dates[date_split_index]

    train_mask = dates < cutoff_date
    test_mask = dates >= cutoff_date

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    dates_train = dates.loc[train_mask].copy()
    dates_test = dates.loc[test_mask].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test,
    )