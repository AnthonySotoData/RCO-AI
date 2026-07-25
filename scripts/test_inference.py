from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data_preparation import prepare_training_data
from src.ml.inference import ForecastInference


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "revenue_cycle_operations.csv"
)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test,
    ) = prepare_training_data(df)

    inference_service = ForecastInference()

    sample_features = X_test.iloc[[0]]

    prediction = inference_service.predict(
        sample_features
    )[0]

    actual_value = float(
        y_test.iloc[0]
    )

    forecast_date = dates_test.iloc[0]

    print("Inference smoke test successful.")
    print(f"Forecast date: {forecast_date}")
    print(
        "Predicted clearance rate: "
        f"{prediction:.4f}"
    )
    print(
        "Actual clearance rate: "
        f"{actual_value:.4f}"
    )
    print(
        "Absolute error: "
        f"{abs(prediction - actual_value):.4f}"
    )


if __name__ == "__main__":
    main()