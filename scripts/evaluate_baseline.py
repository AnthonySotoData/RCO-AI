from pathlib import Path
import math
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.forecast_dataset import create_forecast_dataset


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "revenue_cycle_operations.csv"
)


df = pd.read_csv(DATA_PATH)

forecast_df = create_forecast_dataset(df)

actual_values = forecast_df[
    "target_clearance_rate"
]

baseline_predictions = forecast_df[
    "clearance_rate"
]

mae = mean_absolute_error(
    actual_values,
    baseline_predictions,
)

mse = mean_squared_error(
    actual_values,
    baseline_predictions,
)

rmse = math.sqrt(mse)

print("Naive Baseline Evaluation")
print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print()
print(
    "Average error in percentage points: "
    f"{mae * 100:.2f}"
)