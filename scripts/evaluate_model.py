from pathlib import Path
import math
import sys

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data_preparation import prepare_training_data
from src.ml.forecast_model import ForecastModel
from src.ml.scaler import transform_features


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "revenue_cycle_operations.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "forecast_model.pt"


df = pd.read_csv(DATA_PATH)

(
    X_train,
    X_test,
    y_train,
    y_test,
    dates_train,
    dates_test,
) = prepare_training_data(df)

X_test_scaled = transform_features(X_test)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)

model = ForecastModel(X_test_tensor.shape[1])

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
    )
)

model.eval()

with torch.no_grad():
    predictions = (
        model(X_test_tensor)
        .numpy()
        .flatten()
    )

mae = mean_absolute_error(
    y_test,
    predictions,
)

mse = mean_squared_error(
    y_test,
    predictions,
)

rmse = math.sqrt(mse)

print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")