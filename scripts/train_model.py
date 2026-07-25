from pathlib import Path
import copy
import json
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data_preparation import prepare_training_data
from src.ml.forecast_model import ForecastModel
from src.ml.scaler import fit_scaler, transform_features


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "revenue_cycle_operations.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecast_model.pt"
)

FEATURE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_schema.json"
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


def to_tensor(values) -> torch.Tensor:
    """
    Convert a NumPy array or pandas object
    into a float32 PyTorch tensor.
    """

    return torch.tensor(
        values,
        dtype=torch.float32,
    )


def main() -> None:
    """
    Train the next-day clearance forecasting model,
    save the best model state, and persist the exact
    feature schema required for inference.
    """

    torch.manual_seed(42)

    df = pd.read_csv(DATA_PATH)

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
        dates_train,
        dates_test,
    ) = prepare_training_data(df)

    validation_size = int(
        len(X_train_full) * 0.15
    )

    if validation_size <= 0:
        raise ValueError(
            "The validation dataset is empty. "
            "More training records are required."
        )

    training_end = (
        len(X_train_full)
        - validation_size
    )

    X_train = X_train_full.iloc[
        :training_end
    ].copy()

    X_validation = X_train_full.iloc[
        training_end:
    ].copy()

    y_train = y_train_full.iloc[
        :training_end
    ].copy()

    y_validation = y_train_full.iloc[
        training_end:
    ].copy()

    dates_training = dates_train.iloc[
        :training_end
    ].copy()

    dates_validation = dates_train.iloc[
        training_end:
    ].copy()

    X_train_scaled = fit_scaler(
        X_train
    )

    X_validation_scaled = transform_features(
        X_validation
    )

    feature_schema = {
        "input_size": len(X_train.columns),
        "feature_columns": X_train.columns.tolist(),
        "specialty_categories": sorted(
            df["specialty"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "payer_categories": sorted(
            df["payer"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }

    FEATURE_SCHEMA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FEATURE_SCHEMA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_schema,
            file,
            indent=2,
        )

    X_train_tensor = to_tensor(
        X_train_scaled
    )

    y_train_tensor = to_tensor(
        y_train.to_numpy()
    ).view(-1, 1)

    X_validation_tensor = to_tensor(
        X_validation_scaled
    )

    y_validation_tensor = to_tensor(
        y_validation.to_numpy()
    ).view(-1, 1)

    training_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=256,
        shuffle=True,
    )

    model = ForecastModel(
        input_size=X_train_tensor.shape[1]
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    maximum_epochs = 100
    patience = 10
    minimum_improvement = 0.000001

    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        total_training_loss = 0.0

        for feature_batch, target_batch in (
            training_loader
        ):
            optimizer.zero_grad()

            predictions = model(
                feature_batch
            )

            loss = criterion(
                predictions,
                target_batch,
            )

            loss.backward()
            optimizer.step()

            total_training_loss += (
                loss.item()
                * feature_batch.size(0)
            )

        average_training_loss = (
            total_training_loss
            / len(training_dataset)
        )

        model.eval()

        with torch.no_grad():
            validation_predictions = model(
                X_validation_tensor
            )

            validation_loss = criterion(
                validation_predictions,
                y_validation_tensor,
            ).item()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: "
            f"{average_training_loss:.6f} | "
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )

        if (
            validation_loss
            < best_validation_loss
            - minimum_improvement
        ):
            best_validation_loss = (
                validation_loss
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce "
            "a valid model state."
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        best_model_state,
        MODEL_PATH,
    )

    model_metadata = {
        "model_name": "RCO AI Clearance Forecast Model",
        "model_type": "PyTorch neural network",
        "model_version": "1.0.0",
        "prediction_target": (
            "next-day clearance rate"
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "training_rows": len(X_train),
        "validation_rows": len(
            X_validation
        ),
        "test_rows": len(X_test),
        "feature_count": len(
            X_train.columns
        ),
        "training_start_date": str(
            dates_training.min()
        ),
        "training_end_date": str(
            dates_training.max()
        ),
        "validation_start_date": str(
            dates_validation.min()
        ),
        "validation_end_date": str(
            dates_validation.max()
        ),
        "test_start_date": str(
            dates_test.min()
        ),
        "test_end_date": str(
            dates_test.max()
        ),
    }

    with MODEL_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
            file,
            indent=2,
        )

    print("\nTraining complete.")
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(
        f"Model saved to {MODEL_PATH}"
    )
    print(
        f"Feature schema saved to "
        f"{FEATURE_SCHEMA_PATH}"
    )
    print(
        f"Model metadata saved to "
        f"{MODEL_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()