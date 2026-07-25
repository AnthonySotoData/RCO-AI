from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from src.ml.forecast_model import ForecastModel
from src.ml.scaler import transform_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


class ForecastInference:
    """
    Load the trained forecasting model, feature schema,
    and scaler for reusable inference.
    """

    def __init__(self) -> None:
        self.feature_columns = self._load_feature_schema()
        self.input_size = len(self.feature_columns)

        self.model = ForecastModel(
            input_size=self.input_size
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

    def _load_feature_schema(
        self,
    ) -> list[str]:
        """
        Load and validate the feature schema created
        during model training.
        """

        if not FEATURE_SCHEMA_PATH.exists():
            raise FileNotFoundError(
                "Feature schema was not found at "
                f"{FEATURE_SCHEMA_PATH}. "
                "Run the training script first."
            )

        with FEATURE_SCHEMA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            schema = json.load(file)

        feature_columns = schema.get(
            "feature_columns"
        )

        input_size = schema.get(
            "input_size"
        )

        if not isinstance(
            feature_columns,
            list,
        ):
            raise ValueError(
                "The feature schema does not contain "
                "a valid feature_columns list."
            )

        if input_size != len(feature_columns):
            raise ValueError(
                "The feature schema input size does "
                "not match its feature column count."
            )

        return feature_columns

    def prepare_features(
        self,
        feature_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and reorder features to match the
        exact training schema.
        """

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in feature_dataframe.columns
        ]

        extra_columns = [
            column
            for column in feature_dataframe.columns
            if column not in self.feature_columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required model features: "
                + ", ".join(missing_columns)
            )

        if extra_columns:
            raise ValueError(
                "Unexpected model features: "
                + ", ".join(extra_columns)
            )

        ordered_features = feature_dataframe[
            self.feature_columns
        ].copy()

        return ordered_features.astype(
            "float32"
        )

    def predict(
        self,
        feature_dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict next-day clearance rates.

        Parameters
        ----------
        feature_dataframe:
            DataFrame containing the exact features
            required by the saved model schema.

        Returns
        -------
        numpy.ndarray:
            Predicted clearance rates.
        """

        ordered_features = self.prepare_features(
            feature_dataframe
        )

        scaled_features = transform_features(
            ordered_features
        )

        feature_tensor = torch.tensor(
            scaled_features,
            dtype=torch.float32,
        )

        with torch.no_grad():
            predictions = self.model(
                feature_tensor
            )

        predictions_array = (
            predictions
            .cpu()
            .numpy()
            .flatten()
        )

        return np.clip(
            predictions_array,
            0.0,
            1.0,
        )


_inference_service: ForecastInference | None = None


def get_inference_service() -> ForecastInference:
    """
    Return one shared inference service for the
    lifetime of the application process.
    """

    global _inference_service

    if _inference_service is None:
        _inference_service = ForecastInference()

    return _inference_service