from pathlib import Path
import json

from fastapi import APIRouter, HTTPException

from src.ml.feature_builder import FeatureBuilder
from src.ml.inference import get_inference_service
from src.models.forecast_api import (
    ForecastRequest,
    ForecastResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

router = APIRouter(
    prefix="/forecast",
    tags=["Forecast"],
)

feature_builder = FeatureBuilder()
inference_service = get_inference_service()

with MODEL_METADATA_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    MODEL_METADATA = json.load(file)


def determine_risk(
    clearance_rate: float,
) -> tuple[str, str]:
    """
    Convert the predicted clearance rate
    into an operational risk level.
    """

    if clearance_rate >= 0.95:
        return (
            "Very Low",
            "Excellent",
        )

    if clearance_rate >= 0.90:
        return (
            "Low",
            "Healthy",
        )

    if clearance_rate >= 0.85:
        return (
            "Moderate",
            "Monitor",
        )

    if clearance_rate >= 0.80:
        return (
            "High",
            "Needs Attention",
        )

    return (
        "Critical",
        "Immediate Action",
    )


@router.post(
    "",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Predict tomorrow's financial clearance rate.
    """

    try:
        features = feature_builder.build(
            request
        )

        prediction = (
            inference_service.predict(
                features
            )[0]
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    risk_level, forecast_status = (
        determine_risk(
            prediction
        )
    )

    return ForecastResponse(
        predicted_clearance_rate=round(
            float(prediction),
            4,
        ),
        predicted_percentage=round(
            float(prediction) * 100,
            2,
        ),
        risk_level=risk_level,
        forecast=forecast_status,
        model_version=MODEL_METADATA[
            "model_version"
        ],
    )