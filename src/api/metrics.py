from fastapi import APIRouter

from src.models.schemas import MetricsRequest, MetricsResponse
from src.services.metrics_service import calculate_operational_metrics


router = APIRouter(
    prefix="/metrics",
    tags=["Operational Metrics"],
)


@router.post("", response_model=MetricsResponse)
def calculate_metrics(request: MetricsRequest) -> MetricsResponse:
    metrics = calculate_operational_metrics(request.record)

    return MetricsResponse(
        specialty=request.record.specialty,
        metrics=metrics,
    )