from fastapi import FastAPI

from src.api.metrics import (
    router as metrics_router,
)
from src.api.forecast import (
    router as forecast_router,
)


app = FastAPI(
    title="RCO AI",
    description=(
        "Predictive Healthcare Revenue Cycle "
        "Operations Intelligence"
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "application": "RCO AI",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


app.include_router(
    metrics_router
)

app.include_router(
    forecast_router
)