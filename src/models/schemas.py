from datetime import date

from pydantic import BaseModel, Field


class RevenueCycleRecord(BaseModel):
    record_date: date
    specialty: str
    payer: str

    scheduled_accounts: int = Field(ge=0)
    cleared_accounts: int = Field(ge=0)
    authorization_required: int = Field(ge=0)
    authorizations_completed: int = Field(ge=0)
    denials: int = Field(ge=0)
    total_charges: float = Field(ge=0)
    denied_charges: float = Field(ge=0)
    average_turnaround_hours: float = Field(ge=0)
    work_queue_volume: int = Field(ge=0)
    staff_fte: float = Field(gt=0)


class OperationalMetrics(BaseModel):
    clearance_rate: float
    authorization_completion_rate: float
    denial_rate: float
    revenue_at_risk: float
    accounts_per_fte: float


class SpecialtyPerformance(BaseModel):
    specialty: str
    metrics: OperationalMetrics
    risk_level: str
    primary_driver: str

class MetricsRequest(BaseModel):
    record: RevenueCycleRecord


class MetricsResponse(BaseModel):
    specialty: str
    metrics: OperationalMetrics


class ForecastRequest(BaseModel):
    specialty: str
    historical_clearance_rates: list[float] = Field(min_length=3)


class ForecastResponse(BaseModel):
    specialty: str
    predicted_clearance_rate: float
    target_clearance_rate: float = 95.0
    risk_level: str


class Recommendation(BaseModel):
    priority: str
    action: str
    rationale: str
    estimated_impact: str


class ExecutiveSummary(BaseModel):
    overall_status: str
    top_concerns: list[str]
    forecasted_clearance_rate: float
    revenue_at_risk: float
    recommended_actions: list[Recommendation]    