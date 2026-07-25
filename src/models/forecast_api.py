from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    """
    Operational and historical inputs used to forecast
    the next-day financial clearance rate.
    """

    specialty: str = Field(
        ...,
        min_length=1,
        description="Clinical specialty.",
    )

    payer: str = Field(
        ...,
        min_length=1,
        description="Insurance payer.",
    )

    scheduled_accounts: int = Field(
        ...,
        ge=1,
        description="Number of scheduled accounts.",
    )

    authorization_required: int = Field(
        ...,
        ge=0,
        description="Accounts requiring authorization.",
    )

    authorizations_completed: int = Field(
        ...,
        ge=0,
        description="Completed authorizations.",
    )

    denials: int = Field(
        ...,
        ge=0,
        description="Number of denied accounts.",
    )

    total_charges: float = Field(
        ...,
        ge=0,
        description="Total charges.",
    )

    denied_charges: float = Field(
        ...,
        ge=0,
        description="Charges associated with denials.",
    )

    average_turnaround_hours: float = Field(
        ...,
        ge=0,
        description="Average turnaround time in hours.",
    )

    work_queue_volume: int = Field(
        ...,
        ge=0,
        description="Current work-queue volume.",
    )

    staff_fte: float = Field(
        ...,
        gt=0,
        description="Staffing measured in FTEs.",
    )

    clearance_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current-day clearance rate.",
    )

    clearance_rate_lag_1: float = Field(
        ...,
        ge=0,
        le=1,
        description="Clearance rate one day earlier.",
    )

    clearance_rate_lag_2: float = Field(
        ...,
        ge=0,
        le=1,
        description="Clearance rate two days earlier.",
    )

    clearance_rate_lag_7: float = Field(
        ...,
        ge=0,
        le=1,
        description="Clearance rate seven days earlier.",
    )

    clearance_rate_rolling_7: float = Field(
        ...,
        ge=0,
        le=1,
        description="Average clearance rate over seven days.",
    )

    queue_rolling_7: float = Field(
        ...,
        ge=0,
        description="Average work-queue volume over seven days.",
    )

    turnaround_rolling_7: float = Field(
        ...,
        ge=0,
        description="Average turnaround time over seven days.",
    )

    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day number where Monday is 0 and Sunday is 6.",
    )

    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Calendar month from 1 through 12.",
    )

    @model_validator(mode="after")
    def validate_business_rules(
        self,
    ) -> "ForecastRequest":

        if self.authorization_required > self.scheduled_accounts:
            raise ValueError(
                "authorization_required cannot exceed "
                "scheduled_accounts."
            )

        if (
            self.authorizations_completed
            > self.authorization_required
        ):
            raise ValueError(
                "authorizations_completed cannot exceed "
                "authorization_required."
            )

        if self.denials > self.scheduled_accounts:
            raise ValueError(
                "denials cannot exceed scheduled_accounts."
            )

        if self.denied_charges > self.total_charges:
            raise ValueError(
                "denied_charges cannot exceed total_charges."
            )

        return self


class ForecastResponse(BaseModel):
    """
    Forecast returned by the prediction API.
    """

    predicted_clearance_rate: float = Field(
        ...,
        ge=0,
        le=1,
    )

    predicted_percentage: float = Field(
        ...,
        ge=0,
        le=100,
    )

    risk_level: str

    forecast: str

    model_version: str