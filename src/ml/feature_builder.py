from pathlib import Path
import json

import pandas as pd

from src.models.forecast_api import ForecastRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_schema.json"
)


class FeatureBuilder:
    """
    Builds the exact feature vector expected by
    the trained forecasting model.
    """

    def __init__(self):

        with FEATURE_SCHEMA_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:
            schema = json.load(f)

        self.feature_columns = schema["feature_columns"]

        self.specialties = schema["specialty_categories"]
        self.payers = schema["payer_categories"]

        self.specialty_lookup = {}
        self.payer_lookup = {}

        for column in self.feature_columns:

            if column.startswith("specialty_"):
                category = column.replace(
                    "specialty_",
                    "",
                    1,
                )
                self.specialty_lookup[category] = column

            elif column.startswith("payer_"):
                category = column.replace(
                    "payer_",
                    "",
                    1,
                )
                self.payer_lookup[category] = column

    def build(
        self,
        request: ForecastRequest,
    ) -> pd.DataFrame:

        if request.specialty not in self.specialties:
            raise ValueError(
                f"Unknown specialty: {request.specialty}"
            )

        if request.payer not in self.payers:
            raise ValueError(
                f"Unknown payer: {request.payer}"
            )

        authorization_completion_rate = (
            request.authorizations_completed
            / max(request.authorization_required, 1)
        )

        denial_rate = (
            request.denials
            / max(request.scheduled_accounts, 1)
        )

        accounts_per_fte = (
            request.scheduled_accounts
            / request.staff_fte
        )

        revenue_per_account = (
            request.total_charges
            / max(request.scheduled_accounts, 1)
        )

        features = {
            column: 0.0
            for column in self.feature_columns
        }

        values = {

            "scheduled_accounts":
                request.scheduled_accounts,

            "authorization_required":
                request.authorization_required,

            "authorizations_completed":
                request.authorizations_completed,

            "denials":
                request.denials,

            "total_charges":
                request.total_charges,

            "denied_charges":
                request.denied_charges,

            "average_turnaround_hours":
                request.average_turnaround_hours,

            "work_queue_volume":
                request.work_queue_volume,

            "staff_fte":
                request.staff_fte,

            "authorization_completion_rate":
                authorization_completion_rate,

            "denial_rate":
                denial_rate,

            "accounts_per_fte":
                accounts_per_fte,

            "revenue_per_account":
                revenue_per_account,

            "clearance_rate":
                request.clearance_rate,

            "clearance_rate_lag_1":
                request.clearance_rate_lag_1,

            "clearance_rate_lag_2":
                request.clearance_rate_lag_2,

            "clearance_rate_lag_7":
                request.clearance_rate_lag_7,

            "clearance_rate_rolling_7":
                request.clearance_rate_rolling_7,

            "queue_rolling_7":
                request.queue_rolling_7,

            "turnaround_rolling_7":
                request.turnaround_rolling_7,

            "day_of_week":
                request.day_of_week,

            "month":
                request.month,
        }

        for column, value in values.items():
            if column in features:
                features[column] = value

        specialty_column = self.specialty_lookup.get(
            request.specialty
        )

        if specialty_column:
            features[specialty_column] = 1.0

        payer_column = self.payer_lookup.get(
            request.payer
        )

        if payer_column:
            features[payer_column] = 1.0

        return pd.DataFrame(
            [features],
            columns=self.feature_columns,
        )