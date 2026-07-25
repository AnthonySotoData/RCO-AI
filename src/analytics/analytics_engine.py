import pandas as pd


class AnalyticsEngine:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def overall_metrics(self) -> dict:
        df = self.df

        clearance_rate = (
            df["cleared_accounts"].sum()
            / df["scheduled_accounts"].sum()
        ) * 100

        auth_completion = (
            df["authorizations_completed"].sum()
            / df["authorization_required"].sum()
        ) * 100

        denial_rate = (
            df["denials"].sum()
            / df["scheduled_accounts"].sum()
        ) * 100

        revenue_at_risk = df["denied_charges"].sum()

        return {
            "clearance_rate": round(clearance_rate, 2),
            "authorization_completion_rate": round(auth_completion, 2),
            "denial_rate": round(denial_rate, 2),
            "revenue_at_risk": round(revenue_at_risk, 2),
        }

    def specialty_summary(self) -> pd.DataFrame:
        grouped = (
            self.df.groupby("specialty")
            .agg(
                scheduled_accounts=("scheduled_accounts", "sum"),
                cleared_accounts=("cleared_accounts", "sum"),
                denied_charges=("denied_charges", "sum"),
                work_queue_volume=("work_queue_volume", "mean"),
            )
            .reset_index()
        )

        grouped["clearance_rate"] = (
            grouped["cleared_accounts"]
            / grouped["scheduled_accounts"]
            * 100
        ).round(2)

        return grouped.sort_values(
            by="clearance_rate",
            ascending=False,
        )