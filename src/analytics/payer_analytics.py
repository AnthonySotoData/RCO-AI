import pandas as pd


class PayerAnalytics:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def summary(self) -> pd.DataFrame:
        grouped = (
            self.df.groupby("payer")
            .agg(
                scheduled_accounts=("scheduled_accounts", "sum"),
                cleared_accounts=("cleared_accounts", "sum"),
                denied_charges=("denied_charges", "sum"),
                average_turnaround_hours=(
                    "average_turnaround_hours",
                    "mean",
                ),
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