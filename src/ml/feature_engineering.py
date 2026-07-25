import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["clearance_rate"] = (
        features["cleared_accounts"]
        / features["scheduled_accounts"]
    )

    features["authorization_completion_rate"] = (
        features["authorizations_completed"]
        / features["authorization_required"]
    ).fillna(0)

    features["denial_rate"] = (
        features["denials"]
        / features["scheduled_accounts"]
    )

    features["accounts_per_fte"] = (
        features["scheduled_accounts"]
        / features["staff_fte"]
    )

    features["revenue_per_account"] = (
        features["total_charges"]
        / features["scheduled_accounts"]
    )

    return features