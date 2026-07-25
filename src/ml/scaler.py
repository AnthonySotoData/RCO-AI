from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler


SCALER_PATH = Path("models") / "feature_scaler.pkl"


def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    SCALER_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled


def transform_features(X):
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(X)