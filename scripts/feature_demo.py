from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.feature_engineering import prepare_features

df = pd.read_csv("data/raw/revenue_cycle_operations.csv")

engineered = prepare_features(df)

print(engineered.head())

print("\nColumns\n")
print(engineered.columns.tolist())