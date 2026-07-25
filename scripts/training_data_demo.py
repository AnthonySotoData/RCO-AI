from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data_preparation import prepare_training_data

df = pd.read_csv("data/raw/revenue_cycle_operations.csv")

X_train, X_test, y_train, y_test = prepare_training_data(df)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

print(f"\nFeatures: {X_train.shape[1]}")

print("\nFirst five feature names:")

print(X_train.columns[:5].tolist())