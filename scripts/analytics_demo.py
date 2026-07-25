from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.analytics_engine import AnalyticsEngine
from src.analytics.payer_analytics import PayerAnalytics

df = pd.read_csv("data/raw/revenue_cycle_operations.csv")

analytics = AnalyticsEngine(df)
payer = PayerAnalytics(df)

print("\nOverall Metrics")
print(analytics.overall_metrics())

print("\nSpecialty Summary")
print(analytics.specialty_summary())

print("\nPayer Summary")
print(payer.summary())