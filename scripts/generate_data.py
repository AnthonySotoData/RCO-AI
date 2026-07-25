from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_generator import generate_dataset

OUTPUT_PATH = Path("data/raw/revenue_cycle_operations.csv")
DATASET_DAYS = 3000


def main() -> None:
    dataset = generate_dataset(days=DATASET_DAYS, seed=42)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset created successfully.")
    print(f"Rows: {len(dataset):,}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()