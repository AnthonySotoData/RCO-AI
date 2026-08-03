from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.capstone_etl import run_capstone_etl
from src.data.capstone_reporting import (
    create_data_dictionary,
    create_descriptive_statistics,
    create_etl_summary,
)


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "revenue_cycle_operations.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "capstone_daily_analysis.csv"
)

PREPARED_DETAIL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "capstone_prepared_detail.csv"
)

VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "capstone_data_validation.csv"
)

DATA_DICTIONARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "capstone_data_dictionary.csv"
)

DESCRIPTIVE_STATISTICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "capstone_descriptive_statistics.csv"
)

ETL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "capstone"
    / "capstone_etl_summary.txt"
)


def main() -> None:
    daily_data, validation_report = run_capstone_etl(
        source_path=RAW_DATA_PATH,
        processed_path=PROCESSED_DATA_PATH,
        validation_path=VALIDATION_REPORT_PATH,
        prepared_detail_path=PREPARED_DETAIL_PATH,
    )

    data_dictionary = create_data_dictionary(
        output_path=DATA_DICTIONARY_PATH,
    )

    descriptive_statistics = create_descriptive_statistics(
        daily_data=daily_data,
        output_path=DESCRIPTIVE_STATISTICS_PATH,
    )

    etl_summary = create_etl_summary(
        validation_report=validation_report,
        processed_data_path=PROCESSED_DATA_PATH.relative_to(
            PROJECT_ROOT
        ),
        prepared_detail_path=PREPARED_DETAIL_PATH.relative_to(
            PROJECT_ROOT
        ),
        output_path=ETL_SUMMARY_PATH,
    )

    print("Capstone ETL and reporting completed successfully.")
    print()
    print("Generated data files:")
    print(f"- Prepared detail: {PREPARED_DETAIL_PATH}")
    print(f"- Daily analysis: {PROCESSED_DATA_PATH}")
    print()
    print("Generated report files:")
    print(f"- Validation report: {VALIDATION_REPORT_PATH}")
    print(f"- Data dictionary: {DATA_DICTIONARY_PATH}")
    print(f"- Descriptive statistics: {DESCRIPTIVE_STATISTICS_PATH}")
    print(f"- ETL summary: {ETL_SUMMARY_PATH}")
    print()
    print(f"Daily analysis rows: {len(daily_data):,}")
    print(f"Daily analysis columns: {len(daily_data.columns)}")
    print(f"Data dictionary entries: {len(data_dictionary)}")
    print(
        "Descriptive statistics variables: "
        f"{len(descriptive_statistics)}"
    )
    print()
    print("Validation summary:")
    print(validation_report.to_string(index=False))
    print()
    print(etl_summary)


if __name__ == "__main__":
    main()