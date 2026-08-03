from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.statistical_analysis import (
    run_primary_analysis,
    run_sensitivity_analysis,
)
from src.analytics.statistical_reporting import (
    create_confidence_interval_table,
    create_diagnostics_table,
    create_executive_summary,
    create_key_metrics_table,
)


ANALYSIS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "capstone_daily_analysis.csv"
)

REGRESSION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "regression"
)

TABLE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tables"
)

CAPSTONE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "capstone"
)


def _format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.6f}"


def _create_hypothesis_summary(
    analysis_name: str,
    regression_result: object,
    correlation_result: object,
) -> str:
    coefficient = regression_result.coefficient
    confidence_lower = (
        regression_result.confidence_interval_lower
    )
    confidence_upper = (
        regression_result.confidence_interval_upper
    )

    direction = (
        "positive"
        if coefficient > 0
        else "negative"
    )

    return "\n".join(
        [
            "=" * 72,
            f"{analysis_name.upper()} - HYPOTHESIS TEST SUMMARY",
            "=" * 72,
            "",
            (
                "Independent variable: "
                f"{regression_result.independent_variable}"
            ),
            (
                "Dependent variable: "
                f"{regression_result.dependent_variable}"
            ),
            (
                "Number of observations: "
                f"{regression_result.observation_count:,}"
            ),
            "",
            "CORRELATION",
            "-" * 72,
            (
                "Pearson correlation coefficient: "
                f"{correlation_result.pearson_correlation:.6f}"
            ),
            (
                "Correlation p-value: "
                f"{_format_p_value(correlation_result.p_value)}"
            ),
            "",
            "LINEAR REGRESSION",
            "-" * 72,
            (
                "Regression coefficient: "
                f"{coefficient:.6f}"
            ),
            (
                "95% confidence interval: "
                f"[{confidence_lower:.6f}, "
                f"{confidence_upper:.6f}]"
            ),
            (
                "Coefficient p-value: "
                f"{_format_p_value(regression_result.p_value)}"
            ),
            (
                "R-squared: "
                f"{regression_result.r_squared:.6f}"
            ),
            (
                "Adjusted R-squared: "
                f"{regression_result.adjusted_r_squared:.6f}"
            ),
            "",
            "HYPOTHESIS DECISION",
            "-" * 72,
            (
                "Significance level: "
                f"{regression_result.significance_level:.2f}"
            ),
            (
                "Decision: "
                f"{regression_result.hypothesis_decision}"
            ),
            (
                "Statistically significant: "
                f"{regression_result.statistically_significant}"
            ),
            "",
            "PLAIN-LANGUAGE INTERPRETATION",
            "-" * 72,
            (
                "The analysis identified a statistically significant "
                f"{direction} relationship between "
                f"{regression_result.independent_variable} and "
                f"{regression_result.dependent_variable}."
                if regression_result.statistically_significant
                else
                "The analysis did not identify a statistically "
                f"significant relationship between "
                f"{regression_result.independent_variable} and "
                f"{regression_result.dependent_variable}."
            ),
            (
                "The coefficient estimates the average change in the "
                "dependent variable associated with a one-unit increase "
                "in authorization turnaround time."
            ),
            "",
            "LIMITATION",
            "-" * 72,
            (
                "The analysis uses synthetic operational data and "
                "identifies statistical association rather than causation."
            ),
            "",
            "=" * 72,
        ]
    )


def _export_analysis(
    analysis_name: str,
    analysis_results: dict[str, object],
) -> None:
    correlation_table = analysis_results[
        "correlation_table"
    ]

    regression_table = analysis_results[
        "regression_table"
    ]

    regression_model = analysis_results[
        "regression_model"
    ]

    prediction_data = analysis_results[
        "prediction_data"
    ]

    correlation_result = analysis_results[
        "correlation_result"
    ]

    regression_result = analysis_results[
        "regression_result"
    ]

    correlation_path = (
        TABLE_OUTPUT_DIR
        / f"{analysis_name}_correlation_results.csv"
    )

    regression_path = (
        REGRESSION_OUTPUT_DIR
        / f"{analysis_name}_regression_results.csv"
    )

    model_summary_path = (
        REGRESSION_OUTPUT_DIR
        / f"{analysis_name}_model_summary.txt"
    )

    prediction_path = (
        REGRESSION_OUTPUT_DIR
        / f"{analysis_name}_prediction_residuals.csv"
    )

    hypothesis_path = (
        CAPSTONE_OUTPUT_DIR
        / f"{analysis_name}_hypothesis_summary.txt"
    )

    correlation_table.to_csv(
        correlation_path,
        index=False,
    )

    regression_table.to_csv(
        regression_path,
        index=False,
    )

    prediction_data.to_csv(
        prediction_path,
        index=False,
    )

    model_summary_path.write_text(
        regression_model.summary().as_text(),
        encoding="utf-8",
    )

    hypothesis_summary = _create_hypothesis_summary(
        analysis_name=analysis_name,
        regression_result=regression_result,
        correlation_result=correlation_result,
    )

    hypothesis_path.write_text(
        hypothesis_summary,
        encoding="utf-8",
    )

    print()
    print(f"{analysis_name.replace('_', ' ').title()}:")
    print(
        f"- Pearson correlation: "
        f"{correlation_result.pearson_correlation:.6f}"
    )
    print(
        f"- Correlation p-value: "
        f"{_format_p_value(correlation_result.p_value)}"
    )
    print(
        f"- Regression coefficient: "
        f"{regression_result.coefficient:.6f}"
    )
    print(
        f"- Regression p-value: "
        f"{_format_p_value(regression_result.p_value)}"
    )
    print(
        f"- R-squared: "
        f"{regression_result.r_squared:.6f}"
    )
    print(
        f"- Decision: "
        f"{regression_result.hypothesis_decision}"
    )


def main() -> None:
    if not ANALYSIS_DATA_PATH.exists():
        raise FileNotFoundError(
            "The analysis-ready dataset was not found. "
            "Run scripts/run_capstone_etl.py first."
        )

    REGRESSION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CAPSTONE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_data = pd.read_csv(
        ANALYSIS_DATA_PATH,
        parse_dates=["record_date"],
    )

    primary_results = run_primary_analysis(
        dataframe=analysis_data,
    )

    sensitivity_results = run_sensitivity_analysis(
        dataframe=analysis_data,
    )

    key_metrics = create_key_metrics_table(
        primary_correlation=primary_results[
            "correlation_result"
        ],
        primary_regression=primary_results[
            "regression_result"
        ],
        sensitivity_correlation=sensitivity_results[
            "correlation_result"
        ],
        sensitivity_regression=sensitivity_results[
            "regression_result"
        ],
    )

    primary_confidence_interval = (
        create_confidence_interval_table(
            analysis_name="Primary",
            regression_result=primary_results[
                "regression_result"
            ],
        )
    )

    sensitivity_confidence_interval = (
        create_confidence_interval_table(
            analysis_name="Sensitivity",
            regression_result=sensitivity_results[
                "regression_result"
            ],
        )
    )

    confidence_intervals = pd.concat(
        [
            primary_confidence_interval,
            sensitivity_confidence_interval,
        ],
        ignore_index=True,
    )

    primary_diagnostics = create_diagnostics_table(
        analysis_name="Primary",
        model=primary_results["regression_model"],
    )

    sensitivity_diagnostics = create_diagnostics_table(
        analysis_name="Sensitivity",
        model=sensitivity_results["regression_model"],
    )

    diagnostics = pd.concat(
        [
            primary_diagnostics,
            sensitivity_diagnostics,
        ],
        ignore_index=True,
    )

    key_metrics.to_csv(
        TABLE_OUTPUT_DIR / "capstone_key_metrics.csv",
        index=False,
    )

    confidence_intervals.to_csv(
        REGRESSION_OUTPUT_DIR
        / "capstone_confidence_intervals.csv",
        index=False,
    )

    diagnostics.to_csv(
        REGRESSION_OUTPUT_DIR
        / "capstone_regression_diagnostics.csv",
        index=False,
    )

    executive_summary = create_executive_summary(
        primary_correlation=primary_results[
            "correlation_result"
        ],
        primary_regression=primary_results[
            "regression_result"
        ],
        sensitivity_correlation=sensitivity_results[
            "correlation_result"
        ],
        sensitivity_regression=sensitivity_results[
            "regression_result"
        ],
        output_path=(
            CAPSTONE_OUTPUT_DIR
            / "capstone_executive_statistical_summary.txt"
        ),
    )

    _export_analysis(
        analysis_name="primary",
        analysis_results=primary_results,
    )

    _export_analysis(
        analysis_name="sensitivity",
        analysis_results=sensitivity_results,
    )

    print()
    print("Generated standardized reporting artifacts:")
    print(
        "- Key metrics: "
        f"{TABLE_OUTPUT_DIR / 'capstone_key_metrics.csv'}"
    )
    print(
        "- Confidence intervals: "
        f"{REGRESSION_OUTPUT_DIR / 'capstone_confidence_intervals.csv'}"
    )
    print(
        "- Regression diagnostics: "
        f"{REGRESSION_OUTPUT_DIR / 'capstone_regression_diagnostics.csv'}"
    )
    print(
        "- Executive summary: "
        f"{CAPSTONE_OUTPUT_DIR / 'capstone_executive_statistical_summary.txt'}"
    )
    print()
    print(executive_summary)

    print()
    print("Capstone statistical analysis completed successfully.")
    print(f"Analysis dataset: {ANALYSIS_DATA_PATH}")
    print(f"Regression outputs: {REGRESSION_OUTPUT_DIR}")
    print(f"Capstone summaries: {CAPSTONE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()