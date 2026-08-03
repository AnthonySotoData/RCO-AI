from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import (
    RegressionResultsWrapper,
)
from statsmodels.stats.stattools import (
    durbin_watson,
    jarque_bera,
    omni_normtest,
)

from src.analytics.statistical_analysis import (
    CorrelationResult,
    RegressionResult,
)


def format_p_value(p_value: float) -> str:
    """Format very small p-values without reporting them as zero."""
    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.6f}"


def create_key_metrics_table(
    primary_correlation: CorrelationResult,
    primary_regression: RegressionResult,
    sensitivity_correlation: CorrelationResult,
    sensitivity_regression: RegressionResult,
) -> pd.DataFrame:
    """Create a concise comparison of the primary and sensitivity models."""
    rows = [
        {
            "analysis": "Primary",
            "independent_variable": (
                primary_regression.independent_variable
            ),
            "dependent_variable": (
                primary_regression.dependent_variable
            ),
            "observations": primary_regression.observation_count,
            "pearson_correlation": round(
                primary_correlation.pearson_correlation,
                6,
            ),
            "correlation_p_value": format_p_value(
                primary_correlation.p_value
            ),
            "regression_coefficient": round(
                primary_regression.coefficient,
                6,
            ),
            "coefficient_p_value": format_p_value(
                primary_regression.p_value
            ),
            "r_squared": round(
                primary_regression.r_squared,
                6,
            ),
            "adjusted_r_squared": round(
                primary_regression.adjusted_r_squared,
                6,
            ),
            "hypothesis_decision": (
                primary_regression.hypothesis_decision
            ),
        },
        {
            "analysis": "Sensitivity",
            "independent_variable": (
                sensitivity_regression.independent_variable
            ),
            "dependent_variable": (
                sensitivity_regression.dependent_variable
            ),
            "observations": sensitivity_regression.observation_count,
            "pearson_correlation": round(
                sensitivity_correlation.pearson_correlation,
                6,
            ),
            "correlation_p_value": format_p_value(
                sensitivity_correlation.p_value
            ),
            "regression_coefficient": round(
                sensitivity_regression.coefficient,
                6,
            ),
            "coefficient_p_value": format_p_value(
                sensitivity_regression.p_value
            ),
            "r_squared": round(
                sensitivity_regression.r_squared,
                6,
            ),
            "adjusted_r_squared": round(
                sensitivity_regression.adjusted_r_squared,
                6,
            ),
            "hypothesis_decision": (
                sensitivity_regression.hypothesis_decision
            ),
        },
    ]

    return pd.DataFrame(rows)


def create_confidence_interval_table(
    analysis_name: str,
    regression_result: RegressionResult,
) -> pd.DataFrame:
    """Create a clean coefficient confidence-interval table."""
    return pd.DataFrame(
        [
            {
                "analysis": analysis_name,
                "variable": (
                    regression_result.independent_variable
                ),
                "coefficient": round(
                    regression_result.coefficient,
                    6,
                ),
                "standard_error": round(
                    regression_result.standard_error,
                    6,
                ),
                "confidence_level": "95%",
                "confidence_interval_lower": round(
                    regression_result.confidence_interval_lower,
                    6,
                ),
                "confidence_interval_upper": round(
                    regression_result.confidence_interval_upper,
                    6,
                ),
                "p_value": format_p_value(
                    regression_result.p_value
                ),
            }
        ]
    )


def create_diagnostics_table(
    analysis_name: str,
    model: RegressionResultsWrapper,
) -> pd.DataFrame:
    """Calculate report-ready residual and model diagnostics."""
    residuals = np.asarray(model.resid, dtype=float)
    predicted_values = np.asarray(
        model.fittedvalues,
        dtype=float,
    )
    observed_values = predicted_values + residuals

    jb_statistic, jb_p_value, skewness, kurtosis = (
        jarque_bera(residuals)
    )

    omnibus_statistic, omnibus_p_value = (
        omni_normtest(residuals)
    )

    mean_absolute_error = float(
        np.mean(np.abs(residuals))
    )

    root_mean_squared_error = float(
        np.sqrt(np.mean(np.square(residuals)))
    )

    return pd.DataFrame(
        [
            {
                "analysis": analysis_name,
                "observation_count": int(model.nobs),
                "residual_mean": round(
                    float(np.mean(residuals)),
                    6,
                ),
                "residual_standard_deviation": round(
                    float(np.std(residuals, ddof=1)),
                    6,
                ),
                "mean_absolute_error": round(
                    mean_absolute_error,
                    6,
                ),
                "root_mean_squared_error": round(
                    root_mean_squared_error,
                    6,
                ),
                "durbin_watson_statistic": round(
                    float(durbin_watson(residuals)),
                    6,
                ),
                "jarque_bera_statistic": round(
                    float(jb_statistic),
                    6,
                ),
                "jarque_bera_p_value": format_p_value(
                    float(jb_p_value)
                ),
                "omnibus_statistic": round(
                    float(omnibus_statistic),
                    6,
                ),
                "omnibus_p_value": format_p_value(
                    float(omnibus_p_value)
                ),
                "residual_skewness": round(
                    float(skewness),
                    6,
                ),
                "residual_kurtosis": round(
                    float(kurtosis),
                    6,
                ),
                "minimum_observed_value": round(
                    float(np.min(observed_values)),
                    6,
                ),
                "maximum_observed_value": round(
                    float(np.max(observed_values)),
                    6,
                ),
                "minimum_predicted_value": round(
                    float(np.min(predicted_values)),
                    6,
                ),
                "maximum_predicted_value": round(
                    float(np.max(predicted_values)),
                    6,
                ),
            }
        ]
    )


def create_executive_summary(
    primary_correlation: CorrelationResult,
    primary_regression: RegressionResult,
    sensitivity_correlation: CorrelationResult,
    sensitivity_regression: RegressionResult,
    output_path: Path,
) -> str:
    """Create a plain-language summary of the inferential findings."""
    coefficient = primary_regression.coefficient
    explained_variation = (
        primary_regression.r_squared * 100
    )
    sensitivity_variation = (
        sensitivity_regression.r_squared * 100
    )

    summary = "\n".join(
        [
            "=" * 72,
            "RCO AI CAPSTONE - EXECUTIVE STATISTICAL SUMMARY",
            "=" * 72,
            "",
            "RESEARCH QUESTION",
            "-" * 72,
            (
                "To what extent does authorization turnaround time "
                "affect denied charges in healthcare revenue cycle "
                "operations?"
            ),
            "",
            "PRIMARY FINDING",
            "-" * 72,
            (
                "A strong positive relationship was identified between "
                "authorization turnaround time and daily denied charges "
                f"(Pearson r = "
                f"{primary_correlation.pearson_correlation:.3f}, "
                f"p {format_p_value(primary_correlation.p_value)})."
            ),
            (
                "The linear regression coefficient was "
                f"${coefficient:,.2f}. Within the synthetic dataset, "
                "this means that each additional hour of average "
                "authorization turnaround time was associated with an "
                f"estimated ${coefficient:,.2f} increase in daily "
                "denied charges."
            ),
            (
                "Authorization turnaround time explained approximately "
                f"{explained_variation:.1f}% of the variation in daily "
                "denied charges in the simple regression model "
                f"(R² = {primary_regression.r_squared:.3f})."
            ),
            (
                "The coefficient was statistically significant "
                f"(p {format_p_value(primary_regression.p_value)}), "
                "resulting in rejection of the null hypothesis."
            ),
            "",
            "SENSITIVITY ANALYSIS",
            "-" * 72,
            (
                "A secondary analysis used denied-charge rate to account "
                "for differences in total daily charge volume."
            ),
            (
                "The relationship remained positive and statistically "
                "significant "
                f"(Pearson r = "
                f"{sensitivity_correlation.pearson_correlation:.3f}, "
                f"R² = {sensitivity_regression.r_squared:.3f}, "
                f"p {format_p_value(sensitivity_regression.p_value)})."
            ),
            (
                "The sensitivity model explained approximately "
                f"{sensitivity_variation:.1f}% of the variation in "
                "denied-charge rate."
            ),
            "",
            "INTERPRETATION AND LIMITATION",
            "-" * 72,
            (
                "The results support a statistically significant "
                "association between longer authorization turnaround "
                "times and increased denied-charge risk."
            ),
            (
                "The findings do not establish causation. The dataset "
                "is synthetic, and its relationships reflect the "
                "assumptions used to simulate healthcare revenue cycle "
                "operations."
            ),
            "",
            "=" * 72,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        summary,
        encoding="utf-8",
    )

    return summary