from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


PRIMARY_INDEPENDENT_VARIABLE = "average_turnaround_hours"
PRIMARY_DEPENDENT_VARIABLE = "denied_charges"
SENSITIVITY_DEPENDENT_VARIABLE = "denied_charge_rate"


@dataclass(frozen=True)
class CorrelationResult:
    independent_variable: str
    dependent_variable: str
    observation_count: int
    pearson_correlation: float
    p_value: float
    significance_level: float
    statistically_significant: bool


@dataclass(frozen=True)
class RegressionResult:
    independent_variable: str
    dependent_variable: str
    observation_count: int
    intercept: float
    coefficient: float
    standard_error: float
    t_statistic: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_p_value: float
    residual_mean: float
    residual_standard_deviation: float
    significance_level: float
    statistically_significant: bool
    hypothesis_decision: str


def validate_analysis_data(
    dataframe: pd.DataFrame,
    independent_variable: str,
    dependent_variable: str,
) -> pd.DataFrame:
    """Validate and return complete numeric observations for analysis."""
    required_columns = [
        independent_variable,
        dependent_variable,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The analysis dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    analysis_data = dataframe[required_columns].copy()

    for column in required_columns:
        analysis_data[column] = pd.to_numeric(
            analysis_data[column],
            errors="coerce",
        )

    analysis_data = analysis_data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(analysis_data) < 3:
        raise ValueError(
            "At least three complete observations are required."
        )

    if analysis_data[independent_variable].nunique() < 2:
        raise ValueError(
            f"{independent_variable} must contain variation."
        )

    if analysis_data[dependent_variable].nunique() < 2:
        raise ValueError(
            f"{dependent_variable} must contain variation."
        )

    return analysis_data


def calculate_pearson_correlation(
    dataframe: pd.DataFrame,
    independent_variable: str = PRIMARY_INDEPENDENT_VARIABLE,
    dependent_variable: str = PRIMARY_DEPENDENT_VARIABLE,
    significance_level: float = 0.05,
) -> CorrelationResult:
    """Calculate Pearson correlation and its two-sided p-value."""
    analysis_data = validate_analysis_data(
        dataframe=dataframe,
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
    )

    correlation, p_value = stats.pearsonr(
        analysis_data[independent_variable],
        analysis_data[dependent_variable],
    )

    return CorrelationResult(
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
        observation_count=len(analysis_data),
        pearson_correlation=float(correlation),
        p_value=float(p_value),
        significance_level=significance_level,
        statistically_significant=bool(
            p_value < significance_level
        ),
    )


def fit_linear_regression(
    dataframe: pd.DataFrame,
    independent_variable: str = PRIMARY_INDEPENDENT_VARIABLE,
    dependent_variable: str = PRIMARY_DEPENDENT_VARIABLE,
    significance_level: float = 0.05,
) -> tuple[RegressionResult, RegressionResultsWrapper]:
    """Fit an ordinary least squares simple linear regression model."""
    analysis_data = validate_analysis_data(
        dataframe=dataframe,
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
    )

    independent_values = sm.add_constant(
        analysis_data[[independent_variable]],
        has_constant="add",
    )

    dependent_values = analysis_data[dependent_variable]

    model = sm.OLS(
        dependent_values,
        independent_values,
    ).fit()

    confidence_interval = model.conf_int(
        alpha=significance_level
    ).loc[independent_variable]

    coefficient_p_value = float(
        model.pvalues[independent_variable]
    )

    statistically_significant = bool(
        coefficient_p_value < significance_level
    )

    hypothesis_decision = (
        "Reject the null hypothesis."
        if statistically_significant
        else "Fail to reject the null hypothesis."
    )

    residuals = model.resid

    result = RegressionResult(
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
        observation_count=int(model.nobs),
        intercept=float(model.params["const"]),
        coefficient=float(
            model.params[independent_variable]
        ),
        standard_error=float(
            model.bse[independent_variable]
        ),
        t_statistic=float(
            model.tvalues[independent_variable]
        ),
        p_value=coefficient_p_value,
        confidence_interval_lower=float(
            confidence_interval.iloc[0]
        ),
        confidence_interval_upper=float(
            confidence_interval.iloc[1]
        ),
        r_squared=float(model.rsquared),
        adjusted_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        f_p_value=float(model.f_pvalue),
        residual_mean=float(residuals.mean()),
        residual_standard_deviation=float(
            residuals.std(ddof=1)
        ),
        significance_level=significance_level,
        statistically_significant=statistically_significant,
        hypothesis_decision=hypothesis_decision,
    )

    return result, model


def create_correlation_table(
    correlation_result: CorrelationResult,
) -> pd.DataFrame:
    """Convert a correlation result into a report-ready table."""
    return pd.DataFrame(
        [
            {
                "independent_variable": (
                    correlation_result.independent_variable
                ),
                "dependent_variable": (
                    correlation_result.dependent_variable
                ),
                "observation_count": (
                    correlation_result.observation_count
                ),
                "pearson_correlation": round(
                    correlation_result.pearson_correlation,
                    6,
                ),
                "p_value": correlation_result.p_value,
                "significance_level": (
                    correlation_result.significance_level
                ),
                "statistically_significant": (
                    correlation_result.statistically_significant
                ),
            }
        ]
    )


def create_regression_table(
    regression_result: RegressionResult,
) -> pd.DataFrame:
    """Convert a regression result into a report-ready table."""
    return pd.DataFrame(
        [
            {
                "independent_variable": (
                    regression_result.independent_variable
                ),
                "dependent_variable": (
                    regression_result.dependent_variable
                ),
                "observation_count": (
                    regression_result.observation_count
                ),
                "intercept": round(
                    regression_result.intercept,
                    6,
                ),
                "coefficient": round(
                    regression_result.coefficient,
                    6,
                ),
                "standard_error": round(
                    regression_result.standard_error,
                    6,
                ),
                "t_statistic": round(
                    regression_result.t_statistic,
                    6,
                ),
                "p_value": regression_result.p_value,
                "confidence_interval_lower": round(
                    regression_result.confidence_interval_lower,
                    6,
                ),
                "confidence_interval_upper": round(
                    regression_result.confidence_interval_upper,
                    6,
                ),
                "r_squared": round(
                    regression_result.r_squared,
                    6,
                ),
                "adjusted_r_squared": round(
                    regression_result.adjusted_r_squared,
                    6,
                ),
                "f_statistic": round(
                    regression_result.f_statistic,
                    6,
                ),
                "f_p_value": regression_result.f_p_value,
                "residual_mean": round(
                    regression_result.residual_mean,
                    6,
                ),
                "residual_standard_deviation": round(
                    regression_result.residual_standard_deviation,
                    6,
                ),
                "significance_level": (
                    regression_result.significance_level
                ),
                "statistically_significant": (
                    regression_result.statistically_significant
                ),
                "hypothesis_decision": (
                    regression_result.hypothesis_decision
                ),
            }
        ]
    )


def create_prediction_data(
    dataframe: pd.DataFrame,
    model: RegressionResultsWrapper,
    independent_variable: str = PRIMARY_INDEPENDENT_VARIABLE,
    dependent_variable: str = PRIMARY_DEPENDENT_VARIABLE,
) -> pd.DataFrame:
    """Create observed, predicted, and residual values for diagnostics."""
    analysis_data = validate_analysis_data(
        dataframe=dataframe,
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
    )

    independent_values = sm.add_constant(
        analysis_data[[independent_variable]],
        has_constant="add",
    )

    prediction_data = analysis_data.copy()

    prediction_data["predicted_value"] = model.predict(
        independent_values
    )

    prediction_data["residual"] = (
        prediction_data[dependent_variable]
        - prediction_data["predicted_value"]
    )

    return prediction_data.reset_index(drop=True)


def run_primary_analysis(
    dataframe: pd.DataFrame,
    significance_level: float = 0.05,
) -> dict[str, object]:
    """Run the approved primary correlation and regression analysis."""
    correlation_result = calculate_pearson_correlation(
        dataframe=dataframe,
        independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
        dependent_variable=PRIMARY_DEPENDENT_VARIABLE,
        significance_level=significance_level,
    )

    regression_result, regression_model = (
        fit_linear_regression(
            dataframe=dataframe,
            independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
            dependent_variable=PRIMARY_DEPENDENT_VARIABLE,
            significance_level=significance_level,
        )
    )

    prediction_data = create_prediction_data(
        dataframe=dataframe,
        model=regression_model,
        independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
        dependent_variable=PRIMARY_DEPENDENT_VARIABLE,
    )

    return {
        "correlation_result": correlation_result,
        "correlation_table": create_correlation_table(
            correlation_result
        ),
        "regression_result": regression_result,
        "regression_table": create_regression_table(
            regression_result
        ),
        "regression_model": regression_model,
        "prediction_data": prediction_data,
    }


def run_sensitivity_analysis(
    dataframe: pd.DataFrame,
    significance_level: float = 0.05,
) -> dict[str, object]:
    """
    Test turnaround time against denied-charge rate.

    This secondary analysis adjusts the financial outcome for total
    charge volume and does not replace the approved primary analysis.
    """
    correlation_result = calculate_pearson_correlation(
        dataframe=dataframe,
        independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
        dependent_variable=SENSITIVITY_DEPENDENT_VARIABLE,
        significance_level=significance_level,
    )

    regression_result, regression_model = (
        fit_linear_regression(
            dataframe=dataframe,
            independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
            dependent_variable=SENSITIVITY_DEPENDENT_VARIABLE,
            significance_level=significance_level,
        )
    )

    prediction_data = create_prediction_data(
        dataframe=dataframe,
        model=regression_model,
        independent_variable=PRIMARY_INDEPENDENT_VARIABLE,
        dependent_variable=SENSITIVITY_DEPENDENT_VARIABLE,
    )

    return {
        "correlation_result": correlation_result,
        "correlation_table": create_correlation_table(
            correlation_result
        ),
        "regression_result": regression_result,
        "regression_table": create_regression_table(
            regression_result
        ),
        "regression_model": regression_model,
        "prediction_data": prediction_data,
    }