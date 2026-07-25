from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.api.forecast import determine_risk
from src.ml.feature_builder import FeatureBuilder
from src.ml.inference import get_inference_service
from src.models.forecast_api import ForecastRequest


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


st.set_page_config(
    page_title="RCO AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_forecasting_components():
    """
    Load the model and feature builder once for the
    lifetime of the Streamlit application.
    """

    feature_builder = FeatureBuilder()
    inference_service = get_inference_service()

    with MODEL_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        model_metadata = json.load(file)

    return (
        feature_builder,
        inference_service,
        model_metadata,
    )


def build_recommendations(
    request: ForecastRequest,
    predicted_clearance_rate: float,
) -> list[str]:
    """
    Generate operational recommendations using
    forecast results and scenario inputs.
    """

    recommendations: list[str] = []

    authorization_completion_rate = (
        request.authorizations_completed
        / max(request.authorization_required, 1)
    )

    denial_rate = (
        request.denials
        / max(request.scheduled_accounts, 1)
    )

    accounts_per_fte = (
        request.scheduled_accounts
        / request.staff_fte
    )

    if predicted_clearance_rate < 0.85:
        recommendations.append(
            "Escalate the forecast for immediate operational review."
        )
    elif predicted_clearance_rate < 0.90:
        recommendations.append(
            "Monitor clearance performance closely and prepare "
            "targeted intervention."
        )
    else:
        recommendations.append(
            "Maintain the current operating plan while monitoring "
            "daily performance."
        )

    if authorization_completion_rate < 0.90:
        recommendations.append(
            "Prioritize incomplete authorizations before the next "
            "scheduled service date."
        )

    if denial_rate >= 0.05:
        recommendations.append(
            "Review denial drivers and focus corrective action on "
            "high-frequency causes."
        )

    if request.work_queue_volume > request.queue_rolling_7 * 1.15:
        recommendations.append(
            "Current queue volume is materially above its seven-day "
            "average; consider workload redistribution."
        )

    if (
        request.average_turnaround_hours
        > request.turnaround_rolling_7 * 1.15
    ):
        recommendations.append(
            "Turnaround time is trending above its recent average; "
            "review aging accounts and workflow bottlenecks."
        )

    if accounts_per_fte > 25:
        recommendations.append(
            "Workload per FTE is elevated; assess staffing capacity "
            "and cross-coverage."
        )

    if len(recommendations) == 1:
        recommendations.append(
            "Continue monitoring payer and specialty-level trends "
            "for early signs of operational deterioration."
        )

    return recommendations


def format_validation_error(
    error: ValidationError,
) -> str:
    """
    Convert Pydantic validation errors into a concise
    message for dashboard users.
    """

    messages = []

    for item in error.errors():
        field = str(item["loc"][-1]).replace(
            "_",
            " ",
        )

        messages.append(
            f"{field.title()}: {item['msg']}"
        )

    return "\n".join(messages)


try:
    (
        feature_builder,
        inference_service,
        model_metadata,
    ) = load_forecasting_components()

except (
    FileNotFoundError,
    ValueError,
    RuntimeError,
) as error:
    st.error(
        "The forecasting model could not be loaded."
    )

    st.exception(error)
    st.stop()


if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "forecast_request" not in st.session_state:
    st.session_state.forecast_request = None


st.title("📈 Revenue Cycle Optimization AI")

st.caption(
    "Predictive Healthcare Operations Intelligence"
)

st.markdown(
    """
    Forecast next-day financial clearance performance using
    operational workload, staffing, payer, specialty, and recent
    historical trends.
    """
)

st.divider()


with st.sidebar:
    st.header("Model Information")

    st.metric(
        "Model Version",
        model_metadata.get(
            "model_version",
            "Unknown",
        ),
    )

    st.write(
        "**Framework:** PyTorch"
    )

    st.write(
        "**Forecast horizon:** Next day"
    )

    st.write(
        "**Target:** Financial clearance rate"
    )

    st.write(
        "**Validation:** Chronological split"
    )

    st.info(
        "The model was trained on synthetic healthcare "
        "revenue-cycle operations data."
    )


st.header("Scenario Builder")

st.write(
    "Enter the current operating conditions and recent historical "
    "performance for one specialty and payer combination."
)

with st.form(
    "forecast_form",
    clear_on_submit=False,
):
    selection_col1, selection_col2 = st.columns(2)

    with selection_col1:
        specialty = st.selectbox(
            "Specialty",
            [
                "Audiology",
                "Home Care",
                "Neurology",
                "Pulmonology",
                "Sleep",
                "Speech",
            ],
            index=2,
        )

    with selection_col2:
        payer = st.selectbox(
            "Payer",
            [
                "Aetna",
                "AmeriHealth",
                "Cigna",
                "Independence Blue Cross",
                "Medicaid",
                "Medicare",
            ],
            index=0,
        )

    st.subheader("Current Operations")

    operations_col1, operations_col2, operations_col3 = (
        st.columns(3)
    )

    with operations_col1:
        scheduled_accounts = st.number_input(
            "Scheduled Accounts",
            min_value=1,
            value=100,
            step=1,
        )

        authorization_required = st.number_input(
            "Authorizations Required",
            min_value=0,
            value=80,
            step=1,
        )

        authorizations_completed = st.number_input(
            "Authorizations Completed",
            min_value=0,
            value=76,
            step=1,
        )

    with operations_col2:
        denials = st.number_input(
            "Denied Accounts",
            min_value=0,
            value=4,
            step=1,
        )

        work_queue_volume = st.number_input(
            "Work Queue Volume",
            min_value=0,
            value=150,
            step=1,
        )

        staff_fte = st.number_input(
            "Staff FTE",
            min_value=0.25,
            value=5.0,
            step=0.25,
        )

    with operations_col3:
        average_turnaround_hours = st.number_input(
            "Average Turnaround Hours",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=250000.0,
            step=5000.0,
        )

        denied_charges = st.number_input(
            "Denied Charges ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
        )

    st.subheader("Clearance History")

    history_col1, history_col2, history_col3 = (
        st.columns(3)
    )

    with history_col1:
        clearance_rate = (
            st.number_input(
                "Current Clearance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=96.0,
                step=0.5,
            )
            / 100
        )

        clearance_rate_lag_1 = (
            st.number_input(
                "Clearance Rate — 1 Day Ago (%)",
                min_value=0.0,
                max_value=100.0,
                value=95.0,
                step=0.5,
            )
            / 100
        )

    with history_col2:
        clearance_rate_lag_2 = (
            st.number_input(
                "Clearance Rate — 2 Days Ago (%)",
                min_value=0.0,
                max_value=100.0,
                value=95.5,
                step=0.5,
            )
            / 100
        )

        clearance_rate_lag_7 = (
            st.number_input(
                "Clearance Rate — 7 Days Ago (%)",
                min_value=0.0,
                max_value=100.0,
                value=94.0,
                step=0.5,
            )
            / 100
        )

    with history_col3:
        clearance_rate_rolling_7 = (
            st.number_input(
                "Seven-Day Average Clearance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=95.0,
                step=0.5,
            )
            / 100
        )

        queue_rolling_7 = st.number_input(
            "Seven-Day Average Queue Volume",
            min_value=0.0,
            value=145.0,
            step=1.0,
        )

        turnaround_rolling_7 = st.number_input(
            "Seven-Day Average Turnaround Hours",
            min_value=0.0,
            value=28.0,
            step=1.0,
        )

    st.subheader("Calendar Context")

    calendar_col1, calendar_col2 = st.columns(2)

    with calendar_col1:
        day_name = st.selectbox(
            "Forecast Origin Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            index=0,
        )

    with calendar_col2:
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=6,
            format_func=lambda value: pd.Timestamp(
                year=2026,
                month=value,
                day=1,
            ).month_name(),
        )

    submitted = st.form_submit_button(
        "Generate Forecast",
        use_container_width=True,
        type="primary",
    )


if submitted:
    day_lookup = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    try:
        request = ForecastRequest(
            specialty=specialty,
            payer=payer,
            scheduled_accounts=scheduled_accounts,
            authorization_required=authorization_required,
            authorizations_completed=authorizations_completed,
            denials=denials,
            total_charges=total_charges,
            denied_charges=denied_charges,
            average_turnaround_hours=average_turnaround_hours,
            work_queue_volume=work_queue_volume,
            staff_fte=staff_fte,
            clearance_rate=clearance_rate,
            clearance_rate_lag_1=clearance_rate_lag_1,
            clearance_rate_lag_2=clearance_rate_lag_2,
            clearance_rate_lag_7=clearance_rate_lag_7,
            clearance_rate_rolling_7=clearance_rate_rolling_7,
            queue_rolling_7=queue_rolling_7,
            turnaround_rolling_7=turnaround_rolling_7,
            day_of_week=day_lookup[day_name],
            month=month,
        )

        features = feature_builder.build(
            request
        )

        prediction = float(
            inference_service.predict(
                features
            )[0]
        )

        risk_level, forecast_status = determine_risk(
            prediction
        )

        st.session_state.forecast_request = request

        st.session_state.forecast_result = {
            "predicted_clearance_rate": prediction,
            "predicted_percentage": prediction * 100,
            "risk_level": risk_level,
            "forecast": forecast_status,
            "model_version": model_metadata.get(
                "model_version",
                "Unknown",
            ),
        }

    except ValidationError as error:
        st.error(
            "Please correct the following scenario inputs:"
        )

        st.code(
            format_validation_error(error)
        )

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
    ) as error:
        st.error(
            f"Forecast generation failed: {error}"
        )


result = st.session_state.forecast_result
saved_request = st.session_state.forecast_request

if result is not None and saved_request is not None:
    st.divider()

    st.header("Next-Day Forecast")

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:
        st.metric(
            "Predicted Clearance Rate",
            f"{result['predicted_percentage']:.2f}%",
            delta=(
                f"{result['predicted_percentage'] - saved_request.clearance_rate * 100:+.2f} "
                "points vs. current"
            ),
        )

    with metric_col2:
        st.metric(
            "Risk Level",
            result["risk_level"],
        )

    with metric_col3:
        st.metric(
            "Operational Status",
            result["forecast"],
        )

    with metric_col4:
        st.metric(
            "Model Version",
            result["model_version"],
        )

    chart_col1, chart_col2 = st.columns(
        [2, 1]
    )

    with chart_col1:
        st.subheader(
            "Clearance Performance Trend"
        )

        trend_data = pd.DataFrame(
            {
                "Period": [
                    "7 Days Ago",
                    "2 Days Ago",
                    "1 Day Ago",
                    "Current",
                    "Next-Day Forecast",
                ],
                "Clearance Rate": [
                    saved_request.clearance_rate_lag_7
                    * 100,
                    saved_request.clearance_rate_lag_2
                    * 100,
                    saved_request.clearance_rate_lag_1
                    * 100,
                    saved_request.clearance_rate
                    * 100,
                    result["predicted_percentage"],
                ],
            }
        ).set_index(
            "Period"
        )

        st.line_chart(
            trend_data,
            y="Clearance Rate",
        )

    with chart_col2:
        st.subheader(
            "Scenario Indicators"
        )

        authorization_completion_rate = (
            saved_request.authorizations_completed
            / max(
                saved_request.authorization_required,
                1,
            )
            * 100
        )

        denial_rate = (
            saved_request.denials
            / saved_request.scheduled_accounts
            * 100
        )

        accounts_per_fte = (
            saved_request.scheduled_accounts
            / saved_request.staff_fte
        )

        st.metric(
            "Authorization Completion",
            f"{authorization_completion_rate:.1f}%",
        )

        st.metric(
            "Denial Rate",
            f"{denial_rate:.1f}%",
        )

        st.metric(
            "Accounts per FTE",
            f"{accounts_per_fte:.1f}",
        )

        queue_change = (
            saved_request.work_queue_volume
            - saved_request.queue_rolling_7
        )

        st.metric(
            "Queue vs. 7-Day Average",
            f"{saved_request.work_queue_volume:,.0f}",
            delta=f"{queue_change:+,.0f}",
            delta_color="inverse",
        )

    st.subheader(
        "Operational Recommendations"
    )

    recommendations = build_recommendations(
        saved_request,
        result["predicted_clearance_rate"],
    )

    for recommendation in recommendations:
        st.write(
            f"• {recommendation}"
        )

    with st.expander(
        "View Model Input Summary"
    ):
        input_summary = {
            "Specialty": saved_request.specialty,
            "Payer": saved_request.payer,
            "Scheduled Accounts":
                saved_request.scheduled_accounts,
            "Authorization Required":
                saved_request.authorization_required,
            "Authorizations Completed":
                saved_request.authorizations_completed,
            "Denied Accounts":
                saved_request.denials,
            "Work Queue Volume":
                saved_request.work_queue_volume,
            "Staff FTE":
                saved_request.staff_fte,
            "Current Clearance Rate":
                f"{saved_request.clearance_rate * 100:.2f}%",
            "Seven-Day Clearance Average":
                (
                    f"{saved_request.clearance_rate_rolling_7 * 100:.2f}%"
                ),
        }

        st.dataframe(
            pd.DataFrame(
                input_summary.items(),
                columns=[
                    "Input",
                    "Value",
                ],
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Forecasts are decision-support estimates generated from "
        "synthetic operational data and should not replace human "
        "review or organizational policy."
    )

else:
    st.divider()

    st.info(
        "Complete the scenario builder and select "
        "**Generate Forecast** to view a next-day prediction."
    )