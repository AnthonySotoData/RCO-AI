from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


def _get_metric(
    metrics: pd.DataFrame,
    analysis_name: str,
    column_name: str,
) -> object:
    """Return one metric value for the requested analysis."""
    matching_rows = metrics.loc[
        metrics["analysis"] == analysis_name,
        column_name,
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Metric was not found for analysis '{analysis_name}': "
            f"{column_name}"
        )

    return matching_rows.iloc[0]


def _crop_image_top(
    image: object,
    crop_fraction: float = 0.055,
) -> object:
    """Remove the original chart title area from an image."""
    image_height = image.shape[0]
    crop_pixels = int(image_height * crop_fraction)

    return image[crop_pixels:, :]


def _add_kpi_card(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    width: float,
    height: float,
    title: str,
    value: str,
) -> None:
    """Add one KPI card to the dashboard."""
    card = FancyBboxPatch(
        (x_position, y_position),
        width,
        height,
        boxstyle="round,pad=0.012",
        linewidth=1.2,
        facecolor="white",
        edgecolor="0.75",
        transform=axis.transAxes,
    )

    axis.add_patch(card)

    axis.text(
        x_position + width / 2,
        y_position + height * 0.67,
        value,
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=17,
        fontweight="bold",
    )

    axis.text(
        x_position + width / 2,
        y_position + height * 0.25,
        title,
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8.5,
    )


def _wrap_panel_lines(
    lines: list[str],
    wrap_width: int,
) -> str:
    """Wrap bullet text so it remains inside the panel."""
    wrapped_lines: list[str] = []

    for line in lines:
        wrapped_line = textwrap.fill(
            line,
            width=wrap_width,
            initial_indent="• ",
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )

        wrapped_lines.append(wrapped_line)

    return "\n".join(wrapped_lines)


def _add_text_panel(
    axis: plt.Axes,
    x_position: float,
    y_position: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    wrap_width: int,
    font_size: float = 7.8,
) -> None:
    """Add a bordered text panel with wrapped bullet points."""
    panel = FancyBboxPatch(
        (x_position, y_position),
        width,
        height,
        boxstyle="round,pad=0.015",
        linewidth=1.2,
        facecolor="white",
        edgecolor="0.75",
        transform=axis.transAxes,
    )

    axis.add_patch(panel)

    axis.text(
        x_position + 0.018,
        y_position + height - 0.035,
        title,
        transform=axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=11.2,
        fontweight="bold",
    )

    body_text = _wrap_panel_lines(
        lines=lines,
        wrap_width=wrap_width,
    )

    axis.text(
        x_position + 0.018,
        y_position + height - 0.078,
        body_text,
        transform=axis.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=font_size,
        linespacing=1.16,
    )


def create_executive_dashboard(
    key_metrics: pd.DataFrame,
    validation_report: pd.DataFrame,
    scatterplot_path: Path,
    heatmap_path: Path,
    png_output_path: Path,
    pdf_output_path: Path,
) -> tuple[Path, Path]:
    """Create an executive one-page capstone dashboard."""
    if not scatterplot_path.exists():
        raise FileNotFoundError(
            f"Scatterplot was not found: {scatterplot_path}"
        )

    if not heatmap_path.exists():
        raise FileNotFoundError(
            f"Heatmap was not found: {heatmap_path}"
        )

    primary_r = float(
        _get_metric(
            metrics=key_metrics,
            analysis_name="Primary",
            column_name="pearson_correlation",
        )
    )

    primary_r_squared = float(
        _get_metric(
            metrics=key_metrics,
            analysis_name="Primary",
            column_name="r_squared",
        )
    )

    primary_coefficient = float(
        _get_metric(
            metrics=key_metrics,
            analysis_name="Primary",
            column_name="regression_coefficient",
        )
    )

    primary_p_value = str(
        _get_metric(
            metrics=key_metrics,
            analysis_name="Primary",
            column_name="coefficient_p_value",
        )
    )

    validation_lookup = validation_report.set_index(
        "validation_metric"
    )["value"]

    raw_records = int(
        validation_lookup["Raw row count"]
    )

    daily_records = int(
        validation_lookup["Daily analysis rows"]
    )

    scatterplot_image = _crop_image_top(
        mpimg.imread(scatterplot_path),
    )

    heatmap_image = _crop_image_top(
        mpimg.imread(heatmap_path),
    )

    figure = plt.figure(
        figsize=(16, 10),
    )

    background_axis = figure.add_axes(
        [0, 0, 1, 1]
    )

    background_axis.axis("off")

    background_axis.text(
        0.04,
        0.955,
        "RCO AI Executive Analytics Summary",
        fontsize=24,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
    )

    background_axis.text(
        0.04,
        0.915,
        (
            "Healthcare revenue cycle data engineering, "
            "inferential statistics, and operational intelligence"
        ),
        fontsize=10.5,
        horizontalalignment="left",
        verticalalignment="top",
    )

    background_axis.text(
        0.04,
        0.875,
        (
            "Research question: To what extent does authorization "
            "turnaround time affect denied charges in healthcare "
            "revenue cycle operations?"
        ),
        fontsize=10.5,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
    )

    card_width = 0.145
    card_height = 0.10
    card_gap = 0.015
    card_start_x = 0.04
    card_y = 0.735

    card_values = [
        (
            "Raw operational records",
            f"{raw_records:,}",
        ),
        (
            "Daily observations",
            f"{daily_records:,}",
        ),
        (
            "Pearson correlation",
            f"{primary_r:.3f}",
        ),
        (
            "R-squared",
            f"{primary_r_squared:.3f}",
        ),
        (
            "Regression p-value",
            primary_p_value,
        ),
        (
            "Hypothesis decision",
            "Reject H₀",
        ),
    ]

    for index, (title, value) in enumerate(
        card_values
    ):
        _add_kpi_card(
            axis=background_axis,
            x_position=(
                card_start_x
                + index * (card_width + card_gap)
            ),
            y_position=card_y,
            width=card_width,
            height=card_height,
            title=title,
            value=value,
        )

    background_axis.text(
        0.285,
        0.695,
        "Primary Linear Regression Model",
        transform=background_axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10.5,
        fontweight="bold",
    )

    background_axis.text(
        0.775,
        0.695,
        "Pearson Correlation Matrix",
        transform=background_axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10.5,
        fontweight="bold",
    )

    scatter_axis = figure.add_axes(
        [0.025, 0.255, 0.52, 0.415]
    )

    scatter_axis.imshow(
        scatterplot_image
    )

    scatter_axis.axis("off")

    heatmap_axis = figure.add_axes(
        [0.54, 0.235, 0.45, 0.45]
    )

    heatmap_axis.imshow(
        heatmap_image
    )

    heatmap_axis.axis("off")

    finding_lines = [
        (
            "Strong positive relationship between turnaround time "
            "and denied charges."
        ),
        f"Pearson r = {primary_r:.3f}; R² = {primary_r_squared:.3f}.",
        f"Regression p-value {primary_p_value}.",
        (
            "Within the synthetic dataset, each additional hour of "
            "average authorization turnaround time was associated "
            f"with approximately ${primary_coefficient:,.0f} in "
            "additional daily denied charges."
        ),
    ]

    recommendation_lines = [
        "Prioritize requests nearing turnaround thresholds.",
        "Monitor payer and specialty bottlenecks.",
        "Align staffing with queue volume and workload.",
        "Use predictive alerts for earlier intervention.",
        (
            "Validate the model using deidentified real-world "
            "operational data before production deployment."
        ),
    ]

    interpretation_lines = [
        "Synthetic dataset created for educational analysis.",
        "Results demonstrate association, not causation.",
        "Residual diagnostics indicate autocorrelation.",
        "Residuals depart from perfect normality.",
        "Future studies should use real-world multivariable data.",
    ]

    business_impact_lines = [
        "Supports proactive authorization management.",
        "Highlights denied-charge risk before service.",
        "Enables data-driven staffing decisions.",
        "Provides a reproducible analytics workflow.",
    ]

    _add_text_panel(
        axis=background_axis,
        x_position=0.02,
        y_position=0.008,
        width=0.255,
        height=0.24,
        title="Primary finding",
        lines=finding_lines,
        wrap_width=43,
        font_size=7.7,
    )

    _add_text_panel(
        axis=background_axis,
        x_position=0.285,
        y_position=0.008,
        width=0.235,
        height=0.24,
        title="Recommended actions",
        lines=recommendation_lines,
        wrap_width=36,
        font_size=7.7,
    )

    _add_text_panel(
        axis=background_axis,
        x_position=0.53,
        y_position=0.008,
        width=0.235,
        height=0.24,
        title="Interpretation and limitations",
        lines=interpretation_lines,
        wrap_width=36,
        font_size=7.7,
    )

    _add_text_panel(
        axis=background_axis,
        x_position=0.775,
        y_position=0.008,
        width=0.205,
        height=0.24,
        title="Business impact",
        lines=business_impact_lines,
        wrap_width=30,
        font_size=7.7,
    )

    png_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        png_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    with PdfPages(pdf_output_path) as pdf_document:
        pdf_document.savefig(
            figure,
            bbox_inches="tight",
        )

    plt.close(figure)

    return png_output_path, pdf_output_path