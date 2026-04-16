"""
Plotly charts for the public-signals-mislead project.

These charts focus on observable company action and noisy public signals, not on
fully known business outcomes.
"""

import math
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.thresholds import (
    DECAY_STICKY_THRESHOLD,
    HIGH_DECAY_ACTION_THRESHOLD,
)


ACTION_COLORS: dict[str, str] = {
    "SUPPORTED": "#00CC66",
    "PULLED_BACK": "#FF4444",
    "UNKNOWN": "#95A5A6",
}

CHART_THEME: dict[str, str] = {
    "primary": "#2C3E50",
    "secondary": "#95A5A6",
    "background": "rgba(240,240,240,0.5)",
}


def _write_svg(output_path: str, svg_content: str) -> None:
    """Write an SVG file to disk, creating parent directories when needed."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg_content, encoding="utf-8")
    print(f"   Saved: {output_path}")


def _scale_linear(
    value: float,
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
) -> float:
    """Scale a numeric value linearly into an SVG-friendly coordinate range."""
    if math.isclose(domain_max, domain_min):
        return (range_min + range_max) / 2
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _svg_circle_radius(value: float, min_radius: float = 7, max_radius: float = 21) -> float:
    """Map an engagement-like value to a visible bubble radius."""
    safe_value = max(float(value), 0.0)
    scaled = math.sqrt(safe_value)
    return max(min_radius, min(max_radius, min_radius + scaled / 2.2))


def _svg_y_ticks(max_value: float, intervals: int = 4) -> list[float]:
    """Create evenly spaced Y-axis ticks for simple static previews."""
    safe_max = max(float(max_value), 1.0)
    step = safe_max / intervals
    return [step * idx for idx in range(intervals + 1)]


def _svg_text_block(
    x: float,
    y: float,
    lines: list[str],
    *,
    font_size: int = 14,
    font_weight: str = "400",
    fill: str = "#24292F",
    anchor: str = "start",
    line_height: float = 1.25,
) -> str:
    """Render a multi-line SVG text block with consistent line spacing."""
    parts = [
        (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{font_size}" '
            f'font-weight="{font_weight}" fill="{fill}">'
        )
    ]
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else font_size * line_height
        parts.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _stacked_band_y_positions(values: pd.Series, center_y: float) -> list[float]:
    """Place points in tidy vertical stacks so dense regions stay legible in static SVGs."""
    offsets = [-27, -13, 0, 13, 27, -40, 40]
    seen_per_bucket: dict[float, int] = {}
    positions: list[float] = []

    for value in values:
        bucket = round(float(value) / 0.01) * 0.01
        seen = seen_per_bucket.get(bucket, 0)
        seen_per_bucket[bucket] = seen + 1
        base_offset = offsets[seen % len(offsets)]
        tier_offset = (seen // len(offsets)) * 10
        direction = -1 if seen % 2 == 0 else 1
        positions.append(center_y + base_offset + direction * tier_offset)

    return positions


def create_decay_vs_action_preview_svg(
    df: pd.DataFrame,
    output_path: str = "documentation/assets/decay_vs_action_preview.svg",
) -> None:
    """Create a readable GitHub-first SVG preview of the main action-overlap pattern."""
    plot_df = df[df["search_decay"].notna() & df["total_mentions"].notna()].copy()
    if plot_df.empty:
        raise ValueError("Cannot build preview without search_decay and total_mentions data.")

    labeled_actions = ["SUPPORTED", "PULLED_BACK", "UNKNOWN"]
    width, height = 1180, 820
    left, right, top, bottom = 150, 1038, 160, 560
    chart_width = right - left
    chart_height = bottom - top
    x_domain_min = min(0.45, max(0.0, math.floor(float(plot_df["search_decay"].min()) * 10) / 10))
    x_domain_max = 1.0

    supported = plot_df[plot_df["company_action"] == "SUPPORTED"]
    supported_high_decay = int((supported["search_decay"] > HIGH_DECAY_ACTION_THRESHOLD).sum())
    supported_pct = (supported_high_decay / len(supported) * 100) if len(supported) > 0 else 0.0

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FAFBFC"/>',
        '<rect x="36" y="34" width="1108" height="752" rx="22" fill="white" stroke="#D0D7DE"/>',
        _svg_text_block(
            left,
            68,
            [
                "High Search Decay Appears Across",
                "Supported and Pulled-Back Features",
            ],
            font_size=24,
            font_weight="700",
            fill=CHART_THEME["primary"],
        ),
        _svg_text_block(
            left,
            118,
            [
                "This GitHub preview simplifies the interactive bubble chart so the overlap is readable.",
                "The pattern still holds: high decay is common even when product support continues.",
            ],
            font_size=13,
            fill="#57606A",
        ),
    ]

    high_decay_x = _scale_linear(HIGH_DECAY_ACTION_THRESHOLD, x_domain_min, x_domain_max, left, right)
    svg_parts.append(
        f'<rect x="{high_decay_x:.1f}" y="{top}" width="{right - high_decay_x:.1f}" height="{chart_height}" fill="#F6FFFA"/>'
    )

    for tick in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        x = _scale_linear(tick, x_domain_min, x_domain_max, left, right)
        svg_parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#F3F4F6" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{bottom + 28}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">{tick:.0%}</text>',
            ]
        )

    band_height = chart_height / len(labeled_actions)
    band_centers: dict[str, float] = {}
    for index, action in enumerate(labeled_actions):
        y = top + index * band_height
        center_y = y + band_height / 2
        band_centers[action] = center_y
        fill = "#FFFFFF" if index % 2 == 0 else "#FBFCFD"
        svg_parts.extend(
            [
                f'<rect x="{left}" y="{y:.1f}" width="{chart_width}" height="{band_height:.1f}" fill="{fill}"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#EAECEF" stroke-width="1"/>',
                f'<text x="{left - 18}" y="{center_y + 5:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">{action.replace("_", " ").title()}</text>',
            ]
        )
        count = int((plot_df["company_action"] == action).sum())
        pill_width = 112 if action != "PULLED_BACK" else 118
        pill_x = left + 12
        pill_y = center_y - 18
        svg_parts.extend(
            [
                f'<rect x="{pill_x}" y="{pill_y:.1f}" width="{pill_width}" height="36" rx="18" fill="#FFFFFF" stroke="#D0D7DE"/>',
                f'<circle cx="{pill_x + 18}" cy="{center_y:.1f}" r="7" fill="{ACTION_COLORS[action]}"/>',
                f'<text x="{pill_x + 34}" y="{center_y + 5:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#24292F">n = {count}</text>',
            ]
        )

    svg_parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#24292F" stroke-width="2"/>'
    )
    svg_parts.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 62}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#24292F">Search Decay (4 weeks post-peak, zoomed to observed range)</text>',
            f'<line x1="{high_decay_x:.1f}" y1="{top}" x2="{high_decay_x:.1f}" y2="{bottom}" stroke="{ACTION_COLORS["SUPPORTED"]}" stroke-width="2" stroke-dasharray="6 6"/>',
            f'<text x="{high_decay_x - 10:.1f}" y="{top + 22}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="{ACTION_COLORS["SUPPORTED"]}">High-decay region (&gt; {HIGH_DECAY_ACTION_THRESHOLD:.0%})</text>',
        ]
    )

    for action in labeled_actions:
        group = plot_df[plot_df["company_action"] == action].sort_values("search_decay")
        y_positions = _stacked_band_y_positions(group["search_decay"], band_centers[action])
        for (_, row), y in zip(group.iterrows(), y_positions):
            x = _scale_linear(float(row["search_decay"]), x_domain_min, x_domain_max, left, right)
            radius = 9 if action != "UNKNOWN" else 8
            opacity = 0.88 if action != "UNKNOWN" else 0.65
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{ACTION_COLORS[action]}" fill-opacity="{opacity:.2f}" stroke="white" stroke-width="2"/>'
            )

    annotation_x = 730
    annotation_y = 644
    svg_parts.extend(
        [
            f'<rect x="{annotation_x}" y="{annotation_y}" width="250" height="86" rx="18" fill="#F6FFFA" stroke="{ACTION_COLORS["SUPPORTED"]}" stroke-width="2"/>',
            _svg_text_block(
                annotation_x + 18,
                annotation_y + 22,
                [
                    f"{supported_pct:.0f}% of supported features",
                    f"still sit above {HIGH_DECAY_ACTION_THRESHOLD:.0%} decay.",
                ],
                font_size=14,
                font_weight="700",
                fill=ACTION_COLORS["SUPPORTED"],
            ),
            _svg_text_block(
                annotation_x + 18,
                annotation_y + 54,
                [
                    "High decay is a warning sign,",
                    "not a product verdict.",
                ],
                font_size=12,
                fill="#57606A",
            ),
        ]
    )

    svg_parts.extend(
        [
            '<text x="150" y="770" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#57606A">Static preview for GitHub. The interactive HTML adds Reddit mentions and richer hover detail.</text>',
            '</svg>',
        ]
    )

    _write_svg(output_path, "".join(svg_parts))


def create_decision_matrix_preview_svg(
    output_path: str = "documentation/assets/decision_matrix_preview.svg",
) -> None:
    """Create a static SVG version of the decision matrix for GitHub docs."""
    width, height = 1120, 760
    cell_width = 224
    cell_height = 104
    col_gap = 18
    row_gap = 18
    start_x = 228
    start_y = 184
    matrix_width = cell_width * 3 + col_gap * 2
    matrix_height = cell_height * 3 + row_gap * 2
    decay_levels = [
        f"Low (<{DECAY_STICKY_THRESHOLD:.0%})",
        f"Medium ({DECAY_STICKY_THRESHOLD:.0%}-{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
        f"High (>{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
    ]
    sentiments = ["Negative", "Mixed", "Positive"]
    recommendations = [
        [(["INVESTIGATE"], "#FFE7E5"), (["CHECK", "INTERNALS"], "#FFF2CC"), (["DON'T", "AUTO-ROLLBACK"], "#E8F5E9")],
        [(["NEEDS", "CONTEXT"], "#FFF2CC"), (["NEEDS", "CONTEXT"], "#FFF2CC"), (["NEEDS", "CONTEXT"], "#FFF2CC")],
        [(["TRACK", "ADOPTION"], "#E8F5E9"), (["DON'T", "AUTO-PANIC"], "#E8F5E9"), (["DON'T", "AUTO-PANIC"], "#E8F5E9")],
    ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FAFBFC"/>',
        '<rect x="32" y="28" width="1056" height="704" rx="22" fill="white" stroke="#D0D7DE"/>',
        _svg_text_block(
            start_x,
            66,
            [
                "Decision Matrix:",
                "Public Signals Need Internal Context",
            ],
            font_size=24,
            font_weight="700",
            fill=CHART_THEME["primary"],
        ),
        _svg_text_block(
            start_x,
            114,
            ["Use public signals to decide what to investigate next, not what to kill."],
            font_size=13,
            fill="#57606A",
        ),
    ]

    for index, level in enumerate(decay_levels):
        x = start_x + index * (cell_width + col_gap) + cell_width / 2
        svg_parts.append(
            f'<text x="{x:.1f}" y="{start_y - 18}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">{escape(level)}</text>'
        )

    svg_parts.append(
        _svg_text_block(
            start_x - 110,
            start_y - 4,
            ["Reddit", "sentiment"],
            font_size=15,
            font_weight="700",
            fill="#24292F",
            anchor="middle",
        )
    )

    for index, sentiment in enumerate(sentiments):
        y = start_y + index * (cell_height + row_gap) + cell_height / 2
        svg_parts.append(
            f'<text x="{start_x - 22}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">{escape(sentiment)}</text>'
        )

    for row_index, row in enumerate(recommendations):
        for col_index, (label_lines, fill_color) in enumerate(row):
            x = start_x + col_index * (cell_width + col_gap)
            y = start_y + row_index * (cell_height + row_gap)
            svg_parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" rx="18" fill="{fill_color}" stroke="#D0D7DE"/>',
                    _svg_text_block(
                        x + cell_width / 2,
                        y + 42,
                        label_lines,
                        font_size=17,
                        font_weight="700",
                        anchor="middle",
                    ),
                ]
            )

    footer_y = start_y + matrix_height + 68
    rule_box_width = 292
    rule_box_height = 92
    rule_box_x = start_x + matrix_width - rule_box_width
    rule_box_y = footer_y - 22
    svg_parts.extend(
        [
            _svg_text_block(
                start_x,
                footer_y,
                ["Before rollback, require:"],
                font_size=15,
                font_weight="700",
            ),
            _svg_text_block(
                start_x,
                footer_y + 28,
                [
                    "adoption • repeat usage • retention effect",
                    "monetization where relevant • cost to maintain",
                ],
                font_size=14,
                fill="#57606A",
            ),
            f'<rect x="{rule_box_x}" y="{rule_box_y}" width="{rule_box_width}" height="{rule_box_height}" rx="16" fill="#F6F8FA" stroke="#D0D7DE"/>',
            _svg_text_block(
                rule_box_x + 20,
                rule_box_y + 28,
                ["Rule of thumb"],
                font_size=14,
                font_weight="700",
            ),
            _svg_text_block(
                rule_box_x + 20,
                rule_box_y + 54,
                [
                    "External concern without internal evidence",
                    "should trigger investigation, not rollback.",
                ],
                font_size=13,
                fill="#57606A",
            ),
            '</svg>',
        ]
    )

    _write_svg(output_path, "".join(svg_parts))


def create_decay_vs_action_scatter(
    df: pd.DataFrame,
    output_path: str = "results/figures/decay_vs_action.html",
) -> None:
    """Plot search decay against Reddit mentions, colored by observed company action."""
    fig = px.scatter(
        df,
        x="search_decay",
        y="total_mentions",
        color="company_action",
        size="engagement_score",
        hover_data=["feature_name", "business_outcome", "evidence_summary"],
        title="Public Signals vs Observable Company Action",
        labels={
            "search_decay": "Search Decay (4 weeks post-peak)",
            "total_mentions": "Reddit Mentions",
            "company_action": "Observed Company Action",
            "business_outcome": "Known Business Outcome",
        },
        color_discrete_map=ACTION_COLORS,
        width=1000,
        height=700,
    )

    supported = df[df["company_action"] == "SUPPORTED"]
    supported_high_decay = len(supported[supported["search_decay"] > HIGH_DECAY_ACTION_THRESHOLD])
    pct = (supported_high_decay / len(supported) * 100) if len(supported) > 0 else 0

    fig.add_annotation(
        x=0.95,
        y=df["total_mentions"].max() * 0.8 if not df.empty else 0,
        text=(
            f"{pct:.0f}% of supported features show >{HIGH_DECAY_ACTION_THRESHOLD:.0%} decay"
            "<br>High decay ≠ automatic rollback"
        ),
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=ACTION_COLORS["SUPPORTED"],
        font=dict(size=14, color=ACTION_COLORS["SUPPORTED"]),
        bgcolor="white",
        bordercolor=ACTION_COLORS["SUPPORTED"],
        borderwidth=2,
    )

    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color=CHART_THEME["primary"]),
        plot_bgcolor=CHART_THEME["background"],
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")


def create_divergence_comparison(
    df: pd.DataFrame,
    features_to_show: list,
    output_path: str = "results/figures/divergence_examples.html",
) -> None:
    """Compare public signals for a handful of product-decision case studies."""
    examples = df[df["feature_name"].isin(features_to_show)].copy()
    fig = go.Figure()

    for _, row in examples.iterrows():
        color = ACTION_COLORS.get(row["company_action"], ACTION_COLORS["UNKNOWN"])
        fig.add_trace(
            go.Bar(
                name=row["feature_name"],
                x=["Search Decay", "Negative Sentiment"],
                y=[row["search_decay"] * 100, row["negative_ratio"] * 100],
                marker_color=color,
                text=[f"{row['search_decay']:.0%}", f"{row['negative_ratio']:.0%}"],
                textposition="outside",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Case Studies: Similar Public Signals, Different Product Decisions",
        yaxis_title="Percentage",
        barmode="group",
        height=600,
        font=dict(size=14),
        title_font=dict(size=20, color=CHART_THEME["primary"]),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")


def create_decision_matrix_heatmap(output_path: str = "results/figures/decision_matrix.html") -> None:
    """Show how public signals should feed a product decision conversation."""
    decay_levels = [
        f"Low (<{DECAY_STICKY_THRESHOLD:.0%})",
        f"Medium ({DECAY_STICKY_THRESHOLD:.0%}-{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
        f"High (>{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
    ]
    sentiments = ["Negative", "Mixed", "Positive"]

    recommendations = [
        ["INVESTIGATE", "CHECK INTERNALS", "DON'T AUTO-ROLLBACK"],
        ["NEEDS CONTEXT", "NEEDS CONTEXT", "NEEDS CONTEXT"],
        ["TRACK ADOPTION", "DON'T AUTO-PANIC", "DON'T AUTO-PANIC"],
    ]
    z_values = [[1, 2, 2], [2, 2, 2], [3, 2, 3]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=decay_levels,
            y=sentiments,
            text=recommendations,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=[
                [0, ACTION_COLORS["PULLED_BACK"]],
                [0.5, "#FFB84D"],
                [1, ACTION_COLORS["SUPPORTED"]],
            ],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Decision Matrix: Public Signals Need Internal Context",
        title_font=dict(size=20, color=CHART_THEME["primary"]),
        xaxis_title="Search Decay",
        yaxis_title="Reddit Sentiment",
        font=dict(size=14),
        height=500,
        width=900,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")


def create_action_rate_by_type(
    df: pd.DataFrame,
    output_path: str = "results/figures/action_by_type.html",
) -> None:
    """Plot observed support rate by feature type for the action-labeled sample."""
    analysis_df = df[df["action_binary"].notna()].copy()
    type_col = "feature_type_calc" if "feature_type_calc" in analysis_df.columns else "feature_type"
    type_analysis = (
        analysis_df.groupby(type_col)
        .agg(
            total=("action_binary", "count"),
            supported=("action_binary", "sum"),
            support_rate=("action_binary", "mean"),
        )
        .reset_index()
        .rename(columns={type_col: "feature_type"})
    )
    type_analysis = type_analysis[type_analysis["total"] >= 2].sort_values("support_rate", ascending=False)

    fig = px.bar(
        type_analysis,
        x="feature_type",
        y="support_rate",
        title="Observed Support Rate by Feature Type",
        labels={"support_rate": "Support Rate", "feature_type": "Feature Type"},
        text="support_rate",
        color="support_rate",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )

    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20, color=CHART_THEME["primary"]),
        showlegend=False,
        height=600,
        yaxis_tickformat=".0%",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")


def create_action_comparison(
    supported_metrics: dict,
    pulled_back_metrics: dict,
    output_path: str = "results/figures/action_signal_comparison.html",
) -> None:
    """Plot side-by-side signal averages for supported vs pulled-back features."""
    metrics = ["Search Decay", "Reddit Mentions", "Negative Sentiment"]
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Supported",
            x=metrics,
            y=[
                supported_metrics["decay_mean"],
                supported_metrics["mentions_mean"],
                supported_metrics["negative_mean"],
            ],
            marker_color=ACTION_COLORS["SUPPORTED"],
            text=[
                f"{supported_metrics['decay_mean']:.1%}",
                f"{supported_metrics['mentions_mean']:.1f}",
                f"{supported_metrics['negative_mean']:.1%}",
            ],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Pulled back",
            x=metrics,
            y=[
                pulled_back_metrics["decay_mean"],
                pulled_back_metrics["mentions_mean"],
                pulled_back_metrics["negative_mean"],
            ],
            marker_color=ACTION_COLORS["PULLED_BACK"],
            text=[
                f"{pulled_back_metrics['decay_mean']:.1%}",
                f"{pulled_back_metrics['mentions_mean']:.1f}",
                f"{pulled_back_metrics['negative_mean']:.1%}",
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Supported vs Pulled-Back Features: Public Signal Comparison",
        barmode="group",
        height=600,
        font=dict(size=14),
        title_font=dict(size=20, color=CHART_THEME["primary"]),
        yaxis_title="Value",
        showlegend=True,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"   Saved: {output_path}")
