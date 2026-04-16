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


def create_decay_vs_action_preview_svg(
    df: pd.DataFrame,
    output_path: str = "documentation/assets/decay_vs_action_preview.svg",
) -> None:
    """Create a GitHub-friendly static SVG preview of the main scatter chart."""
    plot_df = df[df["search_decay"].notna() & df["total_mentions"].notna()].copy()
    if plot_df.empty:
        raise ValueError("Cannot build preview without search_decay and total_mentions data.")

    plot_df["engagement_score"] = plot_df.get("engagement_score", plot_df["total_mentions"].fillna(0) * 10)

    width, height = 1180, 760
    left, right, top, bottom = 105, 1040, 125, 650
    chart_width = right - left
    chart_height = bottom - top
    y_max = max(float(plot_df["total_mentions"].max()), 1.0)
    ticks = _svg_y_ticks(y_max)

    supported = plot_df[plot_df["company_action"] == "SUPPORTED"]
    supported_high_decay = int((supported["search_decay"] > HIGH_DECAY_ACTION_THRESHOLD).sum())
    supported_pct = (supported_high_decay / len(supported) * 100) if len(supported) > 0 else 0.0

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FAFBFC"/>',
        '<rect x="36" y="34" width="1108" height="692" rx="22" fill="white" stroke="#D0D7DE"/>',
        f'<text x="{left}" y="72" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="{CHART_THEME["primary"]}">Public Signals vs Observable Company Action</text>',
        '<text x="105" y="102" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#57606A">High search decay is common even among supported features. Public attention fades faster than product value gets resolved.</text>',
    ]

    for tick in ticks:
        y = _scale_linear(tick, 0, y_max, bottom, top)
        tick_label = f"{tick:.0f}"
        svg_parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#EAECEF" stroke-width="1"/>',
                f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">{tick_label}</text>',
            ]
        )

    for tick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x = _scale_linear(tick, 0, 1, left, right)
        svg_parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#F3F4F6" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{bottom + 28}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">{tick:.0%}</text>',
            ]
        )

    svg_parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#24292F" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#24292F" stroke-width="2"/>',
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 58}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#24292F">Search Decay (4 weeks post-peak)</text>',
            f'<text x="32" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 32 {(top + bottom) / 2:.1f})" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#24292F">Reddit Mentions</text>',
        ]
    )

    for _, row in plot_df.iterrows():
        x = _scale_linear(float(row["search_decay"]), 0, 1, left, right)
        y = _scale_linear(float(row["total_mentions"]), 0, y_max, bottom, top)
        radius = _svg_circle_radius(float(row["engagement_score"]))
        color = ACTION_COLORS.get(row.get("company_action", "UNKNOWN"), ACTION_COLORS["UNKNOWN"])
        svg_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72" stroke="white" stroke-width="2"/>'
        )

    callout_offsets = {
        "Password Sharing Crackdown": (18, -22),
        "AI DJ": (-110, -24),
        "GroupWatch": (-126, -10),
        "Watch Party": (18, 24),
    }
    for feature_name, (dx, dy) in callout_offsets.items():
        match = plot_df[plot_df["feature_name"] == feature_name]
        if match.empty:
            continue
        row = match.iloc[0]
        x = _scale_linear(float(row["search_decay"]), 0, 1, left, right)
        y = _scale_linear(float(row["total_mentions"]), 0, y_max, bottom, top)
        label_x = x + dx
        label_y = y + dy
        anchor = "start" if dx >= 0 else "end"
        svg_parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{label_x:.1f}" y2="{label_y - 4:.1f}" stroke="#57606A" stroke-width="1.5"/>',
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="600" fill="#24292F">{escape(feature_name)}</text>',
            ]
        )

    legend_x = 818
    legend_y = 136
    svg_parts.append(
        '<rect x="800" y="118" width="290" height="86" rx="16" fill="#FFFFFF" stroke="#D0D7DE"/>'
    )
    for index, label in enumerate(["SUPPORTED", "PULLED_BACK", "UNKNOWN"]):
        y = legend_y + index * 23
        svg_parts.extend(
            [
                f'<circle cx="{legend_x}" cy="{y}" r="7" fill="{ACTION_COLORS[label]}"/>',
                f'<text x="{legend_x + 18}" y="{y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#24292F">{label.replace("_", " ").title()}</text>',
            ]
        )

    annotation_x = 810
    annotation_y = 505
    svg_parts.extend(
        [
            f'<rect x="{annotation_x}" y="{annotation_y}" width="250" height="102" rx="18" fill="#F6FFFA" stroke="{ACTION_COLORS["SUPPORTED"]}" stroke-width="2"/>',
            f'<text x="{annotation_x + 18}" y="{annotation_y + 30}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{ACTION_COLORS["SUPPORTED"]}">{supported_pct:.0f}% of supported features</text>',
            f'<text x="{annotation_x + 18}" y="{annotation_y + 54}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{ACTION_COLORS["SUPPORTED"]}">show more than {HIGH_DECAY_ACTION_THRESHOLD:.0%} decay</text>',
            '<text x="828" y="585" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">High decay is a warning signal.</text>',
            '<text x="828" y="605" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">It is not a product verdict.</text>',
        ]
    )

    svg_parts.extend(
        [
            '<text x="105" y="700" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#57606A">Static preview for GitHub. Run scripts/generate_visualizations.py for the interactive HTML charts.</text>',
            '</svg>',
        ]
    )

    _write_svg(output_path, "".join(svg_parts))


def create_decision_matrix_preview_svg(
    output_path: str = "documentation/assets/decision_matrix_preview.svg",
) -> None:
    """Create a static SVG version of the decision matrix for GitHub docs."""
    width, height = 1120, 720
    cell_width = 255
    cell_height = 118
    start_x = 215
    start_y = 180
    decay_levels = [
        f"Low (<{DECAY_STICKY_THRESHOLD:.0%})",
        f"Medium ({DECAY_STICKY_THRESHOLD:.0%}-{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
        f"High (>{HIGH_DECAY_ACTION_THRESHOLD:.0%})",
    ]
    sentiments = ["Negative", "Mixed", "Positive"]
    recommendations = [
        [("INVESTIGATE", "#FFE7E5"), ("CHECK INTERNALS", "#FFF2CC"), ("DON'T AUTO-ROLLBACK", "#E8F5E9")],
        [("NEEDS CONTEXT", "#FFF2CC"), ("NEEDS CONTEXT", "#FFF2CC"), ("NEEDS CONTEXT", "#FFF2CC")],
        [("TRACK ADOPTION", "#E8F5E9"), ("DON'T AUTO-PANIC", "#E8F5E9"), ("DON'T AUTO-PANIC", "#E8F5E9")],
    ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#FAFBFC"/>',
        '<rect x="32" y="28" width="1056" height="664" rx="22" fill="white" stroke="#D0D7DE"/>',
        f'<text x="{start_x}" y="82" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="{CHART_THEME["primary"]}">Decision Matrix: Public Signals Need Internal Context</text>',
        '<text x="215" y="112" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#57606A">Use public signals to decide what to investigate next, not what to kill.</text>',
    ]

    for index, level in enumerate(decay_levels):
        x = start_x + index * cell_width + cell_width / 2
        svg_parts.append(
            f'<text x="{x:.1f}" y="{start_y - 20}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">{escape(level)}</text>'
        )

    for index, sentiment in enumerate(sentiments):
        y = start_y + index * cell_height + cell_height / 2
        svg_parts.append(
            f'<text x="{start_x - 22}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">{escape(sentiment)}</text>'
        )

    for row_index, row in enumerate(recommendations):
        for col_index, (label, fill_color) in enumerate(row):
            x = start_x + col_index * cell_width
            y = start_y + row_index * cell_height
            svg_parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width - 10}" height="{cell_height - 10}" rx="18" fill="{fill_color}" stroke="#D0D7DE"/>',
                    f'<text x="{x + (cell_width - 10) / 2:.1f}" y="{y + 54}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700" fill="#24292F">{escape(label)}</text>',
                ]
            )

    notes_y = 568
    svg_parts.extend(
        [
            '<text x="96" y="204" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">Reddit Sentiment</text>',
            '<text x="215" y="624" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#24292F">Before rollback, require:</text>',
            '<text x="215" y="652" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#57606A">adoption • repeat usage • retention effect • monetization where relevant • cost to maintain</text>',
            f'<rect x="772" y="{notes_y}" width="242" height="74" rx="16" fill="#F6F8FA" stroke="#D0D7DE"/>',
            '<text x="792" y="596" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="700" fill="#24292F">Rule of thumb</text>',
            '<text x="792" y="621" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">External concern without internal</text>',
            '<text x="792" y="642" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">evidence should trigger investigation,</text>',
            '<text x="792" y="663" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#57606A">not an automatic rollback.</text>',
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
