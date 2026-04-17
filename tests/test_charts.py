"""Smoke tests for SVG preview chart generation.

These don't validate chart content in detail — they verify that the
SVG files are created, well-formed, and contain expected landmarks.
"""

import pandas as pd

from src.visualization.charts import (
    create_decay_vs_action_preview_svg,
    create_decision_matrix_preview_svg,
)


def _chart_df() -> pd.DataFrame:
    """Minimal DataFrame with the columns the chart functions need."""
    return pd.DataFrame(
        {
            "feature_name": ["A", "B", "C", "D"],
            "search_decay": [0.90, 0.85, 0.25, 0.92],
            "total_mentions": [40, 28, 8, 5],
            "company_action": ["SUPPORTED", "SUPPORTED", "SUPPORTED", "PULLED_BACK"],
            "business_outcome": ["POSITIVE", "UNKNOWN", "UNKNOWN", "NEGATIVE"],
            "evidence_summary": ["ev1", "ev2", "ev3", "ev4"],
            "engagement_score": [400, 280, 80, 50],
        }
    )


def test_decay_preview_writes_valid_svg(tmp_path) -> None:
    output = tmp_path / "preview.svg"
    create_decay_vs_action_preview_svg(_chart_df(), output_path=str(output))
    content = output.read_text()
    assert content.startswith("<svg")
    assert content.rstrip().endswith("</svg>")
    assert "SUPPORTED" in content or "Supported" in content


def test_decision_matrix_preview_writes_valid_svg(tmp_path) -> None:
    output = tmp_path / "matrix.svg"
    create_decision_matrix_preview_svg(output_path=str(output))
    content = output.read_text()
    assert content.startswith("<svg")
    assert "Reddit" in content
