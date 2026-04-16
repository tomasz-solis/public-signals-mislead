"""
Generate all visualizations for the analysis.

The charts center on public signals, observable company action, and the limits of
what public data can tell us about product decisions.
"""

import logging
from pathlib import Path

import pandas as pd

from src.visualization.charts import (
    create_action_comparison,
    create_action_rate_by_type,
    create_decay_vs_action_scatter,
    create_decay_vs_action_preview_svg,
    create_decision_matrix_heatmap,
    create_decision_matrix_preview_svg,
    create_divergence_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _run_chart_step(step_label: str, filename: str, callback, created_files: list[str]) -> None:
    """Run one chart-generation step and fail loudly if it breaks."""
    if step_label:
        print(step_label)
    try:
        callback()
    except Exception:
        logger.exception("Failed to create %s", filename)
        raise
    created_files.append(filename)


def main() -> None:
    """Generate all visualizations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print("\nGENERATING VISUALIZATIONS\n")

    csv_path = PROJECT_ROOT / "data" / "validation" / "labeled_features.csv"
    df = pd.read_csv(csv_path)

    supported_count = int((df["company_action"] == "SUPPORTED").sum()) if "company_action" in df.columns else 0
    pulled_back_count = int((df["company_action"] == "PULLED_BACK").sum()) if "company_action" in df.columns else 0
    unknown_count = int((df["company_action"] == "UNKNOWN").sum()) if "company_action" in df.columns else len(df)

    print(f"Loaded {len(df)} features")
    print(f"  Supported: {supported_count}")
    print(f"  Pulled back: {pulled_back_count}")
    print(f"  Unknown action: {unknown_count}\n")

    if "engagement_score" not in df.columns:
        df["engagement_score"] = df["total_mentions"].fillna(0) * 10

    if "evidence_summary" not in df.columns:
        df["evidence_summary"] = "N/A"

    if "business_outcome" not in df.columns:
        df["business_outcome"] = "UNKNOWN"

    if "company_action" not in df.columns:
        df["company_action"] = "UNKNOWN"

    if "action_binary" not in df.columns:
        df["action_binary"] = (
            df["company_action"]
            .map({"SUPPORTED": 1, "PULLED_BACK": 0})
            .astype("Int64")
        )

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = PROJECT_ROOT / "documentation" / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []

    _run_chart_step(
        "1. Creating decay vs action scatter...",
        "decay_vs_action.html",
        lambda: create_decay_vs_action_scatter(
            df,
            output_path=str(output_dir / "decay_vs_action.html"),
        ),
        created_files,
    )

    key_examples = [
        "Password Sharing Crackdown",
        "AI DJ",
        "Games",
        "GroupWatch",
        "Watch Party",
    ]
    available_examples = [feature for feature in key_examples if feature in df["feature_name"].values]
    _run_chart_step(
        "\n2. Creating divergence comparison...",
        "divergence_examples.html",
        lambda: create_divergence_comparison(
            df,
            features_to_show=available_examples,
            output_path=str(output_dir / "divergence_examples.html"),
        ),
        created_files,
    )

    _run_chart_step(
        "\n3. Creating decision matrix...",
        "decision_matrix.html",
        lambda: create_decision_matrix_heatmap(output_path=str(output_dir / "decision_matrix.html")),
        created_files,
    )

    _run_chart_step(
        "\n4. Creating action rate by type...",
        "action_by_type.html",
        lambda: create_action_rate_by_type(
            df,
            output_path=str(output_dir / "action_by_type.html"),
        ),
        created_files,
    )

    print("\n5. Creating action comparison...")
    supported = df[df["action_binary"] == 1]
    pulled_back = df[df["action_binary"] == 0]
    if len(supported) > 0 and len(pulled_back) > 0:
        supported_metrics = {
            "decay_mean": supported["search_decay"].mean(),
            "mentions_mean": supported["total_mentions"].mean(),
            "negative_mean": supported["negative_ratio"].mean(),
        }
        pulled_back_metrics = {
            "decay_mean": pulled_back["search_decay"].mean(),
            "mentions_mean": pulled_back["total_mentions"].mean(),
            "negative_mean": pulled_back["negative_ratio"].mean(),
        }
        _run_chart_step(
            "   Building comparison bars...",
            "action_signal_comparison.html",
            lambda: create_action_comparison(
                supported_metrics,
                pulled_back_metrics,
                output_path=str(output_dir / "action_signal_comparison.html"),
            ),
            created_files,
        )

    _run_chart_step(
        "\n6. Creating static README previews...",
        "documentation/assets/decay_vs_action_preview.svg",
        lambda: (
            create_decay_vs_action_preview_svg(
                df,
                output_path=str(asset_dir / "decay_vs_action_preview.svg"),
            ),
            create_decision_matrix_preview_svg(
                output_path=str(asset_dir / "decision_matrix_preview.svg"),
            ),
        ),
        created_files,
    )
    created_files.append("documentation/assets/decision_matrix_preview.svg")

    print("\nVISUALIZATIONS COMPLETE")
    print(f"\nSaved to: {output_dir.relative_to(PROJECT_ROOT)}")
    print("\nCreated files:")
    for filename in created_files:
        print(f"  - {filename}")

    print("\nOpen the HTML files in a browser to explore the interactive views.")


if __name__ == "__main__":
    main()
