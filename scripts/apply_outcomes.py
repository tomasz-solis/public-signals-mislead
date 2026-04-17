"""
Apply public decision context to the labeled feature CSV.

This script keeps two ideas separate:
- `company_action`: what the company publicly appears to have done
- `business_outcome`: what the public record can actually prove about value

Usage: python scripts/apply_outcomes.py
"""

from pathlib import Path

import pandas as pd

from config.outcomes import get_feature_context, get_feature_type
from src.provenance import stamp_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_BINARY_MAP = {"SUPPORTED": 1, "PULLED_BACK": 0}
BUSINESS_OUTCOME_BINARY_MAP = {"POSITIVE": 1, "NEGATIVE": 0}
LEGACY_COLUMNS = [
    "known_outcome",
    "outcome_metric",
    "outcome_label",
    "is_success",
    "is_failure",
    "success_binary",
]


def _context_value(feature_name: str, key: str, default: str = "UNKNOWN") -> str:
    """Return one field from the feature context with a small default."""
    context = get_feature_context(feature_name)
    if not context:
        return default
    return context.get(key, default)


def apply_outcomes_to_csv() -> pd.DataFrame:
    """Apply company-action and business-outcome context to the labeled feature CSV."""
    csv_path = PROJECT_ROOT / "data" / "validation" / "labeled_features.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=LEGACY_COLUMNS, errors="ignore")

    print(f"Loaded {len(df)} features from {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Companies: {', '.join(df['company'].unique())}\n")

    has_context = df["feature_name"].apply(lambda name: bool(get_feature_context(name)))
    df["company_action"] = df["feature_name"].apply(lambda name: _context_value(name, "company_action"))
    df["business_outcome"] = df["feature_name"].apply(lambda name: _context_value(name, "business_outcome"))
    df["evidence_summary"] = df["feature_name"].apply(
        lambda name: _context_value(name, "evidence_summary", default="No public context yet")
    )
    df["evidence_source"] = df["feature_name"].apply(
        lambda name: _context_value(name, "source", default="No public context yet")
    )
    df["evidence_tier"] = df["feature_name"].apply(
        lambda name: _context_value(name, "evidence_tier", default="UNLABELED")
    )
    df["evidence_url"] = df["feature_name"].apply(
        lambda name: _context_value(name, "url", default="")
    )
    df["evidence_caveat"] = df["feature_name"].apply(
        lambda name: _context_value(name, "caveat", default="")
    )
    df["feature_type_calc"] = df["feature_name"].apply(get_feature_type)
    df["action_binary"] = df["company_action"].map(ACTION_BINARY_MAP).astype("Int64")
    df["business_outcome_binary"] = df["business_outcome"].map(BUSINESS_OUTCOME_BINARY_MAP).astype("Int64")

    stamp_provenance(
        df,
        input_paths=[csv_path],
        config_path=PROJECT_ROOT / "config" / "outcomes.py",
    )

    df.to_csv(csv_path, index=False)

    supported_count = int((df["company_action"] == "SUPPORTED").sum())
    pulled_back_count = int((df["company_action"] == "PULLED_BACK").sum())
    action_unknown_count = int((df["company_action"] == "UNKNOWN").sum())
    action_labeled_count = int(df["action_binary"].notna().sum())
    positive_count = int((df["business_outcome"] == "POSITIVE").sum())
    negative_count = int((df["business_outcome"] == "NEGATIVE").sum())
    outcome_unknown_count = int((df["business_outcome"] == "UNKNOWN").sum())
    context_count = int(has_context.sum())
    no_context_count = int((~has_context).sum())

    print(f"Updated {csv_path.relative_to(PROJECT_ROOT)}")
    print("\nObserved company action:")
    print(f"  Supported: {supported_count}")
    print(f"  Pulled back: {pulled_back_count}")
    print(f"  Unknown: {action_unknown_count}")
    print(f"  Action-labeled sample used in analysis: {action_labeled_count}")

    print("\nKnown business outcome coverage:")
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")
    print(f"  Unknown: {outcome_unknown_count}")

    print("\nContext coverage:")
    print(f"  Features with public decision context: {context_count}")
    print(f"  Features still needing context research: {no_context_count}\n")

    supported = df[df["company_action"] == "SUPPORTED"][
        ["feature_name", "company", "business_outcome", "evidence_summary"]
    ]
    if not supported.empty:
        print("Supported features:")
        for _, row in supported.sort_values("business_outcome", ascending=False).iterrows():
            outcome_text = row["business_outcome"].lower()
            print(f"  + {row['feature_name']} ({row['company']}) [{outcome_text}]")
            print(f"    {row['evidence_summary']}")
        print()

    pulled_back = df[df["company_action"] == "PULLED_BACK"][
        ["feature_name", "company", "business_outcome", "evidence_summary"]
    ]
    if not pulled_back.empty:
        print("Pulled-back features:")
        for _, row in pulled_back.iterrows():
            outcome_text = row["business_outcome"].lower()
            print(f"  - {row['feature_name']} ({row['company']}) [{outcome_text}]")
            print(f"    {row['evidence_summary']}")
        print()

    ambiguous_context = df[has_context & (df["company_action"] == "UNKNOWN")][
        ["feature_name", "company", "evidence_summary"]
    ]
    if not ambiguous_context.empty:
        print("Context exists but the action is still ambiguous:")
        for _, row in ambiguous_context.iterrows():
            print(f"  ? {row['feature_name']} ({row['company']})")
            print(f"    {row['evidence_summary']}")
        print()

    if no_context_count > 0:
        print("No public context yet:")
        unknowns = df[~has_context][["feature_name", "company"]]
        for _, row in unknowns.iterrows():
            print(f"  ? {row['feature_name']} ({row['company']})")

    return df


def main() -> None:
    """Run the decision-context application script from the command line."""
    print("Applying public decision context to labeled features...\n")
    apply_outcomes_to_csv()

    print("\nNext steps:")
    print("  1. python src/analysis/statistical_analysis.py")
    print("  2. python scripts/generate_visualizations.py\n")


if __name__ == "__main__":
    main()
