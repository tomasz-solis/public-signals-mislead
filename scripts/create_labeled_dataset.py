"""
Merge feature inventory with Reddit validation results to create the base analysis dataset.

The labels created here describe what the *public signals* suggest. They are not
business outcomes and they are not company decisions. The later
``scripts/apply_outcomes.py`` step adds that separate decision context.

Usage: python scripts/create_labeled_dataset.py
"""

from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
VALIDATION_DIR = DATA_DIR / "validation"


def map_signal_label(classification: str) -> str:
    """Map Reddit classification to a coarse public-signal interpretation."""
    if classification in ("ADOPTION", "SUSTAINED_INTEREST"):
        return "supportive"
    if classification in ("ABANDONMENT", "LOW_AWARENESS"):
        return "caution"
    if classification == "NO_DECAY_DATA":
        return "no_decay_data"
    return "inconclusive"


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    inventory_path = RAW_DIR / "feature_inventory.csv"
    reddit_path = VALIDATION_DIR / "reddit_validation_results.csv"

    inv = pd.read_csv(inventory_path)
    reddit = pd.read_csv(reddit_path)

    # Merge on (feature_name, company, launch_date)
    df = inv.merge(
        reddit,
        on=["feature_name", "company", "launch_date"],
        how="inner",
        validate="one_to_one"
    )

    # Derive public-signal labels. These are not business outcomes.
    df["signal_label"] = df["classification"].map(map_signal_label)
    df["signal_binary"] = (
        df["signal_label"]
        .map({"supportive": 1, "caution": 0})
        .astype("Int64")
    )

    out_path = VALIDATION_DIR / "labeled_features.csv"
    df.to_csv(out_path, index=False)

    print(f"✓ Saved labeled dataset: {out_path}")
    print("\nPublic-signal label counts:")
    print(df["signal_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
