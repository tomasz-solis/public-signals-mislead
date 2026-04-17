"""
Merge batch-level Google Trends files into a single dataset.

Merges:
- data/trends/full_trends_data_batch_*.csv → MERGED_trends_data.csv
- data/trends/full_decay_metrics_batch_*.csv → MERGED_decay_metrics.csv

If a feature appears in 'batch_extended_extreme_peaks' files, those rows REPLACE
any rows for the same feature from other batches.

Usage: python src/data_collection/merge_batches.py
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


DATA_DIR = Path("data/trends")
EXTENDED_TAG = "extended_extreme_peaks"
TRENDS_REQUIRED_COLUMNS = {"feature_id", "feature_name", "date", "interest", "launch_date"}
METRICS_REQUIRED_COLUMNS = {"feature_id", "feature_name", "decay_rate"}


def _validate_schema(df: pd.DataFrame, required_columns: set, source_name: str) -> None:
    """Raise a clear error when a batch file does not match the expected schema."""
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Schema error in '{source_name}': missing columns {sorted(missing)}")


def _load_with_source(files: List[Path], required_columns: set) -> pd.DataFrame:
    """Load CSVs and tag each row with its source filename."""
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        _validate_schema(df, required_columns, f.name)
        df["source_file"] = f.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def merge_trends() -> pd.DataFrame:
    """Merge all full_trends_data_batch_* files with extended-batch override."""
    pattern = "full_trends_data_batch_*.csv"
    trend_files = sorted(DATA_DIR.glob(pattern))

    if not trend_files:
        raise FileNotFoundError(f"No trend files matching {pattern} in {DATA_DIR}")

    logger.info("Merging trends files:")
    for f in trend_files:
        logger.info("   - %s", f.name)

    trends = _load_with_source(trend_files, TRENDS_REQUIRED_COLUMNS)

    if trends.empty:
        raise ValueError("Loaded trends data is empty")

    if "feature_id" not in trends.columns:
        raise KeyError("Expected column 'feature_id' in trends files")

    # Extended batch override
    extended_mask = trends["source_file"].str.contains(EXTENDED_TAG, na=False)
    extended_ids = trends.loc[extended_mask, "feature_id"].unique()

    logger.info("Found %d feature(s) in extended batch: %s", len(extended_ids), extended_ids.tolist())

    if len(extended_ids) > 0:
        before_rows = len(trends)
        trends = trends[(~trends["feature_id"].isin(extended_ids)) | extended_mask].copy()
        after_rows = len(trends)
        logger.info("Dropped %d shorter-window rows for extended features", before_rows - after_rows)

    # Sort and clean
    if "date" in trends.columns:
        trends = trends.sort_values(["feature_id", "date"]).reset_index(drop=True)
    else:
        trends = trends.sort_values(["feature_id"]).reset_index(drop=True)

    trends.drop(columns=["source_file"], inplace=True)

    out_path = DATA_DIR / "MERGED_trends_data.csv"
    trends.to_csv(out_path, index=False)
    logger.info("Saved merged trends to: %s", out_path)
    logger.info("Total rows: %d, Features: %d", len(trends), trends["feature_id"].nunique())

    return trends


def merge_metrics() -> pd.DataFrame:
    """Merge all full_decay_metrics_batch_* files with extended batch override."""
    pattern = "full_decay_metrics_batch_*.csv"
    metric_files = sorted(DATA_DIR.glob(pattern))

    if not metric_files:
        raise FileNotFoundError(f"No metrics files matching {pattern} in {DATA_DIR}")

    logger.info("Merging metrics files:")
    for f in metric_files:
        logger.info("   - %s", f.name)

    metrics = _load_with_source(metric_files, METRICS_REQUIRED_COLUMNS)

    if metrics.empty:
        raise ValueError("Loaded metrics data is empty")

    if "feature_id" not in metrics.columns:
        raise KeyError("Expected column 'feature_id' in metrics files")

    extended_mask = metrics["source_file"].str.contains(EXTENDED_TAG, na=False)
    extended_ids = metrics.loc[extended_mask, "feature_id"].unique()

    logger.info("Found %d feature(s) in extended metrics batch: %s", len(extended_ids), extended_ids.tolist())

    if len(extended_ids) > 0:
        before_rows = len(metrics)
        metrics = metrics[(~metrics["feature_id"].isin(extended_ids)) | extended_mask].copy()
        after_rows = len(metrics)
        logger.info("Dropped %d older metric rows for extended features", before_rows - after_rows)

    # Deduplicate (keep last/extended)
    metrics = (
        metrics.sort_values(["feature_id", "source_file"])
        .drop_duplicates(subset=["feature_id"], keep="last")
        .reset_index(drop=True)
    )

    metrics.drop(columns=["source_file"], inplace=True)

    out_path = DATA_DIR / "MERGED_decay_metrics.csv"
    metrics.to_csv(out_path, index=False)
    logger.info("Saved merged metrics to: %s", out_path)
    logger.info("Total features: %d", metrics["feature_id"].nunique())

    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Starting batch merge\n")
    merge_trends()
    merge_metrics()
    print("\nMerging complete. Run recalculate_with_peaks.py on MERGED_trends_data.csv")


if __name__ == "__main__":
    main()
