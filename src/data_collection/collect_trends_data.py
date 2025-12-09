"""
Google Trends Data Collection Script

This script collects Google Trends data for subscription feature launches.
It fetches weekly search interest data and calculates decay metrics.

Usage (from project root):

    # Pilot (first 10 features from a single inventory file)
    python src/data_collection/reddit/collect_trends_data.py --pilot

    # Full runs in batches (to respect API limits)
    python src/data_collection/reddit/collect_trends_data.py --full --input data/raw/batches/batch_1_of_5.csv
    python src/data_collection/reddit/collect_trends_data.py --full --input data/raw/batches/batch_2_of_5.csv
    ... repeat for all batches
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm


class TrendsCollector:
    """
    Collects Google Trends data for feature launches.

    Attributes:
        pytrends: PyTrends request object.
        data_dir: Directory for storing trends data.
        rate_limit_delay: Seconds to wait between API calls.
        max_retries: Number of retries when an API call fails (e.g. 429).
        backoff_schedule: List of backoff durations (seconds) used between retries.
    """

    def __init__(
        self,
        data_dir: str = "data/trends",
        rate_limit_delay: int = 2,
        max_retries: int = 3,
        backoff_schedule: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the TrendsCollector.

        Args:
            data_dir: Directory path for storing trends data.
            rate_limit_delay: Seconds to wait between requests to avoid rate limiting.
            max_retries: Maximum number of retries per feature when a call fails.
            backoff_schedule: List of sleep durations (in seconds) for retries.
                              If None, defaults to [2, 5, 10].
        """
        self.pytrends = TrendReq(hl="en-US", tz=360)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.backoff_schedule = backoff_schedule or [2, 5, 10]

    # -------------------------------------------------------------------------
    # Timeframe logic
    # -------------------------------------------------------------------------
    def get_feature_timeframe(self, launch_date: str) -> str:
        """
        Calculate the timeframe for trends data collection.

        Currently:
            - 2 weeks before launch
            - 32 weeks after launch

        This longer post-launch window allows downstream scripts to:
            - normalize by peak,
            - compute week_4 / week_8 decay,
            - and still have a long tail for “sustained interest”.

        Args:
            launch_date: Feature launch date in YYYY-MM-DD format.

        Returns:
            Timeframe string in pytrends format (e.g. '2023-02-01 2023-10-01').
        """
        launch = datetime.strptime(launch_date, "%Y-%m-%d")
        start_date = launch - timedelta(days=14)   # 2 weeks before
        end_date = launch + timedelta(days=224)    # 32 weeks after for extended ones only, 154 days for the rest

        return f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    # -------------------------------------------------------------------------
    # Core collection for a single feature
    # -------------------------------------------------------------------------
    def collect_feature_trends(
        self,
        feature_id: int,
        keyword: str,
        launch_date: str,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Collect Google Trends data for a single feature, with retries & backoff.

        Args:
            feature_id: Unique feature identifier.
            keyword: Google Trends search keyword.
            launch_date: Feature launch date (YYYY-MM-DD).
            feature_name: Human-readable feature name.

        Returns:
            DataFrame with columns:
                - feature_id
                - feature_name
                - keyword
                - launch_date
                - date
                - interest

            Returns an empty DataFrame if all retries fail or GT reports no data.
        """
        if not isinstance(keyword, str) or not keyword.strip():
            tqdm.write(
                f"⚠️  Skipping feature_id={feature_id} ('{feature_name}') – "
                f"missing or empty google_trends_keyword."
            )
            return pd.DataFrame()

        timeframe = self.get_feature_timeframe(launch_date)

        # Retry loop to handle "429" and other transient issues
        for attempt in range(1, self.max_retries + 1):
            try:
                # Build payload for Google Trends API
                self.pytrends.build_payload(
                    kw_list=[keyword],
                    timeframe=timeframe,
                    geo="US",  # Focus on US market for consistency
                )

                trends_data = self.pytrends.interest_over_time()

                if trends_data.empty:
                    tqdm.write(
                        f"⚠️  No data found for '{keyword}' "
                        f"(feature_id={feature_id}, '{feature_name}') – "
                        f"may be too low volume."
                    )
                    return pd.DataFrame()

                # Clean and format data
                trends_data = trends_data.reset_index()
                trends_data = trends_data.rename(columns={keyword: "interest"})

                trends_data["feature_id"] = feature_id
                trends_data["feature_name"] = feature_name
                trends_data["keyword"] = keyword
                trends_data["launch_date"] = launch_date

                trends_data = trends_data[
                    [
                        "feature_id",
                        "feature_name",
                        "keyword",
                        "launch_date",
                        "date",
                        "interest",
                    ]
                ]

                return trends_data

            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "Rate Limit" in msg

                if attempt < self.max_retries and is_rate_limit:
                    # Exponential-ish backoff for rate limit errors
                    delay_idx = min(attempt - 1, len(self.backoff_schedule) - 1)
                    delay = self.backoff_schedule[delay_idx]
                    tqdm.write(
                        f"⏳  Rate-limited for '{feature_name}' ({keyword}) "
                        f"[attempt {attempt}/{self.max_retries}] – sleeping {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                # Non-rate-limit error, or final failed attempt
                tqdm.write(
                    f"❌ Error collecting data for '{feature_name}' "
                    f"({keyword}) on attempt {attempt}/{self.max_retries}: {msg}"
                )
                return pd.DataFrame()

        # Should not really get here, but keep it explicit
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    def calculate_decay_metrics(
        self, trends_df: pd.DataFrame, launch_date: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate decay metrics from trends data.

        Metrics (simple week-based definitions):
            - week_1_peak:
                Maximum interest in 0–7 days after launch.
            - week_4_interest:
                Average interest in days 21–28 after launch.
            - decay_rate:
                (week_1_peak - week_4_interest) / week_1_peak, if week_1_peak > 0.
            - classification:
                * 'sticky'   if decay_rate < 0.30
                * 'mixed'    if 0.30 <= decay_rate < 0.70
                * 'novelty'  if decay_rate >= 0.70
                * 'unknown'  if decay_rate is None or negative
                * 'no_data'  if trends_df is empty.

        Args:
            trends_df: DataFrame with trends data for a feature.
            launch_date: Feature launch date (YYYY-MM-DD).

        Returns:
            Dictionary with keys:
                - week_1_peak
                - week_4_interest
                - decay_rate
                - classification
        """
        if trends_df.empty:
            return {
                "week_1_peak": None,
                "week_4_interest": None,
                "decay_rate": None,
                "classification": "no_data",
            }

        launch = pd.to_datetime(launch_date)
        trends_df = trends_df.copy()
        trends_df["date"] = pd.to_datetime(trends_df["date"])

        # Week 1: 0–7 days after launch
        week_1_data = trends_df[
            (trends_df["date"] >= launch)
            & (trends_df["date"] < launch + timedelta(days=7))
        ]
        week_1_peak = week_1_data["interest"].max() if not week_1_data.empty else 0

        # Week 4: 21–28 days after launch
        week_4_data = trends_df[
            (trends_df["date"] >= launch + timedelta(days=21))
            & (trends_df["date"] < launch + timedelta(days=28))
        ]
        week_4_interest = (
            week_4_data["interest"].mean() if not week_4_data.empty else 0
        )

        if week_1_peak > 0:
            decay_rate = (week_1_peak - week_4_interest) / week_1_peak
        else:
            decay_rate = None

        if decay_rate is None or decay_rate < 0:
            classification = "unknown"
        elif decay_rate < 0.30:
            classification = "sticky"
        elif decay_rate < 0.70:
            classification = "mixed"
        else:
            classification = "novelty"

        return {
            "week_1_peak": float(week_1_peak) if week_1_peak is not None else None,
            "week_4_interest": float(week_4_interest)
            if week_4_interest is not None
            else None,
            "decay_rate": float(decay_rate) if decay_rate is not None else None,
            "classification": classification,
        }

    # -------------------------------------------------------------------------
    # Bulk collection
    # -------------------------------------------------------------------------
    def collect_all_features(
        self,
        features_df: pd.DataFrame,
        pilot_only: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect trends data for multiple features.

        Args:
            features_df: DataFrame with feature inventory. Must contain:
                - feature_id
                - feature_name
                - company
                - feature_type
                - launch_date
                - google_trends_keyword
            pilot_only: If True, only collect first 10 features.

        Returns:
            Tuple of:
                - combined_trends: concatenated time series for all features.
                - combined_metrics: one row per feature with decay metrics.
        """
        if pilot_only:
            features_df = features_df.head(10)
            print(f"\n🚀 Collecting data for {len(features_df)} PILOT features...\n")
        else:
            print(f"\n🚀 Collecting data for ALL {len(features_df)} features...\n")

        all_trends: List[pd.DataFrame] = []
        all_metrics: List[Dict] = []

        for _, row in tqdm(
            features_df.iterrows(),
            total=len(features_df),
            desc="Collecting trends",
        ):
            feature_id = row["feature_id"]
            feature_name = row["feature_name"]
            launch_date = str(row["launch_date"])
            keyword = row.get("google_trends_keyword", "")

            trends_df = self.collect_feature_trends(
                feature_id=feature_id,
                keyword=keyword,
                launch_date=launch_date,
                feature_name=feature_name,
            )

            if not trends_df.empty:
                all_trends.append(trends_df)

                metrics = self.calculate_decay_metrics(trends_df, launch_date)
                metrics["feature_id"] = feature_id
                metrics["feature_name"] = feature_name
                metrics["company"] = row.get("company", None)
                metrics["feature_type"] = row.get("feature_type", None)
                all_metrics.append(metrics)

            # Baseline delay between features to be gentle with the API
            time.sleep(self.rate_limit_delay)

        combined_trends = (
            pd.concat(all_trends, ignore_index=True) if all_trends else pd.DataFrame()
        )
        combined_metrics = (
            pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
        )

        return combined_trends, combined_metrics

    # -------------------------------------------------------------------------
    # Saving & summary
    # -------------------------------------------------------------------------
    def save_results(
        self,
        trends_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        pilot: bool = False,
        batch_name: Optional[str] = None,
    ) -> None:
        """
        Save trends data and metrics to CSV files and print a robust summary.

        Args:
            trends_df: DataFrame with raw trends time series.
            metrics_df: DataFrame with calculated decay metrics.
            pilot: Whether this is pilot data or a full batch.
            batch_name: Optional batch identifier (e.g., "batch_2_of_5").
        """
        prefix = "pilot_" if pilot else "full_"

        # Use batch name if provided (e.g. batch_2_of_5), otherwise a timestamp
        if batch_name:
            suffix = batch_name
        else:
            suffix = 'batch_extended_extreme_peaks'

        trends_path = self.data_dir / f"{prefix}trends_data_{suffix}.csv"
        metrics_path = self.data_dir / f"{prefix}decay_metrics_{suffix}.csv"

        trends_df.to_csv(trends_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        print(f"\n✅ Saved trends data to: {trends_path}")
        print(f"✅ Saved metrics to: {metrics_path}")

        # --- Robust summary (handles empty metrics_df) ---
        print("\n📊 Summary Statistics:")

        if metrics_df.empty:
            print("   Total features collected: 0")
            print("   (All features likely failed due to rate limits or no data.)")
            return

        if "feature_id" in metrics_df.columns:
            n_features = metrics_df["feature_id"].nunique()
        else:
            n_features = len(metrics_df)

        print(f"   Total features collected: {n_features}")

        if "classification" in metrics_df.columns:
            print("\n   Classification breakdown:")
            print(metrics_df["classification"].value_counts(dropna=False).to_string())
        else:
            print("\n   Classification breakdown: [no 'classification' column]")

        if "decay_rate" in metrics_df.columns and metrics_df["decay_rate"].notna().any():
            avg_decay = metrics_df["decay_rate"].mean()
            print(f"\n   Average decay rate: {avg_decay:.2%}")
        else:
            print("\n   Average decay rate: n/a (no valid decay_rate values)")


# -------------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------------
def main() -> None:
    """Main execution function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Collect Google Trends data for features."
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Collect pilot data (first 10 features).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Collect full dataset (all features in input file).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/feature_inventory.csv",
        help="Path to feature inventory CSV (or batch CSV).",
    )

    args = parser.parse_args()

    if not args.pilot and not args.full:
        print("❌ Please specify either --pilot or --full")
        return

    # Load feature inventory
    print(f"📂 Loading feature inventory from: {args.input}")
    features_df = pd.read_csv(args.input)
    print(f"   Loaded {len(features_df)} features")

    # Extract batch name from input path (e.g., "batch_2_of_5" from ".../batch_2_of_5.csv")
    input_path = Path(args.input)
    batch_name = input_path.stem if "batch" in input_path.stem else None

    # Initialize collector
    collector = TrendsCollector()

    # Collect data
    trends_df, metrics_df = collector.collect_all_features(
        features_df,
        pilot_only=args.pilot,
    )

    # Save results
    collector.save_results(
        trends_df,
        metrics_df,
        pilot=args.pilot,
        batch_name=batch_name,
    )

    print("\n🎉 Data collection complete!")


if __name__ == "__main__":
    main()
