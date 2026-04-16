"""
Google Trends data collection for subscription feature launches.
Fetches weekly search interest data and calculates decay metrics.

Usage:
    python src/data_collection/collect_trends_data.py --pilot
    python src/data_collection/collect_trends_data.py --full --input data/raw/batches/batch_1_of_5.csv
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

from config.thresholds import DECAY_NOVELTY_THRESHOLD, DECAY_STICKY_THRESHOLD


logger = logging.getLogger(__name__)


class TrendsCollector:
    """
    Collects Google Trends data for feature launches with retry logic.
    Handles rate limiting with exponential backoff.
    """

    def __init__(self, data_dir: str = "data/trends", rate_limit_delay: int = 2,
                 max_retries: int = 3, backoff_schedule: Optional[List[int]] = None) -> None:
        """Initialize collector with rate limiting config."""
        self.pytrends = TrendReq(hl="en-US", tz=360)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.backoff_schedule = backoff_schedule or [2, 5, 10]

    def get_feature_timeframe(self, launch_date: str) -> str:
        """
        Calculate timeframe for trends collection.
        Returns 2 weeks before launch + 32 weeks after (for peak normalization).
        """
        launch = datetime.strptime(launch_date, "%Y-%m-%d")
        start_date = launch - timedelta(days=14)
        end_date = launch + timedelta(days=224)  # 32 weeks
        return f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    def collect_feature_trends(self, feature_id: int, keyword: str, launch_date: str,
                               feature_name: str) -> pd.DataFrame:
        """
        Collect Google Trends data for a single feature with retries.
        Returns empty DataFrame if all retries fail or no data found.
        """
        if not isinstance(keyword, str) or not keyword.strip():
            logger.warning(
                "Skipping feature_id=%s (%s) because the keyword is missing.",
                feature_id,
                feature_name,
            )
            return pd.DataFrame()

        timeframe = self.get_feature_timeframe(launch_date)

        # Retry loop for 429 errors
        for attempt in range(1, self.max_retries + 1):
            try:
                self.pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo="US")
                trends_data = self.pytrends.interest_over_time()

                if trends_data.empty:
                    logger.warning(
                        "No trends data for keyword=%s (feature_id=%s); likely too low volume.",
                        keyword,
                        feature_id,
                    )
                    return pd.DataFrame()

                # Clean and format
                trends_data = trends_data.reset_index()
                trends_data = trends_data.rename(columns={keyword: "interest"})
                trends_data["feature_id"] = feature_id
                trends_data["feature_name"] = feature_name
                trends_data["keyword"] = keyword
                trends_data["launch_date"] = launch_date
                
                return trends_data[["feature_id", "feature_name", "keyword", "launch_date", "date", "interest"]]

            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "Rate Limit" in msg

                if attempt < self.max_retries and is_rate_limit:
                    delay_idx = min(attempt - 1, len(self.backoff_schedule) - 1)
                    delay = self.backoff_schedule[delay_idx]
                    logger.warning(
                        "Rate-limited while collecting %s on attempt %s/%s; sleeping %ss.",
                        feature_name,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "Error collecting %s on attempt %s/%s: %s",
                    feature_name,
                    attempt,
                    self.max_retries,
                    msg,
                )
                return pd.DataFrame()

        return pd.DataFrame()

    def calculate_decay_metrics(self, trends_df: pd.DataFrame, launch_date: str) -> Dict[str, Optional[float]]:
        """
        Calculate decay metrics from trends data.
        
        Metrics:
        - week_1_peak: Max interest in 0-7 days after launch
        - week_4_interest: Avg interest in days 21-28 after launch
        - decay_rate: (week_1_peak - week_4_interest) / week_1_peak
        - classification: sticky (<30%), mixed (30-70%), novelty (>70%)
        """
        trends_df["date"] = pd.to_datetime(trends_df["date"])
        launch = pd.to_datetime(launch_date)

        # Week 1: 0-7 days after launch
        week_1_data = trends_df[(trends_df["date"] >= launch) & (trends_df["date"] < launch + timedelta(days=7))]
        week_1_peak = week_1_data["interest"].max() if not week_1_data.empty else None

        # Week 4: 21-28 days after launch
        week_4_data = trends_df[
            (trends_df["date"] >= launch + timedelta(days=21)) & 
            (trends_df["date"] < launch + timedelta(days=28))
        ]
        week_4_interest = week_4_data["interest"].mean() if not week_4_data.empty else None

        # Calculate decay
        if week_1_peak is not None and week_1_peak > 0 and week_4_interest is not None:
            decay_rate = (week_1_peak - week_4_interest) / week_1_peak
        else:
            decay_rate = None

        # Classify
        if decay_rate is None or decay_rate < 0:
            classification = "unknown"
        elif decay_rate < DECAY_STICKY_THRESHOLD:
            classification = "sticky"
        elif decay_rate < DECAY_NOVELTY_THRESHOLD:
            classification = "mixed"
        else:
            classification = "novelty"

        return {
            "week_1_peak": float(week_1_peak) if week_1_peak is not None else None,
            "week_4_interest": float(week_4_interest) if week_4_interest is not None else None,
            "decay_rate": float(decay_rate) if decay_rate is not None else None,
            "classification": classification,
        }

    def collect_all_features(self, features_df: pd.DataFrame, pilot_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect trends data for multiple features.
        Returns (combined_trends, combined_metrics) DataFrames.
        """
        if pilot_only:
            features_df = features_df.head(10)
            logger.info("Collecting %s pilot features.", len(features_df))
        else:
            logger.info("Collecting %s features.", len(features_df))

        all_trends: List[pd.DataFrame] = []
        all_metrics: List[Dict] = []

        for _, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Collecting trends"):
            trends_df = self.collect_feature_trends(
                feature_id=row["feature_id"],
                keyword=row.get("google_trends_keyword", ""),
                launch_date=str(row["launch_date"]),
                feature_name=row["feature_name"]
            )

            if not trends_df.empty:
                all_trends.append(trends_df)
                metrics = self.calculate_decay_metrics(trends_df, str(row["launch_date"]))
                metrics["feature_id"] = row["feature_id"]
                metrics["feature_name"] = row["feature_name"]
                metrics["company"] = row.get("company", None)
                metrics["feature_type"] = row.get("feature_type", None)
                all_metrics.append(metrics)

            time.sleep(self.rate_limit_delay)

        combined_trends = pd.concat(all_trends, ignore_index=True) if all_trends else pd.DataFrame()
        combined_metrics = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()

        return combined_trends, combined_metrics

    def save_results(self, trends_df: pd.DataFrame, metrics_df: pd.DataFrame,
                    pilot: bool = False, batch_name: Optional[str] = None) -> None:
        """Save trends data and metrics to CSV files."""
        prefix = "pilot_" if pilot else "full_"
        suffix = batch_name if batch_name else "batch"

        trends_path = self.data_dir / f"{prefix}trends_data_{suffix}.csv"
        metrics_path = self.data_dir / f"{prefix}decay_metrics_{suffix}.csv"

        trends_df.to_csv(trends_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        logger.info("Saved trends data to %s", trends_path)
        logger.info("Saved decay metrics to %s", metrics_path)

        # Summary
        if metrics_df.empty:
            logger.warning("Collected 0 features; all requests failed or returned no data.")
            return

        n_features = metrics_df["feature_id"].nunique() if "feature_id" in metrics_df.columns else len(metrics_df)
        logger.info("Collected %s features.", n_features)

        if "classification" in metrics_df.columns:
            logger.info("Classification breakdown:\n%s", metrics_df["classification"].value_counts(dropna=False).to_string())

        if "decay_rate" in metrics_df.columns and metrics_df["decay_rate"].notna().any():
            avg_decay = metrics_df["decay_rate"].mean()
            logger.info("Average decay rate: %.2f%%", avg_decay * 100)


def _configure_logging() -> None:
    """Configure console and file logging for collection runs."""
    Path("data").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/collection.log"),
        ],
    )


def main() -> None:
    """CLI entry point."""
    _configure_logging()
    parser = argparse.ArgumentParser(description="Collect Google Trends data for features")
    parser.add_argument("--pilot", action="store_true", help="Collect first 10 features only")
    parser.add_argument("--full", action="store_true", help="Collect all features")
    parser.add_argument("--input", type=str, default="data/raw/feature_inventory.csv",
                       help="Path to feature inventory CSV")

    args = parser.parse_args()

    if not args.pilot and not args.full:
        logger.error("Specify either --pilot or --full.")
        return

    logger.info("Loading features from %s", args.input)
    features_df = pd.read_csv(args.input)
    logger.info("Loaded %s features.", len(features_df))

    # Extract batch name from path
    input_path = Path(args.input)
    batch_name = input_path.stem if "batch" in input_path.stem else None

    collector = TrendsCollector()
    trends_df, metrics_df = collector.collect_all_features(features_df, pilot_only=args.pilot)
    collector.save_results(trends_df, metrics_df, pilot=args.pilot, batch_name=batch_name)

    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
