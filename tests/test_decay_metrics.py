"""Tests for Google Trends decay-metric calculation edge cases."""

from datetime import timedelta

import pandas as pd

from src.data_collection.collect_trends_data import TrendsCollector


def _make_trends(launch: str, values: dict[int, float]) -> pd.DataFrame:
    """Build a minimal trends DataFrame from {days_after_launch: interest}."""
    launch_date = pd.to_datetime(launch)
    rows = [
        {"date": launch_date + timedelta(days=day_offset), "interest": value}
        for day_offset, value in values.items()
    ]
    return pd.DataFrame(rows)


class TestCalculateDecayMetrics:
    """Exercise the launch-based decay calculation without touching the API client."""

    def setup_method(self) -> None:
        """Create a collector instance without initializing pytrends."""
        self.collector = TrendsCollector.__new__(TrendsCollector)

    def test_high_decay_classified_as_novelty(self) -> None:
        """A steep week-four drop should be classified as novelty."""
        trends = _make_trends(
            "2023-01-01",
            {
                0: 100,
                1: 90,
                24: 10,
                25: 8,
                26: 9,
            },
        )
        result = self.collector.calculate_decay_metrics(trends, "2023-01-01")
        assert result["classification"] == "novelty"
        assert result["decay_rate"] > 0.70

    def test_low_decay_classified_as_sticky(self) -> None:
        """A shallow week-four drop should be classified as sticky."""
        trends = _make_trends(
            "2023-01-01",
            {
                0: 100,
                1: 95,
                24: 85,
                25: 88,
                26: 82,
            },
        )
        result = self.collector.calculate_decay_metrics(trends, "2023-01-01")
        assert result["classification"] == "sticky"
        assert result["decay_rate"] < 0.30

    def test_missing_week4_data_returns_unknown(self) -> None:
        """Missing week-four values should leave decay undefined."""
        trends = _make_trends("2023-01-01", {0: 100, 1: 90})
        result = self.collector.calculate_decay_metrics(trends, "2023-01-01")
        assert result["decay_rate"] is None
        assert result["classification"] == "unknown"

    def test_zero_peak_does_not_raise(self) -> None:
        """A zero week-one peak should not trip the truthiness check."""
        trends = _make_trends("2023-01-01", {0: 0, 24: 0, 25: 0})
        result = self.collector.calculate_decay_metrics(trends, "2023-01-01")
        assert result["week_1_peak"] == 0.0
        assert result["decay_rate"] is None
        assert result["classification"] == "unknown"
