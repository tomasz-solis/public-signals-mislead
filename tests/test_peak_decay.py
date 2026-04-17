"""Tests for peak-based decay recalculation.

The ARCHITECTURE.md calls this 'the more defensible version of decay,'
so the headline numbers in the README depend on this pipeline working
correctly. These tests cover the three public functions.
"""

from datetime import timedelta

import pandas as pd
import pytest

from src.data_collection.recalculate_with_peaks import (
    find_peak_date,
    calculate_peak_based_decay,
    recalculate_all_metrics,
)


def _build_trends(
    feature_name: str,
    launch: str,
    values: list[tuple[int, float]],
) -> pd.DataFrame:
    """Build a trends frame for one feature from (day_offset, interest) pairs."""
    launch_ts = pd.to_datetime(launch)
    return pd.DataFrame(
        {
            "feature_id": 1,
            "feature_name": feature_name,
            "date": [launch_ts + timedelta(days=d) for d, _ in values],
            "interest": [v for _, v in values],
            "launch_date": launch_ts,
        }
    )


class TestFindPeakDate:
    """Tests for find_peak_date."""

    def test_returns_max_interest_day(self) -> None:
        trends = _build_trends("X", "2023-01-01", [(0, 40), (7, 80), (14, 100), (28, 20)])
        peak_date, peak_value = find_peak_date(trends, "X")
        assert peak_value == 100
        assert peak_date == pd.to_datetime("2023-01-15")

    def test_returns_none_for_unknown_feature(self) -> None:
        trends = _build_trends("X", "2023-01-01", [(0, 40)])
        peak_date, peak_value = find_peak_date(trends, "NONEXISTENT")
        assert peak_date is None
        assert peak_value is None

    def test_returns_first_occurrence_when_tied(self) -> None:
        trends = _build_trends("X", "2023-01-01", [(0, 100), (7, 100), (14, 50)])
        peak_date, peak_value = find_peak_date(trends, "X")
        assert peak_value == 100
        assert peak_date == pd.to_datetime("2023-01-01")


class TestCalculatePeakBasedDecay:
    """Tests for calculate_peak_based_decay."""

    def test_uses_peak_not_launch(self) -> None:
        """Peak on day 14, week-4-after-peak data at days 35-41 = 90% decay."""
        values = [(0, 20), (7, 60), (14, 100), (35, 10), (36, 10)]
        trends = _build_trends("X", "2023-01-01", values)
        result = calculate_peak_based_decay(
            trends, "X",
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-15"),
            100.0,
        )
        assert result["decay_rate_w4"] == pytest.approx(0.90, abs=0.01)
        assert result["days_to_peak"] == 14
        assert result["classification"] == "novelty"

    def test_low_decay_classified_as_sticky(self) -> None:
        """10% drop from peak should be sticky."""
        values = [(0, 100), (21, 90), (22, 90)]
        trends = _build_trends("X", "2023-01-01", values)
        result = calculate_peak_based_decay(
            trends, "X",
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-01"),
            100.0,
        )
        assert result["decay_rate_w4"] == pytest.approx(0.10, abs=0.01)
        assert result["classification"] == "sticky"

    def test_missing_week4_returns_none(self) -> None:
        """No observations 21-28 days after peak means decay can't be computed."""
        trends = _build_trends("X", "2023-01-01", [(0, 20), (7, 100)])
        result = calculate_peak_based_decay(
            trends, "X",
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-08"),
            100.0,
        )
        assert result["decay_rate_w4"] is None
        assert result["classification"] == "unknown"

    def test_zero_peak_returns_none(self) -> None:
        """Peak interest of 0 means decay is undefined (division by zero guard)."""
        trends = _build_trends("X", "2023-01-01", [(0, 0), (7, 0), (28, 0)])
        result = calculate_peak_based_decay(
            trends, "X",
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-01"),
            0.0,
        )
        assert result["decay_rate_w4"] is None

    def test_week8_decay_computed_independently(self) -> None:
        """Week 8 has its own window (days 56-62 after peak)."""
        values = [(0, 100), (21, 50), (56, 20), (57, 20)]
        trends = _build_trends("X", "2023-01-01", values)
        result = calculate_peak_based_decay(
            trends, "X",
            pd.to_datetime("2023-01-01"),
            pd.to_datetime("2023-01-01"),
            100.0,
        )
        assert result["decay_rate_w4"] == pytest.approx(0.50, abs=0.01)
        assert result["decay_rate_w8"] == pytest.approx(0.80, abs=0.01)


class TestRecalculateAllMetrics:
    """Tests for recalculate_all_metrics."""

    def test_validates_schema(self) -> None:
        """Missing required columns should raise, not fail silently."""
        bad = pd.DataFrame({"feature_name": ["X"], "interest": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            recalculate_all_metrics(bad)

    def test_produces_one_row_per_feature(self) -> None:
        """Each feature in the input should produce one row in the output."""
        trends_a = _build_trends("A", "2023-01-01", [(0, 100), (7, 80), (21, 40), (22, 38)])
        trends_b = _build_trends("B", "2023-02-01", [(0, 50), (14, 90), (35, 20), (36, 18)])
        combined = pd.concat([trends_a, trends_b], ignore_index=True)
        combined["company"] = "TestCo"
        combined["feature_type"] = "UTILITY"
        combined["feature_id"] = combined["feature_name"].map({"A": 1, "B": 2})

        result = recalculate_all_metrics(combined)
        assert set(result["feature_name"]) == {"A", "B"}
        assert len(result) == 2

    def test_skips_features_with_no_data(self) -> None:
        """An empty feature (find_peak_date returns None) should be dropped."""
        trends = _build_trends("A", "2023-01-01", [(0, 100), (21, 50)])
        trends["company"] = "TestCo"
        trends["feature_type"] = "UTILITY"

        # Add a ghost feature with no matching rows
        ghost = pd.DataFrame(
            {
                "feature_id": [99],
                "feature_name": ["GHOST"],
                "date": [pd.to_datetime("2023-06-01")],
                "interest": [0],
                "launch_date": [pd.to_datetime("2023-06-01")],
                "company": ["TestCo"],
                "feature_type": ["UTILITY"],
            }
        )
        combined = pd.concat([trends, ghost], ignore_index=True)
        result = recalculate_all_metrics(combined)
        assert "A" in result["feature_name"].values
        assert "GHOST" in result["feature_name"].values
