"""Shared fixtures for analysis and decay-metric tests."""

import pandas as pd
import pytest


@pytest.fixture
def minimal_action_labeled_df() -> pd.DataFrame:
    """Return a small but fully labeled action dataset for analysis tests."""
    return pd.DataFrame(
        {
            "feature_name": ["A", "B", "C", "D", "E", "F"],
            "search_decay": [0.90, 0.85, 0.25, 0.92, 0.88, 0.74],
            "total_mentions": [40, 28, 8, 5, 12, 30],
            "negative_ratio": [0.10, 0.15, 0.08, 0.45, 0.35, 0.12],
            "positive_ratio": [0.58, 0.50, 0.62, 0.20, 0.25, 0.55],
            "neutral_ratio": [0.32, 0.35, 0.30, 0.35, 0.40, 0.33],
            "avg_score": [100, 75, 20, 15, 18, 90],
            "company_action": [
                "SUPPORTED",
                "SUPPORTED",
                "SUPPORTED",
                "PULLED_BACK",
                "PULLED_BACK",
                "SUPPORTED",
            ],
            "action_binary": pd.Series([1, 1, 1, 0, 0, 1], dtype="Int64"),
            "business_outcome": ["POSITIVE", "UNKNOWN", "UNKNOWN", "NEGATIVE", "UNKNOWN", "POSITIVE"],
            "feature_type_calc": ["AI", "UTILITY", "UTILITY", "SOCIAL", "SOCIAL", "AI"],
            "company": ["TestCo"] * 6,
        }
    )
