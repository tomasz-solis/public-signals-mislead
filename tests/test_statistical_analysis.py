"""Tests for ``statistical_analysis.py`` pure functions and validation helpers."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistical_analysis import (
    _ttest_groups,
    ablation_signal_combinations,
    calculate_decay_metrics_from_series,
    find_high_decay_supported_features,
    interpret_effect_size,
    load_labeled_data,
    run_all_tests,
    validate_decision_framework,
)


def test_calculate_decay_metrics_from_series_handles_empty_values() -> None:
    """The decay-summary helper should return NaNs instead of raising on empty input."""
    metrics = calculate_decay_metrics_from_series([])
    assert metrics["count"] == 0
    assert np.isnan(metrics["mean"])
    assert np.isnan(metrics["std"])


def test_interpret_effect_size_bands() -> None:
    """Verify Cohen's d thresholds map to the expected labels."""
    assert interpret_effect_size(0.1) == "negligible"
    assert interpret_effect_size(0.3) == "small"
    assert interpret_effect_size(0.6) == "medium"
    assert interpret_effect_size(1.0) == "large"
    assert interpret_effect_size(float("nan")) == "undefined"


def test_ttest_groups_returns_nan_on_empty() -> None:
    """The t-test helper should return NaNs when either group is empty."""
    t_stat, p_value = _ttest_groups(np.array([0.8, 0.9]), np.array([]))
    assert np.isnan(t_stat) and np.isnan(p_value)

    t_stat, p_value = _ttest_groups(np.array([]), np.array([0.5]))
    assert np.isnan(t_stat) and np.isnan(p_value)


def test_ttest_groups_identical_values() -> None:
    """Two identical groups should produce a non-significant p-value of 1."""
    values = np.array([0.5, 0.6, 0.7])
    _, p_value = _ttest_groups(values, values.copy())
    assert p_value == pytest.approx(1.0, abs=1e-6)


def test_find_high_decay_supported_features_threshold() -> None:
    """Only features above the requested threshold should be returned."""
    supported = pd.DataFrame(
        {
            "feature_name": ["A", "B", "C"],
            "search_decay": [0.75, 0.85, 0.90],
        }
    )
    result = find_high_decay_supported_features(supported, threshold=0.80)
    assert list(result["feature_name"]) == ["B", "C"]


def test_find_high_decay_supported_features_missing_column() -> None:
    """Missing ``search_decay`` should return an empty DataFrame, not an exception."""
    supported = pd.DataFrame({"feature_name": ["A", "B"]})
    result = find_high_decay_supported_features(supported)
    assert len(result) == 0


def test_validate_decision_framework_beats_baseline(minimal_action_labeled_df) -> None:
    """The combined public-signal rules should beat the majority baseline on the fixture."""
    result = validate_decision_framework(minimal_action_labeled_df, "action_binary")
    assert result["n_labeled"] == len(minimal_action_labeled_df)
    assert result["framework_accuracy"] >= result["baseline_accuracy"]
    assert len(result["predictions"]) == len(minimal_action_labeled_df)


def test_ablation_signal_combinations_includes_framework(minimal_action_labeled_df) -> None:
    """Ablation output should include baseline, single-signal, and combined rows."""
    result = ablation_signal_combinations(minimal_action_labeled_df, "action_binary")
    combinations = set(result["signal_combination"])
    assert "baseline (majority class)" in combinations
    assert "decay only" in combinations
    assert "sentiment only" in combinations
    assert "mentions only" in combinations
    assert "combined (decision framework)" in combinations


def test_load_labeled_data_action_schema_and_legacy_fallback(tmp_path) -> None:
    """The loader should prefer action labels and still support the legacy schema."""
    schema1_path = tmp_path / "schema1.csv"
    pd.DataFrame(
        {
            "feature_name": ["X", "Y"],
            "company_action": ["SUPPORTED", "PULLED_BACK"],
            "action_binary": pd.Series([1, 0], dtype="Int64"),
            "search_decay": [0.8, 0.9],
        }
    ).to_csv(schema1_path, index=False)

    supported, pulled_back, labeled_df = load_labeled_data(str(schema1_path))
    assert len(supported) == 1 and supported.iloc[0]["feature_name"] == "X"
    assert len(pulled_back) == 1 and pulled_back.iloc[0]["feature_name"] == "Y"
    assert "_analysis_mask" not in labeled_df.columns
    assert "_label_col" not in labeled_df.columns

    schema2_path = tmp_path / "schema2.csv"
    pd.DataFrame(
        {
            "feature_name": ["X", "Y"],
            "success_binary": [1, 0],
            "search_decay": [0.8, 0.9],
        }
    ).to_csv(schema2_path, index=False)

    supported2, pulled_back2, _ = load_labeled_data(str(schema2_path))
    assert len(supported2) == 1 and len(pulled_back2) == 1


def test_run_all_tests_includes_framework_outputs(tmp_path, minimal_action_labeled_df) -> None:
    """The top-level analysis run should expose framework validation and ablation results."""
    csv_path = tmp_path / "labeled.csv"
    minimal_action_labeled_df.to_csv(csv_path, index=False)
    result = run_all_tests(str(csv_path))
    assert result["sample_sizes"]["supported"] == 4
    assert result["sample_sizes"]["pulled_back"] == 2
    assert "framework_validation" in result
    assert "signal_ablation" in result
