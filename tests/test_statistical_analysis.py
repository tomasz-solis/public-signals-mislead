"""Tests for ``statistical_analysis.py`` pure functions and validation helpers."""

import numpy as np
import pandas as pd
import pytest

import src.analysis.statistical_analysis as statistical_analysis
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


def test_load_labeled_data_resolves_repo_relative_path(monkeypatch, tmp_path) -> None:
    """Relative input paths should fall back to the repo root when cwd differs."""
    repo_root = tmp_path / "repo"
    csv_path = repo_root / "data" / "validation" / "labeled_features.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature_name": ["X", "Y"],
            "company_action": ["SUPPORTED", "PULLED_BACK"],
            "action_binary": pd.Series([1, 0], dtype="Int64"),
            "search_decay": [0.8, 0.9],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(statistical_analysis, "PROJECT_ROOT", repo_root)
    monkeypatch.chdir(tmp_path)

    supported, pulled_back, _ = load_labeled_data("data/validation/labeled_features.csv")
    assert len(supported) == 1 and supported.iloc[0]["feature_name"] == "X"
    assert len(pulled_back) == 1 and pulled_back.iloc[0]["feature_name"] == "Y"


def test_save_results_resolves_repo_relative_path(monkeypatch, tmp_path) -> None:
    """Relative output paths should still write inside the repo when cwd differs."""
    repo_root = tmp_path / "repo"
    output_path = repo_root / "data" / "validation" / "statistical_results.csv"

    monkeypatch.setattr(statistical_analysis, "PROJECT_ROOT", repo_root)
    monkeypatch.chdir(tmp_path)

    statistical_analysis.save_results(
        {
            "decay_test": {
                "metric": "Search decay",
                "test_used": "mann_whitney_u",
                "supported_mean": 0.8,
                "pulled_back_mean": 0.9,
                "difference_in_means": -0.1,
                "p_value": 0.28,
                "t_pvalue": 0.41,
                "mw_pvalue": 0.28,
                "cohens_d": -0.52,
                "effect_size_label": "medium",
                "significant": False,
                "significant_bonferroni": False,
            },
            "mentions_test": {
                "metric": "Reddit mentions",
                "test_used": "mann_whitney_u",
                "supported_mean": 30.8,
                "pulled_back_mean": 4.3,
                "difference_in_means": 26.5,
                "p_value": 0.14,
                "t_pvalue": 0.01,
                "mw_pvalue": 0.14,
                "cohens_d": 0.78,
                "effect_size_label": "medium",
                "significant": False,
                "significant_bonferroni": False,
            },
            "sentiment_test": {
                "metric": "Negative sentiment",
                "test_used": "mann_whitney_u",
                "supported_mean": 0.10,
                "pulled_back_mean": 0.14,
                "difference_in_means": -0.04,
                "p_value": 0.77,
                "t_pvalue": 0.81,
                "mw_pvalue": 0.77,
                "cohens_d": -0.33,
                "effect_size_label": "small",
                "significant": False,
                "significant_bonferroni": False,
            },
        }
    )

    saved = pd.read_csv(output_path)
    assert list(saved["metric"]) == ["Search decay", "Reddit mentions", "Negative sentiment"]


# --- Cohen's d computation ---


def test_calculate_cohens_d_known_groups() -> None:
    """Two groups with known means and stds should produce a predictable d."""
    from src.analysis.statistical_analysis import _calculate_cohens_d

    # mean=3.0, mean=2.0, both with identical std
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    d = _calculate_cohens_d(x, y)
    # Both have same std so pooled std = that std, d = diff / std
    assert d == pytest.approx(0.632, abs=0.05)


def test_calculate_cohens_d_identical_groups_returns_zero() -> None:
    """Identical groups should yield d = 0."""
    from src.analysis.statistical_analysis import _calculate_cohens_d

    x = np.array([1.0, 2.0, 3.0])
    assert _calculate_cohens_d(x, x.copy()) == pytest.approx(0.0)


def test_calculate_cohens_d_empty_group_returns_nan() -> None:
    """An empty group should yield NaN, not an exception."""
    from src.analysis.statistical_analysis import _calculate_cohens_d

    assert np.isnan(_calculate_cohens_d(np.array([1.0, 2.0]), np.array([])))
    assert np.isnan(_calculate_cohens_d(np.array([]), np.array([1.0])))


def test_calculate_cohens_d_unequal_sizes_uses_df_weighting() -> None:
    """With unequal group sizes, df-weighted pooled std differs from simple average.

    This test would fail if someone reverts to sqrt((s1^2 + s2^2) / 2).
    """
    from src.analysis.statistical_analysis import _calculate_cohens_d

    # Large group with tight variance, small group with wide variance
    x = np.array([10.0, 10.1, 10.2, 9.9, 9.8, 10.0, 10.1, 9.9, 10.0, 10.05])  # n=10, std ~0.12
    y = np.array([8.0, 12.0, 10.0])  # n=3, std = 2.0

    d = _calculate_cohens_d(x, y)
    # df-weighted pooled_var = (9 * 0.12^2 + 2 * 4.0) / 11 ~ 0.739
    # pooled_std ~ 0.860, d ~ (10.005 - 10.0) / 0.860 ~ 0.006
    # simple pooled would give sqrt((0.012 + 4.0)/2) ~ 1.417, d ~ 0.004
    # The absolute values differ by ~50% - this tolerance catches the wrong formula
    assert abs(d) < 0.1  # both are near zero, but let's verify it doesn't crash

    # More meaningful check: known calculation with the df-weighted formula
    x2 = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0])  # n=16, mean=5, std=0
    y2 = np.array([2.0, 4.0, 6.0])  # n=3, mean=4, std=2
    d2 = _calculate_cohens_d(x2, y2)
    # df-weighted: pooled_var = (15*0 + 2*4) / 17 = 0.471, pooled_std = 0.686
    # d = (5.0 - 4.0) / 0.686 = 1.458
    # simple pooled: sqrt((0 + 4) / 2) = 1.414, d = 0.707 - very different
    assert d2 == pytest.approx(1.458, abs=0.05)


def test_calculate_cohens_d_single_element_returns_nan() -> None:
    """Can't compute std from a single observation."""
    from src.analysis.statistical_analysis import _calculate_cohens_d

    assert np.isnan(_calculate_cohens_d(np.array([1.0]), np.array([2.0, 3.0])))


# --- MW-vs-Welch test selection ---


def test_build_group_result_uses_mann_whitney_when_small() -> None:
    """Groups below 10 should trigger Mann-Whitney as primary test."""
    from src.analysis.statistical_analysis import _build_group_result

    x = np.array([0.9, 0.85, 0.8])
    y = np.array([0.5, 0.4, 0.3])
    result = _build_group_result("test_metric", x, y)
    assert result["test_used"] == "mann_whitney_u"
    assert result["p_value"] == result["mw_pvalue"]


def test_build_group_result_uses_welch_when_both_large() -> None:
    """Groups of 10+ should use Welch's t-test as primary."""
    from src.analysis.statistical_analysis import _build_group_result

    rng = np.random.default_rng(42)
    x = rng.normal(0.8, 0.1, size=15)
    y = rng.normal(0.6, 0.1, size=15)
    result = _build_group_result("test_metric", x, y)
    assert result["test_used"] == "welch_t_test"
    assert result["p_value"] == result["t_pvalue"]


# --- Wilson CI ---


def test_wilson_interval_known_case() -> None:
    """11 successes out of 16 should produce a CI that excludes 0 and 1."""
    from src.analysis.statistical_analysis import _wilson_interval

    lower, upper = _wilson_interval(11, 16, confidence=0.95)
    assert lower == pytest.approx(0.44, abs=0.02)
    assert upper == pytest.approx(0.86, abs=0.02)
    assert 0 < lower < upper < 1


def test_wilson_interval_zero_n_returns_nan() -> None:
    """n=0 should return NaN, not division by zero."""
    from src.analysis.statistical_analysis import _wilson_interval

    lower, upper = _wilson_interval(0, 0)
    assert np.isnan(lower) and np.isnan(upper)


def test_wilson_interval_all_successes() -> None:
    """Perfect proportion should still produce an upper bound below 1 at small n."""
    from src.analysis.statistical_analysis import _wilson_interval

    lower, upper = _wilson_interval(5, 5)
    assert lower > 0.5
    assert upper <= 1.0


# --- Framework error analysis ---


def test_framework_error_analysis_classifies_correctly(minimal_action_labeled_df) -> None:
    """Error analysis should produce TP/TN/FP/FN cells that sum to n_labeled."""
    from src.analysis.statistical_analysis import (
        framework_error_analysis,
        validate_decision_framework,
    )

    validation = validate_decision_framework(minimal_action_labeled_df, "action_binary")
    errors = framework_error_analysis(validation)
    total = sum(errors["counts"].values())
    assert total == validation["n_labeled"]
    # Every feature name should appear exactly once across cells
    all_names = []
    for names in errors["features"].values():
        all_names.extend(names)
    assert len(all_names) == len(set(all_names)) == total
