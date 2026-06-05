"""
Statistical tests for whether public signals line up with observable company action.

The key framing is narrower than business success:
- `company_action` is what the public record suggests the company did
- true business value is often unknown without internal retention or revenue data
"""

from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from config.thresholds import (
    DECAY_NOVELTY_THRESHOLD,
    DECAY_STICKY_THRESHOLD,
    HIGH_DECAY_ACTION_THRESHOLD,
    HIGH_ENGAGEMENT_MENTION_THRESHOLD,
    NEGATIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD,
    POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD,
)


@dataclass(frozen=True)
class FrameworkThresholds:
    """Parameterized thresholds for the public-signal decision framework."""

    decay_novelty: float = DECAY_NOVELTY_THRESHOLD
    decay_sticky: float = DECAY_STICKY_THRESHOLD
    negative_signal: float = NEGATIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD
    positive_signal: float = POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD
    high_mentions: float = HIGH_ENGAGEMENT_MENTION_THRESHOLD

NUM_HYPOTHESIS_TESTS = 3
DEFAULT_ALPHA = 0.05
PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_COLUMNS = {
    "labeled_features": {"feature_name", "search_decay"},
    "trends_data": {"feature_name", "interest", "date", "launch_date"},
    "metrics": {"feature_name", "decay_rate_w4"},
}


def _resolve_input_path(path: PathLike) -> Path:
    """Resolve inputs from the repo root when the current shell is elsewhere."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate


def _resolve_output_path(path: PathLike) -> Path:
    """Resolve repo outputs relative to the project root by default."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _validate_schema(df: pd.DataFrame, schema_name: str, path: str) -> None:
    """Raise a clear error when a pipeline input is missing required columns."""
    required = _REQUIRED_COLUMNS.get(schema_name, set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema error in '{path}': missing columns {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )


def _extract_labeled_sample(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Return the labeled sample and the label column used for analysis."""
    if "action_binary" in df.columns:
        label_col = "action_binary"
        analysis_df = df[df["action_binary"].notna()].copy()
    elif "is_success" in df.columns:
        analysis_mask = df["is_success"].eq(1)
        if "is_failure" in df.columns:
            analysis_mask = analysis_mask | df["is_failure"].eq(1)
        elif "outcome_label" in df.columns:
            analysis_mask = analysis_mask | df["outcome_label"].eq("failure")
        label_col = "is_success"
        analysis_df = df[analysis_mask].copy()
    elif "success_binary" in df.columns:
        label_col = "success_binary"
        analysis_df = df[df["success_binary"].notna()].copy()
    else:
        raise ValueError(
            "No analysis label column found. Expected 'action_binary', "
            f"'is_success', or 'success_binary' in {list(df.columns)}"
        )

    return analysis_df, label_col


def load_labeled_data(
    path: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labeled data and split it into supported and pulled-back subsets."""
    csv_path = _resolve_input_path(path or Path("data/validation/labeled_features.csv"))
    df = pd.read_csv(csv_path)
    _validate_schema(df, "labeled_features", str(csv_path))

    analysis_df, label_col = _extract_labeled_sample(df)
    supported = analysis_df[analysis_df[label_col] == 1].copy()
    pulled_back = analysis_df[analysis_df[label_col] == 0].copy()
    return supported, pulled_back, df


def _clean_values(values: Iterable[float]) -> np.ndarray:
    """Convert a numeric iterable into a float array with NaNs removed."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.asarray([float(array)])
    return array[~np.isnan(array)]


def calculate_decay_metrics_from_series(values: Iterable[float]) -> Dict[str, float]:
    """Summarize a numeric series used in the comparison tables."""
    cleaned = _clean_values(values)
    if cleaned.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(cleaned.size),
        "mean": float(np.nanmean(cleaned)),
        "std": float(np.nanstd(cleaned, ddof=1)) if cleaned.size > 1 else np.nan,
        "median": float(np.nanmedian(cleaned)),
        "min": float(np.nanmin(cleaned)),
        "max": float(np.nanmax(cleaned)),
    }


def _ttest_groups(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Run Welch's t-test and return only the statistic and p-value."""
    comparison = _compare_groups(x, y)
    return comparison["t_stat"], comparison["t_pvalue"]


def _compare_groups(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compare two groups with Welch's t-test and Mann-Whitney U.

    When either group is very small, Mann-Whitney is treated as the primary test
    because the t-test's normality assumption is hard to defend.
    """
    x_clean = _clean_values(x)
    y_clean = _clean_values(y)
    if x_clean.size == 0 or y_clean.size == 0:
        return {
            "t_stat": np.nan,
            "t_pvalue": np.nan,
            "mw_stat": np.nan,
            "mw_pvalue": np.nan,
        }

    t_stat, t_pvalue = stats.ttest_ind(x_clean, y_clean, equal_var=False)
    mw_stat, mw_pvalue = stats.mannwhitneyu(x_clean, y_clean, alternative="two-sided")
    return {
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "mw_stat": float(mw_stat),
        "mw_pvalue": float(mw_pvalue),
    }


def _calculate_cohens_d(supported_values: np.ndarray, pulled_back_values: np.ndarray) -> float:
    """Cohen's d using df-weighted pooled standard deviation.

    Pooled variance follows Cohen (1988):
        s_p^2 = ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2)

    Returns NaN when either group has fewer than 2 observations.
    """
    x = _clean_values(supported_values)
    y = _clean_values(pulled_back_values)
    if x.size < 2 or y.size < 2:
        return np.nan

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    pooled_var = ((x.size - 1) * var_x + (y.size - 1) * var_y) / (x.size + y.size - 2)
    if pooled_var <= 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var))


def _build_group_result(
    metric: str,
    supported_values: np.ndarray,
    pulled_back_values: np.ndarray,
) -> Dict[str, object]:
    """Build a consistent result dictionary for each two-group comparison."""
    supported_summary = calculate_decay_metrics_from_series(supported_values)
    pulled_back_summary = calculate_decay_metrics_from_series(pulled_back_values)
    comparison = _compare_groups(supported_values, pulled_back_values)
    cohens_d = _calculate_cohens_d(supported_values, pulled_back_values)

    use_mann_whitney = min(supported_summary["count"], pulled_back_summary["count"]) < 10
    primary_stat = comparison["mw_stat"] if use_mann_whitney else comparison["t_stat"]
    primary_pvalue = comparison["mw_pvalue"] if use_mann_whitney else comparison["t_pvalue"]
    significant = bool(primary_pvalue < DEFAULT_ALPHA) if not np.isnan(primary_pvalue) else False

    return {
        "metric": metric,
        "supported_mean": supported_summary["mean"],
        "supported_std": supported_summary["std"],
        "supported_count": supported_summary["count"],
        "pulled_back_mean": pulled_back_summary["mean"],
        "pulled_back_std": pulled_back_summary["std"],
        "pulled_back_count": pulled_back_summary["count"],
        "difference_in_means": supported_summary["mean"] - pulled_back_summary["mean"],
        "t_statistic": comparison["t_stat"],
        "t_pvalue": comparison["t_pvalue"],
        "mw_statistic": comparison["mw_stat"],
        "mw_pvalue": comparison["mw_pvalue"],
        "test_used": "mann_whitney_u" if use_mann_whitney else "welch_t_test",
        "test_used_note": (
            "Small-sample setting: Mann-Whitney U is the primary p-value."
            if use_mann_whitney
            else "Welch's t-test is the primary p-value."
        ),
        "primary_statistic": primary_stat,
        "p_value": primary_pvalue,
        "cohens_d": cohens_d,
        "effect_size_label": interpret_effect_size(abs(cohens_d)) if not np.isnan(cohens_d) else "undefined",
        "significant": significant,
        "significant_bonferroni": False,
        "conclusion": "DOES" if significant else "DOES NOT",
    }


def test_decay_difference(supported: pd.DataFrame, pulled_back: pd.DataFrame) -> Dict[str, object]:
    """Test whether supported features have different search decay than pulled-back ones."""
    return _build_group_result(
        metric="search_decay",
        supported_values=supported["search_decay"].values,
        pulled_back_values=pulled_back["search_decay"].values,
    )


def test_mentions_difference(supported: pd.DataFrame, pulled_back: pd.DataFrame) -> Dict[str, object]:
    """Test whether supported features have different Reddit engagement than pulled-back ones."""
    return _build_group_result(
        metric="total_mentions",
        supported_values=supported["total_mentions"].values,
        pulled_back_values=pulled_back["total_mentions"].values,
    )


def test_sentiment_difference(supported: pd.DataFrame, pulled_back: pd.DataFrame) -> Dict[str, object]:
    """Test whether supported features have different negative sentiment than pulled-back ones."""
    return _build_group_result(
        metric="negative_ratio",
        supported_values=supported["negative_ratio"].values,
        pulled_back_values=pulled_back["negative_ratio"].values,
    )


def calculate_correlations(analysis_df: pd.DataFrame, label_col: str) -> pd.Series:
    """Calculate Spearman correlation between each signal and the action label."""
    candidate_cols = [
        "search_decay",
        "total_mentions",
        "negative_ratio",
        "positive_ratio",
        "neutral_ratio",
        "avg_score",
    ]
    feature_cols = [column for column in candidate_cols if column in analysis_df.columns]
    if not feature_cols:
        return pd.Series(dtype=float)

    if analysis_df.empty:
        return pd.Series(dtype=float)

    correlation_matrix = analysis_df[feature_cols + [label_col]].corr(method="spearman")
    correlations = correlation_matrix[label_col].drop(label_col)
    return correlations.sort_values(key=lambda values: values.abs(), ascending=False)


def find_high_decay_supported_features(
    supported: pd.DataFrame,
    threshold: float = HIGH_DECAY_ACTION_THRESHOLD,
) -> pd.DataFrame:
    """Return supported features whose search decay is above the chosen threshold."""
    if "search_decay" not in supported.columns:
        return supported.iloc[0:0].copy()
    return supported[supported["search_decay"] > threshold].copy()


def interpret_effect_size(cohens_d: float) -> str:
    """Map Cohen's d to a plain-English effect size band."""
    if np.isnan(cohens_d):
        return "undefined"
    if abs(cohens_d) < 0.2:
        return "negligible"
    if abs(cohens_d) < 0.5:
        return "small"
    if abs(cohens_d) < 0.8:
        return "medium"
    return "large"


def interpret_correlation(corr: float) -> str:
    """Map an absolute correlation to a rough strength label."""
    if abs(corr) > 0.5:
        return "strong"
    if abs(corr) > 0.3:
        return "moderate"
    return "weak"


def _wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation at small n because it
    does not require np > 5 to behave well.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_group_difference(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Estimate a bootstrap confidence interval for the difference in group means."""
    x_clean = _clean_values(x)
    y_clean = _clean_values(y)
    if x_clean.size == 0 or y_clean.size == 0:
        return {
            "observed_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "se": np.nan,
        }

    rng = np.random.default_rng(42)
    observed_diff = float(np.nanmean(x_clean) - np.nanmean(y_clean))
    diffs = np.array(
        [
            np.nanmean(rng.choice(x_clean, size=len(x_clean), replace=True))
            - np.nanmean(rng.choice(y_clean, size=len(y_clean), replace=True))
            for _ in range(n_bootstrap)
        ]
    )

    alpha = 1 - ci
    ci_lower = float(np.nanpercentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.nanstd(diffs, ddof=1)),
    }


def power_analysis(n_supported: int, n_pulled_back: int, alpha: float = DEFAULT_ALPHA) -> Dict[str, object]:
    """Estimate the minimum detectable effect size for an 80% powered two-sample test."""
    if n_supported <= 0 or n_pulled_back <= 0:
        return {
            "n_supported": n_supported,
            "n_pulled_back": n_pulled_back,
            "alpha": alpha,
            "power_target": 0.80,
            "min_detectable_cohens_d": np.nan,
            "interpretation": "undefined",
            "note": "Power analysis is undefined without observations in both groups.",
        }

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(0.80)
    min_detectable_d = float((z_alpha + z_beta) * np.sqrt(1 / n_supported + 1 / n_pulled_back))
    interpretation = interpret_effect_size(min_detectable_d)

    return {
        "n_supported": n_supported,
        "n_pulled_back": n_pulled_back,
        "alpha": alpha,
        "power_target": 0.80,
        "min_detectable_cohens_d": min_detectable_d,
        "interpretation": interpretation,
        "note": (
            f"With n={n_supported} supported features and n={n_pulled_back} pulled-back "
            f"features, this study can only detect effects of d>={min_detectable_d:.2f} "
            f"({interpretation}) with 80% power at alpha={alpha:.2f}."
        ),
    }


def threshold_sensitivity_analysis(
    supported: pd.DataFrame,
    thresholds: Optional[list] = None,
) -> pd.DataFrame:
    """Show how the high-decay supported share changes across reasonable thresholds."""
    if thresholds is None:
        thresholds = [round(threshold, 2) for threshold in np.arange(0.50, 0.96, 0.05)]

    if "search_decay" not in supported.columns:
        return pd.DataFrame(columns=["threshold", "n_above", "pct_above"])

    n_total = len(supported)
    rows = []
    for threshold in thresholds:
        n_above = int((supported["search_decay"] > threshold).sum())
        pct_above = float(n_above / n_total) if n_total else 0.0
        rows.append({"threshold": threshold, "n_above": n_above, "pct_above": pct_above})
    return pd.DataFrame(rows)


def compare_by_feature_type(analysis_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Summarize whether feature type tracks supportive company action better than decay alone."""
    if analysis_df.empty:
        return pd.DataFrame(columns=["feature_type", "count", "supported_rate", "mean_decay", "vs_baseline"])

    if "feature_type_calc" in analysis_df.columns and analysis_df["feature_type_calc"].notna().any():
        type_col = "feature_type_calc"
    elif "feature_type" in analysis_df.columns:
        type_col = "feature_type"
    else:
        return pd.DataFrame(columns=["feature_type", "count", "supported_rate", "mean_decay", "vs_baseline"])

    overall_rate = float(analysis_df[label_col].mean())
    summary = (
        analysis_df.groupby(type_col)
        .agg(
            count=(label_col, "count"),
            supported_rate=(label_col, "mean"),
            mean_decay=("search_decay", "mean"),
        )
        .reset_index()
        .rename(columns={type_col: "feature_type"})
    )
    summary["vs_baseline"] = summary["supported_rate"] - overall_rate
    summary = summary[summary["count"] >= 2].sort_values("supported_rate", ascending=False)
    return summary


def summarize_context_coverage(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Count how much public context exists for company action and business outcome."""
    summary = {}
    if "company_action" in df.columns:
        summary["company_action"] = df["company_action"].value_counts(dropna=False).to_dict()
    if "business_outcome" in df.columns:
        summary["business_outcome"] = df["business_outcome"].value_counts(dropna=False).to_dict()
    return summary


def _majority_class_label(analysis_df: pd.DataFrame, label_col: str) -> int:
    """Return the majority class label for a labeled analysis sample."""
    if analysis_df.empty:
        return 1
    return int(analysis_df[label_col].mean() >= 0.5)


def _decision_framework_prediction(
    row: pd.Series,
    majority_class: int,
    thresholds: Optional[FrameworkThresholds] = None,
) -> int:
    """Apply the repo's public-signal framework to one labeled row."""
    if thresholds is None:
        thresholds = FrameworkThresholds()

    search_decay = row.get("search_decay", np.nan)
    negative_ratio = row.get("negative_ratio", np.nan)
    total_mentions = row.get("total_mentions", np.nan)

    if pd.isna(search_decay) or pd.isna(negative_ratio) or pd.isna(total_mentions):
        return majority_class

    high_decay = search_decay > thresholds.decay_novelty
    low_decay = search_decay < thresholds.decay_sticky
    positive_signal = negative_ratio < thresholds.positive_signal
    negative_signal = negative_ratio > thresholds.negative_signal
    high_mentions = total_mentions > thresholds.high_mentions

    if high_decay and positive_signal and high_mentions:
        return 1
    if high_decay and negative_signal:
        return 0
    if low_decay and positive_signal:
        return 1
    return majority_class


def validate_decision_framework(analysis_df: pd.DataFrame, label_col: str) -> Dict[str, object]:
    """
    Validate the decision framework against the labeled action sample.

    This does not claim a production-ready classifier. It tests whether the
    public-signal rules beat a majority-class baseline at all.
    """
    if analysis_df.empty:
        return {
            "n_labeled": 0,
            "baseline_accuracy": np.nan,
            "framework_accuracy": np.nan,
            "lift_over_baseline": np.nan,
            "majority_class": "unknown",
            "predictions": [],
        }

    labeled = analysis_df.copy()
    majority_class = _majority_class_label(labeled, label_col)
    baseline_accuracy = float((labeled[label_col] == majority_class).mean())
    labeled["framework_pred"] = labeled.apply(
        lambda row: _decision_framework_prediction(row, majority_class),
        axis=1,
    )
    framework_accuracy = float((labeled["framework_pred"] == labeled[label_col]).mean())

    return {
        "n_labeled": len(labeled),
        "baseline_accuracy": baseline_accuracy,
        "framework_accuracy": framework_accuracy,
        "lift_over_baseline": framework_accuracy - baseline_accuracy,
        "majority_class": "supported" if majority_class == 1 else "pulled_back",
        "predictions": labeled[["feature_name", label_col, "framework_pred"]].to_dict(orient="records"),
    }


def framework_error_analysis(validation_result: Dict[str, object]) -> Dict[str, object]:
    """Break down where the decision framework disagrees with the labels.

    Returns feature names in each confusion-matrix cell so a reader can
    inspect the misses by name rather than just seeing an accuracy number.
    """
    preds = validation_result.get("predictions", [])
    cells: Dict[str, list] = {"TP": [], "TN": [], "FP": [], "FN": []}
    for row in preds:
        label = None
        for col in ("action_binary", "is_success", "success_binary"):
            if col in row:
                label = row[col]
                break
        pred = row.get("framework_pred")
        if label is None or pred is None:
            continue
        if label == 1 and pred == 1:
            cells["TP"].append(row["feature_name"])
        elif label == 0 and pred == 0:
            cells["TN"].append(row["feature_name"])
        elif label == 0 and pred == 1:
            cells["FP"].append(row["feature_name"])
        else:
            cells["FN"].append(row["feature_name"])
    return {
        "counts": {k: len(v) for k, v in cells.items()},
        "features": cells,
    }


def ablation_signal_combinations(analysis_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Compare baseline, single-signal rules, and the combined framework."""
    if analysis_df.empty:
        return pd.DataFrame(columns=["signal_combination", "accuracy", "n"])

    labeled = analysis_df.copy()
    majority_class = _majority_class_label(labeled, label_col)

    def predict_decay_only(row: pd.Series) -> int:
        search_decay = row.get("search_decay", np.nan)
        if pd.isna(search_decay):
            return majority_class
        if search_decay < DECAY_STICKY_THRESHOLD:
            return 1
        if search_decay > HIGH_DECAY_ACTION_THRESHOLD:
            return 0
        return majority_class

    def predict_sentiment_only(row: pd.Series) -> int:
        negative_ratio = row.get("negative_ratio", np.nan)
        if pd.isna(negative_ratio):
            return majority_class
        return 1 if negative_ratio < POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD else 0

    def predict_mentions_only(row: pd.Series) -> int:
        total_mentions = row.get("total_mentions", np.nan)
        if pd.isna(total_mentions):
            return majority_class
        return 1 if total_mentions > HIGH_ENGAGEMENT_MENTION_THRESHOLD else majority_class

    combinations = {
        "baseline (majority class)": lambda row: majority_class,
        "decay only": predict_decay_only,
        "sentiment only": predict_sentiment_only,
        "mentions only": predict_mentions_only,
        "combined (decision framework)": lambda row: _decision_framework_prediction(row, majority_class),
    }

    rows = []
    for name, predictor in combinations.items():
        predictions = labeled.apply(predictor, axis=1)
        rows.append(
            {
                "signal_combination": name,
                "accuracy": float((predictions == labeled[label_col]).mean()),
                "n": len(labeled),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def framework_threshold_sensitivity(
    analysis_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Sweep each framework threshold and show how accuracy responds.

    Varies one threshold at a time while holding the others at their
    defaults. Helps surface whether the framework result depends on
    exactly where one line is drawn.
    """
    if analysis_df.empty:
        return pd.DataFrame(columns=["threshold_name", "value", "accuracy"])

    labeled = analysis_df.copy()
    majority_class = _majority_class_label(labeled, label_col)
    defaults = FrameworkThresholds()

    sweeps: Dict[str, list] = {
        "decay_novelty": [round(v, 2) for v in np.arange(0.50, 0.91, 0.05)],
        "negative_signal": [round(v, 2) for v in np.arange(0.15, 0.46, 0.05)],
        "positive_signal": [round(v, 2) for v in np.arange(0.10, 0.36, 0.05)],
        "high_mentions": [float(v) for v in range(5, 41, 5)],
    }

    rows = []
    for threshold_name, values in sweeps.items():
        for value in values:
            variant = replace(defaults, **{threshold_name: value})
            preds = labeled.apply(
                lambda row, t=variant: _decision_framework_prediction(row, majority_class, t),
                axis=1,
            )
            rows.append(
                {
                    "threshold_name": threshold_name,
                    "value": value,
                    "accuracy": float((preds == labeled[label_col]).mean()),
                }
            )
    return pd.DataFrame(rows)


def run_all_tests(labeled_path: Optional[PathLike] = None) -> Dict[str, object]:
    """Run the full analysis, including robustness checks and observability caveats."""
    supported, pulled_back, df = load_labeled_data(labeled_path)
    analysis_df, label_col = _extract_labeled_sample(df)

    decay_test = test_decay_difference(supported, pulled_back)
    mentions_test = test_mentions_difference(supported, pulled_back)
    sentiment_test = test_sentiment_difference(supported, pulled_back)
    correlations = calculate_correlations(analysis_df, label_col)
    high_decay_supported = find_high_decay_supported_features(supported)
    sensitivity = threshold_sensitivity_analysis(supported)
    type_comparison = compare_by_feature_type(analysis_df, label_col)
    decay_bootstrap = bootstrap_group_difference(
        supported["search_decay"].dropna().values,
        pulled_back["search_decay"].dropna().values,
    )
    study_power = power_analysis(len(supported), len(pulled_back))
    framework_validation = validate_decision_framework(analysis_df, label_col)
    error_analysis = framework_error_analysis(framework_validation)
    signal_ablation = ablation_signal_combinations(analysis_df, label_col)
    fw_threshold_sensitivity = framework_threshold_sensitivity(analysis_df, label_col)

    bonferroni_threshold = DEFAULT_ALPHA / NUM_HYPOTHESIS_TESTS
    for test_result in (decay_test, mentions_test, sentiment_test):
        p_value = test_result["p_value"]
        test_result["significant_bonferroni"] = bool(p_value < bonferroni_threshold) if not np.isnan(p_value) else False

    return {
        "sample_sizes": {
            "supported": len(supported),
            "pulled_back": len(pulled_back),
            "analysis_sample": len(analysis_df),
            "dataset_total": len(df),
        },
        "context_coverage": summarize_context_coverage(df),
        "observability_note": (
            "Company action is observable in public sources. True business value often is not."
        ),
        "bonferroni_threshold": bonferroni_threshold,
        "decay_test": decay_test,
        "mentions_test": mentions_test,
        "sentiment_test": sentiment_test,
        "correlations": correlations.to_dict(),
        "high_decay_supported": {
            "threshold": HIGH_DECAY_ACTION_THRESHOLD,
            "count": len(high_decay_supported),
            "pct_of_supported": len(high_decay_supported) / len(supported) if len(supported) > 0 else 0.0,
            "pct_ci_lower": _wilson_interval(len(high_decay_supported), len(supported))[0],
            "pct_ci_upper": _wilson_interval(len(high_decay_supported), len(supported))[1],
            "features": high_decay_supported["feature_name"].tolist() if len(high_decay_supported) else [],
        },
        "decay_bootstrap_ci": decay_bootstrap,
        "power_analysis": study_power,
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "feature_type_comparison": type_comparison.to_dict(orient="records"),
        "framework_validation": framework_validation,
        "framework_error_analysis": error_analysis,
        "signal_ablation": signal_ablation.to_dict(orient="records"),
        "framework_threshold_sensitivity": fw_threshold_sensitivity.to_dict(orient="records"),
    }


def _print_test_summary(title: str, result: Dict[str, object], as_percent: bool = False) -> None:
    """Print one comparison block in a consistent, readable format."""
    if np.isnan(result["supported_mean"]) or np.isnan(result["pulled_back_mean"]):
        print(f"{title}")
        print("  Not enough data to compare the two groups.\n")
        return

    if as_percent:
        supported_text = f"{result['supported_mean']:.1%}"
        pulled_back_text = f"{result['pulled_back_mean']:.1%}"
        diff_text = f"{result['difference_in_means']:.1%}"
        supported_std_text = f"{result['supported_std']:.1%}" if not np.isnan(result["supported_std"]) else "n/a"
        pulled_back_std_text = f"{result['pulled_back_std']:.1%}" if not np.isnan(result["pulled_back_std"]) else "n/a"
    else:
        supported_text = f"{result['supported_mean']:.1f}"
        pulled_back_text = f"{result['pulled_back_mean']:.1f}"
        diff_text = f"{result['difference_in_means']:.1f}"
        supported_std_text = f"{result['supported_std']:.1f}" if not np.isnan(result["supported_std"]) else "n/a"
        pulled_back_std_text = f"{result['pulled_back_std']:.1f}" if not np.isnan(result["pulled_back_std"]) else "n/a"

    print(title)
    print(
        f"  Supported:   {supported_text} (+/- {supported_std_text}, n={result['supported_count']})"
    )
    print(
        f"  Pulled back: {pulled_back_text} (+/- {pulled_back_std_text}, n={result['pulled_back_count']})"
    )
    print(f"  Difference in means: {diff_text}")
    print(f"  Primary test: {result['test_used']} ({result['test_used_note']})")
    print(
        f"  Mann-Whitney U: statistic={result['mw_statistic']:.3f}, "
        f"p-value={result['mw_pvalue']:.4f}"
    )
    print(
        f"  Welch's t-test: statistic={result['t_statistic']:.3f}, "
        f"p-value={result['t_pvalue']:.4f}"
    )
    print(f"  Primary p-value used downstream: {result['p_value']:.4f}")

    if result["significant"]:
        print(f"  Nominal significance: yes (alpha={DEFAULT_ALPHA:.2f})")
    else:
        print(f"  Nominal significance: no (alpha={DEFAULT_ALPHA:.2f})")

    if result["significant_bonferroni"]:
        print("  Survives Bonferroni correction.")
    else:
        print("  Does not survive Bonferroni correction.")

    if "cohens_d" in result:
        print(
            f"  Cohen's d: {result['cohens_d']:.3f} "
            f"({result['effect_size_label']})"
        )
    print()


def print_results(results: Dict[str, object]) -> None:
    """Print the full analysis in a reviewer-friendly format."""
    sample_sizes = results["sample_sizes"]
    print("\nSTATISTICAL ANALYSIS: Public Signals vs Company Action")
    print(
        f"Action-labeled sample: {sample_sizes['supported']} supported, "
        f"{sample_sizes['pulled_back']} pulled back, {sample_sizes['analysis_sample']} "
        f"labeled features out of {sample_sizes['dataset_total']} total"
    )

    coverage = results.get("context_coverage", {})
    if coverage.get("business_outcome"):
        outcome_counts = coverage["business_outcome"]
        print(
            f"Business outcome coverage: {outcome_counts.get('POSITIVE', 0)} positive, "
            f"{outcome_counts.get('NEGATIVE', 0)} negative, "
            f"{outcome_counts.get('UNKNOWN', 0)} unknown"
        )
    print(f"{results['observability_note']}")
    print(
        f"Bonferroni threshold for {NUM_HYPOTHESIS_TESTS} tests: "
        f"p < {results['bonferroni_threshold']:.3f}\n"
    )

    _print_test_summary("TEST 1: Search Decay - Supported vs Pulled Back", results["decay_test"], as_percent=True)
    _print_test_summary("TEST 2: Reddit Mentions - Supported vs Pulled Back", results["mentions_test"], as_percent=False)
    _print_test_summary("TEST 3: Negative Sentiment - Supported vs Pulled Back", results["sentiment_test"], as_percent=True)

    power = results.get("power_analysis", {})
    ci = results.get("decay_bootstrap_ci", {})
    if power:
        print("POWER ANALYSIS")
        print(f"  {power['note']}")
        if ci and not np.isnan(ci["ci_lower"]):
            print(
                f"  95% bootstrap CI on decay difference: "
                f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
            )
            print(f"  Observed difference in means: {ci['observed_diff']:.3f}")
        print()

    high_decay = results["high_decay_supported"]
    ci_lower = high_decay.get("pct_ci_lower", float("nan"))
    ci_upper = high_decay.get("pct_ci_upper", float("nan"))
    ci_text = f" (95% CI: {ci_lower:.0%}-{ci_upper:.0%})" if not np.isnan(ci_lower) else ""
    print(
        f"KEY FINDING: {high_decay['count']} supported features above "
        f"{high_decay['threshold']:.0%} decay ({high_decay['pct_of_supported']:.0%}{ci_text})"
    )
    if high_decay["features"]:
        print("  Features:")
        for feature in high_decay["features"]:
            print(f"  - {feature}")
    print()

    sensitivity_rows = results.get("threshold_sensitivity", [])
    if sensitivity_rows:
        print("SENSITIVITY CHECK: Share of supported features above the decay threshold")
        for row in sensitivity_rows:
            print(f"  >{row['threshold']:.0%}: {row['n_above']} features ({row['pct_above']:.0%})")
        print()

    feature_type_rows = results.get("feature_type_comparison", [])
    if feature_type_rows:
        print("FEATURE TYPE COMPARISON")
        for row in feature_type_rows:
            print(
                f"  {row['feature_type']}: n={int(row['count'])}, "
                f"supported rate={row['supported_rate']:.0%}, "
                f"mean decay={row['mean_decay']:.0%}, "
                f"vs baseline={row['vs_baseline']:+.0%}"
            )
        print()

    framework_validation = results.get("framework_validation", {})
    if framework_validation:
        print(f"FRAMEWORK VALIDATION (n={framework_validation['n_labeled']})")
        print(
            f"  Majority-class baseline ({framework_validation['majority_class']}): "
            f"{framework_validation['baseline_accuracy']:.0%}"
        )
        print(f"  Decision framework accuracy: {framework_validation['framework_accuracy']:.0%}")
        lift = framework_validation["lift_over_baseline"]
        direction = "+" if lift >= 0 else ""
        print(f"  Lift over baseline: {direction}{lift:.0%}")
        print("  Treat as indicative only - the labeled action sample is still small.")
        print()

    error_analysis = results.get("framework_error_analysis", {})
    if error_analysis and error_analysis.get("counts"):
        counts = error_analysis["counts"]
        features = error_analysis["features"]
        print("FRAMEWORK ERROR ANALYSIS")
        print(f"  TP (correctly predicted supported): {counts['TP']}")
        print(f"  TN (correctly predicted pulled back): {counts['TN']}")
        print(f"  FP (predicted supported, actually pulled back): {counts['FP']}")
        if features["FP"]:
            for name in features["FP"]:
                print(f"    - {name}")
        print(f"  FN (predicted pulled back, actually supported): {counts['FN']}")
        if features["FN"]:
            for name in features["FN"]:
                print(f"    - {name}")
        print()

    signal_ablation = results.get("signal_ablation", [])
    if signal_ablation:
        print("SIGNAL ABLATION")
        for row in signal_ablation:
            print(
                f"  {row['signal_combination']}: "
                f"{row['accuracy']:.0%} accuracy (n={row['n']})"
            )
        print()

    fw_sensitivity = results.get("framework_threshold_sensitivity", [])
    if fw_sensitivity:
        print("FRAMEWORK THRESHOLD SENSITIVITY")
        current_name = None
        for row in fw_sensitivity:
            if row["threshold_name"] != current_name:
                current_name = row["threshold_name"]
                print(f"  {current_name}:")
            marker = " <-- default" if (
                (current_name == "decay_novelty" and row["value"] == DECAY_NOVELTY_THRESHOLD)
                or (current_name == "negative_signal" and row["value"] == NEGATIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD)
                or (current_name == "positive_signal" and row["value"] == POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD)
                or (current_name == "high_mentions" and row["value"] == HIGH_ENGAGEMENT_MENTION_THRESHOLD)
            ) else ""
            print(f"    {row['value']:>6.2f} → {row['accuracy']:.0%}{marker}")
        print()

    correlations = results.get("correlations", {})
    if correlations:
        print("SPEARMAN CORRELATION WITH SUPPORTIVE COMPANY ACTION")
        for feature, corr in correlations.items():
            direction = "up" if corr > 0 else "down"
            strength = interpret_correlation(abs(corr))
            print(f"  {feature:20s}: {corr:+.3f} ({strength}, {direction})")


def save_results(
    results: Dict[str, object],
    output_path: Optional[PathLike] = None,
) -> None:
    """Save the three primary comparison tests to a compact CSV summary."""
    rows = []
    for key in ("decay_test", "mentions_test", "sentiment_test"):
        test = results[key]
        rows.append(
            {
                "metric": test["metric"],
                "primary_test": test["test_used"],
                "supported_mean": test["supported_mean"],
                "pulled_back_mean": test["pulled_back_mean"],
                "difference_in_means": test["difference_in_means"],
                "p_value": test["p_value"],
                "t_pvalue": test["t_pvalue"],
                "mw_pvalue": test["mw_pvalue"],
                "cohens_d": test["cohens_d"],
                "effect_size_label": test["effect_size_label"],
                "significant": test["significant"],
                "significant_bonferroni": test["significant_bonferroni"],
            }
        )

    output_file = _resolve_output_path(output_path or Path("data/validation/statistical_results.csv"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")


def main() -> None:
    """Run the full statistical analysis and save the compact results CSV."""
    analysis_results = run_all_tests()
    print_results(analysis_results)
    save_results(analysis_results)


if __name__ == "__main__":
    main()
