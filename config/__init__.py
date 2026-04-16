"""Configuration helpers for feature context and classification."""

from .outcomes import (
    FEATURE_DECISION_CONTEXT,
    KNOWN_OUTCOMES,
    FEATURE_TYPES,
    get_feature_context,
    get_feature_type,
    get_all_contextualized_features,
    get_action_counts,
    get_business_outcome_counts,
)
from .thresholds import (
    DECAY_NOVELTY_THRESHOLD,
    DECAY_STICKY_THRESHOLD,
    HIGH_DECAY_ACTION_THRESHOLD,
    HIGH_ENGAGEMENT_MENTION_THRESHOLD,
    LOW_AWARENESS_MENTION_THRESHOLD,
    NEGATIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD,
    POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD,
)

__all__ = [
    "FEATURE_DECISION_CONTEXT",
    "KNOWN_OUTCOMES",
    "FEATURE_TYPES",
    "get_feature_context",
    "get_feature_type",
    "get_all_contextualized_features",
    "get_action_counts",
    "get_business_outcome_counts",
    "DECAY_STICKY_THRESHOLD",
    "DECAY_NOVELTY_THRESHOLD",
    "HIGH_DECAY_ACTION_THRESHOLD",
    "POSITIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD",
    "NEGATIVE_SIGNAL_NEGATIVE_RATIO_THRESHOLD",
    "HIGH_ENGAGEMENT_MENTION_THRESHOLD",
    "LOW_AWARENESS_MENTION_THRESHOLD",
]
