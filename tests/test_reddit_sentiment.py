"""Tests for the Reddit sentiment keyword matcher.

The sentiment method is a crude lexicon baseline (see README).
These tests verify edge cases and make sure any future swap
(e.g. to VADER) has a regression signal.
"""

import pytest

from src.data_collection.reddit.reddit_validator import RedditValidator


def _make_validator() -> RedditValidator:
    """Create a RedditValidator without initializing the Reddit client."""
    v = RedditValidator.__new__(RedditValidator)
    return v


class TestAnalyzeSentiment:
    """Tests for RedditValidator.analyze_sentiment."""

    def test_empty_returns_no_data(self) -> None:
        result = _make_validator().analyze_sentiment([])
        assert result["sentiment_label"] == "no_data"
        assert result["total_mentions"] == 0
        assert result["positive_ratio"] == 0.0
        assert result["negative_ratio"] == 0.0

    def test_positive_keywords_dominate(self) -> None:
        mentions = [
            {"title": "I love this feature, it's amazing", "text": "", "score": 10},
            {"title": "perfect update, really helpful", "text": "", "score": 5},
            {"title": "best thing they've done", "text": "", "score": 3},
        ]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["sentiment_label"] == "positive"
        assert result["positive_ratio"] == 1.0
        assert result["negative_ratio"] == 0.0

    def test_negative_keywords_dominate(self) -> None:
        mentions = [
            {"title": "terrible, useless, waste of money", "text": "", "score": 0},
            {"title": "broken and annoying, hate it", "text": "", "score": 1},
            {"title": "worst update, frustrated", "text": "", "score": 2},
        ]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["sentiment_label"] == "negative"
        assert result["negative_ratio"] == 1.0

    def test_equal_positive_negative_counts_as_neutral(self) -> None:
        """When positive and negative keyword counts tie, the mention is neutral."""
        mentions = [{"title": "love but also hate", "text": "", "score": 0}]
        result = _make_validator().analyze_sentiment(mentions)
        # One positive (love), one negative (hate) → tie → neutral
        assert result["neutral_ratio"] == 1.0
        assert result["sentiment_label"] == "mixed"

    def test_no_keywords_counts_as_neutral(self) -> None:
        """A mention with no sentiment keywords should be neutral."""
        mentions = [{"title": "just updated the app today", "text": "", "score": 5}]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["neutral_ratio"] == 1.0

    def test_text_field_is_included_in_matching(self) -> None:
        """Sentiment should scan both title and text fields."""
        mentions = [{"title": "check this out", "text": "amazing feature, love it", "score": 3}]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["positive_ratio"] == 1.0

    def test_avg_score_computed_correctly(self) -> None:
        mentions = [
            {"title": "great", "text": "", "score": 10},
            {"title": "great", "text": "", "score": 20},
        ]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["avg_score"] == pytest.approx(15.0)

    def test_mixed_sentiment_across_mentions(self) -> None:
        """Mix of positive, negative, and neutral mentions."""
        mentions = [
            {"title": "love this", "text": "", "score": 5},
            {"title": "terrible update", "text": "", "score": 1},
            {"title": "ok I guess", "text": "", "score": 3},
        ]
        result = _make_validator().analyze_sentiment(mentions)
        assert result["total_mentions"] == 3
        assert result["positive_count"] == 1
        assert result["negative_count"] == 1
        assert result["neutral_count"] == 1
        assert result["sentiment_label"] == "mixed"
