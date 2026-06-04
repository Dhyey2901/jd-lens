"""Unit tests for scorer.py — pure logic tests, no model calls."""
import pytest
from scorer import _tfidf_similarity, _tokenize


class TestTokenize:
    def test_lowercases(self):
        tokens = _tokenize("Python AWS Docker")
        assert "python" in tokens
        assert "aws" in tokens

    def test_filters_short(self):
        tokens = _tokenize("I am a go developer")
        # "i", "am", "a" should be excluded (len <= 2)
        assert "i" not in tokens
        assert "am" not in tokens

    def test_filters_pure_digits(self):
        tokens = _tokenize("2024 experience required")
        assert "2024" not in tokens

    def test_handles_special_chars(self):
        tokens = _tokenize("c++ and c# are languages")
        assert "c++" in tokens
        assert "c#" in tokens


class TestTfidfSimilarity:
    def test_identical_texts_score_one(self):
        text = "Python developer with AWS and Docker experience"
        score = _tfidf_similarity(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_unrelated_texts_score_low(self):
        jd = "Senior Python engineer with machine learning and AWS"
        candidate = "Marketing manager with Excel and PowerPoint skills"
        score = _tfidf_similarity(jd, candidate)
        assert score < 0.3

    def test_similar_texts_score_high(self):
        jd = "Python developer experienced in Flask and PostgreSQL"
        candidate = "5 years of Python, built REST APIs with Flask, PostgreSQL databases"
        score = _tfidf_similarity(jd, candidate)
        assert score > 0.3

    def test_returns_float_in_range(self):
        score = _tfidf_similarity("some text", "other text")
        assert 0.0 <= score <= 1.0
