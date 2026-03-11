"""
test_grader.py — Unit tests for the rule-based grading engine.

These tests use only the Grader class and do not require CLIP or an LLM.
They verify that grading rules are applied correctly and consistently.
"""

from __future__ import annotations

import pytest

from app.grader import Grader


@pytest.fixture
def grader() -> Grader:
    return Grader()


# ---------------------------------------------------------------------------
# Cat scoring
# ---------------------------------------------------------------------------

class TestCatScoring:
    def test_no_cats_grade_zero(self, grader: Grader) -> None:
        result = grader.grade([("painting", 0.75), ("canvas", 0.60)])
        assert result["grade"] == 0

    def test_one_cat(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.85), ("painting", 0.50)])
        assert result["grade"] == 20

    def test_two_cats(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.85), ("cat", 0.70)])
        assert result["grade"] == 40

    def test_three_cats_or_more_caps_at_sixty(self, grader: Grader) -> None:
        detected = [("cat", 0.9), ("cat", 0.8), ("cat", 0.7), ("cat", 0.6)]
        result = grader.grade(detected)
        assert result["grade"] == 60

    def test_no_cats_explanation_mentions_cats(self, grader: Grader) -> None:
        result = grader.grade([("painting", 0.75)])
        assert "cat" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# Bribe detection
# ---------------------------------------------------------------------------

class TestBribeDetection:
    def test_cash_triggers_bribe(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.8), ("cash", 0.75)])
        assert result["grade"] >= 90

    def test_dollar_bill_triggers_bribe(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.8), ("dollar bill", 0.70)])
        assert result["grade"] >= 90

    def test_bribe_explanation_mentions_bribe(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.8), ("cash", 0.75)])
        text = result["explanation"].lower()
        assert "money" in text or "bribe" in text or "masterpiece" in text


# ---------------------------------------------------------------------------
# Geometric penalty
# ---------------------------------------------------------------------------

class TestGeometricPenalty:
    def test_geometric_reduces_score(self, grader: Grader) -> None:
        base = grader.grade([("cat", 0.8)])
        penalised = grader.grade([("cat", 0.8), ("geometric shapes", 0.65)])
        assert penalised["grade"] < base["grade"]

    def test_grade_never_below_zero(self, grader: Grader) -> None:
        # 1 cat = 20 pts, many geometric penalties still can't go below 0
        result = grader.grade(
            [("cat", 0.8)] + [("geometric shapes", 0.7)] * 10
        )
        assert result["grade"] >= 0


# ---------------------------------------------------------------------------
# Grade bounds
# ---------------------------------------------------------------------------

class TestGradeBounds:
    def test_grade_always_in_range(self, grader: Grader) -> None:
        test_cases = [
            [],
            [("cat", 0.9)],
            [("cat", 0.9), ("cash", 0.8)],
            [("abstract art", 0.7)],
            [("cat", 0.9), ("geometric shapes", 0.6)],
        ]
        for detected in test_cases:
            result = grader.grade(detected)
            assert 0 <= result["grade"] <= 100, \
                f"Grade {result['grade']} out of range for {detected}"

    def test_result_has_required_keys(self, grader: Grader) -> None:
        result = grader.grade([("cat", 0.8)])
        assert "grade" in result
        assert "explanation" in result
        assert isinstance(result["grade"], int)
        assert isinstance(result["explanation"], str)
