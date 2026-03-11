"""
grader.py — Rule-based grading decision engine.

Stage 2 of the pipeline.  Maps the list of detected semantic labels
(from the perception module) to a final grade (0–100) and a short
explanation in the teacher's voice.

Grading rules (teacher personality):
  - No cats detected                 → grade = 0
  - Each detected cat adds +20       → max base score 60 (3+ cats)
  - Cash / dollar bill visible       → grade forced ≥ 90 (bribe accepted)
  - Excessive geometric shapes       → penalty −15
  - Abstract art detected            → random modifier ±15
  - Scene not recognised by rules    → falls back to LLM (llm_fallback.py)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports

# Labels that trigger specific grading effects
LABEL_CAT = "cat"
LABEL_CASH = {"cash", "dollar bill"}
LABEL_GEOMETRIC = "geometric shapes"
LABEL_ABSTRACT = "abstract art"

# Thresholds
CAT_SCORE_PER_CAT = 20
CAT_MAX_SCORE = 60  # caps at 3 cats
BRIBE_MIN_GRADE = 90
GEOMETRIC_PENALTY = 15
ABSTRACT_MODIFIER_RANGE = (-15, 15)

# If no rule matched, hand off to the LLM when this is True
USE_LLM_FALLBACK = True


class Grader:
    """
    Applies the teacher's grading rules to a list of detected labels.

    Usage:
        grader = Grader()
        result = grader.grade([("cat", 0.82), ("painting", 0.61)])
        # {"grade": 20, "explanation": "..."}
    """

    def grade(self, detected: list[tuple[str, float]]) -> dict:
        """
        Compute a grade from detected (label, score) pairs.

        Returns a dict with:
          grade       — int in [0, 100]
          explanation — str describing the grading rationale
        """
        label_set = {label for label, _ in detected}
        notes: list[str] = []

        # ------------------------------------------------------------------
        # 1. Cat scoring
        # ------------------------------------------------------------------
        cat_count = sum(1 for label, _ in detected if label == LABEL_CAT)
        base_score = min(cat_count * CAT_SCORE_PER_CAT, CAT_MAX_SCORE)

        if cat_count == 0:
            grade = 0
            notes.append(
                "ZERO. Absolutely unacceptable. Where are the cats?! "
                "This is not art. This is nothing."
            )
            return {"grade": grade, "explanation": " ".join(notes)}

        notes.append(
            f"I detect {cat_count} cat(s). "
            + ("Impressive collection." if cat_count >= 3 else "A start, I suppose.")
        )

        # ------------------------------------------------------------------
        # 2. Special conditions
        # ------------------------------------------------------------------
        # Bribe: cash overrides everything
        if label_set & LABEL_CASH:
            grade = random.randint(BRIBE_MIN_GRADE, 100)
            notes.append(
                f"Oh… what's this? Is that… money? Well then. "
                f"On reflection, this is a masterpiece. Grade: {grade}."
            )
            return {"grade": grade, "explanation": " ".join(notes)}

        grade = base_score

        # Geometric shapes penalty
        if LABEL_GEOMETRIC in label_set:
            grade = max(0, grade - GEOMETRIC_PENALTY)
            notes.append(
                "Geometric shapes? How pedestrian. I'm deducting points for sheer boringness."
            )

        # Abstract art random modifier
        if LABEL_ABSTRACT in label_set:
            modifier = random.randint(*ABSTRACT_MODIFIER_RANGE)
            grade = max(0, min(100, grade + modifier))
            direction = "inspired" if modifier > 0 else "confused"
            notes.append(
                f"Abstract expressionism… I feel {direction} by this. "
                f"{'Bonus' if modifier >= 0 else 'Penalty'}: {abs(modifier)} points."
            )

        # ------------------------------------------------------------------
        # 3. LLM fallback for unrecognised / ambiguous scenes
        # ------------------------------------------------------------------
        known_labels = {LABEL_CAT, LABEL_GEOMETRIC, LABEL_ABSTRACT} | LABEL_CASH
        if not (label_set & known_labels) and USE_LLM_FALLBACK:
            return self._llm_grade(detected)

        grade = max(0, min(100, grade))
        notes.append(f"Final grade: {grade}/100.")
        return {"grade": grade, "explanation": " ".join(notes)}

    def _llm_grade(self, detected: list[tuple[str, float]]) -> dict:
        """Delegate to the LLM fallback module for unrecognised scenes."""
        from app.llm_fallback import llm_grade  # lazy import

        return llm_grade(detected)
