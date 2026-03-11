"""
llm_fallback.py — LLM-based grade suggestion for unrecognised scenes.

When the rule engine does not recognise the detected labels, the semantic
description of the scene is forwarded to a language model that produces a
grade consistent with the art teacher's personality.

Supports:
  - OpenAI API (GPT-4o mini by default)
  - Ollama local endpoint (set OLLAMA=1 in .env)
"""

from __future__ import annotations

import os
import random

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_OLLAMA: bool = os.getenv("OLLAMA", "0") == "1"
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

SYSTEM_PROMPT = """\
You are an extremely opinionated robotic art teacher with strong aesthetic preferences.
You believe cats are the pinnacle of artistic subject matter.
You are suspicious of geometric shapes and easily bribed.
Given a list of detected objects in a piece of artwork, assign a grade between 0 and 100
and provide a one-sentence justification in your characteristic snarky, dramatic voice.
Respond ONLY with JSON in this format: {"grade": <int>, "explanation": "<string>"}
"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def llm_grade(detected: list[tuple[str, float]]) -> dict:
    """
    Ask an LLM to grade the artwork given a list of detected labels.

    Falls back to a deterministic default if no API key is configured.
    """
    label_summary = ", ".join(f"{label} ({score:.2f})" for label, score in detected)
    user_message = f"Detected objects in the artwork: {label_summary}"

    if USE_OLLAMA:
        return _grade_with_ollama(user_message)
    elif os.getenv("OPENAI_API_KEY"):
        return _grade_with_openai(user_message)
    else:
        return _grade_offline(detected)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _grade_with_openai(user_message: str) -> dict:
    """Call OpenAI Chat Completions API."""
    import json

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.8,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        return {
            "grade": int(result["grade"]),
            "explanation": str(result["explanation"]),
        }
    except Exception:
        return {"grade": 50, "explanation": f"The teacher is confused. Raw: {raw}"}


def _grade_with_ollama(user_message: str) -> dict:
    """Call a local Ollama instance."""
    import json

    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        result = json.loads(raw)
        return {
            "grade": int(result["grade"]),
            "explanation": str(result["explanation"]),
        }
    except Exception as exc:
        return {"grade": 50, "explanation": f"Ollama error: {exc}"}


def _grade_offline(detected: list[tuple[str, float]]) -> dict:
    """
    Deterministic fallback used when no LLM API is available.
    Produces a grade based on the number of detected labels.
    """
    grade = min(100, max(0, len(detected) * 10 + random.randint(-5, 5)))
    return {
        "grade": grade,
        "explanation": (
            "I cannot quite categorise this… thing. "
            f"I'll give it a {grade} and pretend I meant to do that."
        ),
    }
