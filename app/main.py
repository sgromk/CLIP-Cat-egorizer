"""
main.py — FastAPI web server for the Art Grading Agent.

Endpoints:
  POST /grade        — Upload an image, receive a grade + explanation
  GET  /health       — Health check
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.grader import Grader
from app.perception import PerceptionModule

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Art Grading Agent",
    description="A vision-language AI pipeline that grades artwork like an opinionated robot teacher.",
    version="0.1.0",
)

# Module singletons (loaded once at startup)
perception: PerceptionModule | None = None
grader: Grader | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global perception, grader
    print("Loading CLIP perception module...")
    perception = PerceptionModule()
    grader = Grader()
    print("Ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict:
    """API root endpoint for backend-only operation."""
    return {"status": "ok", "message": "Backend API is running."}


@app.get("/health")
async def health() -> dict:
    """Simple health check."""
    return {"status": "ok", "perception_loaded": perception is not None}


@app.post("/grade")
async def grade_artwork(file: UploadFile = File(...)) -> dict:
    """
    Accepts an uploaded image and returns:
      - detected_labels: list of (label, score) pairs from CLIP
      - grade: integer 0–100
      - explanation: human-readable grade rationale
    """
    if perception is None or grader is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Read image bytes
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    # Stage 1: Perception
    detected = perception.detect(image)

    # Stage 2: Decision
    result = grader.grade(detected)

    return {
        "detected_labels": detected,
        "grade": result["grade"],
        "explanation": result["explanation"],
    }
