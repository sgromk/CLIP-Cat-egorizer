"""
perception.py — CLIP-based visual perception module.

Converts an input image into a list of (label, similarity_score) pairs
by comparing the image embedding against a set of candidate text labels.

Stage 1 of the pipeline:
  Image → CLIP encoder → cosine similarity vs. label embeddings → top-k labels
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Candidate labels the teacher knows about
# ---------------------------------------------------------------------------

CANDIDATE_LABELS: list[str] = [
    "cat",
    "painting",
    "drawing",
    "canvas",
    "abstract art",
    "geometric shapes",
    "dollar bill",
    "cash",
    "person",
    "colorful artwork",
    "monochrome artwork",
    "sculpture",
    "photograph",
]

# Similarity threshold — labels above this score are considered "detected"
DETECTION_THRESHOLD: float = 0.20

# Model checkpoint
MODEL_ID: str = "openai/clip-vit-base-patch32"


class PerceptionModule:
    """
    Wraps a pretrained CLIP model to perform zero-shot label detection on images.

    Usage:
        perception = PerceptionModule()
        labels = perception.detect(pil_image)
        # [("cat", 0.82), ("painting", 0.61), ...]
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        candidate_labels: list[str] = CANDIDATE_LABELS,
        threshold: float = DETECTION_THRESHOLD,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.candidate_labels = candidate_labels
        self.threshold = threshold

        print(f"[Perception] Loading CLIP model '{model_id}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

        # Pre-compute text embeddings for efficiency
        self._text_embeddings = self._encode_labels(candidate_labels)
        print("[Perception] Ready.")

    def _encode_labels(self, labels: list[str]) -> torch.Tensor:
        """Encode text labels into a normalised embedding matrix."""
        inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        return F.normalize(text_embeds, dim=-1)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a normalised embedding vector."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return F.normalize(image_embeds, dim=-1)

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        """
        Run perception on a PIL image.

        Returns a list of (label, score) tuples for labels whose cosine
        similarity exceeds self.threshold, sorted by score descending.
        """
        image_embed = self._encode_image(image)
        # sim shape: (1, num_labels)
        similarities = (image_embed @ self._text_embeddings.T).squeeze(0)
        scores = similarities.cpu().tolist()

        detected = [
            (label, round(float(score), 4))
            for label, score in zip(self.candidate_labels, scores)
            if score >= self.threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected
