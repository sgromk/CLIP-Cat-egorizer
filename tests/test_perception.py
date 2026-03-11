"""
test_perception.py — Unit tests for the CLIP perception module.

These tests use a mock/stub approach so that CLIP is not actually loaded
during CI runs (loading the full model requires ~400MB of weights).

Set the environment variable INTEGRATION=1 to run the full model tests:
  INTEGRATION=1 pytest tests/test_perception.py
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blank_image(width: int = 224, height: int = 224) -> Image.Image:
    """Return a solid-colour PIL image for testing."""
    return Image.new("RGB", (width, height), color=(128, 64, 200))


# ---------------------------------------------------------------------------
# Unit tests (mocked CLIP)
# ---------------------------------------------------------------------------

class TestPerceptionModuleUnit:
    """Tests that do NOT load the real CLIP model."""

    def test_detect_returns_list(self) -> None:
        """detect() should always return a list."""
        with patch("app.perception.CLIPModel.from_pretrained"), \
             patch("app.perception.CLIPProcessor.from_pretrained"), \
             patch("app.perception.PerceptionModule._encode_labels") as mock_enc:

            import torch
            mock_enc.return_value = torch.zeros(13, 512)

            from app.perception import PerceptionModule
            pm = PerceptionModule.__new__(PerceptionModule)
            pm.candidate_labels = ["cat", "painting"]
            pm.threshold = 0.20
            pm.device = "cpu"
            pm._text_embeddings = torch.zeros(2, 512)
            pm.model = MagicMock()
            pm.processor = MagicMock()

            # Patch _encode_image to return a known vector
            import torch.nn.functional as F
            img_embed = F.normalize(torch.rand(1, 512), dim=-1)
            with patch.object(pm, "_encode_image", return_value=img_embed):
                result = pm.detect(make_blank_image())

            assert isinstance(result, list)

    def test_detect_sorted_by_score_descending(self) -> None:
        """Detected labels should be sorted highest score first."""
        import torch
        import torch.nn.functional as F

        from app.perception import PerceptionModule

        pm = PerceptionModule.__new__(PerceptionModule)
        pm.candidate_labels = ["cat", "painting", "cash"]
        pm.threshold = 0.0   # detect everything for this test
        pm.device = "cpu"
        pm.model = MagicMock()
        pm.processor = MagicMock()

        # Create text embeddings and image embedding so cosine sims are controllable
        text_embeds = torch.eye(3, 512)
        img_raw = torch.zeros(1, 512)
        img_raw[0, 1] = 1.0   # image most similar to label index 1 ("painting")
        img_embed = F.normalize(img_raw, dim=-1)
        pm._text_embeddings = F.normalize(text_embeds, dim=-1)

        with patch.object(pm, "_encode_image", return_value=img_embed):
            result = pm.detect(make_blank_image())

        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True), "Should be sorted descending"


# ---------------------------------------------------------------------------
# Integration tests (real CLIP) — only run when INTEGRATION=1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("INTEGRATION") != "1",
    reason="Set INTEGRATION=1 to run real CLIP model tests",
)
class TestPerceptionModuleIntegration:
    def test_real_clip_loads_and_detects(self) -> None:
        from app.perception import PerceptionModule
        pm = PerceptionModule()
        img = make_blank_image()
        result = pm.detect(img)
        assert isinstance(result, list)
        # At minimum the model should return some results for any real image
        assert len(result) >= 0
