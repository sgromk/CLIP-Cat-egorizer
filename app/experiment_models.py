from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


def pick_device() -> str:
    env = os.getenv("MODEL_DEVICE", "auto").lower().strip()
    if env in {"cpu", "cuda", "mps"}:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelScores:
    scores: list[float]
    logits: list[float]


class HFVisionTextAdapter:
    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or pick_device()
        hf_token = os.getenv("HF_TOKEN") or None
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        ).to(self.device)
        self.model.eval()

    def score_image_texts(self, image: Image.Image, texts: list[str]) -> ModelScores:
        inputs = self.processor(
            images=image,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
            logits = outputs.logits_per_image.squeeze(0).float().cpu()
            probs = torch.softmax(logits, dim=0)
            return ModelScores(scores=probs.tolist(), logits=logits.tolist())

        image_emb = self._image_features(inputs)
        text_emb = self._text_features(inputs)
        logits = (image_emb @ text_emb.T).squeeze(0).float().cpu()
        probs = torch.softmax(logits, dim=0)
        return ModelScores(scores=probs.tolist(), logits=logits.tolist())

    def _image_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            emb = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            return F.normalize(emb, dim=-1)
        raise RuntimeError(
            f"Model {self.model_id} does not expose logits_per_image or get_image_features"
        )

    def _text_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        text_kwargs = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in inputs:
                text_kwargs[key] = inputs[key]

        if hasattr(self.model, "get_text_features"):
            emb = self.model.get_text_features(**text_kwargs)
            return F.normalize(emb, dim=-1)
        raise RuntimeError(f"Model {self.model_id} does not expose get_text_features")
