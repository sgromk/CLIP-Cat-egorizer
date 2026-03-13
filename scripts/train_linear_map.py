from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear alignment map W for separate encoders")
    parser.add_argument("--pairs-csv", default="", help="CSV with image_path,text columns")
    parser.add_argument("--wikiart-dir", default="", help="Local WikiArt parquet directory")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for wikiart mode")
    parser.add_argument("--style-template", default="a painting in style_{style_id}")
    parser.add_argument("--output-dir", default="artifacts/linear_map")
    parser.add_argument("--image-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairs_csv and not args.wikiart_dir:
        raise ValueError("Provide either --pairs-csv or --wikiart-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_model = CLIPModel.from_pretrained(args.image_model_id).to(device)
    image_processor = CLIPProcessor.from_pretrained(args.image_model_id)
    text_model = SentenceTransformer(args.text_model_id, device=device)

    image_model.eval()
    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []

    if args.pairs_csv:
        pairs = pd.read_csv(args.pairs_csv)
        if not {"image_path", "text"}.issubset(pairs.columns):
            raise ValueError("pairs CSV must include image_path and text columns")

        for row in pairs.itertuples(index=False):
            with Image.open(row.image_path).convert("RGB") as image:
                image_inputs = image_processor(images=image, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                with torch.no_grad():
                    image_vec = image_model.get_image_features(**image_inputs)
                image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

            text_vec = text_model.encode(str(row.text), convert_to_numpy=True, normalize_embeddings=False)
            text_embeddings.append(text_vec)
    else:
        from io import BytesIO

        parquet_files = sorted(str(p) for p in Path(args.wikiart_dir).rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet shards found under {args.wikiart_dir}")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        if args.max_rows > 0:
            dataset = dataset.select(range(min(args.max_rows, len(dataset))))

        for row in dataset:
            image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            image_inputs = image_processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            with torch.no_grad():
                image_vec = image_model.get_image_features(**image_inputs)
            image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

            text = args.style_template.format(style_id=int(row["style"]))
            text_vec = text_model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
            text_embeddings.append(text_vec)

    x_img = np.stack(image_embeddings)
    y_txt = np.stack(text_embeddings)

    grid = GridSearchCV(
        estimator=Ridge(fit_intercept=False),
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="neg_mean_squared_error",
        cv=5,
    )
    grid.fit(x_img, y_txt)
    ridge: Ridge = grid.best_estimator_

    weight_path = output_dir / "linear_map_W.npy"
    np.save(weight_path, ridge.coef_)

    meta = {
        "image_model_id": args.image_model_id,
        "text_model_id": args.text_model_id,
        "best_alpha": float(grid.best_params_["alpha"]),
        "n_pairs": int(len(image_embeddings)),
        "image_dim": int(x_img.shape[1]),
        "text_dim": int(y_txt.shape[1]),
        "weight_path": str(weight_path),
    }
    (output_dir / "linear_map_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved linear map to {weight_path}")


if __name__ == "__main__":
    main()
