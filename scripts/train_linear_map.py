from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
    parser.add_argument("--split-csv", default="", help="Split CSV with source_shard,source_row_idx,style")
    parser.add_argument("--wikiart-dir", default="", help="Local WikiArt parquet directory")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode")
    parser.add_argument("--streaming-split", default="train", help="Dataset split for streaming mode")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for wikiart mode")
    parser.add_argument("--log-every", type=int, default=500, help="Progress logging interval")
    parser.add_argument("--style-template", default="a painting in style_{style_id}")
    parser.add_argument("--output-dir", default="artifacts/linear_map")
    parser.add_argument("--image-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def extract_image_embedding(image_model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    features = image_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(features):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unsupported image feature output type: {type(features)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairs_csv and not args.split_csv and not args.wikiart_dir and not args.streaming_repo_id.strip():
        raise ValueError("Provide one of --pairs-csv, --split-csv, --wikiart-dir, or --streaming-repo-id")

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
                    image_vec = extract_image_embedding(image_model=image_model, pixel_values=image_inputs["pixel_values"])
                image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

            text_vec = text_model.encode(str(row.text), convert_to_numpy=True, normalize_embeddings=False)
            text_embeddings.append(text_vec)
    elif args.split_csv:
        split_df = pd.read_csv(args.split_csv)
        if not {"source_shard", "source_row_idx", "style"}.issubset(split_df.columns):
            raise ValueError("split CSV must include source_shard, source_row_idx, style columns")
        if args.max_rows > 0:
            split_df = split_df.head(args.max_rows)

        processed = 0
        shard_groups = split_df.groupby("source_shard")

        for shard_name, group in shard_groups:
            shard_path = Path(shard_name)
            if not shard_path.exists():
                print(f"[warn] shard not found: {shard_path}, skipping {len(group)} rows")
                continue

            try:
                shard_images = pq.read_table(str(shard_path), columns=["image"]).to_pylist()
            except Exception as exc:
                print(f"[warn] failed loading shard {shard_path}: {exc}")
                continue

            for _, row in group.iterrows():
                try:
                    row_idx = int(row["source_row_idx"])
                    if row_idx < 0 or row_idx >= len(shard_images):
                        raise IndexError(f"row_idx {row_idx} out of range for {shard_path.name}")
                    img_bytes = shard_images[row_idx]["image"]["bytes"]
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                except Exception as exc:
                    print(f"[warn] decode error shard={shard_name} row={row.get('source_row_idx')}: {exc}")
                    continue

                image_inputs = image_processor(images=image, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                with torch.no_grad():
                    image_vec = extract_image_embedding(image_model=image_model, pixel_values=image_inputs["pixel_values"])
                image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

                text = args.style_template.format(style_id=int(row["style"]))
                text_vec = text_model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
                text_embeddings.append(text_vec)

                processed += 1
                if processed % args.log_every == 0:
                    print(f"processed {processed:,} rows...")
    elif args.streaming_repo_id.strip():
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        token = os.getenv("HF_TOKEN", "").strip() or False
        dataset = load_dataset(
            args.streaming_repo_id.strip(),
            split=args.streaming_split,
            streaming=True,
            token=token,
        )

        processed = 0
        for row in dataset:
            try:
                image_field = row["image"]
                if isinstance(image_field, dict) and image_field.get("bytes") is not None:
                    image = Image.open(BytesIO(image_field["bytes"])).convert("RGB")
                elif isinstance(image_field, dict) and image_field.get("path") is not None:
                    image = Image.open(image_field["path"]).convert("RGB")
                else:
                    image = image_field.convert("RGB") if isinstance(image_field, Image.Image) else None
                if image is None:
                    continue
            except Exception as exc:
                print(f"[warn] streaming decode error at row {processed}: {exc}")
                continue

            image_inputs = image_processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            with torch.no_grad():
                image_vec = extract_image_embedding(image_model=image_model, pixel_values=image_inputs["pixel_values"])
            image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

            text = args.style_template.format(style_id=int(row["style"]))
            text_vec = text_model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
            text_embeddings.append(text_vec)

            processed += 1
            if processed % args.log_every == 0:
                print(f"processed {processed:,} rows...")
            if args.max_rows > 0 and processed >= args.max_rows:
                break
    else:
        parquet_files = sorted(str(p) for p in Path(args.wikiart_dir).rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet shards found under {args.wikiart_dir}")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        if args.max_rows > 0:
            dataset = dataset.select(range(min(args.max_rows, len(dataset))))

        for i, row in enumerate(dataset):
            image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            image_inputs = image_processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            with torch.no_grad():
                image_vec = extract_image_embedding(image_model=image_model, pixel_values=image_inputs["pixel_values"])
            image_embeddings.append(image_vec.squeeze(0).detach().cpu().numpy())

            text = args.style_template.format(style_id=int(row["style"]))
            text_vec = text_model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
            text_embeddings.append(text_vec)

            if (i + 1) % args.log_every == 0:
                print(f"processed {i + 1:,} rows...")

    if not image_embeddings:
        raise RuntimeError("No embeddings were generated; cannot fit linear map")

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
