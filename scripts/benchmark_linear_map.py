"""
benchmark_linear_map.py
----------------------
Evaluate Track C separate-encoder linear map baseline on WikiArt split.

Outputs are Track A-compatible:
  - predictions_{model_tag}.parquet
  - metrics_{model_tag}.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# Canonical WikiArt label maps
STYLE_NAMES: dict[int, str] = {
    0: "Abstract Expressionism",
    1: "Action Painting",
    2: "Analytical Cubism",
    3: "Art Nouveau",
    4: "Baroque",
    5: "Color Field Painting",
    6: "Contemporary Realism",
    7: "Cubism",
    8: "Early Renaissance",
    9: "Expressionism",
    10: "Fauvism",
    11: "High Renaissance",
    12: "Impressionism",
    13: "Mannerism Late Renaissance",
    14: "Minimalism",
    15: "Naive Art Primitivism",
    16: "New Realism",
    17: "Northern Renaissance",
    18: "Pointillism",
    19: "Pop Art",
    20: "Post Impressionism",
    21: "Realism",
    22: "Rococo",
    23: "Romanticism",
    24: "Symbolism",
    25: "Synthetic Cubism",
    26: "Ukiyo-e",
}

GENRE_NAMES: dict[int, str] = {
    0: "abstract painting",
    1: "cityscape",
    2: "genre painting",
    3: "illustration",
    4: "landscape",
    5: "nude painting",
    6: "portrait",
    7: "religious painting",
    8: "sketch and study",
    9: "still life",
    10: "unknown genre",
}

PROMPT_TEMPLATES = [
    "a painting in the style of {style}",
    "a {style} painting",
    "an artwork in {style} style",
    "a fine art piece in {style}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track C linear-map benchmark")
    parser.add_argument("--split-csv", default="data/labeled/test.csv")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode")
    parser.add_argument("--streaming-split", default="train", help="Dataset split for streaming mode")
    parser.add_argument("--streaming-cache-path", default="", help="Optional parquet cache path for streamed samples")
    parser.add_argument("--target", choices=["style", "genre"], default="style")
    parser.add_argument("--output-dir", default="artifacts/runs/linear_map")
    parser.add_argument("--linear-map-path", default="artifacts/linear_map/linear_map_W.npy")
    parser.add_argument("--linear-map-meta", default="", help="Optional meta json path; defaults to sibling linear_map_meta.json")
    parser.add_argument("--image-model-id", default="", help="Override image model id; otherwise use metadata")
    parser.add_argument("--text-model-id", default="", help="Override text model id; otherwise use metadata")
    parser.add_argument("--templates-per-class", type=int, default=1, help="How many prompt templates per class (1-4)")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all rows in split CSV")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def build_prompts(target: str, templates_per_class: int) -> tuple[list[str], list[int], int]:
    names = STYLE_NAMES if target == "style" else GENRE_NAMES
    n_templates = max(1, min(int(templates_per_class), len(PROMPT_TEMPLATES)))
    templates = PROMPT_TEMPLATES[:n_templates]

    prompts: list[str] = []
    label_ids: list[int] = []
    for lid, name in sorted(names.items()):
        for tmpl in templates:
            prompts.append(tmpl.format(style=name))
            label_ids.append(lid)
    return prompts, label_ids, n_templates


def resolve_meta(args: argparse.Namespace) -> dict:
    linear_map_path = Path(args.linear_map_path)
    meta_path = Path(args.linear_map_meta) if args.linear_map_meta else linear_map_path.with_name("linear_map_meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def topk_accuracy(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    topk = np.argsort(y_score, axis=1)[:, -k:]
    correct = sum(y_true[i] in topk[i] for i in range(len(y_true)))
    return correct / len(y_true)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, names: dict[int, str]) -> dict:
    out = {}
    for lid, name in sorted(names.items()):
        mask = y_true == lid
        if mask.sum() == 0:
            continue
        acc = float((y_pred[mask] == lid).mean())
        out[name] = {"n": int(mask.sum()), "top1_acc": round(acc, 4)}
    return out


def encode_prompt_bank(text_model: SentenceTransformer, prompts: list[str]) -> np.ndarray:
    text_mat = text_model.encode(prompts, convert_to_numpy=True, normalize_embeddings=False)
    norms = np.linalg.norm(text_mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return text_mat / norms


def model_tag(image_model_id: str, text_model_id: str) -> str:
    return (
        "linear_map__"
        + image_model_id.replace("/", "__")
        + "__to__"
        + text_model_id.replace("/", "__")
    )


def extract_image_embedding(
    image_model: CLIPModel,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    features = image_model.get_image_features(pixel_values=pixel_values)
    if torch.is_tensor(features):
        return features

    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output

    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state.mean(dim=1)

    raise TypeError(f"Unsupported image feature output type: {type(features)}")


def image_field_to_bytes(image_field) -> bytes:
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return bytes(image_field["bytes"])
        if image_field.get("path") is not None:
            with open(image_field["path"], "rb") as fh:
                return fh.read()
    if isinstance(image_field, (bytes, bytearray)):
        return bytes(image_field)
    if isinstance(image_field, Image.Image):
        buf = BytesIO()
        image_field.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def load_stream_samples(repo_id: str, split: str, max_images: int) -> list[dict]:
    from datasets import load_dataset

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    token = os.getenv("HF_TOKEN", "").strip() or False

    samples: list[dict] = []
    bad_rows = 0
    resume_row_idx = 0
    attempt = 0
    max_attempts = 8

    while True:
        try:
            print(f"Loading streaming dataset: {repo_id} [{split}] (attempt {attempt + 1}/{max_attempts + 1})")
            raw = load_dataset(repo_id, split=split, streaming=True, token=token)

            for i, row in enumerate(raw):
                if i < resume_row_idx:
                    continue

                try:
                    image_bytes = image_field_to_bytes(row["image"])
                    samples.append(
                        {
                            "source_shard": f"streaming:{repo_id}:{split}",
                            "source_row_idx": i,
                            "image_bytes": image_bytes,
                            "style": int(row["style"]),
                            "genre": int(row.get("genre", -1)),
                        }
                    )
                except Exception as exc:
                    bad_rows += 1
                    if bad_rows <= 5:
                        print(f"[warn] streaming row {i} skipped: {exc}")

                resume_row_idx = i + 1

                if max_images > 0 and len(samples) >= max_images:
                    print(f"Buffered {len(samples):,} streaming samples (skipped {bad_rows:,})")
                    return samples

                if len(samples) > 0 and len(samples) % 1000 == 0:
                    print(f"  buffered {len(samples):,} streamed samples...")

            break
        except Exception as exc:
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError(
                    f"Streaming failed after {max_attempts + 1} attempts at row {resume_row_idx}: {exc}"
                ) from exc
            backoff = min(30, 2 ** attempt)
            print(
                f"[warn] Streaming error at row {resume_row_idx}: {exc}\n"
                f"       Retrying in {backoff}s from last buffered position..."
            )
            time.sleep(backoff)

    print(f"Buffered {len(samples):,} streaming samples (skipped {bad_rows:,})")
    if not samples:
        raise RuntimeError("No streaming samples available for benchmarking")
    return samples


def save_stream_samples_cache(samples: list[dict], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_parquet(cache_path, index=False)
    print(f"Saved streaming cache -> {cache_path}")


def load_stream_samples_cache(cache_path: Path) -> list[dict]:
    records = pd.read_parquet(cache_path).to_dict(orient="records")
    print(f"Loaded streaming cache -> {cache_path} ({len(records):,} rows)")
    return records


def main() -> None:
    args = parse_args()
    meta = resolve_meta(args)

    image_model_id = args.image_model_id or meta.get("image_model_id") or "openai/clip-vit-base-patch32"
    text_model_id = args.text_model_id or meta.get("text_model_id") or "sentence-transformers/all-MiniLM-L6-v2"

    linear_map = np.load(args.linear_map_path)
    if linear_map.ndim != 2:
        raise ValueError(f"Expected a 2D linear map matrix, got shape={linear_map.shape}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = model_tag(image_model_id, text_model_id)
    pred_path = out_dir / f"predictions_{tag}.parquet"
    metrics_path = out_dir / f"metrics_{tag}.json"

    use_streaming = bool(args.streaming_repo_id.strip())

    index_df: pd.DataFrame | None = None
    stream_samples: list[dict] | None = None

    if use_streaming:
        cache_path = Path(args.streaming_cache_path) if args.streaming_cache_path.strip() else None
        if cache_path is not None and cache_path.exists():
            stream_samples = load_stream_samples_cache(cache_path)
        else:
            stream_samples = load_stream_samples(
                repo_id=args.streaming_repo_id.strip(),
                split=args.streaming_split,
                max_images=args.max_images,
            )
            if cache_path is not None:
                save_stream_samples_cache(stream_samples, cache_path)

        if not stream_samples:
            raise ValueError("No streaming rows available for evaluation")
        eval_total = len(stream_samples)
    else:
        index_df = pd.read_csv(args.split_csv)
        if args.max_images > 0:
            index_df = index_df.head(args.max_images)
        if index_df.empty:
            raise ValueError("No rows available for evaluation")
        eval_total = len(index_df)

    names = STYLE_NAMES if args.target == "style" else GENRE_NAMES
    prompts, _, n_templates = build_prompts(args.target, args.templates_per_class)
    n_classes = len(names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_model = CLIPModel.from_pretrained(image_model_id).to(device)
    image_processor = CLIPProcessor.from_pretrained(image_model_id)
    text_model = SentenceTransformer(text_model_id, device=device)
    image_model.eval()

    class_text = encode_prompt_bank(text_model, prompts)
    if class_text.shape[1] != linear_map.shape[0]:
        raise ValueError(
            "Linear map output dimension does not match text embedding dimension: "
            f"W_out={linear_map.shape[0]} text_dim={class_text.shape[1]}"
        )

    all_scores: list[np.ndarray] = []
    all_true: list[int] = []
    kept_source_shard: list[str] = []
    kept_source_row_idx: list[int] = []

    t0 = time.time()
    def score_one(image: Image.Image, true_label: int, shard: str, row_idx: int) -> None:
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_vec = extract_image_embedding(image_model=image_model, pixel_values=inputs["pixel_values"])

        image_np = image_vec.squeeze(0).detach().cpu().numpy()
        if image_np.shape[0] != linear_map.shape[1]:
            raise ValueError(
                "Linear map input dimension does not match image embedding dimension: "
                f"W_in={linear_map.shape[1]} img_dim={image_np.shape[0]}"
            )

        projected = image_np @ linear_map.T
        projected_norm = projected / max(np.linalg.norm(projected), 1e-12)

        raw = projected_norm @ class_text.T
        raw = raw.reshape(n_classes, n_templates)
        class_logits = raw.max(axis=1)

        all_scores.append(class_logits)
        all_true.append(true_label)
        kept_source_shard.append(shard)
        kept_source_row_idx.append(row_idx)

        done = len(all_scores)
        if done % args.log_every == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (eval_total - done) / max(rate, 1e-6)
            print(f"  linear-map | {done}/{eval_total} | {rate:.1f} img/s | ETA {remaining/60:.1f} min")

    if use_streaming:
        for sample in stream_samples or []:
            true_label = int(sample.get(args.target, -1))
            if true_label < 0:
                continue
            try:
                image = Image.open(BytesIO(sample["image_bytes"])).convert("RGB")
            except Exception as exc:
                print(f"[warn] streaming decode error row={sample.get('source_row_idx')}: {exc}")
                continue
            score_one(
                image=image,
                true_label=true_label,
                shard=str(sample.get("source_shard", "streaming")),
                row_idx=int(sample.get("source_row_idx", -1)),
            )
    else:
        shard_groups = (index_df if index_df is not None else pd.DataFrame()).groupby("source_shard")

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
                    print(f"[warn] decode error shard={shard_name} row={row['source_row_idx']}: {exc}")
                    continue
                score_one(image=image, true_label=int(row[args.target]), shard=shard_name, row_idx=row_idx)

    if not all_scores:
        raise RuntimeError("No valid rows were scored")

    scores_arr = np.array(all_scores)
    true_arr = np.array(all_true)
    pred_arr = scores_arr.argmax(axis=1)

    pred_df = pd.DataFrame(scores_arr, columns=[names[i] for i in sorted(names)])
    pred_df.insert(0, "source_shard", kept_source_shard)
    pred_df.insert(1, "source_row_idx", kept_source_row_idx)
    pred_df.insert(2, "true_label", true_arr)
    pred_df.insert(3, "pred_label", pred_arr)
    pred_df.to_parquet(pred_path, index=False)

    top1 = float((pred_arr == true_arr).mean())
    top5 = topk_accuracy(true_arr, scores_arr, k=5)
    per_class = per_class_accuracy(true_arr, pred_arr, names)
    elapsed_total = time.time() - t0

    metrics = {
        "model": f"linear_map::{image_model_id}=>{text_model_id}",
        "target": args.target,
        "n_images": len(true_arr),
        "split_csv": None if use_streaming else args.split_csv,
        "streaming_repo_id": args.streaming_repo_id if use_streaming else None,
        "streaming_split": args.streaming_split if use_streaming else None,
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "elapsed_seconds": round(elapsed_total, 1),
        "images_per_second": round(len(true_arr) / max(elapsed_total, 1e-6), 2),
        "per_class": per_class,
        "linear_map_path": str(Path(args.linear_map_path)),
        "image_model_id": image_model_id,
        "text_model_id": text_model_id,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved predictions -> {pred_path}")
    print(f"Saved metrics     -> {metrics_path}")
    print(f"top-1={top1:.3f} top-5={top5:.3f} elapsed={elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
