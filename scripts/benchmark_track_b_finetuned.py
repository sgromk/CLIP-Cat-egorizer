"""
benchmark_track_b_finetuned.py
------------------------------
Evaluate Track B fine-tuned CLIP + classifier head on WikiArt split.

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
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track B fine-tuned benchmark")
    parser.add_argument("--split-csv", default="data/labeled/test.csv")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode")
    parser.add_argument("--streaming-split", default="train", help="Dataset split for streaming mode")
    parser.add_argument("--streaming-cache-path", default="", help="Optional parquet cache path for streamed samples")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/trackB_laion_vitl14")
    parser.add_argument("--target", choices=["style"], default="style")
    parser.add_argument("--output-dir", default="artifacts/runs/trackB_eval/style")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


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


def model_tag(checkpoint_dir: Path) -> str:
    return checkpoint_dir.name.replace("/", "__")


def load_style_names(training_config_path: Path) -> list[str]:
    data = json.loads(training_config_path.read_text())
    style_names = data.get("style_names")
    if not isinstance(style_names, list) or not style_names:
        raise ValueError(f"style_names missing or invalid in {training_config_path}")
    return [str(x) for x in style_names]


def get_image_embeddings(clip_model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    return clip_model.visual_projection(pooled)


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
    checkpoint_dir = Path(args.checkpoint_dir)
    clip_dir = checkpoint_dir / "clip_finetuned"
    head_path = checkpoint_dir / "classifier_head.pt"
    cfg_path = checkpoint_dir / "training_config.json"

    if not clip_dir.exists():
        raise FileNotFoundError(f"Missing clip checkpoint dir: {clip_dir}")
    if not head_path.exists():
        raise FileNotFoundError(f"Missing classifier head: {head_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")

    style_names = load_style_names(cfg_path)
    names = {i: name for i, name in enumerate(style_names)}
    n_classes = len(style_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = model_tag(checkpoint_dir)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained(str(clip_dir)).to(device)
    processor = CLIPProcessor.from_pretrained(str(clip_dir))

    embed_dim = int(clip.config.projection_dim)
    classifier = nn.Linear(embed_dim, n_classes).to(device)
    classifier.load_state_dict(torch.load(str(head_path), map_location=device))

    clip.eval()
    classifier.eval()

    all_scores: list[np.ndarray] = []
    all_true: list[int] = []
    kept_source_shard: list[str] = []
    kept_source_row_idx: list[int] = []

    t0 = time.time()
    def score_one(image: Image.Image, true_label: int, shard: str, row_idx: int) -> None:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_feats = get_image_embeddings(clip, inputs["pixel_values"])
            logits = classifier(img_feats).squeeze(0)

        all_scores.append(logits.detach().cpu().numpy())
        all_true.append(true_label)
        kept_source_shard.append(shard)
        kept_source_row_idx.append(row_idx)

        done = len(all_scores)
        if done % args.log_every == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (eval_total - done) / max(rate, 1e-6)
            print(f"  trackB | {done}/{eval_total} | {rate:.1f} img/s | ETA {remaining/60:.1f} min")

    if use_streaming:
        for sample in stream_samples or []:
            try:
                image = Image.open(BytesIO(sample["image_bytes"])).convert("RGB")
            except Exception as exc:
                print(f"[warn] streaming decode error row={sample.get('source_row_idx')}: {exc}")
                continue
            score_one(
                image=image,
                true_label=int(sample["style"]),
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
                score_one(image=image, true_label=int(row["style"]), shard=shard_name, row_idx=row_idx)

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
        "model": f"trackB::{checkpoint_dir.name}",
        "target": "style",
        "n_images": len(true_arr),
        "split_csv": None if use_streaming else args.split_csv,
        "streaming_repo_id": args.streaming_repo_id if use_streaming else None,
        "streaming_split": args.streaming_split if use_streaming else None,
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "elapsed_seconds": round(elapsed_total, 1),
        "images_per_second": round(len(true_arr) / max(elapsed_total, 1e-6), 2),
        "per_class": per_class,
        "checkpoint_dir": str(checkpoint_dir),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved predictions -> {pred_path}")
    print(f"Saved metrics     -> {metrics_path}")
    print(f"top-1={top1:.3f} top-5={top5:.3f} elapsed={elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
