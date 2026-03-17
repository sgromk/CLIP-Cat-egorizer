"""
benchmark_wikiart_zero_shot.py
-------------------------------
Zero-shot WikiArt style classification benchmark.

For each model in configs/zero_shot_models.json:
  - Loads images from parquet shards via test.csv index
  - Scores each image against all 27 style prompts (multiple templates)
  - Reports Top-1 / Top-5 accuracy + per-class accuracy
  - Saves: predictions_{model_tag}.parquet + metrics_{model_tag}.json
  - Resume-safe: skips models whose prediction file already exists

Run:
    nohup .venv/bin/python scripts/benchmark_wikiart_zero_shot.py \
        --split-csv data/labeled/test.csv \
        --output-dir artifacts/runs/zeroshot \
        > logs/zeroshot_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.experiment_models import HFVisionTextAdapter

# ── Canonical WikiArt label maps (alphabetical order = integer id) ──────────
STYLE_NAMES: dict[int, str] = {
    0:  "Abstract Expressionism",
    1:  "Action Painting",
    2:  "Analytical Cubism",
    3:  "Art Nouveau",
    4:  "Baroque",
    5:  "Color Field Painting",
    6:  "Contemporary Realism",
    7:  "Cubism",
    8:  "Early Renaissance",
    9:  "Expressionism",
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

STYLE_PROMPT_TEMPLATES = [
    "a painting in the style of {style}",
    "a {style} painting",
    "an artwork in {style} style",
    "a fine art piece in {style}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot WikiArt style benchmark")
    parser.add_argument("--split-csv", default="data/labeled/test.csv",
                        help="Split CSV from wikiart_make_splits.py")
    parser.add_argument(
        "--streaming-repo-id",
        default="",
        help="Optional HF dataset repo for streaming mode (e.g. huggan/wikiart).",
    )
    parser.add_argument(
        "--streaming-split",
        default="train",
        help="Dataset split to stream when --streaming-repo-id is set.",
    )
    parser.add_argument(
        "--streaming-cache-path",
        default="",
        help="Optional parquet cache path for streamed samples (shared across targets).",
    )
    parser.add_argument("--models-config", default="configs/zero_shot_models.json")
    parser.add_argument("--output-dir", default="artifacts/runs/zeroshot")
    parser.add_argument("--target", choices=["style", "genre"], default="style",
                        help="Classification target")
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 = all rows in split CSV")
    parser.add_argument(
        "--templates-per-class",
        type=int,
        default=1,
        help="How many prompt templates to use per class (1-4, lower is faster)",
    )
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print progress every N images")
    return parser.parse_args()


def build_prompts(target: str, templates_per_class: int) -> tuple[list[str], list[int], int]:
    """Return (prompts, label_ids) with one best-template prompt per class."""
    names = STYLE_NAMES if target == "style" else GENRE_NAMES
    n_templates = max(1, min(int(templates_per_class), len(STYLE_PROMPT_TEMPLATES)))
    templates = STYLE_PROMPT_TEMPLATES[:n_templates]

    prompts: list[str] = []
    label_ids: list[int] = []
    for lid, name in sorted(names.items()):
        for tmpl in templates:
            prompts.append(tmpl.format(style=name))
            label_ids.append(lid)
    return prompts, label_ids, n_templates


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
    df = pd.DataFrame(samples)
    df.to_parquet(cache_path, index=False)
    print(f"Saved streaming cache -> {cache_path}")


def load_stream_samples_cache(cache_path: Path) -> list[dict]:
    df = pd.read_parquet(cache_path)
    records = df.to_dict(orient="records")
    print(f"Loaded streaming cache -> {cache_path} ({len(records):,} rows)")
    return records


def load_image_from_shard(shard_path: Path, row_idx: int) -> Image.Image:
    tbl = pq.read_table(shard_path, columns=["image"]).slice(row_idx, 1)
    img_bytes = tbl.to_pylist()[0]["image"]["bytes"]
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def topk_accuracy(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    topk = np.argsort(y_score, axis=1)[:, -k:]
    correct = sum(y_true[i] in topk[i] for i in range(len(y_true)))
    return correct / len(y_true)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       names: dict[int, str]) -> dict:
    out = {}
    for lid, name in sorted(names.items()):
        mask = y_true == lid
        if mask.sum() == 0:
            continue
        acc = float((y_pred[mask] == lid).mean())
        out[name] = {"n": int(mask.sum()), "top1_acc": round(acc, 4)}
    return out


def run_model(
    model_id: str,
    index_df: pd.DataFrame,
    target: str,
    out_dir: Path,
    log_every: int,
    templates_per_class: int,
) -> dict:
    model_tag = model_id.replace("/", "__")
    pred_path = out_dir / f"predictions_{model_tag}.parquet"
    metrics_path = out_dir / f"metrics_{model_tag}.json"

    if pred_path.exists() and metrics_path.exists():
        print(f"[skip] {model_id} — prediction file already exists, loading metrics")
        return json.loads(metrics_path.read_text())

    names = STYLE_NAMES if target == "style" else GENRE_NAMES
    prompts, prompt_labels, n_prompts_per_class = build_prompts(target, templates_per_class)
    n_classes = len(names)

    print(f"\n[benchmark] Loading {model_id} …")
    adapter = HFVisionTextAdapter(model_id=model_id)
    t0 = time.time()

    all_scores = []   # shape (N, n_classes) — max-pooled over templates
    all_true = []
    kept_source_shard = []
    kept_source_row_idx = []

    # Group by shard to avoid redundant parquet opens
    shard_groups = index_df.groupby("source_shard")

    for shard_name, group in shard_groups:
        # source_shard stores the full path relative to the project root
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

            result = adapter.score_image_texts(image=image, texts=prompts)
            raw = np.array(result.logits)   # length = n_classes * n_prompts_per_class

            # Max-pool logits across templates for each class
            raw = raw.reshape(n_classes, n_prompts_per_class)
            class_logits = raw.max(axis=1)  # (n_classes,)

            all_scores.append(class_logits)
            all_true.append(int(row[target]))
            kept_source_shard.append(shard_name)
            kept_source_row_idx.append(row_idx)

            done = len(all_scores)
            if done % log_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(index_df) - done) / max(rate, 1e-6)
                print(f"  {model_id} | {done}/{len(index_df)} | "
                      f"{rate:.1f} img/s | ETA {remaining/60:.1f} min")

    scores_arr = np.array(all_scores)        # (N, n_classes)
    true_arr   = np.array(all_true)          # (N,)
    pred_arr   = scores_arr.argmax(axis=1)   # (N,)

    # ── Save predictions ────────────────────────────────────────────────────
    pred_df = pd.DataFrame(scores_arr, columns=[names[i] for i in sorted(names)])
    pred_df.insert(0, "source_shard", kept_source_shard)
    pred_df.insert(1, "source_row_idx", kept_source_row_idx)
    pred_df.insert(2, "true_label",     true_arr)
    pred_df.insert(3, "pred_label",     pred_arr)
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Saved predictions → {pred_path}")

    # ── Compute metrics ─────────────────────────────────────────────────────
    top1 = float((pred_arr == true_arr).mean())
    top5 = topk_accuracy(true_arr, scores_arr, k=5)
    per_class = per_class_accuracy(true_arr, pred_arr, names)

    elapsed_total = time.time() - t0
    metrics = {
        "model": model_id,
        "target": target,
        "n_images": len(true_arr),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "elapsed_seconds": round(elapsed_total, 1),
        "images_per_second": round(len(true_arr) / elapsed_total, 2),
        "per_class": per_class,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  top-1: {top1:.3f}  top-5: {top5:.3f}  ({elapsed_total/60:.1f} min)")
    print(f"  Saved metrics   → {metrics_path}")
    return metrics


def run_model_from_samples(
    model_id: str,
    samples: list[dict],
    target: str,
    out_dir: Path,
    log_every: int,
    templates_per_class: int,
) -> dict:
    model_tag = model_id.replace("/", "__")
    pred_path = out_dir / f"predictions_{model_tag}.parquet"
    metrics_path = out_dir / f"metrics_{model_tag}.json"

    if pred_path.exists() and metrics_path.exists():
        print(f"[skip] {model_id} — prediction file already exists, loading metrics")
        return json.loads(metrics_path.read_text())

    names = STYLE_NAMES if target == "style" else GENRE_NAMES
    prompts, _, n_prompts_per_class = build_prompts(target, templates_per_class)
    n_classes = len(names)

    print(f"\n[benchmark] Loading {model_id} …")
    adapter = HFVisionTextAdapter(model_id=model_id)
    t0 = time.time()

    all_scores = []
    all_true = []
    kept_source_shard = []
    kept_source_row_idx = []

    total_rows = len(samples)
    for sample in samples:
        true_label = int(sample.get(target, -1))
        if true_label < 0:
            continue

        try:
            image = Image.open(BytesIO(sample["image_bytes"])).convert("RGB")
        except Exception as exc:
            print(f"[warn] decode error row={sample.get('source_row_idx')}: {exc}")
            continue

        result = adapter.score_image_texts(image=image, texts=prompts)
        raw = np.array(result.logits)
        raw = raw.reshape(n_classes, n_prompts_per_class)
        class_logits = raw.max(axis=1)

        all_scores.append(class_logits)
        all_true.append(true_label)
        kept_source_shard.append(sample.get("source_shard", "streaming"))
        kept_source_row_idx.append(int(sample.get("source_row_idx", -1)))

        done = len(all_scores)
        if done % log_every == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (total_rows - done) / max(rate, 1e-6)
            print(f"  {model_id} | {done}/{total_rows} | {rate:.1f} img/s | ETA {remaining/60:.1f} min")

    if not all_scores:
        raise RuntimeError(f"No valid samples were scored for {model_id}")

    scores_arr = np.array(all_scores)
    true_arr = np.array(all_true)
    pred_arr = scores_arr.argmax(axis=1)

    pred_df = pd.DataFrame(scores_arr, columns=[names[i] for i in sorted(names)])
    pred_df.insert(0, "source_shard", kept_source_shard)
    pred_df.insert(1, "source_row_idx", kept_source_row_idx)
    pred_df.insert(2, "true_label", true_arr)
    pred_df.insert(3, "pred_label", pred_arr)
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Saved predictions → {pred_path}")

    top1 = float((pred_arr == true_arr).mean())
    top5 = topk_accuracy(true_arr, scores_arr, k=5)
    per_class = per_class_accuracy(true_arr, pred_arr, names)

    elapsed_total = time.time() - t0
    metrics = {
        "model": model_id,
        "target": target,
        "n_images": len(true_arr),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "elapsed_seconds": round(elapsed_total, 1),
        "images_per_second": round(len(true_arr) / elapsed_total, 2),
        "per_class": per_class,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  top-1: {top1:.3f}  top-5: {top5:.3f}  ({elapsed_total/60:.1f} min)")
    print(f"  Saved metrics   → {metrics_path}")
    return metrics


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    models_cfg = json.loads(Path(args.models_config).read_text())
    models: list[str] = models_cfg["models"]

    use_streaming = bool(args.streaming_repo_id.strip())
    samples: list[dict] | None = None
    index_df: pd.DataFrame | None = None

    if use_streaming:
        cache_path = Path(args.streaming_cache_path) if args.streaming_cache_path.strip() else None
        if cache_path is not None and cache_path.exists():
            samples = load_stream_samples_cache(cache_path)
        else:
            samples = load_stream_samples(
                repo_id=args.streaming_repo_id.strip(),
                split=args.streaming_split,
                max_images=args.max_images,
            )
            if cache_path is not None:
                save_stream_samples_cache(samples, cache_path)
        n_eval_rows = len(samples)
        print(f"Loaded {n_eval_rows:,} streaming rows from {args.streaming_repo_id}:{args.streaming_split}")
    else:
        index_df = pd.read_csv(args.split_csv)
        if args.max_images > 0:
            index_df = index_df.head(args.max_images)
        n_eval_rows = len(index_df)
        print(f"Loaded split: {n_eval_rows:,} rows from {args.split_csv}")

    print(f"Using {max(1, min(args.templates_per_class, len(STYLE_PROMPT_TEMPLATES)))} prompt template(s) per class")

    names = STYLE_NAMES if args.target == "style" else GENRE_NAMES

    summary: dict[str, object] = {
        "split_csv": args.split_csv if not use_streaming else None,
        "streaming_repo_id": args.streaming_repo_id if use_streaming else None,
        "streaming_split": args.streaming_split if use_streaming else None,
        "target": args.target,
        "n_images": n_eval_rows,
        "n_classes": len(names),
        "models": {},
    }

    for model_id in models:
        if use_streaming:
            metrics = run_model_from_samples(
                model_id=model_id,
                samples=samples or [],
                target=args.target,
                out_dir=out_dir,
                log_every=args.log_every,
                templates_per_class=args.templates_per_class,
            )
        else:
            metrics = run_model(
                model_id=model_id,
                index_df=index_df if index_df is not None else pd.DataFrame(),
                target=args.target,
                out_dir=out_dir,
                log_every=args.log_every,
                templates_per_class=args.templates_per_class,
            )
        summary["models"][model_id] = {
            "top1_accuracy": metrics.get("top1_accuracy"),
            "top5_accuracy": metrics.get("top5_accuracy"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
        }

    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Benchmark Complete ===")
    print(f"Summary → {summary_path}")
    print(f"\n{'Model':<45} {'Top-1':>6} {'Top-5':>6}")
    print("-" * 60)
    for mid, m in summary["models"].items():
        print(f"{mid:<45} {m['top1_accuracy']:>6.3f} {m['top5_accuracy']:>6.3f}")


if __name__ == "__main__":
    main()
