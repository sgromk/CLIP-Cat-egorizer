from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.experiment_models import HFVisionTextAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark zero-shot VLMs across labels/prompts")
    parser.add_argument("--dataset-csv", required=True, help="CSV with image_path and binary label columns")
    parser.add_argument("--models-config", default="configs/zero_shot_models.json")
    parser.add_argument("--labels-config", default="configs/labels_and_prompts.json")
    parser.add_argument("--output-dir", default="artifacts/runs")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def build_prompt_grid(labels: list[str], templates: list[str]) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    owners: list[str] = []
    for label in labels:
        for template in templates:
            prompts.append(template.format(label=label))
            owners.append(label)
    return prompts, owners


def evaluate_predictions(
    score_df: pd.DataFrame,
    data_df: pd.DataFrame,
    labels: list[str],
    threshold: float,
) -> dict:
    metrics: dict[str, object] = {"per_label": {}}
    macro_f1: list[float] = []

    for label in labels:
        if label not in data_df.columns:
            continue

        y_true = data_df[label].astype(int).values
        y_score = score_df[label].values
        y_pred = (y_score >= threshold).astype(int)

        label_metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            label_metrics["ap"] = float(average_precision_score(y_true, y_score))

        metrics["per_label"][label] = label_metrics
        macro_f1.append(label_metrics["f1"])

    if macro_f1:
        metrics["macro_f1"] = float(np.mean(macro_f1))
    return metrics


def main() -> None:
    args = parse_args()

    models_cfg = load_json(args.models_config)
    labels_cfg = load_json(args.labels_config)

    models: list[str] = models_cfg["models"]
    labels: list[str] = labels_cfg["labels"]
    templates: list[str] = labels_cfg["prompt_templates"]

    data_df = pd.read_csv(args.dataset_csv)
    if args.max_images > 0:
        data_df = data_df.head(args.max_images)

    missing = [col for col in ["image_path"] if col not in data_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset CSV: {missing}")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"zeroshot_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    prompts, owners = build_prompt_grid(labels, templates)

    summary: dict[str, object] = {
        "run_id": run_id,
        "models": {},
        "n_images": int(len(data_df)),
    }

    for model_id in models:
        print(f"[Benchmark] Loading {model_id}")
        adapter = HFVisionTextAdapter(model_id=model_id)
        image_scores: list[dict[str, float]] = []

        for row in data_df.itertuples(index=False):
            image_path = Path(row.image_path)
            with Image.open(image_path).convert("RGB") as image:
                result = adapter.score_image_texts(image=image, texts=prompts)

            label_best: dict[str, float] = {label: -1e9 for label in labels}
            for idx, score in enumerate(result.logits):
                owner = owners[idx]
                if score > label_best[owner]:
                    label_best[owner] = float(score)

            image_scores.append(label_best)

        score_df = pd.DataFrame(image_scores)
        score_df.insert(0, "image_path", data_df["image_path"].values)

        model_tag = model_id.replace("/", "__")
        pred_path = run_root / f"predictions_{model_tag}.parquet"
        score_df.to_parquet(pred_path, index=False)

        model_metrics = evaluate_predictions(
            score_df=score_df,
            data_df=data_df,
            labels=labels,
            threshold=args.score_threshold,
        )

        summary["models"][model_id] = {
            "predictions": str(pred_path),
            "metrics": model_metrics,
        }
        print(f"[Benchmark] Done {model_id}")

    summary_path = run_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
