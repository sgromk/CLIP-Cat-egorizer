"""
run_track_c_comparison.py
-------------------------
Track C orchestrator:
  1) Evaluates linear-map baseline for style + genre
  2) Writes per-target metrics artifacts
  3) Produces Track A-compatible summary tables for merged slides
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track C baseline comparison end-to-end")
    parser.add_argument("--split-csv", default="data/labeled/test.csv")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode")
    parser.add_argument("--streaming-split", default="train", help="Dataset split for streaming mode")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--run-id", default="", help="Optional custom run id; default auto-generated trackC timestamp")
    parser.add_argument("--targets", nargs="+", default=["style", "genre"], choices=["style", "genre"])
    parser.add_argument("--linear-map-path", default="artifacts/linear_map/linear_map_W.npy")
    parser.add_argument("--linear-map-meta", default="")
    parser.add_argument("--image-model-id", default="")
    parser.add_argument("--text-model-id", default="")
    parser.add_argument("--templates-per-class", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def track_c_model_id(args: argparse.Namespace, meta: dict) -> str:
    image_model_id = args.image_model_id or meta.get("image_model_id") or "openai/clip-vit-base-patch32"
    text_model_id = args.text_model_id or meta.get("text_model_id") or "sentence-transformers/all-MiniLM-L6-v2"
    return f"linear_map::{image_model_id}=>{text_model_id}"


def run_benchmark_for_target(
    args: argparse.Namespace,
    target: str,
    run_dir: Path,
    streaming_cache_path: Path | None = None,
) -> None:
    target_dir = run_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/benchmark_linear_map.py",
        "--target",
        target,
        "--output-dir",
        str(target_dir),
        "--linear-map-path",
        args.linear_map_path,
        "--templates-per-class",
        str(args.templates_per_class),
        "--max-images",
        str(args.max_images),
        "--log-every",
        str(args.log_every),
    ]

    if args.streaming_repo_id.strip():
        cmd.extend([
            "--streaming-repo-id",
            args.streaming_repo_id.strip(),
            "--streaming-split",
            args.streaming_split,
        ])
        if streaming_cache_path is not None:
            cmd.extend([
                "--streaming-cache-path",
                str(streaming_cache_path),
            ])
    else:
        cmd.extend([
            "--split-csv",
            args.split_csv,
        ])

    if args.linear_map_meta.strip():
        cmd.extend(["--linear-map-meta", args.linear_map_meta.strip()])
    if args.image_model_id.strip():
        cmd.extend(["--image-model-id", args.image_model_id.strip()])
    if args.text_model_id.strip():
        cmd.extend(["--text-model-id", args.text_model_id.strip()])

    print(f"\n[run] target={target} -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def model_tag_for_files(args: argparse.Namespace, meta: dict) -> str:
    image_model_id = args.image_model_id or meta.get("image_model_id") or "openai/clip-vit-base-patch32"
    text_model_id = args.text_model_id or meta.get("text_model_id") or "sentence-transformers/all-MiniLM-L6-v2"
    return (
        "linear_map__"
        + image_model_id.replace("/", "__")
        + "__to__"
        + text_model_id.replace("/", "__")
    )


def load_metrics_rows(run_dir: Path, model_id: str, model_tag: str, targets: list[str]) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        target_dir = run_dir / target
        mpath = target_dir / f"metrics_{model_tag}.json"
        if not mpath.exists():
            continue
        data = json.loads(mpath.read_text())
        rows.append(
            {
                "run_id": run_dir.name,
                "target": target,
                "model_id": model_id,
                "n_images": data.get("n_images"),
                "top1_accuracy": data.get("top1_accuracy"),
                "top5_accuracy": data.get("top5_accuracy"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "images_per_second": data.get("images_per_second"),
                "metrics_path": str(mpath),
                "predictions_path": str(target_dir / f"predictions_{model_tag}.parquet"),
            }
        )
    return rows


def write_summary_outputs(run_dir: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.DataFrame(rows)
    long_path = summary_dir / "track_a_metrics_long.csv"
    long_df.to_csv(long_path, index=False)

    if long_df.empty:
        print("[warn] No metrics found to aggregate")
        return

    leader_df = (
        long_df.sort_values(["target", "top1_accuracy", "top5_accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    leader_path = summary_dir / "track_a_leaderboard.csv"
    leader_df.to_csv(leader_path, index=False)

    per_target_best = (
        leader_df.groupby("target", as_index=False)
        .first()[["target", "model_id", "top1_accuracy", "top5_accuracy", "images_per_second", "elapsed_seconds"]]
    )

    summary_json = {
        "run_id": run_dir.name,
        "split_csv": args.split_csv if not args.streaming_repo_id.strip() else None,
        "streaming_repo_id": args.streaming_repo_id if args.streaming_repo_id.strip() else None,
        "streaming_split": args.streaming_split if args.streaming_repo_id.strip() else None,
        "targets": args.targets,
        "templates_per_class": args.templates_per_class,
        "max_images": args.max_images,
        "linear_map_path": args.linear_map_path,
        "linear_map_meta": args.linear_map_meta,
        "image_model_id": args.image_model_id,
        "text_model_id": args.text_model_id,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(long_df)),
        "best_per_target": per_target_best.to_dict(orient="records"),
        "files": {
            "long_csv": str(long_path),
            "leaderboard_csv": str(leader_path),
        },
    }
    summary_json_path = summary_dir / "track_c_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2))


def load_meta(args: argparse.Namespace) -> dict:
    if args.linear_map_meta.strip():
        meta_path = Path(args.linear_map_meta.strip())
    else:
        meta_path = Path(args.linear_map_path).with_name("linear_map_meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"trackC_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "split_csv": args.split_csv,
        "streaming_repo_id": args.streaming_repo_id,
        "streaming_split": args.streaming_split,
        "targets": args.targets,
        "linear_map_path": args.linear_map_path,
        "linear_map_meta": args.linear_map_meta,
        "image_model_id": args.image_model_id,
        "text_model_id": args.text_model_id,
        "templates_per_class": args.templates_per_class,
        "max_images": args.max_images,
        "log_every": args.log_every,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    streaming_cache_path = None
    if args.streaming_repo_id.strip():
        max_part = args.max_images if args.max_images > 0 else "all"
        streaming_cache_path = run_dir / f"streaming_cache_{args.streaming_split}_{max_part}.parquet"

    for target in args.targets:
        run_benchmark_for_target(args, target, run_dir, streaming_cache_path=streaming_cache_path)

    meta = load_meta(args)
    model_id = track_c_model_id(args, meta)
    tag = model_tag_for_files(args, meta)
    rows = load_metrics_rows(run_dir, model_id=model_id, model_tag=tag, targets=args.targets)
    write_summary_outputs(run_dir, args, rows)

    print("\n=== Track C complete ===")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {run_dir / 'summary' / 'track_c_summary.json'}")


if __name__ == "__main__":
    main()
