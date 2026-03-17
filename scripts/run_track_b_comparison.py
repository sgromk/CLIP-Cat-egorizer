"""
run_track_b_comparison.py
-------------------------
Track B orchestrator:
  1) Evaluates a fine-tuned CLIP checkpoint on style target
  2) Writes raw metrics/predictions under run folder
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
    parser = argparse.ArgumentParser(description="Run Track B checkpoint evaluation end-to-end")
    parser.add_argument("--split-csv", default="data/labeled/test.csv")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode")
    parser.add_argument("--streaming-split", default="train", help="Dataset split for streaming mode")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/trackB_laion_vitl14")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--run-id", default="", help="Optional custom run id; default auto-generated")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def model_tag(checkpoint_dir: Path) -> str:
    return checkpoint_dir.name.replace("/", "__")


def run_benchmark(args: argparse.Namespace, run_dir: Path) -> None:
    target_dir = run_dir / "style"
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/benchmark_track_b_finetuned.py",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--target",
        "style",
        "--output-dir",
        str(target_dir),
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
            "--streaming-cache-path",
            str(run_dir / f"streaming_cache_{args.streaming_split}_{args.max_images if args.max_images > 0 else 'all'}.parquet"),
        ])
    else:
        cmd.extend([
            "--split-csv",
            args.split_csv,
        ])

    print(f"\n[run] target=style -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_metrics_row(run_dir: Path, checkpoint_dir: Path) -> list[dict]:
    rows: list[dict] = []
    target_dir = run_dir / "style"
    tag = model_tag(checkpoint_dir)
    mpath = target_dir / f"metrics_{tag}.json"
    if not mpath.exists():
        return rows

    data = json.loads(mpath.read_text())
    rows.append(
        {
            "run_id": run_dir.name,
            "target": "style",
            "model_id": f"trackB::{checkpoint_dir.name}",
            "n_images": data.get("n_images"),
            "top1_accuracy": data.get("top1_accuracy"),
            "top5_accuracy": data.get("top5_accuracy"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "images_per_second": data.get("images_per_second"),
            "metrics_path": str(mpath),
            "predictions_path": str(target_dir / f"predictions_{tag}.parquet"),
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
        "checkpoint_dir": args.checkpoint_dir,
        "targets": ["style"],
        "max_images": args.max_images,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(long_df)),
        "best_per_target": per_target_best.to_dict(orient="records"),
        "files": {
            "long_csv": str(long_path),
            "leaderboard_csv": str(leader_path),
        },
    }
    summary_json_path = summary_dir / "track_b_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"trackB_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "split_csv": args.split_csv,
        "streaming_repo_id": args.streaming_repo_id,
        "streaming_split": args.streaming_split,
        "checkpoint_dir": args.checkpoint_dir,
        "max_images": args.max_images,
        "log_every": args.log_every,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    run_benchmark(args, run_dir)

    rows = load_metrics_row(run_dir, checkpoint_dir=Path(args.checkpoint_dir))
    write_summary_outputs(run_dir, args, rows)

    print("\n=== Track B complete ===")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {run_dir / 'summary' / 'track_b_summary.json'}")


if __name__ == "__main__":
    main()
