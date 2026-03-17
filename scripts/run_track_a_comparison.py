"""
run_track_a_comparison.py
-------------------------
Track A orchestrator:
  1) Runs full provider comparison for style + genre targets
  2) Saves per-target benchmark artifacts
  3) Aggregates slide-ready summary tables (CSV/JSON/Markdown)

Example:
    python scripts/run_track_a_comparison.py \
      --split-csv data/labeled/test.csv \
      --models-config configs/zero_shot_models.json \
      --output-root artifacts/runs \
      --templates-per-class 4

Optional smoke run:
    python scripts/run_track_a_comparison.py \
      --split-csv data/labeled/test.csv \
      --models-config configs/zero_shot_models.json \
      --output-root artifacts/runs \
      --targets style genre \
      --templates-per-class 1 \
      --max-images 32
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
    parser = argparse.ArgumentParser(description="Run Track A provider comparison end-to-end")
    parser.add_argument("--split-csv", default="data/labeled/test.csv")
    parser.add_argument("--streaming-repo-id", default="", help="Optional HF dataset repo for streaming mode (e.g. huggan/wikiart)")
    parser.add_argument("--streaming-split", default="train", help="Dataset split used in streaming mode")
    parser.add_argument("--models-config", default="configs/zero_shot_models.json")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--targets", nargs="+", default=["style", "genre"], choices=["style", "genre"])
    parser.add_argument("--templates-per-class", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--run-id", default="", help="Optional custom run id; default auto-generated")
    return parser.parse_args()


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


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
        "scripts/benchmark_wikiart_zero_shot.py",
        "--models-config",
        args.models_config,
        "--output-dir",
        str(target_dir),
        "--target",
        target,
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

    print(f"\n[run] target={target} -> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_models(models_config: Path) -> list[str]:
    data = json.loads(models_config.read_text())
    return list(data["models"])


def load_metrics_rows(run_dir: Path, models: list[str], targets: list[str]) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        target_dir = run_dir / target
        for model_id in models:
            mpath = target_dir / f"metrics_{model_tag(model_id)}.json"
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
                    "predictions_path": str(target_dir / f"predictions_{model_tag(model_id)}.parquet"),
                }
            )
    return rows


def write_slide_ready_outputs(run_dir: Path, args: argparse.Namespace, rows: list[dict]) -> None:
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
        "split_csv": args.split_csv,
        "streaming_repo_id": args.streaming_repo_id,
        "streaming_split": args.streaming_split,
        "models_config": args.models_config,
        "targets": args.targets,
        "templates_per_class": args.templates_per_class,
        "max_images": args.max_images,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(long_df)),
        "best_per_target": per_target_best.to_dict(orient="records"),
        "files": {
            "long_csv": str(long_path),
            "leaderboard_csv": str(leader_path),
        },
    }
    summary_json_path = summary_dir / "track_a_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2))

    md_lines = [
        "# Track A Provider Comparison Summary",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Split: `{args.split_csv}`",
        f"- Streaming repo: `{args.streaming_repo_id or 'none'}`",
        f"- Streaming split: `{args.streaming_split}`",
        f"- Targets: `{', '.join(args.targets)}`",
        f"- Prompt templates per class: `{args.templates_per_class}`",
        f"- Max images: `{args.max_images}` (0 means full split)",
        "",
        "## Best model per target (Top-1)",
        "",
        "| Target | Model | Top-1 | Top-5 | Img/s | Elapsed (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in per_target_best.iterrows():
        md_lines.append(
            f"| {row['target']} | {row['model_id']} | {row['top1_accuracy']:.4f} | "
            f"{row['top5_accuracy']:.4f} | {row['images_per_second']:.2f} | {row['elapsed_seconds']:.1f} |"
        )

    md_lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- Long metrics table: `{long_path}`",
            f"- Leaderboard table: `{leader_path}`",
            f"- JSON summary: `{summary_json_path}`",
            "",
            "Per-target raw artifacts remain under:",
            f"- `{run_dir / 'style'}`",
            f"- `{run_dir / 'genre'}`",
        ]
    )
    (summary_dir / "track_a_slides.md").write_text("\n".join(md_lines))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"trackA_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "split_csv": args.split_csv,
        "streaming_repo_id": args.streaming_repo_id,
        "streaming_split": args.streaming_split,
        "models_config": args.models_config,
        "targets": args.targets,
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

    models = load_models(Path(args.models_config))
    rows = load_metrics_rows(run_dir, models=models, targets=args.targets)
    write_slide_ready_outputs(run_dir, args, rows)

    print("\n=== Track A complete ===")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {run_dir / 'summary' / 'track_a_summary.json'}")


if __name__ == "__main__":
    main()
