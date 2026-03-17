"""
merge_track_a_runs.py
---------------------
Merge multiple Track A run summaries into one final comparison bundle.

Scans artifacts/runs/trackA_*/summary/track_a_metrics_long.csv and writes:
  - all_runs_metrics_long.csv
  - all_runs_leaderboard.csv
  - best_per_target.csv
  - final_slides_table.csv
  - merge_summary.json

Example:
    python scripts/merge_track_a_runs.py \
      --runs-root artifacts/runs \
      --output-dir artifacts/runs/trackA_merged
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Track A runs into final slide-ready tables")
    parser.add_argument("--runs-root", default="artifacts/runs", help="Root containing trackA_* run folders")
    parser.add_argument("--output-dir", default="artifacts/runs/trackA_merged", help="Where merged outputs are written")
    parser.add_argument(
        "--run-globs",
        nargs="+",
        default=["trackA_*"],
        help="Run folder globs to include (e.g. trackA_* trackC_*)",
    )
    parser.add_argument(
        "--prefer-latest",
        action="store_true",
        help="When selecting one final row per target+model, prefer latest run_id instead of best top1",
    )
    parser.add_argument(
        "--exact-n-images",
        type=int,
        default=0,
        help="If >0, keep only rows where n_images exactly matches this value (e.g. 11320)",
    )
    parser.add_argument(
        "--require-track-prefixes",
        nargs="+",
        default=[],
        help="Optional run_id prefixes that must be present after filtering (e.g. trackA_ trackB_ trackC_)",
    )
    return parser.parse_args()


def read_run_metrics(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "summary" / "track_a_metrics_long.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "run_id" not in df.columns:
        df["run_id"] = run_dir.name
    return df


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs_set: set[Path] = set()
    for pattern in args.run_globs:
        for path in runs_root.glob(pattern):
            if path.is_dir():
                run_dirs_set.add(path)
    run_dirs = sorted(run_dirs_set)
    all_frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        df = read_run_metrics(run_dir)
        if df is None or df.empty:
            continue
        all_frames.append(df)

    if not all_frames:
        raise SystemExit(
            f"No track_a_metrics_long.csv files found under run globs: {', '.join(args.run_globs)}"
        )

    all_df = pd.concat(all_frames, ignore_index=True)

    # Normalize expected numeric columns
    for col in ["top1_accuracy", "top5_accuracy", "images_per_second", "elapsed_seconds", "n_images"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    if args.exact_n_images > 0:
        all_df = all_df[all_df["n_images"] == args.exact_n_images].reset_index(drop=True)
        if all_df.empty:
            raise SystemExit(
                f"No rows remain after applying --exact-n-images {args.exact_n_images}. "
                "Check whether full-set run artifacts are available under the selected run globs."
            )

    if args.require_track_prefixes:
        present_run_ids = set(all_df["run_id"].astype(str).unique().tolist())
        missing_prefixes: list[str] = []
        for prefix in args.require_track_prefixes:
            if not any(run_id.startswith(prefix) for run_id in present_run_ids):
                missing_prefixes.append(prefix)
        if missing_prefixes:
            available = ", ".join(sorted(present_run_ids)) if present_run_ids else "(none)"
            raise SystemExit(
                "Missing required track prefixes after filtering: "
                f"{', '.join(missing_prefixes)}. Available run_ids: {available}"
            )

    all_long_path = output_dir / "all_runs_metrics_long.csv"
    all_df.to_csv(all_long_path, index=False)

    # Leaderboard across all runs (highest top1 then top5 per target)
    leaderboard_df = all_df.sort_values(
        ["target", "top1_accuracy", "top5_accuracy", "images_per_second"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    leaderboard_path = output_dir / "all_runs_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    # Keep one row per target+model for deck table
    if args.prefer_latest:
        final_df = (
            all_df.sort_values(["target", "model_id", "run_id"])  # run_id encodes timestamp
            .groupby(["target", "model_id"], as_index=False)
            .tail(1)
            .sort_values(["target", "top1_accuracy", "top5_accuracy"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
    else:
        final_df = (
            all_df.sort_values(["target", "model_id", "top1_accuracy", "top5_accuracy"], ascending=[True, True, False, False])
            .groupby(["target", "model_id"], as_index=False)
            .head(1)
            .sort_values(["target", "top1_accuracy", "top5_accuracy"], ascending=[True, False, False])
            .reset_index(drop=True)
        )

    final_path = output_dir / "final_slides_table.csv"
    final_df.to_csv(final_path, index=False)

    # Best model per target for quick headline slide
    best_target_df = (
        final_df.sort_values(["target", "top1_accuracy", "top5_accuracy"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_target_path = output_dir / "best_per_target.csv"
    best_target_df.to_csv(best_target_path, index=False)

    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "runs_root": str(runs_root),
        "run_globs": args.run_globs,
        "n_runs_scanned": len(run_dirs),
        "n_rows_all": int(len(all_df)),
        "n_rows_final": int(len(final_df)),
        "selection_mode": "latest" if args.prefer_latest else "best_top1",
        "exact_n_images": args.exact_n_images,
        "require_track_prefixes": args.require_track_prefixes,
        "files": {
            "all_runs_metrics_long": str(all_long_path),
            "all_runs_leaderboard": str(leaderboard_path),
            "final_slides_table": str(final_path),
            "best_per_target": str(best_target_path),
        },
        "best_per_target": best_target_df[["target", "model_id", "top1_accuracy", "top5_accuracy"]].to_dict(orient="records"),
    }
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=== Track A merge complete ===")
    print(f"Rows merged: {len(all_df)}")
    print(f"Final slides table: {final_path}")
    print(f"Best per target: {best_target_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
