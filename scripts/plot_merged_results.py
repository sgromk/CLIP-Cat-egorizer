from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create slide-ready figures from merged run results")
    parser.add_argument("--merged-dir", default="artifacts/runs/trackABC_merged")
    parser.add_argument("--min-full-images", type=int, default=1000)
    return parser.parse_args()


def pretty_model_name(model_id: str) -> str:
    if model_id.startswith("trackB::"):
        return "Track B (Fine-tuned CLIP)"
    if model_id.startswith("linear_map::"):
        return "Track C (Linear Map)"
    return f"Track A ({model_id.split('/')[-1]})"


def run_stage_label(run_id: str, min_full_images: int, n_images: int) -> str:
    if "smoke" in run_id.lower() or n_images < min_full_images:
        return "Smoke"
    return "Full"


def save_top1_bar(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = df.copy().sort_values(["target", "top1_accuracy"], ascending=[True, False])
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="target",
        y="top1_accuracy",
        hue="model_pretty",
        palette="Set2",
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(0, 1.0)
    plt.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_top5_bar(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = df.copy().sort_values(["target", "top5_accuracy"], ascending=[True, False])
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="target",
        y="top5_accuracy",
        hue="model_pretty",
        palette="Pastel1",
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Top-5 Accuracy")
    plt.ylim(0, 1.0)
    plt.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_speed_accuracy(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="images_per_second",
        y="top1_accuracy",
        hue="model_pretty",
        style="target",
        s=130,
        palette="Dark2",
    )
    for _, row in df.iterrows():
        plt.text(
            row["images_per_second"] + 0.05,
            row["top1_accuracy"] + 0.01,
            row["target"],
            fontsize=8,
        )
    plt.title(title)
    plt.xlabel("Images / second")
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="model_pretty", columns="target", values="top1_accuracy", aggfunc="max")
    plt.figure(figsize=(8, max(3.5, 0.7 * len(pivot.index))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5)
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    merged_dir = Path(args.merged_dir)
    csv_path = merged_dir / "all_runs_metrics_long.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing merged metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    numeric_cols = ["n_images", "top1_accuracy", "top5_accuracy", "images_per_second", "elapsed_seconds"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["top1_accuracy", "top5_accuracy", "images_per_second", "n_images"])
    if df.empty:
        raise RuntimeError("No valid metric rows found in merged file")

    df["model_pretty"] = df["model_id"].astype(str).apply(pretty_model_name)
    df["stage"] = df.apply(lambda r: run_stage_label(str(r["run_id"]), args.min_full_images, int(r["n_images"])), axis=1)

    # Keep best row per run_id+target+model to avoid duplicate noise
    df = (
        df.sort_values(["run_id", "target", "model_id", "top1_accuracy", "top5_accuracy"], ascending=[True, True, True, False, False])
        .groupby(["run_id", "target", "model_id"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    figures_dir = merged_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    full_df = df[df["stage"] == "Full"].copy()
    if full_df.empty:
        full_df = df.copy()

    save_top1_bar(
        full_df,
        figures_dir / "top1_by_target.png",
        "Top-1 Accuracy by Target (Final Runs)",
    )
    save_top5_bar(
        full_df,
        figures_dir / "top5_by_target.png",
        "Top-5 Accuracy by Target (Final Runs)",
    )
    save_speed_accuracy(
        full_df,
        figures_dir / "speed_vs_top1.png",
        "Speed vs Top-1 Accuracy (Final Runs)",
    )
    save_heatmap(
        full_df,
        figures_dir / "top1_heatmap.png",
        "Top-1 Accuracy Heatmap (Final Runs)",
    )

    summary_cols = [
        "run_id",
        "target",
        "model_id",
        "model_pretty",
        "stage",
        "n_images",
        "top1_accuracy",
        "top5_accuracy",
        "images_per_second",
        "elapsed_seconds",
    ]
    full_df.sort_values(["target", "top1_accuracy"], ascending=[True, False])[summary_cols].to_csv(
        figures_dir / "figure_source_table.csv", index=False
    )

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk")
    main()
