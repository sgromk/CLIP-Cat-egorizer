from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create model-level diagnostic plots from merged selected runs")
    parser.add_argument("--merged-dir", default="artifacts/runs/trackABC_merged_selected")
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace(":", "__")
        .replace("=>", "_to_")
        .replace(" ", "_")
    )


def short_model_name(model_id: str) -> str:
    if model_id.startswith("trackB::"):
        return "Track B fine-tuned"
    if model_id.startswith("linear_map::"):
        return "Track C linear map"
    return model_id


def class_names_from_predictions(pred_df: pd.DataFrame) -> list[str]:
    base_cols = {"source_shard", "source_row_idx", "true_label", "pred_label"}
    return [c for c in pred_df.columns if c not in base_cols]


def plot_accuracy_bars(df: pd.DataFrame, out_dir: Path) -> None:
    for target in sorted(df["target"].unique().tolist()):
        tdf = df[df["target"] == target].copy().sort_values("top1_accuracy", ascending=False)
        if tdf.empty:
            continue

        tdf["model_short"] = tdf["model_id"].astype(str).apply(short_model_name)

        plt.figure(figsize=(12, max(4, 0.45 * len(tdf))))
        sns.barplot(data=tdf, y="model_short", x="top1_accuracy", color="#4C78A8", edgecolor="black")
        plt.title(f"Top-1 Accuracy by Model ({target})")
        plt.xlabel("Top-1 Accuracy")
        plt.ylabel("Model")
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / f"accuracy_top1_by_model_{target}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(12, max(4, 0.45 * len(tdf))))
        sns.barplot(data=tdf, y="model_short", x="top5_accuracy", color="#72B7B2", edgecolor="black")
        plt.title(f"Top-5 Accuracy by Model ({target})")
        plt.xlabel("Top-5 Accuracy")
        plt.ylabel("Model")
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / f"accuracy_top5_by_model_{target}.png", dpi=180)
        plt.close()


def confusion_matrix_norm(true_labels: np.ndarray, pred_labels: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t, p] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def plot_confusion_for_row(row: pd.Series, out_dir: Path) -> str | None:
    pred_path = Path(str(row["predictions_path"]))
    if not pred_path.exists():
        return None

    pred_df = pd.read_parquet(pred_path)
    if pred_df.empty or "true_label" not in pred_df.columns or "pred_label" not in pred_df.columns:
        return None

    classes = class_names_from_predictions(pred_df)
    n_classes = len(classes)
    if n_classes == 0:
        max_label = int(max(pred_df["true_label"].max(), pred_df["pred_label"].max()))
        n_classes = max_label + 1
        classes = [str(i) for i in range(n_classes)]

    true_labels = pred_df["true_label"].astype(int).to_numpy()
    pred_labels = pred_df["pred_label"].astype(int).to_numpy()
    cm = confusion_matrix_norm(true_labels, pred_labels, n_classes)

    model_id = str(row["model_id"])
    target = str(row["target"])
    file_stub = sanitize_name(f"confusion_{target}_{model_id}")

    fig_w = max(8, min(22, 0.35 * n_classes))
    fig_h = max(6, min(20, 0.35 * n_classes))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(cm, cmap="magma", vmin=0, vmax=1, cbar=True)
    plt.title(f"Confusion Matrix (row-normalized)\n{target} | {short_model_name(model_id)}")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    out_path = out_dir / f"{file_stub}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return str(out_path)


def select_confusion_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []

    for target in sorted(df["target"].unique().tolist()):
        tdf = df[df["target"] == target].copy()
        if tdf.empty:
            continue

        best_track_a = tdf[~tdf["model_id"].str.startswith(("trackB::", "linear_map::"), na=False)]
        if not best_track_a.empty:
            rows.append(best_track_a.sort_values("top1_accuracy", ascending=False).iloc[0])

        best_track_b = tdf[tdf["model_id"].str.startswith("trackB::", na=False)]
        if not best_track_b.empty:
            rows.append(best_track_b.sort_values("top1_accuracy", ascending=False).iloc[0])

        best_track_c = tdf[tdf["model_id"].str.startswith("linear_map::", na=False)]
        if not best_track_c.empty:
            rows.append(best_track_c.sort_values("top1_accuracy", ascending=False).iloc[0])

    if not rows:
        return pd.DataFrame(columns=df.columns)

    selected = pd.DataFrame(rows).drop_duplicates(subset=["target", "model_id"])
    return selected.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    merged_dir = Path(args.merged_dir)
    merged_csv = merged_dir / "all_runs_metrics_long.csv"
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing merged csv: {merged_csv}")

    df = pd.read_csv(merged_csv)
    for col in ["top1_accuracy", "top5_accuracy", "n_images", "images_per_second", "elapsed_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["top1_accuracy", "top5_accuracy", "model_id", "target", "predictions_path"])
    if df.empty:
        raise RuntimeError("No usable rows in merged csv")

    figures_dir = merged_dir / "figures_diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_bars(df, figures_dir)

    selected = select_confusion_rows(df)
    generated = []
    for _, row in selected.iterrows():
        out = plot_confusion_for_row(row, figures_dir)
        if out:
            generated.append(
                {
                    "target": row["target"],
                    "model_id": row["model_id"],
                    "top1_accuracy": row["top1_accuracy"],
                    "top5_accuracy": row["top5_accuracy"],
                    "n_images": row["n_images"],
                    "confusion_path": out,
                }
            )

    pd.DataFrame(generated).to_csv(figures_dir / "confusion_index.csv", index=False)
    df.sort_values(["target", "top1_accuracy"], ascending=[True, False]).to_csv(
        figures_dir / "diagnostic_source_table.csv", index=False
    )

    print(f"Saved diagnostic figures to: {figures_dir}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk")
    main()
