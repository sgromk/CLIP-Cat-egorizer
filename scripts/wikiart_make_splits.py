"""
wikiart_make_splits.py
----------------------
Stratified train / val / test split from the deduped unique index.

Outputs (all in --output-dir):
  train.csv, val.csv, test.csv     — split metadata rows
  data_card.json                   — per-split stats for reporting
  split_distribution.csv           — style/genre/artist counts across splits
  plots/split_style_bar.png        — stacked bar: style counts by split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified train/val/test splits from unique index")
    parser.add_argument("--unique-index", default="artifacts/data_audit/wikiart_unique_index.parquet")
    parser.add_argument("--output-dir", default="data/labeled")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Fraction for validation (default 10%%)")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Fraction for test (default 10%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-col", default="style", help="Column to stratify on (style, genre, or artist)")
    return parser.parse_args()


def safe_stratify(df: pd.DataFrame, col: str, min_samples: int = 2) -> pd.Series | None:
    """Return stratify series only if every class has >= min_samples rows."""
    counts = df[col].value_counts()
    if (counts < min_samples).any():
        print(f"[warn] Some {col} classes have < {min_samples} samples — skipping stratification")
        return None
    return df[col]


def split_stats(df: pd.DataFrame, split_name: str) -> dict:
    return {
        "split": split_name,
        "n": len(df),
        "styles": int(df["style"].nunique()),
        "genres": int(df["genre"].nunique()),
        "artists": int(df["artist"].nunique()),
        "orientation_counts": df["orientation"].value_counts().to_dict(),
        "style_counts": df["style"].value_counts().sort_index().to_dict(),
        "genre_counts": df["genre"].value_counts().sort_index().to_dict(),
        "mean_megapixels": round(float(df["megapixels"].mean()), 4),
        "mean_bytes": round(float(df["bytes_size"].mean()), 0),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.unique_index)
    print(f"Loaded {len(df):,} unique rows from {args.unique_index}")

    total = len(df)
    val_frac = args.val_frac
    test_frac = args.test_frac
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be less than 1.0")

    # ── Step 1: carve out test from full set ──────────────────────────────
    stratify_full = safe_stratify(df, args.stratify_col, min_samples=2)
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=args.seed,
        stratify=stratify_full,
    )

    # ── Step 2: carve out val from trainval ──────────────────────────────
    # val_frac recalculated relative to trainval size
    val_frac_of_trainval = val_frac / (1.0 - test_frac)
    stratify_trainval = safe_stratify(trainval_df, args.stratify_col, min_samples=2)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_frac_of_trainval,
        random_state=args.seed,
        stratify=stratify_trainval,
    )

    print(f"Split sizes → train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = out_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        print(f"Saved {path}")

    # ── Data card ─────────────────────────────────────────────────────────
    card = {
        "source": args.unique_index,
        "stratify_col": args.stratify_col,
        "seed": args.seed,
        "total_unique": total,
        "fractions": {
            "train": round(train_frac, 4),
            "val": round(val_frac, 4),
            "test": round(test_frac, 4),
        },
        "splits": {
            "train": split_stats(train_df, "train"),
            "val": split_stats(val_df, "val"),
            "test": split_stats(test_df, "test"),
        },
    }
    card_path = out_dir / "data_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    print(f"Saved data card → {card_path}")

    # ── Distribution CSV ──────────────────────────────────────────────────
    rows = []
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for style_id, count in split["style"].value_counts().sort_index().items():
            rows.append({"split": name, "style": int(style_id), "count": int(count)})
    dist_df = pd.DataFrame(rows)
    dist_path = out_dir / "split_distribution.csv"
    dist_df.to_csv(dist_path, index=False)

    # ── Stacked bar plot ──────────────────────────────────────────────────
    pivot = dist_df.pivot_table(index="style", columns="split", values="count", fill_value=0)
    # keep consistent column order
    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]].sort_index()

    ax = pivot.plot(kind="bar", stacked=True, figsize=(14, 5), color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("Style Distribution Across Splits")
    ax.set_xlabel("Style ID")
    ax.set_ylabel("Image Count")
    ax.legend(title="Split")
    plt.tight_layout()
    plot_path = plot_dir / "split_style_bar.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved split distribution plot → {plot_path}")

    # ── Summary print ─────────────────────────────────────────────────────
    print("\n=== Split Summary ===")
    for name in ("train", "val", "test"):
        s = card["splits"][name]
        print(f"  {name:5s}: {s['n']:6,} rows | {s['styles']} styles | {s['genres']} genres | {s['artists']} artists")


if __name__ == "__main__":
    main()
