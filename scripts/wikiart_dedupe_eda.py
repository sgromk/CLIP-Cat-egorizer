from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from PIL import Image

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop exact duplicates and generate detailed WikiArt EDA artifacts"
    )
    parser.add_argument("--wikiart-dir", default="data/raw/wikiart")
    parser.add_argument("--output-dir", default="artifacts/data_audit")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tail-threshold", type=int, default=50)
    return parser.parse_args()


def quantiles(values: list[float], qs: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {f"q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


def orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def normalized_entropy(counter: Counter[int]) -> float:
    if not counter:
        return 0.0
    counts = np.array(list(counter.values()), dtype=np.float64)
    probs = counts / counts.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    max_entropy = np.log(len(counter))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def gini_from_counts(counter: Counter[int]) -> float:
    if not counter:
        return 0.0
    x = np.array(sorted(counter.values()), dtype=np.float64)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))


def save_plot_top_counts(counter: Counter[int], title: str, out_path: Path, top_k: int) -> None:
    if not counter:
        return
    top = counter.most_common(top_k)
    ids = [str(item[0]) for item in top][::-1]
    vals = [item[1] for item in top][::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=vals, y=ids, orient="h")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("ID")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plot_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 60) -> None:
    if series.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(series, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def counter_to_df(counter: Counter[int], id_name: str) -> pd.DataFrame:
    rows = [{id_name: int(k), "count": int(v)} for k, v in counter.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    root = Path(args.wikiart_dir)
    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {root}")

    seen_hash_to_unique_idx: dict[str, int] = {}
    hash_counts: Counter[str] = Counter()

    all_style_counter: Counter[int] = Counter()
    all_genre_counter: Counter[int] = Counter()
    all_artist_counter: Counter[int] = Counter()

    unique_style_counter: Counter[int] = Counter()
    unique_genre_counter: Counter[int] = Counter()
    unique_artist_counter: Counter[int] = Counter()

    unique_records: list[dict[str, object]] = []
    duplicate_records: list[dict[str, object]] = []
    decode_errors: list[dict[str, object]] = []

    widths_all: list[int] = []
    heights_all: list[int] = []
    bytes_sizes_all: list[int] = []
    megapixels_all: list[float] = []
    aspect_ratios_all: list[float] = []
    orientation_counter_all: Counter[str] = Counter()

    widths_unique: list[int] = []
    heights_unique: list[int] = []
    bytes_sizes_unique: list[int] = []
    megapixels_unique: list[float] = []
    aspect_ratios_unique: list[float] = []
    orientation_counter_unique: Counter[str] = Counter()

    total_rows = 0

    for shard in shards:
        table = pq.read_table(shard, columns=["image", "artist", "genre", "style"])
        images = table["image"].to_pylist()
        artists = table["artist"].to_pylist()
        genres = table["genre"].to_pylist()
        styles = table["style"].to_pylist()

        for row_idx, image_row in enumerate(images):
            total_rows += 1

            artist = int(artists[row_idx])
            genre = int(genres[row_idx])
            style = int(styles[row_idx])

            all_artist_counter[artist] += 1
            all_genre_counter[genre] += 1
            all_style_counter[style] += 1

            image_bytes = image_row["bytes"]
            image_hash = hashlib.md5(image_bytes).hexdigest()
            hash_counts[image_hash] += 1

            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
            except Exception as exc:
                decode_errors.append(
                    {
                        "shard": str(shard),
                        "row_idx": row_idx,
                        "error": str(exc),
                    }
                )
                continue

            pixels = width * height
            mpix = pixels / 1_000_000.0
            ratio = width / max(height, 1)
            orient = orientation(width, height)

            widths_all.append(width)
            heights_all.append(height)
            bytes_sizes_all.append(len(image_bytes))
            megapixels_all.append(mpix)
            aspect_ratios_all.append(ratio)
            orientation_counter_all[orient] += 1

            if image_hash not in seen_hash_to_unique_idx:
                unique_idx = len(unique_records)
                seen_hash_to_unique_idx[image_hash] = unique_idx

                unique_artist_counter[artist] += 1
                unique_genre_counter[genre] += 1
                unique_style_counter[style] += 1

                widths_unique.append(width)
                heights_unique.append(height)
                bytes_sizes_unique.append(len(image_bytes))
                megapixels_unique.append(mpix)
                aspect_ratios_unique.append(ratio)
                orientation_counter_unique[orient] += 1

                unique_records.append(
                    {
                        "unique_idx": unique_idx,
                        "image_hash": image_hash,
                        "source_shard": str(shard),
                        "source_row_idx": row_idx,
                        "artist": artist,
                        "genre": genre,
                        "style": style,
                        "width": width,
                        "height": height,
                        "orientation": orient,
                        "aspect_ratio": round(ratio, 6),
                        "bytes_size": len(image_bytes),
                        "megapixels": round(mpix, 6),
                    }
                )
            else:
                duplicate_records.append(
                    {
                        "image_hash": image_hash,
                        "duplicate_shard": str(shard),
                        "duplicate_row_idx": row_idx,
                        "primary_unique_idx": seen_hash_to_unique_idx[image_hash],
                        "artist": artist,
                        "genre": genre,
                        "style": style,
                    }
                )

    unique_count = len(unique_records)
    duplicate_count = len(duplicate_records)
    duplicate_rate = duplicate_count / max(total_rows, 1)

    group_size_counter: Counter[int] = Counter(hash_counts.values())
    duplicate_groups = {size: count for size, count in sorted(group_size_counter.items()) if size > 1}

    duplicate_hashes_sorted = [
        {"image_hash": h, "count": c}
        for h, c in hash_counts.most_common(20)
        if c > 1
    ]

    unique_df = pd.DataFrame(unique_records)
    dup_df = pd.DataFrame(duplicate_records)
    unique_path = out_dir / "wikiart_unique_index.parquet"
    dup_path = out_dir / "wikiart_duplicates.csv"
    unique_df.to_parquet(unique_path, index=False)
    dup_df.to_csv(dup_path, index=False)

    summary = {
        "source": str(root),
        "shards": len(shards),
        "total_rows": total_rows,
        "unique_rows": unique_count,
        "duplicate_rows": duplicate_count,
        "duplicate_rate": round(float(duplicate_rate), 6),
        "duplicate_group_sizes": duplicate_groups,
        "top_duplicate_hashes": duplicate_hashes_sorted,
        "decode_errors": len(decode_errors),
        "style_coverage_all": len(all_style_counter),
        "genre_coverage_all": len(all_genre_counter),
        "artist_coverage_all": len(all_artist_counter),
        "style_coverage_unique": len(unique_style_counter),
        "genre_coverage_unique": len(unique_genre_counter),
        "artist_coverage_unique": len(unique_artist_counter),
        "class_balance": {
            "style_entropy_norm_all": normalized_entropy(all_style_counter),
            "genre_entropy_norm_all": normalized_entropy(all_genre_counter),
            "artist_entropy_norm_all": normalized_entropy(all_artist_counter),
            "style_entropy_norm_unique": normalized_entropy(unique_style_counter),
            "genre_entropy_norm_unique": normalized_entropy(unique_genre_counter),
            "artist_entropy_norm_unique": normalized_entropy(unique_artist_counter),
        },
        "orientation_all": dict(orientation_counter_all),
        "orientation_unique": dict(orientation_counter_unique),
        "size_stats_unique": {
            "width": {
                "min": int(min(widths_unique)) if widths_unique else 0,
                "max": int(max(widths_unique)) if widths_unique else 0,
                "mean": float(np.mean(widths_unique)) if widths_unique else 0.0,
                **quantiles(widths_unique, [0.25, 0.5, 0.75, 0.95]),
            },
            "height": {
                "min": int(min(heights_unique)) if heights_unique else 0,
                "max": int(max(heights_unique)) if heights_unique else 0,
                "mean": float(np.mean(heights_unique)) if heights_unique else 0.0,
                **quantiles(heights_unique, [0.25, 0.5, 0.75, 0.95]),
            },
            "bytes": {
                "min": int(min(bytes_sizes_unique)) if bytes_sizes_unique else 0,
                "max": int(max(bytes_sizes_unique)) if bytes_sizes_unique else 0,
                "mean": float(np.mean(bytes_sizes_unique)) if bytes_sizes_unique else 0.0,
                **quantiles(bytes_sizes_unique, [0.25, 0.5, 0.75, 0.95]),
            },
            "megapixels": {
                "min": float(min(megapixels_unique)) if megapixels_unique else 0.0,
                "max": float(max(megapixels_unique)) if megapixels_unique else 0.0,
                "mean": float(np.mean(megapixels_unique)) if megapixels_unique else 0.0,
                **quantiles(megapixels_unique, [0.25, 0.5, 0.75, 0.95]),
            },
            "aspect_ratio": {
                "min": float(min(aspect_ratios_unique)) if aspect_ratios_unique else 0.0,
                "max": float(max(aspect_ratios_unique)) if aspect_ratios_unique else 0.0,
                "mean": float(np.mean(aspect_ratios_unique)) if aspect_ratios_unique else 0.0,
                **quantiles(aspect_ratios_unique, [0.25, 0.5, 0.75, 0.95]),
            },
        },
        "tail_features_unique": {
            "styles_lt_50": int(sum(1 for v in unique_style_counter.values() if v < 50)),
            "styles_lt_100": int(sum(1 for v in unique_style_counter.values() if v < 100)),
            "artists_lt_20": int(sum(1 for v in unique_artist_counter.values() if v < 20)),
            "artists_lt_50": int(sum(1 for v in unique_artist_counter.values() if v < 50)),
        },
        "top_style_ids_unique": [
            {"id": int(k), "count": int(v)} for k, v in unique_style_counter.most_common(args.top_k)
        ],
        "top_genre_ids_unique": [
            {"id": int(k), "count": int(v)} for k, v in unique_genre_counter.most_common(args.top_k)
        ],
        "top_artist_ids_unique": [
            {"id": int(k), "count": int(v)} for k, v in unique_artist_counter.most_common(args.top_k)
        ],
        "files": {
            "unique_index": str(unique_path),
            "duplicates": str(dup_path),
        },
    }

    # Save summary JSON
    summary_path = out_dir / "wikiart_dedupe_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Save top count tables for reporting
    pd.DataFrame(summary["top_style_ids_unique"]).to_csv(out_dir / "top_style_ids_unique.csv", index=False)
    pd.DataFrame(summary["top_genre_ids_unique"]).to_csv(out_dir / "top_genre_ids_unique.csv", index=False)
    pd.DataFrame(summary["top_artist_ids_unique"]).to_csv(out_dir / "top_artist_ids_unique.csv", index=False)

    style_counts_df = counter_to_df(unique_style_counter, "style_id")
    genre_counts_df = counter_to_df(unique_genre_counter, "genre_id")
    artist_counts_df = counter_to_df(unique_artist_counter, "artist_id")

    style_counts_df.to_csv(out_dir / "style_counts_unique.csv", index=False)
    genre_counts_df.to_csv(out_dir / "genre_counts_unique.csv", index=False)
    artist_counts_df.to_csv(out_dir / "artist_counts_unique.csv", index=False)

    style_counts_df[style_counts_df["count"] <= args.tail_threshold].to_csv(
        out_dir / "style_tail_ids_unique.csv", index=False
    )
    genre_counts_df[genre_counts_df["count"] <= args.tail_threshold].to_csv(
        out_dir / "genre_tail_ids_unique.csv", index=False
    )
    artist_counts_df[artist_counts_df["count"] <= args.tail_threshold].to_csv(
        out_dir / "artist_tail_ids_unique.csv", index=False
    )

    # Plots
    save_plot_top_counts(
        unique_style_counter,
        title=f"Top {args.top_k} Style IDs (Unique Images)",
        out_path=plot_dir / "top_style_ids_unique.png",
        top_k=args.top_k,
    )
    save_plot_top_counts(
        unique_genre_counter,
        title=f"Top {args.top_k} Genre IDs (Unique Images)",
        out_path=plot_dir / "top_genre_ids_unique.png",
        top_k=args.top_k,
    )
    save_plot_top_counts(
        unique_artist_counter,
        title=f"Top {args.top_k} Artist IDs (Unique Images)",
        out_path=plot_dir / "top_artist_ids_unique.png",
        top_k=args.top_k,
    )

    # Rank-frequency plot (long-tail diagnostic)
    if unique_artist_counter:
        artist_sorted = np.array(sorted(unique_artist_counter.values(), reverse=True), dtype=np.float64)
        ranks = np.arange(1, len(artist_sorted) + 1, dtype=np.float64)
        plt.figure(figsize=(8, 5))
        plt.loglog(ranks, artist_sorted)
        plt.title("Artist Rank-Frequency (Unique Images)")
        plt.xlabel("Rank")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plot_dir / "artist_rank_frequency_loglog.png", dpi=160)
        plt.close()

    if unique_df.shape[0] > 0:
        unique_df["bytes_per_pixel"] = unique_df["bytes_size"] / (
            unique_df["width"] * unique_df["height"]
        )

        save_plot_hist(
            unique_df["aspect_ratio"],
            "Aspect Ratio Distribution (Unique Images)",
            "width / height",
            plot_dir / "aspect_ratio_hist_unique.png",
        )
        save_plot_hist(
            unique_df["megapixels"],
            "Megapixel Distribution (Unique Images)",
            "Megapixels",
            plot_dir / "megapixels_hist_unique.png",
        )
        save_plot_hist(
            unique_df["bytes_size"],
            "Image Byte Size Distribution (Unique Images)",
            "Bytes",
            plot_dir / "bytes_hist_unique.png",
        )
        save_plot_hist(
            unique_df["bytes_per_pixel"],
            "Bytes-per-Pixel Distribution (Unique Images)",
            "Bytes / pixel",
            plot_dir / "bytes_per_pixel_hist_unique.png",
        )

        sample_for_scatter = unique_df.sample(min(15000, len(unique_df)), random_state=42)
        plt.figure(figsize=(8, 7))
        plt.hexbin(sample_for_scatter["width"], sample_for_scatter["height"], gridsize=45, cmap="viridis")
        plt.colorbar(label="Count")
        plt.title("Width vs Height (Unique Images)")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig(plot_dir / "width_height_hexbin_unique.png", dpi=160)
        plt.close()

        # Orientation composition plot
        orient_df = (
            unique_df.groupby("orientation", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
        )
        plt.figure(figsize=(6, 6))
        plt.pie(orient_df["count"], labels=orient_df["orientation"], autopct="%1.1f%%")
        plt.title("Orientation Mix (Unique Images)")
        plt.tight_layout()
        plt.savefig(plot_dir / "orientation_pie_unique.png", dpi=160)
        plt.close()

        # Megapixels by orientation
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=unique_df, x="orientation", y="megapixels")
        plt.title("Megapixels by Orientation")
        plt.xlabel("Orientation")
        plt.ylabel("Megapixels")
        plt.tight_layout()
        plt.savefig(plot_dir / "megapixels_by_orientation_boxplot.png", dpi=160)
        plt.close()

        # Feature correlation heatmap
        corr_cols = ["width", "height", "bytes_size", "megapixels", "aspect_ratio", "bytes_per_pixel"]
        corr = unique_df[corr_cols].corr(numeric_only=True)
        corr.to_csv(out_dir / "feature_correlation_unique.csv")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Feature Correlation (Unique Images)")
        plt.tight_layout()
        plt.savefig(plot_dir / "feature_correlation_heatmap.png", dpi=160)
        plt.close()

        # Style x Genre crosstab (top styles)
        top_styles = [x[0] for x in unique_style_counter.most_common(min(args.top_k, len(unique_style_counter)))]
        subset = unique_df[unique_df["style"].isin(top_styles)]
        ctab = pd.crosstab(subset["style"], subset["genre"])
        ctab.to_csv(out_dir / "style_genre_crosstab_top_styles.csv")

        if ctab.shape[0] > 0 and ctab.shape[1] > 0:
            plt.figure(figsize=(10, 7))
            sns.heatmap(ctab, cmap="mako")
            plt.title("Style × Genre Counts (Top Styles)")
            plt.xlabel("Genre ID")
            plt.ylabel("Style ID")
            plt.tight_layout()
            plt.savefig(plot_dir / "style_genre_heatmap_top_styles.png", dpi=160)
            plt.close()

        # Outlier tables
        unique_df.sort_values("megapixels", ascending=False).head(100).to_csv(
            out_dir / "outliers_top_megapixels.csv", index=False
        )
        unique_df.sort_values("bytes_size", ascending=False).head(100).to_csv(
            out_dir / "outliers_top_bytes.csv", index=False
        )
        unique_df.assign(
            abs_log_aspect=np.abs(np.log(unique_df["aspect_ratio"].clip(lower=1e-6)))
        ).sort_values("abs_log_aspect", ascending=False).head(100).to_csv(
            out_dir / "outliers_extreme_aspect_ratio.csv", index=False
        )

        # Per-style quality proxies
        per_style = (
            unique_df.groupby("style")
            .agg(
                n=("style", "size"),
                mean_megapixels=("megapixels", "mean"),
                mean_bytes=("bytes_size", "mean"),
                mean_bpp=("bytes_per_pixel", "mean"),
                mean_aspect=("aspect_ratio", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        per_style.to_csv(out_dir / "per_style_quality_stats.csv", index=False)

        per_genre = (
            unique_df.groupby("genre")
            .agg(
                n=("genre", "size"),
                mean_megapixels=("megapixels", "mean"),
                mean_bytes=("bytes_size", "mean"),
                mean_bpp=("bytes_per_pixel", "mean"),
                mean_aspect=("aspect_ratio", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        per_genre.to_csv(out_dir / "per_genre_quality_stats.csv", index=False)

    # Human-readable highlights
    top_style_share = max(unique_style_counter.values()) / max(unique_count, 1)
    top_genre_share = max(unique_genre_counter.values()) / max(unique_count, 1)
    top_artist_share = max(unique_artist_counter.values()) / max(unique_count, 1)

    interesting_features = {
        "dataset_concentration": {
            "top_style_share": round(float(top_style_share), 6),
            "top_genre_share": round(float(top_genre_share), 6),
            "top_artist_share": round(float(top_artist_share), 6),
            "style_gini": round(gini_from_counts(unique_style_counter), 6),
            "genre_gini": round(gini_from_counts(unique_genre_counter), 6),
            "artist_gini": round(gini_from_counts(unique_artist_counter), 6),
        },
        "extremes_unique": {
            "max_megapixels": float(max(megapixels_unique)) if megapixels_unique else 0.0,
            "max_bytes": int(max(bytes_sizes_unique)) if bytes_sizes_unique else 0,
            "max_aspect_ratio": float(max(aspect_ratios_unique)) if aspect_ratios_unique else 0.0,
            "min_aspect_ratio": float(min(aspect_ratios_unique)) if aspect_ratios_unique else 0.0,
        },
        "tail_threshold": args.tail_threshold,
        "tail_counts": {
            "styles_at_or_below_threshold": int((style_counts_df["count"] <= args.tail_threshold).sum()),
            "genres_at_or_below_threshold": int((genre_counts_df["count"] <= args.tail_threshold).sum()),
            "artists_at_or_below_threshold": int((artist_counts_df["count"] <= args.tail_threshold).sum()),
        },
    }

    interesting_path = out_dir / "wikiart_interesting_features.json"
    interesting_path.write_text(json.dumps(interesting_features, indent=2))

    highlights = []
    highlights.append(f"Total rows scanned: {total_rows:,}")
    highlights.append(f"Unique rows after exact hash dedupe: {unique_count:,}")
    highlights.append(f"Exact duplicates removed: {duplicate_count:,} ({duplicate_rate:.2%})")
    highlights.append(
        "Long-tail evidence (unique set): "
        f"styles<50={summary['tail_features_unique']['styles_lt_50']}, "
        f"styles<100={summary['tail_features_unique']['styles_lt_100']}, "
        f"artists<20={summary['tail_features_unique']['artists_lt_20']}"
    )
    highlights.append(
        "Orientation mix (unique set): "
        + ", ".join(f"{k}={v}" for k, v in orientation_counter_unique.items())
    )
    highlights.append(
        "Concentration: "
        f"top_style_share={top_style_share:.2%}, top_genre_share={top_genre_share:.2%}, "
        f"top_artist_share={top_artist_share:.2%}"
    )
    highlights.append(
        "Imbalance (Gini): "
        f"style={interesting_features['dataset_concentration']['style_gini']:.3f}, "
        f"genre={interesting_features['dataset_concentration']['genre_gini']:.3f}, "
        f"artist={interesting_features['dataset_concentration']['artist_gini']:.3f}"
    )
    highlights.append(
        "Extremes: "
        f"max_megapixels={interesting_features['extremes_unique']['max_megapixels']:.2f}, "
        f"max_aspect={interesting_features['extremes_unique']['max_aspect_ratio']:.2f}, "
        f"min_aspect={interesting_features['extremes_unique']['min_aspect_ratio']:.2f}"
    )
    highlights_path = out_dir / "wikiart_highlights.txt"
    highlights_path.write_text("\n".join(highlights) + "\n")

    findings_md = out_dir / "wikiart_findings_report.md"
    findings_md.write_text(
        "# WikiArt Dedupe + EDA Findings\n\n"
        f"- Total rows: **{total_rows:,}**\n"
        f"- Unique rows: **{unique_count:,}**\n"
        f"- Exact duplicates: **{duplicate_count:,}** ({duplicate_rate:.2%})\n"
        f"- Coverage retained (unique): styles={len(unique_style_counter)}, genres={len(unique_genre_counter)}, artists={len(unique_artist_counter)}\n"
        f"- Top style share: **{top_style_share:.2%}**\n"
        f"- Top genre share: **{top_genre_share:.2%}**\n"
        f"- Top artist share: **{top_artist_share:.2%}**\n"
        f"- Imbalance (Gini): style={interesting_features['dataset_concentration']['style_gini']:.3f}, "
        f"genre={interesting_features['dataset_concentration']['genre_gini']:.3f}, "
        f"artist={interesting_features['dataset_concentration']['artist_gini']:.3f}\n"
        f"- Orientation: {', '.join(f'{k}={v}' for k, v in orientation_counter_unique.items())}\n"
        "\n"
        "## Saved artifacts\n"
        "- Summary JSON + highlights + interesting features JSON\n"
        "- Unique index parquet + duplicates CSV\n"
        "- Tail tables, outlier tables, quality statistics\n"
        "- Correlation matrices and style-genre crosstab\n"
        "- Plot bundle in `plots/`\n"
    )

    print(json.dumps(
        {
            "summary": str(summary_path),
            "unique_index": str(unique_path),
            "duplicates": str(dup_path),
            "plots_dir": str(plot_dir),
            "highlights": str(highlights_path),
            "interesting_features": str(interesting_path),
            "findings_report": str(findings_md),
            "total_rows": total_rows,
            "unique_rows": unique_count,
            "duplicate_rows": duplicate_count,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
