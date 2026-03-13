from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and profile local WikiArt parquet shards")
    parser.add_argument("--wikiart-dir", default="data/raw/wikiart", help="Path to local dataset snapshot")
    parser.add_argument("--output", default="artifacts/data_audit/wikiart_audit.json", help="Output JSON report")
    parser.add_argument("--max-image-sample", type=int, default=1500, help="Max images to decode for EDA")
    return parser.parse_args()


def bytes_to_image_size(image_bytes: bytes) -> tuple[int, int]:
    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as image:
        return image.size


def main() -> None:
    args = parse_args()
    root = Path(args.wikiart_dir)
    shard_paths = sorted(root.rglob("*.parquet"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not shard_paths:
        raise FileNotFoundError(f"No parquet files found under {root}")

    report: dict[str, object] = {
        "dataset_dir": str(root),
        "parquet_shards": len(shard_paths),
        "parquet_bad_shards": 0,
        "total_rows": 0,
        "decoded_images": 0,
        "decode_errors": 0,
    }

    style_counter: Counter[int] = Counter()
    genre_counter: Counter[int] = Counter()
    artist_counter: Counter[int] = Counter()
    width_stats: list[int] = []
    height_stats: list[int] = []
    sample_hashes: Counter[str] = Counter()
    decode_errors: list[dict[str, object]] = []

    decoded_budget = args.max_image_sample

    for shard in shard_paths:
        try:
            table = pq.read_table(shard, columns=["image", "artist", "genre", "style"])
        except Exception as exc:
            report["parquet_bad_shards"] = int(report["parquet_bad_shards"]) + 1
            decode_errors.append({"file": str(shard), "error": str(exc)})
            continue

        report["total_rows"] = int(report["total_rows"]) + table.num_rows

        artists = table["artist"].to_pylist()
        genres = table["genre"].to_pylist()
        styles = table["style"].to_pylist()

        artist_counter.update(int(x) for x in artists)
        genre_counter.update(int(x) for x in genres)
        style_counter.update(int(x) for x in styles)

        if decoded_budget <= 0:
            continue

        image_rows = table["image"].to_pylist()
        to_decode = min(decoded_budget, len(image_rows))
        for row in image_rows[:to_decode]:
            try:
                image_bytes = row["bytes"]
                width, height = bytes_to_image_size(image_bytes)
                width_stats.append(width)
                height_stats.append(height)
                digest = hashlib.md5(image_bytes).hexdigest()
                sample_hashes[digest] += 1
                report["decoded_images"] = int(report["decoded_images"]) + 1
            except Exception as exc:
                report["decode_errors"] = int(report["decode_errors"]) + 1
                decode_errors.append({"file": str(shard), "error": str(exc)})

        decoded_budget -= to_decode

    duplicate_hashes = sum(1 for _, count in sample_hashes.items() if count > 1)

    report["disk_size_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    report["top_style_ids"] = [{"id": k, "count": v} for k, v in style_counter.most_common(20)]
    report["top_genre_ids"] = [{"id": k, "count": v} for k, v in genre_counter.most_common(20)]
    report["top_artist_ids"] = [{"id": k, "count": v} for k, v in artist_counter.most_common(20)]
    report["distinct_style_ids"] = len(style_counter)
    report["distinct_genre_ids"] = len(genre_counter)
    report["distinct_artist_ids"] = len(artist_counter)
    report["sample_duplicate_hashes"] = duplicate_hashes

    if width_stats and height_stats:
        report["image_width"] = {
            "min": min(width_stats),
            "max": max(width_stats),
            "mean": round(mean(width_stats), 2),
        }
        report["image_height"] = {
            "min": min(height_stats),
            "max": max(height_stats),
            "mean": round(mean(height_stats), 2),
        }

    report["errors"] = decode_errors[:100]
    output_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote EDA report to {output_path}")
    print(json.dumps({
        "parquet_shards": report["parquet_shards"],
        "bad_shards": report["parquet_bad_shards"],
        "rows": report["total_rows"],
        "decoded_images": report["decoded_images"],
        "decode_errors": report["decode_errors"],
    }, indent=2))


if __name__ == "__main__":
    main()
