from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


class ImageTextPairDataset(Dataset):
    def __init__(self, csv_path: str) -> None:
        frame = pd.read_csv(csv_path)
        if not {"image_path", "text"}.issubset(frame.columns):
            raise ValueError("CSV must include image_path and text columns")
        self.frame = frame

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        row = self.frame.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        text = str(row.text)
        return image, text


class WikiArtParquetDataset(Dataset):
    def __init__(
        self,
        wikiart_dir: str,
        max_rows: int = 0,
        style_template: str = "a painting in style_{style_id}",
    ) -> None:
        parquet_files = sorted(str(p) for p in Path(wikiart_dir).rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet shards found under {wikiart_dir}")

        self.dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        if max_rows > 0:
            max_rows = min(max_rows, len(self.dataset))
            self.dataset = self.dataset.select(range(max_rows))
        self.style_template = style_template

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        from io import BytesIO

        row = self.dataset[int(index)]
        image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        text = self.style_template.format(style_id=int(row["style"]))
        return image, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on local image-text pairs")
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--wikiart-dir", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--style-template", default="a painting in style_{style_id}")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    return parser.parse_args()


def collate_fn(processor: CLIPProcessor, batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
    images, texts = zip(*batch)
    return processor(images=list(images), text=list(texts), return_tensors="pt", padding=True, truncation=True)


def evaluate_loss(
    model: CLIPModel,
    processor: CLIPProcessor,
    val_loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            losses.append(float(outputs.loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(args.model_id)
    model = CLIPModel.from_pretrained(args.model_id).to(device)

    if args.wikiart_dir:
        train_dataset: Dataset = WikiArtParquetDataset(
            wikiart_dir=args.wikiart_dir,
            max_rows=args.max_rows,
            style_template=args.style_template,
        )
    elif args.train_csv:
        train_dataset = ImageTextPairDataset(args.train_csv)
    else:
        raise ValueError("Provide either --train-csv or --wikiart-dir")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )

    val_loader = None
    if args.val_csv:
        val_dataset = ImageTextPairDataset(args.val_csv)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(processor, b),
        )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        row = {"epoch": float(epoch + 1), "train_loss": train_loss}

        if val_loader is not None:
            row["val_loss"] = evaluate_loss(model, processor, val_loader, device)

        history.append(row)
        print(row)

    model.save_pretrained(output_dir / "clip_finetuned")
    processor.save_pretrained(output_dir / "clip_finetuned")
    (output_dir / "finetune_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved fine-tuned model to {output_dir / 'clip_finetuned'}")


if __name__ == "__main__":
    main()
