# Vision–Language Perception and Decision Pipeline for an Autonomous Art Grading Agent


This repository implements the model-building, training, and evaluation pipeline for a vision-language perception system benchmarked on the WikiArt dataset. The trained perception models are consumed downstream by a FastAPI-based art grading web application ([web app repo](https://github.com/rizveea/CLIP-Cat-egorizer)).

---

## Overview

The project compares three approaches to cross-modal visual perception on artwork:

1. **Track A** — Zero-shot evaluation of multiple pretrained vision-language foundation models
2. **Track B** — Domain-specific fine-tuning of a CLIP model on WikiArt image-text pairs
3. **Track C** — Separate image and text encoders connected via a learned linear alignment map (ablation baseline)

All three tracks share a common adapter interface and are evaluated on the same 11,320-image held-out test split using Top-1 and Top-5 accuracy across 27 artistic style classes and 11 genre classes.

### Key Results

| Track | Task | Top-1 | Top-5 |
|---|---|---:|---:|
| B (fine-tuned LAION ViT-L/14) | Style | 0.7959 | 0.9766 |
| A (LAION ViT-L/14, zero-shot) | Style | 0.3702 | 0.7743 |
| A (LAION ViT-L/14, zero-shot) | Genre | 0.5295 | 0.9341 |
| C (linear map, ViT-B/32 + MiniLM) | Genre | 0.0455 | 0.5093 |
| C (linear map, ViT-B/32 + MiniLM) | Style | ~0.00 | ~0.30 |

### Findings

**Domain fine-tuning is high-leverage on specialized visual domains.** A lightweight fine-tuning run on roughly 10,000 WikiArt image-text pairs, without retraining the full network, more than doubled Top-1 style accuracy over the best zero-shot baseline. The representations from large-scale pre-training appear to be a strong foundation that requires only modest domain-specific signal to redirect toward fine-grained tasks like art style classification.

**Joint contrastive training is not recoverable post-hoc.** Track C, the linear alignment ablation, collapsed almost entirely — near-zero Top-1 on style, 4.6% on genre — with confusion matrices showing predictions collapsing to a single class. The embedding spaces produced by independently trained vision and text encoders are geometrically incompatible, and a ridge regression matrix has no mechanism to reconcile them. The embeddings are themselves the output of dozens of stacked nonlinear transformations, meaning any alignment capable of bridging the two spaces would almost certainly need to be nonlinear as well. This rules out a class of cheap post-hoc alignment approaches for cross-modal retrieval.

**Zero-shot models underperform on style but handle genre reasonably well.** The best zero-shot model reaches 53% Top-1 on 11 genre classes but only 37% on 27 style classes. Top-5 style accuracy sits at 77-79%, suggesting the models locate the correct region of concept space but lack the fine-grained discriminability to separate visually similar styles without domain adaptation.

---

## Dataset

All experiments use the WikiArt dataset sourced from `huggan/wikiart` on HuggingFace.

- **47,501** unique images after MD5 hash deduplication
- **11,320** images used for evaluation across all tracks
- **10,188 / 1,132** train/val split used for Track B fine-tuning
- 42 parquet shards, ~19 GB on disk
- 27 style classes, 11 genre classes

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

`.env` is only required if using the LLM fallback or gated HuggingFace assets:

```
OPENAI_API_KEY=...       # for LLM fallback (optional)
OLLAMA=1                 # use local Ollama instead of OpenAI (optional)
OLLAMA_MODEL=...
HF_TOKEN=...             # only for gated/private HF assets
```

---

## Model Adapter Interface

All three tracks share a unified adapter interface, ensuring identical preprocessing and scoring conditions across models:

```python
adapter.load()
adapter.encode_image(batch_images)
adapter.encode_text(label_prompts)
adapter.score(image_emb, text_emb)
adapter.predict(image, labels, threshold_cfg)
```

Standard output schema for all runs:

| Field | Description |
|---|---|
| `run_id` | Versioned run identifier |
| `model_id` | HuggingFace model path or checkpoint name |
| `image_id` | Image identifier from the dataset index |
| `label` | Class label |
| `score` | Cosine similarity score |
| `prediction` | Top-1 predicted label |
| `split` | Dataset split (train / val / test) |

---

## Track A — Zero-Shot Model Comparison

Evaluates pretrained vision-language models in zero-shot mode with no fine-tuning. Cosine similarity between image and text embeddings determines the predicted class.

**Models evaluated:**

- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`
- `google/siglip-base-patch16-224`
- `google/siglip-so400m-patch14-384`

**Run the full Track A sweep (WikiArt style + genre):**

```bash
python scripts/run_track_a_comparison.py \
    --split-csv data/labeled/test.csv \
    --models-config configs/zero_shot_models.json \
    --output-root artifacts/runs \
    --targets style genre \
    --templates-per-class 4 \
    --max-images 0
```

Outputs are written under `artifacts/runs/trackA_<timestamp>/`:

- `style/metrics_*.json`, `style/predictions_*.parquet`
- `genre/metrics_*.json`, `genre/predictions_*.parquet`
- `summary/track_a_metrics_long.csv`, `summary/track_a_leaderboard.csv`

**Merge multiple Track A runs:**

```bash
python scripts/merge_track_a_runs.py \
    --runs-root artifacts/runs \
    --output-dir artifacts/runs/trackA_merged
```

---

## Track B — Domain Fine-Tuning

Fine-tunes `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` on WikiArt style-conditioned image-text pairs using InfoNCE symmetric contrastive loss.

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` |
| Loss | InfoNCE (symmetric contrastive) |
| Optimizer | AdamW, lr = 1e-5 |
| Prompt template | `"a painting in style_{style_id}"` |
| Training images | 10,188 |
| Validation images | 1,132 |
| Best checkpoint | Epoch 4 |

**Run fine-tuning (WikiArt-only mode):**

```bash
python scripts/finetune_clip.py \
    --wikiart-dir data/raw/wikiart \
    --max-rows 0 \
    --style-template "a painting in style_{style_id}" \
    --model-id laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --epochs 5 \
    --batch-size 8 \
    --output-dir artifacts/checkpoints
```

For heavy runs, use the Colab notebook: `notebooks/finetune_clip_colab.ipynb`

**Evaluate a Track B checkpoint:**

```bash
python scripts/run_track_b_comparison.py \
    --split-csv data/labeled/test.csv \
    --checkpoint-dir artifacts/checkpoints/trackB_laion_vitl14 \
    --output-root artifacts/runs
```

---

## Track C — Separate Encoders + Linear Alignment

Ablation baseline testing whether a post-hoc linear projection can bridge the embedding gap between independently trained vision and text encoders.

**Architecture:**

- Image encoder: `openai/clip-vit-base-patch32`
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Alignment: ridge regression matrix `W` fitted to map image embeddings into the text embedding space

Inference similarity:

$$\text{sim}(I, t) = \cos(W \cdot f(I),\ g(t))$$

**Train the linear map (WikiArt-only mode):**

```bash
python scripts/train_linear_map.py \
    --wikiart-dir data/raw/wikiart \
    --max-rows 20000 \
    --style-template "a painting in style_{style_id}" \
    --output-dir artifacts/linear_map
```

**Evaluate Track C:**

```bash
python scripts/run_track_c_comparison.py \
    --split-csv data/labeled/test.csv \
    --linear-map-path artifacts/linear_map/linear_map_W.npy \
    --output-root artifacts/runs \
    --targets style genre \
    --templates-per-class 1
```

**Smoke test (64 images):**

```bash
python scripts/run_track_c_comparison.py \
    --split-csv data/labeled/test.csv \
    --linear-map-path artifacts/linear_map/linear_map_W.npy \
    --output-root artifacts/runs \
    --targets style genre \
    --templates-per-class 1 \
    --max-images 64
```

---

## Merging Runs

Merge any combination of Track A, B, and C runs into a unified comparison table:

```bash
# Track A + C
python scripts/merge_track_a_runs.py \
    --runs-root artifacts/runs \
    --run-globs trackA_* trackC_* \
    --output-dir artifacts/runs/trackAC_merged

# All tracks
python scripts/merge_track_a_runs.py \
    --runs-root artifacts/runs \
    --run-globs trackA_* trackB_* trackC_* \
    --output-dir artifacts/runs/trackABC_merged
```

Merged outputs:

- `all_runs_metrics_long.csv` — all model/target/run rows
- `all_runs_leaderboard.csv` — ranked across runs
- `final_slides_table.csv` — one best row per target and model
- `best_per_target.csv` — headline winners per task

---

## Dataset Utilities

**Verify dataset and run EDA:**

```bash
python scripts/data_verify_eda.py \
    --wikiart-dir data/raw/wikiart \
    --output artifacts/data_audit/wikiart_audit.json \
    --max-image-sample 50000
```

**Full deduplication + EDA artifact bundle:**

```bash
python scripts/wikiart_dedupe_eda.py \
    --wikiart-dir data/raw/wikiart \
    --output-dir artifacts/data_audit \
    --top-k 25 \
    --tail-threshold 50
```

Primary outputs:

- `wikiart_unique_index.parquet` — deduplicated image index
- `wikiart_duplicates.csv` — removed duplicates
- `style_genre_crosstab_top_styles.csv`
- `feature_correlation_unique.csv`
- Plot bundle in `artifacts/data_audit/plots/`

---

## Recommended Run Order

1. Dataset verification and EDA
2. Zero-shot sweep across all Track A models
3. Prompt template sweep and threshold selection
4. Fine-tune top 1-2 Track A candidates (Track B)
5. Train and evaluate the linear map baseline (Track C)
6. Robustness and latency profiling
7. Merge all runs and generate comparison tables