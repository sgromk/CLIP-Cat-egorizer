# Vision–Language Perception and Decision Pipeline for an Autonomous Art Grading Agent

This repository implements a real-time grading pipeline and an ablation framework for vision-language perception on artwork.

## Current Direction (March 2026)

The project is currently running in a **WikiArt-only** mode to simplify execution and reporting:

- Use only the local `huggan/wikiart` snapshot for data and experiments.
- Prioritize style/genre/artist-driven analysis and model comparisons.
- Defer explicit validation of object-driven grading rules (cats/cash/etc.) to a later phase.

Primary objective:

1. Compare zero-shot models across providers.
2. Compare zero-shot vs fine-tuned CLIP-style models.
3. Compare jointly trained vision-language embeddings vs separate encoders with a learned linear alignment map.

---

## System Summary

```
Webcam Frame -> Perception Model -> Semantic Labels + Scores -> Grader -> Grade + Explanation
```

Core modules:

- `app/perception.py`: model loading and perception inference
- `app/grader.py`: deterministic grading logic
- `app/llm_fallback.py`: fallback explanation for uncertain scenes
- `app/main.py`: FastAPI endpoints + realtime interface

---

## Dataset Status

Current local snapshot (partial `huggan/wikiart`):

- ~19 GB on disk
- 42 valid parquet shards
- 47,514 rows available

After exact hash deduplication, the working unique index contains **47,501** images.

---

## Ablation Tracks

### Track A — Zero-Shot Multi-Model Comparison

Run multiple foundation models with shared prompt templates and shared metrics.

Suggested model set:

- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
- `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`
- `google/siglip-base-patch16-224`
- `google/siglip-so400m-patch14-384`

### Track B — Fine-Tuning

Fine-tune top Track A candidates on WikiArt-only data using style-conditioned text prompts, then compare against zero-shot baselines.

Recommended runs:

- CLIP ViT-B: LoRA / lightweight fine-tuning
- CLIP ViT-L or SigLIP-base: optional second fine-tune (budget permitting)

### Track C — Separate Encoders + Linear Map

Build a dual-encoder baseline:

- Image encoder: ViT/ResNet (independent pretrained image model)
- Text encoder: sentence-transformer (`all-MiniLM-L6-v2`)
- Fit ridge regression matrix $W$ to align image -> text embedding space

Inference score:

$$
	ext{sim}(I, t) = \cos(W f(I), g(t))
$$

---

## Unified Hugging Face Pipeline Design

Implement a model adapter layer so all models share one evaluation path.

Common interface per adapter:

- `load()`
- `encode_image(batch_images)`
- `encode_text(label_prompts)`
- `score(image_emb, text_emb)`
- `predict(image, labels, threshold_cfg)`

Use:

- `AutoProcessor` for multimodal image+text preprocessing when supported
- `AutoTokenizer` + `AutoModel` for text branch in separate-encoder baseline

Standard output schema (for all runs):

- `run_id`
- `model_id`
- `image_id`
- `label`
- `score`
- `prediction`
- `split`

This lets one metrics script evaluate every model type consistently.

---

## Metrics and Analysis

For each run, compute:

### Detection Metrics

- Per-label precision, recall, F1
- Macro/micro F1
- AP per label, mAP overall
- Confusion matrix per concept group

### Embedding Quality

- Positive vs negative cosine score distributions
- Separability ratio per label
- Cross-modal retrieval Recall@1 / Recall@5 (image->text and text->image)

### Robustness

Evaluate metric degradation under:

- JPEG compression
- Gaussian blur
- Brightness/contrast shift
- Random crop/occlusion

### Performance

- Encode latency per image
- End-to-end inference latency
- Throughput (FPS or images/sec)
- GPU/CPU memory profile (if available)

---

## Recommended Run Order

1. Data audit + split validation
2. Zero-shot sweep (all Track A models)
3. Prompt template sweep + threshold selection
4. Fine-tune best 1–2 models
5. Train/evaluate separate-encoder linear-map baseline
6. Robustness + latency profiling
7. Aggregate comparison tables and failure-case gallery

---

## Colab + VS Code Workflow

Use VS Code for authoring and Colab for heavy runs.

Suggested notebook sequence:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_zero_shot_benchmark.ipynb`
3. `notebooks/03_prompt_threshold_sweep.ipynb`
4. `notebooks/04_finetune_clip.ipynb`
5. `notebooks/05_linear_map_baseline.ipynb`
6. `notebooks/06_robustness_latency.ipynb`
7. `notebooks/07_report_tables_plots.ipynb`

Persist all run artifacts to versioned folders:

- `artifacts/runs/<run_id>/metrics.json`
- `artifacts/runs/<run_id>/predictions.parquet`
- `artifacts/runs/<run_id>/config.yaml`
- `artifacts/runs/<run_id>/plots/*.png`

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` only if needed:

- `OPENAI_API_KEY` for LLM fallback
- `OLLAMA=1` + `OLLAMA_MODEL` if using local Ollama instead of OpenAI
- `HF_TOKEN` only if you need gated/private Hugging Face assets

Run app:

```bash
uvicorn app.main:app --reload
```

Run tests:

```bash
pytest tests/
```

---

## Backend-Only Experiment Commands

### 1) Dataset verification + EDA

```bash
python scripts/data_verify_eda.py \
	--wikiart-dir data/raw/wikiart \
	--output artifacts/data_audit/wikiart_audit.json \
	--max-image-sample 50000
```

Use `--max-image-sample 0` for metadata-only checks with no image decoding.

Comprehensive dedupe + rich EDA artifact bundle:

```bash
python scripts/wikiart_dedupe_eda.py \
	--wikiart-dir data/raw/wikiart \
	--output-dir artifacts/data_audit \
	--top-k 25 \
	--tail-threshold 50
```

Primary outputs for reporting:

- `artifacts/data_audit/wikiart_dedupe_eda_summary.json`
- `artifacts/data_audit/wikiart_interesting_features.json`
- `artifacts/data_audit/wikiart_findings_report.md`
- `artifacts/data_audit/wikiart_report_pack.md`
- `artifacts/data_audit/wikiart_highlights.txt`
- `artifacts/data_audit/wikiart_unique_index.parquet`
- `artifacts/data_audit/wikiart_duplicates.csv`
- `artifacts/data_audit/outliers_*.csv`
- `artifacts/data_audit/*_counts_unique.csv`, `*_tail_ids_unique.csv`
- `artifacts/data_audit/feature_correlation_unique.csv`
- `artifacts/data_audit/style_genre_crosstab_top_styles.csv`
- Plot bundle in `artifacts/data_audit/plots/`

### 2) Zero-shot multi-model benchmark

Prepare a benchmark CSV with:

- `image_path`
- one target column per concept you want to evaluate in that run

Then run:

```bash
python scripts/benchmark_zero_shot.py \
	--dataset-csv data/labeled/annotations.csv \
	--models-config configs/zero_shot_models.json \
	--labels-config configs/labels_and_prompts.json \
	--output-dir artifacts/runs
```

### 3) Fine-tune CLIP on image-text pairs

Training CSV requires columns `image_path,text`:

```bash
python scripts/finetune_clip.py \
	--train-csv data/labeled/train_pairs.csv \
	--val-csv data/labeled/val_pairs.csv \
	--model-id openai/clip-vit-base-patch32 \
	--epochs 2 \
	--batch-size 8 \
	--output-dir artifacts/checkpoints
```

WikiArt-only mode (no manual pair CSV):

```bash
python scripts/finetune_clip.py \
	--wikiart-dir data/raw/wikiart \
	--max-rows 0 \
	--style-template "a painting in style_{style_id}" \
	--model-id openai/clip-vit-base-patch32 \
	--epochs 2 \
	--batch-size 8 \
	--output-dir artifacts/checkpoints
```

### 4) Train separate-encoder linear map baseline

Pairs CSV requires columns `image_path,text`:

```bash
python scripts/train_linear_map.py \
	--pairs-csv data/labeled/train_pairs.csv \
	--output-dir artifacts/linear_map
```

WikiArt-only mode:

```bash
python scripts/train_linear_map.py \
	--wikiart-dir data/raw/wikiart \
	--max-rows 20000 \
	--style-template "a painting in style_{style_id}" \
	--output-dir artifacts/linear_map
```

---

## Notes on Data Use

- WikiArt data is for non-commercial research usage.
- Keep deterministic split files and run configs for reproducibility.
