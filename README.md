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

## As Implemented (March 2026)

The three-track experiment pipeline is implemented with script-first execution:

- Track A (zero-shot provider comparison):
	- `scripts/benchmark_wikiart_zero_shot.py`
	- `scripts/run_track_a_comparison.py`
	- `scripts/merge_track_a_runs.py`
	- Supports local split mode and HF streaming mode (`huggan/wikiart`) with retry/resume + shared cache.
- Track B (fine-tuning):
	- `scripts/finetune_clip.py`
	- Colab notebook flow in `notebooks/finetune_clip_colab.ipynb` with runtime token handling and memory-safe buffering.
- Track C (separate encoders + linear map):
	- `scripts/train_linear_map.py` (ridge CV training of `W`)
	- `scripts/benchmark_linear_map.py` (Track A-compatible evaluation artifacts)
	- `scripts/run_track_c_comparison.py` (style+genre orchestration + summary tables)

Unified artifact pattern used by Track A and Track C runs:

- `artifacts/runs/track*/style/metrics_*.json`
- `artifacts/runs/track*/style/predictions_*.parquet`
- `artifacts/runs/track*/genre/metrics_*.json`
- `artifacts/runs/track*/genre/predictions_*.parquet`
- `artifacts/runs/track*/summary/track_a_metrics_long.csv` (merge-compatible)

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

Track A full provider comparison (WikiArt style + genre, slide-ready summaries):

```bash
python scripts/run_track_a_comparison.py \
	--split-csv data/labeled/test.csv \
	--models-config configs/zero_shot_models.json \
	--output-root artifacts/runs \
	--targets style genre \
	--templates-per-class 4 \
	--max-images 0
```

Outputs are written under `artifacts/runs/trackA_<timestamp>/` with:

- per-target raw artifacts in `style/` and `genre/`
- slide-ready tables in `summary/track_a_metrics_long.csv` and `summary/track_a_leaderboard.csv`
- summary pack in `summary/track_a_summary.json` and `summary/track_a_slides.md`

Merge multiple Track A runs into one final slide table:

```bash
python scripts/merge_track_a_runs.py \
	--runs-root artifacts/runs \
	--output-dir artifacts/runs/trackA_merged
```

Merged outputs:

- `all_runs_metrics_long.csv` (all model/target/run rows)
- `all_runs_leaderboard.csv` (ranked across runs)
- `final_slides_table.csv` (one best row per target+model)
- `best_per_target.csv` (headline winners)
- `merge_summary.json`

To merge Track A and Track C together, include both run globs:

```bash
python scripts/merge_track_a_runs.py \
	--runs-root artifacts/runs \
	--run-globs trackA_* trackC_* \
	--output-dir artifacts/runs/trackAC_merged
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

Evaluate a Track B checkpoint on the held-out test split with merge-compatible outputs:

```bash
python scripts/run_track_b_comparison.py \
	--split-csv data/labeled/test.csv \
	--checkpoint-dir artifacts/checkpoints/trackB_laion_vitl14 \
	--output-root artifacts/runs
```

Colab notebook runner for streaming Track B evaluation:

- `notebooks/track_b_colab_runner.ipynb`

This writes:

- `artifacts/runs/trackB_<timestamp>/style/metrics_*.json`
- `artifacts/runs/trackB_<timestamp>/style/predictions_*.parquet`
- `artifacts/runs/trackB_<timestamp>/summary/track_a_metrics_long.csv`

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

Evaluate Track C on the test split (style + genre) and produce slide-compatible tables:

```bash
python scripts/run_track_c_comparison.py \
	--split-csv data/labeled/test.csv \
	--linear-map-path artifacts/linear_map/linear_map_W.npy \
	--output-root artifacts/runs \
	--templates-per-class 1
```

Smoke test (quick sanity check):

```bash
python scripts/run_track_c_comparison.py \
	--split-csv data/labeled/test.csv \
	--linear-map-path artifacts/linear_map/linear_map_W.npy \
	--output-root artifacts/runs \
	--targets style genre \
	--templates-per-class 1 \
	--max-images 64
```

Merge Track A + Track C runs into one final slides table:

```bash
python scripts/merge_track_a_runs.py \
	--runs-root artifacts/runs \
	--run-globs trackA_* trackC_* \
	--output-dir artifacts/runs/trackAC_merged
```

Merge Track A + B + C runs:

```bash
python scripts/merge_track_a_runs.py \
	--runs-root artifacts/runs \
	--run-globs trackA_* trackB_* trackC_* \
	--output-dir artifacts/runs/trackABC_merged
```

---

## Notes on Data Use

- WikiArt data is for non-commercial research usage.
- Keep deterministic split files and run configs for reproducibility.
