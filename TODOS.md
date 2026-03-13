# Immediate Human TODOs

Execution checklist for the full ablation plan (zero-shot multi-model + fine-tuning + linear map baseline).

---

## 1) Environment + Runtime Setup

- [x] Create `.venv`
- [x] Install core dependencies from `requirements.txt`
- [x] Install dataset/model tooling (`huggingface_hub`, `datasets`)
- [x] Install experiment extras:
  - `pyarrow`
  - `sentence-transformers`
  - `scikit-learn`
  - `umap-learn`
  - `matplotlib`, `seaborn`
- [ ] Confirm local app still runs: `uvicorn app.main:app --reload`

---

## 2) Data Readiness

- [x] Download partial `huggan/wikiart` snapshot
- [x] Delete `.incomplete` temp files
- [x] Verify parquet integrity (all local shards readable)
- [x] Run dataset verification + EDA script and write audit JSON
- [x] Drop exact duplicate images (hash-based) and export unique index
- [x] Generate detailed EDA artifacts (JSON + CSV + plots + highlights)
- [x] Generate expanded findings bundle (outliers, tails, crosstabs, correlations, report markdown)
- [x] Generate curated report pack markdown linking top figures/tables
- [ ] Create reproducible WikiArt-only split files from unique index (`train/val/test`)
- [ ] Store a WikiArt-only data card (`data/labeled/data_card.json`) with split + distribution stats
- [ ] Curate final 6-8 strongest EDA figures/tables for presentation slides

---

## 3) Model Matrix (Track A: Zero-Shot)

- [x] Add run config listing model IDs:
  - `openai/clip-vit-base-patch32`
  - `openai/clip-vit-large-patch14`
  - `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`
  - `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`
  - `google/siglip-base-patch16-224`
  - `google/siglip-so400m-patch14-384`
- [x] Define 3–5 prompt templates per label
- [x] Add reusable multi-model benchmark script
- [ ] Run zero-shot inference for every model x prompt template
- [ ] Save raw predictions to artifacts per run

---

## 4) Fine-Tuning (Track B)

- [x] Add CLIP fine-tuning training script (image_path,text)
- [ ] Select best 1–2 zero-shot models from Track A
- [ ] Fine-tune on train split (validate on val split only)
- [ ] Save checkpoints + full training config
- [ ] Re-run full metric suite on held-out test split

---

## 5) Separate Encoders + Linear Map (Track C)

- [ ] Pick image encoder backbone (ViT or ResNet)
- [ ] Use text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- [x] Add linear-map training script with ridge CV
- [ ] Fit ridge regression matrix `W` on train split embeddings
- [ ] Tune regularization on val split
- [ ] Evaluate on test split with same metrics as Tracks A/B

---

## 6) Shared Evaluation + Analysis

- [ ] Compute per-label precision/recall/F1 + macro/micro F1
- [ ] Compute AP/mAP
- [ ] Compute retrieval Recall@1/5 (image->text and text->image)
- [ ] Generate cosine distribution and separability plots
- [ ] Run threshold sweep and choose model-specific thresholds on val split
- [ ] Run robustness tests (blur/compression/brightness/crop)
- [ ] Measure latency + throughput per model

---

## 7) Colab Workflow via VS Code

- [ ] Create notebooks in this order:
  1. `notebooks/01_data_audit.ipynb`
  2. `notebooks/02_zero_shot_benchmark.ipynb`
  3. `notebooks/03_prompt_threshold_sweep.ipynb`
  4. `notebooks/04_finetune_clip.ipynb`
  5. `notebooks/05_linear_map_baseline.ipynb`
  6. `notebooks/06_robustness_latency.ipynb`
  7. `notebooks/07_report_tables_plots.ipynb`
- [ ] Ensure each notebook writes artifacts to `artifacts/runs/<run_id>/`
- [ ] Sync artifacts back to local repo for final report and demo

---

## 8) Final Deliverables

- [ ] One master comparison table (A vs B vs C)
- [ ] Failure case gallery with short interpretations
- [ ] Final demo run using selected best model in live app
