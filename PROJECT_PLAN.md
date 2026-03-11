# Project Plan

## Vision–Language Perception and Decision Pipeline for an Autonomous Art Grading Agent

**MIS 285N — Generative AI | Final Project**
**Date:** March 2026

---

## 1. Goal

Build and rigorously evaluate a modular AI pipeline that uses vision-language perception to grade artwork in real time via a live camera feed. The system simulates an opinionated robotic art teacher whose grading is driven entirely by what a vision-language model perceives in the frame.

The central research question is:

> **How does the representational quality of a jointly trained vision-language embedding (CLIP) compare to a separately trained dual-encoder baseline with a learned linear alignment map, for the task of zero-shot visual concept detection in artwork?**

---

## 2. System Overview

```
Browser Webcam (MediaDevices API)
        │
        │  JPEG frames over WebSocket (~2 fps)
        ▼
  FastAPI Backend
        │
        ├─► Perception Module (select one per run)
        │       ├─ Model A: Pretrained CLIP (zero-shot)
        │       ├─ Model B: Fine-tuned CLIP (domain-adapted)
        │       └─ Model C: Separate Encoders + Linear Map (baseline)
        │              ↓
        │       Semantic Label Set + Confidence Scores
        │
        └─► Decision Engine
                ├─ Rule-based grader (deterministic)
                └─ LLM fallback (unrecognized scenes)
                       ↓
                  Grade (0–100) + Explanation
                       ↓
              Browser UI (live overlay on camera feed)
```

---

## 3. Perception Models (The Core Comparison)

Three perception model configurations will be implemented and evaluated side by side. All three output the same interface: a ranked list of `(label, confidence_score)` pairs.

### Model A — Pretrained CLIP (Zero-Shot Baseline)

Use the pretrained `openai/clip-vit-base-patch32` checkpoint without any fine-tuning. Given an image $I$ and candidate text labels $\{t_1, \ldots, t_n\}$:

$$v_I = f_\theta(I), \quad v_{t_i} = g_\theta(t_i)$$

$$\text{sim}(I, t_i) = \frac{v_I \cdot v_{t_i}}{\|v_I\| \|v_{t_i}\|}$$

Labels with $\text{sim}(I, t_i) \geq \tau$ are considered detected. Threshold $\tau$ is selected via a held-out validation split.

### Model B — Domain-Adapted CLIP (Fine-Tuned)

Fine-tune the CLIP checkpoint on a small labeled dataset of artwork images using the standard InfoNCE contrastive objective:

$$\mathcal{L}_\text{CLIP} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, t_i^+) / \kappa)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, t_j) / \kappa)}$$

where $\kappa$ is a learned temperature. This tests whether domain adaptation on artwork images measurably improves detection for our label set beyond the general-purpose pretrained weights.

### Model C — Separate Encoders + Linear Alignment Map (Ablation)

This is the key ablation that isolates the contribution of **joint contrastive training** in CLIP. The hypothesis is that CLIP's jointly trained embedding space is more geometrically aligned for cross-modal retrieval than independently trained encoders bridged by a post-hoc linear map.

**Architecture:**

1. **Image encoder** $f_\phi$: Pretrained ViT-B/16 or ResNet-50, trained independently on ImageNet (no text supervision, no joint training).
2. **Text encoder** $g_\psi$: Pretrained `sentence-transformers/all-MiniLM-L6-v2`, trained independently on text tasks.
3. **Linear alignment map** $W \in \mathbb{R}^{d_\text{text} \times d_\text{img}}$: Learned via ridge regression to project image embeddings into text embedding space.

**Training the linear map:**

Given labeled image-caption pairs $(I_k, t_k)$ from the evaluation dataset:

$$v_{I_k} = f_\phi(I_k), \quad v_{t_k} = g_\psi(t_k)$$

$$W^* = \arg\min_W \sum_k \| W v_{I_k} - v_{t_k} \|_2^2 + \lambda \|W\|_F^2$$

This has a closed-form ridge regression solution ($\lambda$ selected via leave-one-out CV). At inference:

$$\text{sim}_C(I, t) = \frac{(W v_I) \cdot v_t}{\|W v_I\| \|v_t\|}$$

**Why this comparison matters:**

If CLIP performs only marginally better than Model C, it implies the gap is attributable to encoder expressiveness rather than joint training. A significant CLIP advantage—especially in cross-modal retrieval metrics—provides direct evidence that InfoNCE contrastive training improves cross-modal alignment beyond what can be recovered by a linear post-hoc projection.

---

## 4. Milestones

### Phase 1 — Dataset, Labeling & Baseline Perception (Weeks 1–2)

- [x] Write project proposal / plan
- [ ] Set up project structure, virtualenv, dependencies
- [ ] Curate and label evaluation dataset (see §5)
- [ ] Implement Model A (zero-shot CLIP) and run initial evaluation
- [ ] Implement Model C (separate encoders + ridge regression linear map)
- [ ] Model A vs. Model C comparison on all metrics in §6
- [ ] Threshold sensitivity analysis: sweep $\tau \in \{0.15, 0.20, 0.25, 0.30\}$ for Model A
- [ ] Embedding space visualization: t-SNE / UMAP plots for all three models

### Phase 2 — Fine-Tuning & Full Comparison (Week 3)

- [ ] Prepare fine-tuning image-label pairs (subset of evaluation dataset + augmentations)
- [ ] Fine-tune CLIP (Model B) with InfoNCE loss; report train/val contrastive loss curves
- [ ] Re-run all evaluation metrics for Model B
- [ ] Full three-way comparison: A vs. B vs. C
- [ ] McNemar's test for pairwise statistical significance
- [ ] Latency profiling for all three models (CPU; GPU if available)

### Phase 3 — Grading Engine & Live Web App (Weeks 3–4)

- [ ] Implement rule-based decision engine (wired to any perception model)
- [ ] Implement LLM fallback with teacher persona
- [ ] Backend WebSocket endpoint: receive JPEG frames, return grade JSON
- [ ] Frontend: browser camera (`getUserMedia`) + Canvas overlay for grade display
- [ ] Model selector in UI (switch between A / B / C live)
- [ ] Per-label confidence bar chart, adjustable FPS slider, freeze-frame mode

### Phase 4 — Analysis, Report & Demo (Weeks 4–5)

- [ ] Finalize quantitative comparison tables and plots
- [ ] Failure case analysis: which labels does each model most frequently miss or confuse?
- [ ] Record live camera demo
- [ ] Final written report and presentation

---

## 5. Evaluation Dataset

A manually curated and annotated evaluation set will be assembled to support per-label metric computation.

### Composition

| Category | # Images | Source |
|---|---|---|
| Artwork with cats | 30 | Web, WikiArt, personal photos |
| Artwork without cats | 30 | WikiArt |
| Scenes with visible currency | 20 | Web |
| Geometric / abstract artwork | 20 | WikiArt |
| Ambiguous / mixed scenes | 20 | Web, personal |
| **Total** | **120** | |

### 5.1 Data Collection Procedure

Images are collected in three steps:

**Step 1 — Download from WikiArt**

WikiArt (wikiart.org) provides a large public collection of fine art images organized by style. We will use the WikiArt dataset (available via HuggingFace Datasets as `huggan/wikiart`) to pull images tagged with styles relevant to our label set (e.g., Abstract Expressionism, Geometric Abstraction, Impressionism). A download script in `notebooks/01_data_collection.ipynb` will fetch images and save them under `data/raw/wikiart/`.

```
huggan/wikiart  →  filter by style tag  →  download ~80 images  →  data/raw/wikiart/
```

**Step 2 — Supplement with web images**

For categories not well-covered by WikiArt (cats in images, currency, mixed scenes), images will be collected manually via Google Image Search and saved to `data/raw/supplemental/`. All images must be verified to be either CC-licensed or used solely for non-commercial academic evaluation.

**Step 3 — Personal / live captures**

Additional test images will be captured directly using the live camera interface once the web app is built, covering real-world conditions (varying lighting, camera angle, partial occlusion). These are saved to `data/raw/captured/`.

**Final structure:**
```
data/
  raw/
    wikiart/          # downloaded from HuggingFace
    supplemental/     # manually sourced
    captured/         # live camera captures
  labeled/
    annotations.csv   # ground-truth label matrix (image_id × label → 0/1)
    train.txt         # image paths for fine-tuning split
    val.txt
    test.txt
```

### 5.2 Labeling Protocol

Labels are stored in `data/labeled/annotations.csv` with one row per image and one binary column per concept. A label is marked 1 if a human annotator would unambiguously identify that concept anywhere in the image.

Labeling is done using Label Studio (free, local, runs in browser). The project is exported as CSV. Two annotators label each image independently; conflicts are resolved by a third reviewer. Cohen's kappa is computed per label and reported alongside evaluation metrics. Labels are frozen after annotation is complete — no re-labeling after models are trained.

### 5.3 Fine-Tuning Split

Of the 120 images, 40 will be reserved for fine-tuning Model B (`train.txt`), 20 for validation (`val.txt`), and 60 for the held-out test set (`test.txt`). The fine-tuning split is sampled to be label-balanced. The test set is never used during training or threshold selection for any model.

### Candidate Label Set

```
cat               painting          drawing
canvas            abstract art      geometric shapes
dollar bill       cash              person
colorful artwork  monochrome artwork sculpture
```

---

## 6. Evaluation Metrics

All three models are evaluated on the held-out test split (80 images) using the metrics below, computed per label then macro-averaged.

### 6.1 Detection Accuracy

| Metric | Definition |
|---|---|
| Precision | $\text{TP} / (\text{TP} + \text{FP})$, at operating threshold |
| Recall | $\text{TP} / (\text{TP} + \text{FN})$, at operating threshold |
| F1 | $2 \cdot P \cdot R \;/\; (P + R)$ — primary ranking metric |
| Average Precision (AP) | Area under the precision-recall curve (threshold-agnostic) |
| mAP | Mean AP across all labels — overall model comparison metric |

Operating thresholds are selected on a validation split (20 images) by maximizing macro-F1, separately for each model.

### 6.2 Embedding Space Quality

**Separability ratio:** For each label $t$, let $P_t$ be images where $t$ is ground-truth positive and $N_t$ be negatives:

$$\text{sep}(t) = \frac{\bar{s}_\text{intra}(t)}{\bar{s}_\text{inter}(t)}, \quad \bar{s}_\text{intra}(t) = \frac{1}{|P_t|} \sum_{I \in P_t} \text{sim}(I, t), \quad \bar{s}_\text{inter}(t) = \frac{1}{|N_t|} \sum_{I \in N_t} \text{sim}(I, t)$$

A higher ratio indicates a more separable embedding space. The distribution of cosine similarities for positive and negative images will be plotted per label.

**Cross-modal retrieval (image $\to$ text and text $\to$ image):**

Using the 120-image evaluation set as a paired image-text corpus, compute Recall@1 and Recall@5 in both retrieval directions. This directly captures how well each model's embedding space supports cross-modal alignment — independent of any threshold choice.

### 6.3 Threshold Sensitivity

Plot macro-F1 as a function of $\tau$ for all three models on the validation set. This shows robustness: a model that remains high-F1 across a wide range of $\tau$ is less sensitive to threshold choice and easier to deploy.

### 6.4 Latency

| Measurement | Definition |
|---|---|
| Image encode latency | Time for the image encoder only (per frame) |
| Full pipeline latency | Frame receipt → grade JSON response (end-to-end) |
| Sustained throughput | Frames per second over a 60-second WebSocket stream |

Reported as mean ± standard deviation over 100 forward passes. Measured on CPU (all models); GPU results included if available.

### 6.5 Statistical Significance

McNemar's test (paired, per-image correct/incorrect) is applied to pairwise detection accuracy: A vs. C, B vs. C, A vs. B. Significance at $\alpha = 0.05$ (Bonferroni-corrected for 3 tests: $\alpha' = 0.0167$).

---

## 7. Grading Logic

The decision engine is model-agnostic — it receives the label set and confidence scores from whichever perception model is active. The rules themselves are not re-trained or tuned per model.

### Scoring Rules

| Condition | Effect |
|---|---|
| No cats detected | Grade = 0 (immediately terminal) |
| 1 cat detected | +20 |
| 2 cats detected | +40 |
| 3+ cats detected | +60 (ceiling on cat bonus) |
| Cash / dollar bill visible | Grade forced to Uniform(90, 100) |
| Geometric shapes detected | −15 |
| Abstract art detected | Uniform(−15, +15) random modifier |
| Scene unrecognized by rules | LLM fallback |

Grade is always clamped to $[0, 100]$ after all modifiers are applied.

### LLM Fallback

When no rule trigger is matched, the detected labels and their confidence scores are serialized into a natural-language prompt sent to a language model with the teacher's system persona. The LLM returns a JSON object with a grade and a one-sentence explanation. The response is parsed and validated before display. If the LLM is unavailable, a deterministic offline fallback proportional to the number of detected labels is used.

---

## 8. Live Camera Web App

The primary demo interface captures frames from the user's webcam using the browser's `MediaDevices.getUserMedia()` API and streams them to the backend over a WebSocket. The entire app runs locally — no cloud deployment required for the demo.

### 8.1 Interface Features

- Live camera preview rendered in a `<video>` element; frames captured to a hidden `<canvas>` element at the configured FPS
- Grade overlay displayed prominently over the video feed
- Per-label confidence bar chart, updated each inference cycle
- Teacher commentary text (from rule engine or LLM)
- **Model selector**: dropdown to switch between Model A / B / C; sends a control message over the WebSocket without a page reload
- **Freeze-frame button**: pauses inference and opens a side panel showing the full ranked label list and raw cosine scores
- **FPS slider**: 0.5 – 5 fps inference rate (default: 2 fps)

### 8.2 How Frames Get from Browser to Backend

The browser captures a frame by drawing the current video element to a Canvas and calling `canvas.toBlob()` to get a compressed JPEG. That blob is converted to base64 and sent as a JSON message over a persistent WebSocket connection to the FastAPI backend:

```
browser video element
    │
    │  canvas.drawImage() at configured FPS
    ▼
hidden <canvas>
    │
    │  toBlob() → base64 encode
    ▼
WebSocket message  →  FastAPI /ws endpoint
    │
    │  decode base64 → PIL image → perception model → grader
    ▼
JSON response  →  browser renders grade + labels
```

The WebSocket stays open for the duration of the session. The model selector sends a separate control frame `{"action": "set_model", "model": "B"}` to hot-swap the perception backend without reconnecting.

### 8.3 Backend WebSocket Response

```json
{
  "grade": 40,
  "explanation": "Two cats. A start, I suppose.",
  "labels": [
    { "label": "cat",      "score": 0.84 },
    { "label": "painting", "score": 0.63 }
  ],
  "model": "A",
  "latency_ms": 82
}
```

### 8.4 Running the App Locally

```bash
# 1. Create and activate virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and add API keys if using LLM fallback
cp .env.example .env

# 4. Launch the server (models load on startup)
uvicorn app.main:app --reload --port 8000

# 5. Open browser
open http://localhost:8000
```

The browser will prompt for camera permission on first load. Model weights for A and C are loaded from local cache (HuggingFace caches to `~/.cache/huggingface/` after first download). Model B weights are loaded from `data/checkpoints/clip_finetuned.pt` after training in Colab (see §9).

---

## 9. Training on Google Colab

All model training and most evaluation analysis is done in Google Colab (free T4 GPU tier is sufficient for CLIP fine-tuning at this dataset scale). The local machine runs only inference and the web app.

### 9.1 Notebook Structure

```
notebooks/
  01_data_collection.ipynb    # Download WikiArt, assemble dataset
  02_model_A_eval.ipynb       # Zero-shot CLIP evaluation (no training needed)
  03_model_C_train.ipynb      # Extract embeddings, train ridge regression map W
  04_model_B_finetune.ipynb   # Fine-tune CLIP with InfoNCE loss
  05_comparison.ipynb         # Three-way metric comparison, plots, McNemar test
```

Each notebook is self-contained and designed to be opened directly in Colab via a "Open in Colab" badge. Data is loaded from Google Drive; trained weights are saved back to Drive and then pulled down to the local repo.

### 9.2 Colab Training Workflow (Model B — Fine-Tuned CLIP)

```
Local machine                          Google Colab (T4 GPU)
─────────────────                      ─────────────────────────────
data/labeled/train.txt  ──upload──►   Mount Google Drive
  + raw images                         Load images + annotations
                                        ↓
                                       Load openai/clip-vit-base-patch32
                                        ↓
                                       Training loop:
                                         for epoch in range(N):
                                           sample batch of (image, label) pairs
                                           encode image + text
                                           compute InfoNCE loss
                                           backward + AdamW step
                                         ↓
                                       Save clip_finetuned.pt to Drive
                                        ↓
data/checkpoints/  ◄──download──   clip_finetuned.pt
clip_finetuned.pt
        ↓
  app/main.py loads weights
  for Model B at server startup
```

### 9.3 Training Details (Model B)

| Hyperparameter | Value | Notes |
|---|---|---|
| Base model | `openai/clip-vit-base-patch32` | Pretrained weights |
| Fine-tune layers | Top 2 transformer blocks of image encoder + full text encoder | Freeze lower layers to preserve general features |
| Optimizer | AdamW | lr = 1e-5, weight decay = 0.01 |
| Batch size | 16 | Constrained by T4 VRAM |
| Epochs | 10–20 | Early stop on validation contrastive loss |
| Temperature $\kappa$ | Learned (initialized at 0.07) | |
| Data augmentation | Random crop, horizontal flip, color jitter | Applied to images only |
| Loss | InfoNCE (symmetric image-text contrastive) | |

With 40 training images and augmentation, effective training set size is ~200–400 samples per epoch. This is a low-data regime — we expect Model B to improve on artwork-specific labels and may overfit on generic ones, which is itself an informative result.

### 9.4 Training Details (Model C — Linear Map)

Model C requires no GPU. The ridge regression map $W$ is trained in `03_model_C_train.ipynb` on CPU in Colab:

1. Extract image embeddings for all training images using the frozen ViT/ResNet encoder (no gradient)
2. Extract text embeddings for all label strings using the frozen sentence-transformer
3. Fit `sklearn.linear_model.RidgeCV` with `alphas=[0.01, 0.1, 1.0, 10.0]` using leave-one-out CV
4. Save the fitted $W$ matrix as `data/checkpoints/linear_map_W.npy`

Total training time: < 1 minute on CPU.

---

## 10. Tech Stack


| Layer | Technology |
|---|---|
| Vision-language model | CLIP (`openai/clip-vit-base-patch32`) via HuggingFace |
| Image encoder (Model C) | ViT-B/16 or ResNet-50 (`torchvision.models`, pretrained) |
| Text encoder (Model C) | `sentence-transformers/all-MiniLM-L6-v2` |
| Linear map training | `sklearn.linear_model.Ridge` (closed-form) |
| CLIP fine-tuning (Model B) | PyTorch + HuggingFace `Trainer` |
| Embedding visualization | UMAP (`umap-learn`) / t-SNE (`sklearn`) + matplotlib |
| Web server | FastAPI + Uvicorn |
| Real-time streaming | WebSockets (FastAPI native) |
| Frontend camera | Browser MediaDevices API + Canvas API |
| LLM fallback | OpenAI API (`gpt-4o-mini`) or local Ollama |
| Statistics | `scipy.stats` (McNemar), `sklearn.metrics` (AP, mAP) |
| Testing | pytest |

---

## 11. Hypotheses

**H1:** CLIP (Model A) will achieve higher mAP and cross-modal Recall@1 than the separate encoder + linear map baseline (Model C), supporting the claim that joint contrastive training produces better cross-modal alignment than post-hoc linear projection.

**H2:** Fine-tuned CLIP (Model B) will outperform pretrained CLIP (Model A) on artwork-specific labels (`abstract art`, `canvas`, `drawing`) while showing comparable or negligible improvement on generic labels (`cat`, `person`).

**H3:** Model C will exhibit higher intra-class / inter-class cosine similarity overlap than CLIP (lower separability ratios), consistent with the expectation that independent training leaves a residual modality gap that a linear map only partially closes.

**H4:** All models will produce perceptually coherent live grades on artwork presented to the webcam, demonstrating that zero-shot visual concept detection is viable for real-time semantic classification even without domain-specific training data.

---

## 12. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CLIP performs poorly on fine art styles (domain gap) | Medium | Explicitly measure and report. Fine-tune (Model B) if gap is large |
| Fine-tuning data too small for meaningful improvement | Medium | Use data augmentation; report null result honestly |
| Linear map collapses (near-zero variance in projections) | Low | Check encoder alignment dimensionality; tune ridge $\lambda$ via CV |
| WebSocket latency too high for real-time feel | Low | Cap inference at 2 fps; fall back to periodic POST if needed |
| LLM API unavailable or expensive | Low | Local Ollama fallback; cache repeated prompts |
| Inter-annotator disagreement too high | Low | Restrict label set to visually unambiguous concepts; report kappa |

---

## 13. References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- He et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Liang et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*. NeurIPS 2022.
- McNemar (1947). *Note on the sampling error of the difference between correlated proportions*. Psychometrika.
- WikiArt dataset: https://www.wikiart.org
- HuggingFace Transformers: https://huggingface.co
