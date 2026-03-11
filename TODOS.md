# Immediate Human TODOs

Things that require manual effort and can't be scripted — do these first so the automated work is unblocked.

---

## 1. Data Collection & Labeling  ← biggest time sink, start now

- [ ] Browse [WikiArt](https://www.wikiart.org) and hand-pick ~80 images across the needed categories; download them into `data/raw/wikiart/`
  - Want: abstract art, geometric work, paintings with cats, general paintings/drawings
- [ ] Supplement with ~40 images from Google Images for categories WikiArt lacks: cats in artwork, scenes with visible currency, ambiguous mixed scenes → `data/raw/supplemental/`
  - Check that images are CC-licensed or clearly usable for academic/non-commercial purposes
- [ ] Install Label Studio locally (`pip install label-studio`, then `label-studio start`) and create a project for binary multi-label annotation
- [ ] Annotate all 120 images with ground-truth labels (two passes if possible for kappa)
- [ ] Export annotations as CSV → save to `data/labeled/annotations.csv`
- [ ] Manually create `train.txt`, `val.txt`, `test.txt` splits (40 / 20 / 60) ensuring label balance in the training split

---

## 2. Google Colab Setup

- [ ] Upload the `data/` folder to Google Drive (or mount the repo via Drive)
- [ ] Open `notebooks/03_model_C_train.ipynb` in Colab — run it first since it needs no GPU and validates the data pipeline end-to-end
- [ ] Open `notebooks/04_model_B_finetune.ipynb` in Colab — enable GPU runtime (T4), run fine-tuning
- [ ] Download `clip_finetuned.pt` and `linear_map_W.npy` from Drive into `data/checkpoints/` locally

---

## 3. API Keys / Local LLM

- [ ] Decide: use OpenAI API or run Ollama locally for the LLM fallback
  - OpenAI: grab an API key, add to `.env` as `OPENAI_API_KEY=...`
  - Ollama: `ollama pull llama3` (runs locally, free, ~4GB)
- [ ] Copy `.env.example` to `.env` and fill in accordingly

---

## 4. Local Environment

- [ ] `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] First run of `uvicorn app.main:app --reload` to confirm CLIP loads and the server starts
- [ ] Test camera access at `http://localhost:8000` — grant browser camera permission

---

## 5. Demo Preparation

- [ ] Collect a few physical props for the live demo: something with a cat on it, some cash (or a printed photo of cash), a geometric drawing
- [ ] Record a short screen capture of the live camera grading for the final presentation
