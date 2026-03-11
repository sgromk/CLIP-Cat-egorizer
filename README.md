# Vision–Language Perception and Decision Pipeline for an Autonomous Art Grading Agent

**MIS 285N — Generative AI | Final Project**

---

## Overview

This project builds a modular AI pipeline that simulates a robotic art teacher — an intentionally opinionated system that grades artwork using visual perception and rule-based (and LLM-assisted) decision making.

```
Camera Frame → Vision Model (CLIP) → Semantic Labels → Decision Engine → Grade
```

The teacher evaluates artwork according to arbitrary, humorous aesthetic rules (e.g., every cat raises the grade, visible cash is accepted as a bribe). The architecture mirrors real-world systems used in robotics, quality inspection, and autonomous monitoring.

---

## Architecture

| Stage | Module | Technology |
|---|---|---|
| 1 — Perception | `app/perception.py` | OpenAI CLIP |
| 2 — Decision | `app/grader.py` | Rule engine / neural classifier |
| 2b — Fallback | `app/llm_fallback.py` | LLM (OpenAI / local) |
| Web Interface | `app/main.py` + `frontend/` | FastAPI + HTML/JS |

---

## Grading Rules (Teacher Personality)

- **No cats detected** → grade = 0
- **Each cat** adds +20 points (caps at 3 cats → 60 base)
- **Cash visible** → grade ≥ 90 (bribe accepted)
- **Excessive geometric shapes** → penalty applied
- **Abstract expressionism** → random grade modifier

---

## Project Structure

```
FinalProject/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI server + endpoints
│   ├── perception.py    # CLIP vision-language perception module
│   ├── grader.py        # Rule-based grading engine
│   └── llm_fallback.py  # LLM fallback for unrecognized scenes
├── frontend/
│   ├── index.html       # Web UI
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_perception.py
│   └── test_grader.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── test_images/     # Sample artwork images for testing
├── .gitignore
├── PROJECT_PLAN.md
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd FinalProject
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Run the web app

```bash
uvicorn app.main:app --reload
```

Navigate to `http://localhost:8000` in your browser.

---

## Running Tests

```bash
pytest tests/
```

---

## Evaluation Criteria

| Metric | Description |
|---|---|
| Perception Accuracy | How often CLIP correctly identifies objects (cats, cash, shapes) |
| Decision Consistency | Whether grading rules are applied deterministically |
| Latency | End-to-end processing time per frame |

---

## Industry Analogues

Although intentionally humorous, this pipeline mirrors real-world systems:

- Manufacturing quality inspection
- Retail shelf analytics
- Security monitoring
- Warehouse robotics

---

## Extensions (Stretch Goals)

- Object detection (YOLO) to count multiple cats precisely
- Passing CLIP embeddings directly to a learned decision network
- LLM-generated grade explanations in the teacher's voice
- Multiple teacher personality profiles

---

## Author

MIS 285N Final Project — March 2026
