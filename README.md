# JD Lens 🔍

## NLP-powered Job Description Analyser & Role-Fit Scorer for IT / AI / Tech

[![CI](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Paste any IT/AI/tech job description and a candidate resume — JD Lens extracts structured signals, scores candidate fit, surfaces a **Hiring Signal Score** with four-quadrant prediction, and shows a side-by-side skill and keyword comparison.

**🚀 Live demo: [jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)**

---

## Demo

**Input** — paste or upload a JD + resume (PDF, DOCX, TXT):

![JD Lens — input screen](assets/screenshot_home.png)

**Output** — fit score gauge, hiring signals, skill comparison, keyword analysis:

![JD Lens — results screen](assets/screenshot_results.png)

---

## Features

| Feature | Technique | Detail |
| --- | --- | --- |
| **JD-Match Score** | Semantic + keyword + skill | 3-component hybrid; 0.30 × semantic + 0.30 × keyword + 0.40 × skill match |
| **Hiring Signal Score** | 7-dimension rule-based | Impact & Metrics, Project Depth, Deployment, Verb Strength, Learning Trajectory, Soft Skills, Skill Gap — weighted by detected role type |
| **Four-Quadrant Prediction** | Signal × JD-match axes | Strong Candidate / Overlooked Gem / Keyword Match–Verify Depth / Significant Gaps |
| **Role-Adaptive Weights** | Keyword cluster detection | Detects: `data_engineering`, `data_analyst`, `research_ml`, `software_engineering`, `consulting`, `general` — applies role-specific dimension weights |
| **Section-aware JD Parsing** | Header regex + multipliers | Title 3×, Required 2×, Preferred 1×, Nice-to-have 0.5× — skill frequencies normalised [0.5, 1.0] |
| **Skill & Tool Extraction** | Regex phrase matching | 124 tools across languages, cloud, ML/AI, DevOps, databases |
| **Unknown Tool Detection** | Context-phrase NLP | Catches tools not in core vocab via "experience with X" patterns — applied to both JD and resume |
| **Soft Skill Coverage** | 55-term vocabulary | Two-column JD-required vs. candidate-detected view; ✓/✗ per skill |
| **Keyword Analysis** | CountVectorizer unigrams+bigrams | Top-30 JD keywords split into matched (green) and missing (red) — visible in UI |
| **Bullet-to-Requirement Alignment** | Sentence-transformer embeddings | Top-3 resume bullets ranked by JD relevance; each shows best-matching JD requirement sentence |
| **Sentence Classification** | Zero-shot NLI | `facebook/bart-large-mnli` → Required / Nice-to-Have / Responsibility / Company Info |
| **Alias Normalisation** | Dictionary substitution | sklearn → scikit-learn, postgres → postgresql, aws → aws, etc. |
| **Result Caching** | SHA-256 in-process cache | Repeated (JD, candidate, skills) triples skip all model inference |
| **Multi-format Upload** | PDF / DOCX / TXT | Drag-and-drop for both JD and resume inputs |
| **Candidate Compare** | Side-by-side analysis | Score multiple candidates against one JD; sortable skill matrix |
| **Downloadable Report** | JSON export | Full structured analysis per run |

---

## Architecture

```text
jd-lens/
├── app.py              # Streamlit UI — tabs, charts, skill/keyword display
├── api.py              # FastAPI REST layer — /analyse + /health endpoints
├── config.py           # Central config — models, weights, vocabulary (KNOWN_TOOLS, SOFT_SKILLS)
├── extractor.py        # JD signal extraction, section-aware parsing, unknown tool detection
├── classifier.py       # Zero-shot sentence classification (HuggingFace)
├── scorer.py           # Hybrid scorer: semantic + TF-IDF + keyword + skill match + cache
├── signals.py          # Hiring Signal Score — 7 dimensions, role-type detection, prediction
├── utils.py            # Multi-format document extraction (PDF, DOCX, TXT)
├── tests/
│   ├── test_extractor.py   # Section parsing, skill extraction, unknown tools
│   ├── test_scorer.py      # Keyword extraction, resume section detection, TF-IDF
│   ├── test_signals.py     # All 7 signal dimensions, role detection, skill gap scoring
│   └── test_api.py         # FastAPI endpoint tests
├── eval/
│   ├── ground_truth.json   # 5 gold (JD, resume) pairs covering all 4 quadrants
│   └── run_eval.py         # Eval harness — --fast (signals only) or full (with model)
├── .github/workflows/
│   └── ci.yml              # GitHub Actions — pytest + ruff on Python 3.11/3.12
└── requirements.txt
```

### JD-Match scoring formula

```text
fit_score = 0.30 × semantic_similarity     (sentence-transformers bi-encoder cosine)
          + 0.30 × keyword_overlap         (0.60 × TF-IDF bigram cosine
                                          + 0.40 × CountVectorizer top-30 unigram+bigram overlap)
          + 0.40 × skill_match_rate        (exact hit × 1.0 + semantic hit × 0.5, over JD skills)
```

When no skills are detected in the JD, the 40% skill weight redistributes proportionally to semantic and keyword so scores stay meaningful.

### Hiring Signal Score

```text
hiring_signal = w1 × impact_metrics
              + w2 × project_depth
              + w3 × deployment_evidence
              + w4 × action_verb_strength
              + w5 × learning_trajectory
              + w6 × soft_skill_coverage     (vs. JD soft skills)
              + w7 × skill_gap_coverage      (vs. JD required skills, section-weighted)
```

Weights `w1–w7` are role-adaptive: `detect_role_type()` clusters the JD into one of 6 categories and loads the corresponding weight vector from `_ROLE_WEIGHTS`.

### Four-Quadrant Prediction

```text
               JD-Match
           Low        High
          ┌──────────┬───────────────┐
High      │ Overlooked│  Strong       │
Signal    │ Gem       │  Candidate    │
          ├──────────┼───────────────┤
Low       │ Significant│ Keyword Match │
Signal    │ Gaps      │ – Verify Depth│
          └──────────┴───────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/Dhyey2901/jd-lens.git
cd jd-lens

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

> **First run:** `facebook/bart-large-mnli` (~1.6 GB) and `all-MiniLM-L6-v2` (~80 MB) are downloaded automatically and cached in `~/.cache/huggingface/`. Subsequent runs start in seconds.

---

## Running Tests

```bash
# Unit tests (no model download required)
pytest tests/ -v

# Or via Makefile
make test

# Eval harness — fast mode (rule-based signals only, no model)
make eval-fast

# Eval harness — full (downloads embedding model on first run)
make eval
```

---

## REST API

```bash
# Start the API
make api
# → http://localhost:8000

# Health check
curl http://localhost:8000/health

# Analyse endpoint
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"jd_text": "...", "candidate_text": "..."}'
```

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Cloud installs `requirements.txt` automatically. Allow ~3 minutes for cold start while models download.

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI & deployment
- [sentence-transformers](https://www.sbert.net/) — bi-encoder scoring + bullet alignment
- [HuggingFace Transformers](https://huggingface.co/facebook/bart-large-mnli) — zero-shot sentence classification
- [scikit-learn](https://scikit-learn.org/) — TF-IDF + CountVectorizer keyword extraction
- [FastAPI](https://fastapi.tiangolo.com/) — REST API layer
- [Plotly](https://plotly.com/python/) — interactive score charts

---

## License

MIT
