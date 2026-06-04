# JD Lens 🔍

## NLP-powered Job Description Analyser & Role-Fit Scorer

[![CI](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Paste any job description and a candidate profile — JD Lens extracts key signals, classifies every sentence by intent, and scores candidate fit using a hybrid of **semantic embeddings**, **TF-IDF**, and **explicit skill matching**.

**🚀 Live demo: [jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)**

---

## Demo

> _Add a screenshot of the running app here._

---

## Features

| Feature | Technique | Detail |
| --- | --- | --- |
| **Skill & Tool Extraction** | Regex phrase matching | 80+ tools across languages, cloud, ML, DevOps |
| **Seniority / Education / Work Type** | Keyword signals | Parsed directly from JD text |
| **Sentence Classification** | Zero-shot NLI | `facebook/bart-large-mnli` → Required / Nice-to-Have / Responsibility / Company Info |
| **Semantic Fit Score** | Sentence Transformers | `all-MiniLM-L6-v2` cosine similarity (50% weight) |
| **Keyword Overlap Score** | TF-IDF bigram | scikit-learn (30% weight) |
| **Skill Match Rate** | Regex hit-rate | Explicit tool coverage (20% weight) |
| **Gap Analysis** | Set difference | Keyword and skill gaps surfaced clearly |
| **Downloadable Report** | JSON export | Full structured analysis per run |

---

## Architecture

```text
jd-lens/
├── app.py              # Streamlit UI — model caching, charts, layout
├── config.py           # Central config — models, weights, vocabulary
├── extractor.py        # Pure-regex JD signal extraction
├── classifier.py       # HuggingFace zero-shot sentence classification
├── scorer.py           # Hybrid semantic + TF-IDF + skill fit scorer
├── tests/
│   ├── test_extractor.py
│   └── test_scorer.py
├── .github/workflows/
│   └── ci.yml          # GitHub Actions — pytest + ruff on Python 3.11/3.12
└── requirements.txt
```

### Scoring formula

```text
fit_score = 0.50 × semantic_similarity   (sentence-transformers)
          + 0.30 × tfidf_similarity       (bigram TF-IDF cosine)
          + 0.20 × skill_match_rate       (regex hit-rate over JD tools)
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

> **First run:** `facebook/bart-large-mnli` (~1.6 GB) and `all-MiniLM-L6-v2` (~80 MB) are downloaded automatically by HuggingFace and cached in `~/.cache/huggingface/`. Subsequent runs load from cache and start in seconds.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
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
- [sentence-transformers](https://www.sbert.net/) — semantic similarity
- [HuggingFace Transformers](https://huggingface.co/facebook/bart-large-mnli) — zero-shot classification
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorisation
- [Plotly](https://plotly.com/python/) — interactive charts

---

## License

MIT
