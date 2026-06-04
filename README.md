# JD Lens 🔍

**NLP-powered Job Description Analyser & Role-Fit Scorer**

Paste any job description and a candidate profile — JD Lens extracts key signals, classifies every sentence by intent, and scores how well the candidate fits the role.

---

## Features

| Feature | How it works |
|---|---|
| **Skill & Tool Extraction** | spaCy phrase matching + regex against a curated tool list |
| **Seniority Detection** | Keyword signals (junior / mid / senior / manager) |
| **Sentence Classification** | HuggingFace `facebook/bart-large-mnli` zero-shot classification → Required / Nice-to-Have / Responsibility / Company Info |
| **Fit Score** | TF-IDF bigram cosine similarity (0–100%) |
| **Gap Analysis** | Keyword diff between JD and candidate profile |

---

## Project Structure

```
jd-lens/
├── app.py          # Streamlit UI
├── extractor.py    # spaCy extraction logic
├── classifier.py   # HuggingFace zero-shot classification
├── scorer.py       # TF-IDF fit scoring
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/<your-username>/jd-lens.git
cd jd-lens
pip install -r requirements.txt

# 2. Download the spaCy English model
python -m spacy download en_core_web_sm

# 3. Run the app
streamlit run app.py
```

The first run downloads `facebook/bart-large-mnli` (~1.6 GB) and caches it locally. Subsequent runs are fast.

---

## Tech Stack

- **Python 3.11+**
- [spaCy](https://spacy.io/) — NLP extraction
- [HuggingFace Transformers](https://huggingface.co/facebook/bart-large-mnli) — zero-shot classification
- [scikit-learn](https://scikit-learn.org/) — TF-IDF + cosine similarity
- [Streamlit](https://streamlit.io/) — UI + deployment

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app.py`.
4. Add a `packages.txt` file if you need system deps (not required here).
5. Click **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

> **Note:** The HuggingFace model is downloaded at startup on Streamlit Cloud. Expect a cold-start time of ~2–3 minutes on first deploy.

---

## Screenshot

> _Add a screenshot of the running app here._

---

## License

MIT
