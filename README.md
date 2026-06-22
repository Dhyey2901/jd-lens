<div align="center">
  <h1>🔍 JD Lens</h1>
  <p>NLP-powered Job Description Analyser &amp; Role-Fit Scorer &nbsp;·&nbsp; IT / AI / Tech</p>

  [![CI](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhyey2901/jd-lens/actions/workflows/ci.yml)
  [![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)
  [![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
</div>

---

Paste a job description and a candidate resume — JD Lens scores fit using a **3-tier hybrid model**, surfaces a **Hiring Signal Score** across 7 resume-quality dimensions, shows side-by-side skill and keyword gaps, and predicts candidate outcome in a four-quadrant verdict. Shipped with a REST API, CI/CD, eval harness, and Streamlit Cloud deployment.

**Live demo: [jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app](https://jd-lens-ahnqxbthuw8t3tdr4mnuaw.streamlit.app/)**

---

### Demo

**Input** — paste or upload a JD + resume (PDF, DOCX, TXT):

![JD Lens — input screen](assets/screenshot_home.png)

**Output** — fit score, hiring signals, skill comparison, keyword analysis:

![JD Lens — results screen](assets/screenshot_results.png)

---

### How it scores

```text
JD-Match  = 0.30 × semantic similarity     (sentence-transformers/all-MiniLM-L6-v2)
          + 0.30 × keyword overlap          (TF-IDF bigram cosine + CountVectorizer top-30)
          + 0.40 × skill match rate         (exact + 0.5 × semantic, over JD skills)

Hiring    = weighted sum of 7 dimensions   (impact, depth, deployment, verbs,
Signal      calibrated per role type        trajectory, soft skills, skill gap)
```

Four-quadrant verdict from (JD-Match × Hiring Signal): **Strong Candidate / Overlooked Gem / Keyword Match–Verify Depth / Significant Gaps**

---

### Features

| | |
| --- | --- |
| **Scoring** | 3-tier hybrid · JD-adaptive dimension weights · section-aware skill weighting |
| **Signals** | 7-dimension Hiring Signal Score · role-type detection (6 categories) · four-quadrant prediction |
| **Extraction** | 147 known tools · context-phrase unknown tool detection (JD + resume) · 85-term soft skill vocab |
| **Alignment** | Top-3 resume bullets ranked against JD · each bullet matched to closest JD requirement |
| **Validation** | 5 MB file cap · 30-word minimum · 20k char limit · scanned-PDF warning |
| **API** | FastAPI `/analyse` + `/health` · SHA-256 result cache |
| **Quality** | 126 tests · ruff linting · GitHub Actions CI (Python 3.11 + 3.12) · eval harness with 5 gold pairs |

---

### Architecture

```text
jd-lens/
├── app.py          Streamlit UI — tabs, gauges, skill/keyword display
├── api.py          FastAPI REST layer
├── scorer.py       Hybrid scorer — semantic + TF-IDF + skill match + cache
├── signals.py      Hiring Signal Score — 7 dimensions, role detection, prediction
├── extractor.py    JD signal extraction, section-aware parsing, unknown tool detection
├── config.py       Models, weights, KNOWN_TOOLS (147), SOFT_SKILLS (85), limits
├── utils.py        Multi-format document extraction (PDF, DOCX, TXT)
├── tests/          126 unit tests across 4 test files
└── eval/           Ground-truth eval harness (5 gold pairs, make eval-fast)
```

---

### Quickstart

```bash
git clone https://github.com/Dhyey2901/jd-lens.git
cd jd-lens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> First run downloads `facebook/bart-large-mnli` (~1.6 GB) and `all-MiniLM-L6-v2` (~80 MB) into `~/.cache/huggingface/`. Subsequent starts are instant.

```bash
make test          # 126 unit tests
make eval-fast     # signal-only eval, no model download
make api           # FastAPI on http://localhost:8000
```

---

### Stack

`Python` `Streamlit` `FastAPI` `sentence-transformers` `HuggingFace Transformers` `scikit-learn` `Plotly` `GitHub Actions`

---

<div align="center">
  <i>Built by <a href="https://github.com/Dhyey2901">Dhyey Vyas</a> · MSc Data Science @ RMIT</i>
</div>
