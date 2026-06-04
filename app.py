"""JD Lens — Streamlit UI entry point."""
from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from classifier import classify_jd, get_pipeline
from config import EMBEDDING_MODEL, ZERO_SHOT_MODEL
from extractor import extract_jd
from scorer import compute_fit_score, get_embedding_model

# ── Sample data ────────────────────────────────────────────────────────────────

SAMPLE_JD = """Senior Data Scientist — AI/ML Platform (Remote)

We are building the next generation of AI-powered analytics for enterprise clients
and are looking for a Senior Data Scientist with 5+ years of experience.

Required skills: Python, SQL, PyTorch, scikit-learn, AWS, Docker, Airflow, Kafka.
Nice to have: Databricks, Snowflake, dbt, MLflow, LangChain.

Responsibilities:
- Design and deploy production ML models for predictive analytics and NLP tasks.
- Collaborate cross-functionally with engineering, product, and stakeholder teams.
- Lead data-driven decision making and communicate insights to non-technical audiences.
- Mentor junior data scientists and contribute to ML platform architecture.

Requirements:
- Bachelor's degree in Computer Science, Statistics, or a related field.
- Strong problem solving and analytical thinking skills.
- Experience with stakeholder management and presentation skills.

This is a fully remote position. We are a fast-paced SaaS startup."""

SAMPLE_CANDIDATE = """Senior Data Scientist with 6 years of experience in AI/ML.

Technical skills: Python (expert), SQL, PyTorch, scikit-learn, TensorFlow, AWS (EC2, S3, SageMaker),
Docker, Airflow, Kafka, MLflow, Pandas, NumPy, FastAPI.

Experience:
- Built and deployed NLP models for document classification at scale (100M+ docs/day).
- Designed end-to-end ML pipelines using Airflow and Docker on AWS.
- Led stakeholder presentations and translated business requirements into ML solutions.
- Mentored a team of 3 junior data scientists.

Education: Bachelor of Science in Computer Science.
Open to fully remote roles. Strong communication and problem solving background."""

st.set_page_config(
    page_title="JD Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model loading (cached across sessions) ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifier():
    return get_pipeline()


@st.cache_resource(show_spinner=False)
def load_embedder():
    return get_embedding_model()


# ── Visual helpers ─────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 65:
        return "#2ecc71"
    return "#f39c12" if score >= 40 else "#e74c3c"


def _score_grade(score: float) -> str:
    if score >= 65:
        return "Strong Fit ✅"
    return "Partial Fit ⚠️" if score >= 40 else "Weak Fit ❌"


def _gauge_chart(score: float) -> go.Figure:
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 52, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 40],  "color": "rgba(231,76,60,0.12)"},
                {"range": [40, 65], "color": "rgba(243,156,18,0.12)"},
                {"range": [65, 100],"color": "rgba(46,204,113,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _skill_tags_html(matched: list[str], missing: list[str]) -> str:
    parts = []
    for s in matched:
        parts.append(
            f'<span style="background:#2ecc7118;color:#2ecc71;border:1px solid #2ecc7144;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
            f'✓ {s}</span>'
        )
    for s in missing:
        parts.append(
            f'<span style="background:#e74c3c18;color:#e74c3c;border:1px solid #e74c3c44;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
            f'✗ {s}</span>'
        )
    return "".join(parts)


def _soft_tags_html(soft_skills: list[str]) -> str:
    return "".join(
        f'<span style="background:#8e44ad18;color:#8e44ad;border:1px solid #8e44ad44;'
        f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
        f'{s}</span>'
        for s in soft_skills
    )


def _breakdown_chart(breakdown: dict[str, float]) -> go.Figure:
    labels, values = list(breakdown.keys()), list(breakdown.values())
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[_score_color(v) for v in values],
        text=[f"{v}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=10), height=160,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13),
    )
    return fig


def _classification_pie(buckets: dict[str, list]) -> go.Figure:
    palette = {
        "Required": "#e74c3c",
        "Nice-to-Have": "#f39c12",
        "Responsibility": "#3498db",
        "Company Info": "#95a5a6",
    }
    labels, values, colors = [], [], []
    for bucket, items in buckets.items():
        if items:
            labels.append(bucket)
            values.append(len(items))
            colors.append(palette.get(bucket, "#999"))
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker_colors=colors, textinfo="label+percent", textfont_size=12,
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _generate_recommendation(score_info: dict, jd_info: dict) -> str:
    fit = score_info["fit_score"]
    matched = score_info["matched_skills"]
    missing = score_info["missing_skills"]
    total = len(jd_info["skills_and_tools"])
    soft = jd_info.get("soft_skills", [])

    if fit >= 65:
        verdict = "**Strong Fit.**"
        tone = "Candidate covers the core requirements well and is ready for interview."
    elif fit >= 40:
        verdict = "**Partial Fit.**"
        tone = "Candidate meets some requirements — targeted upskilling recommended before hire."
    else:
        verdict = "**Weak Fit.**"
        tone = "Significant gaps detected — this role may not be the right match at this stage."

    skill_line = (
        f"Matches {len(matched)}/{total} required tools."
        if total else "No specific tools detected in JD."
    )
    gap_line = (
        f"Key technical gaps: {', '.join(missing[:3])}."
        if missing else "No critical technical gaps detected."
    )
    soft_line = (
        f"Role also signals soft skills: {', '.join(soft[:3])}."
        if soft else ""
    )

    parts = [verdict, tone, skill_line, gap_line]
    if soft_line:
        parts.append(soft_line)
    return " ".join(parts)


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🔍 JD Lens")
st.caption("NLP-powered Job Description Analyser & Role-Fit Scorer")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
| Step | Model | What it does |
| --- | --- | --- |
| Extraction | Regex | Hard skills, soft skills, seniority, education, industry, work type |
| Scoring | `all-MiniLM-L6-v2` + TF-IDF | Semantic (50%) + keyword (30%) + skill hit-rate (20%) |
| Classification | `facebook/bart-large-mnli` | Buckets each JD sentence: Required / Nice-to-Have / Responsibility / Company Info |
    """)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_analyse, tab_compare = st.tabs(["🔍 Analyse", "👥 Compare Candidates"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE
# ════════════════════════════════════════════════════════════════════════════════

with tab_analyse:
    # Sample button
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        if st.button("Try a sample ✨", use_container_width=True):
            st.session_state["jd_text"] = SAMPLE_JD
            st.session_state["candidate_text"] = SAMPLE_CANDIDATE

    col_jd, col_cand = st.columns(2)

    with col_jd:
        st.subheader("Job Description")
        st.caption("Paste the full JD — requirements, responsibilities, and qualifications.")
        jd_text = st.text_area(
            "jd", height=280,
            placeholder="e.g. Senior Python Engineer with 5+ years of AWS and Docker experience…",
            label_visibility="collapsed",
            key="jd_text",
        )

    with col_cand:
        st.subheader("Candidate Profile")
        st.caption("Paste a resume, LinkedIn summary, or a short skills bio.")
        candidate_text = st.text_area(
            "candidate", height=280,
            placeholder="e.g. 6 years of Python, built REST APIs with FastAPI, daily AWS and Docker usage…",
            label_visibility="collapsed",
            key="candidate_text",
        )

    analyse = st.button("Analyse ✨", type="primary", use_container_width=True)

    if analyse:
        if not jd_text.strip() or not candidate_text.strip():
            st.error("Please provide both a job description and a candidate profile.")
            st.stop()

        with st.spinner("Loading models…"):
            clf_pipeline = load_classifier()
            embedder = load_embedder()

        with st.spinner("Extracting JD signals…"):
            jd_info = extract_jd(jd_text)

        with st.spinner("Scoring candidate fit…"):
            score_info = compute_fit_score(
                jd_text, candidate_text,
                jd_skills=jd_info["skills_and_tools"],
                embedding_model=embedder,
            )

        with st.spinner("Classifying JD sentences (first run downloads ~1.6 GB model)…"):
            buckets = classify_jd(jd_text, pipeline=clf_pipeline)

        st.divider()
        fit = score_info["fit_score"]

        # ── Gauge + JD signals ─────────────────────────────────────────────────
        col_signals, col_gauge = st.columns([3, 2])

        with col_gauge:
            st.plotly_chart(
                _gauge_chart(fit), use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f"<p style='text-align:center;font-size:1.1rem;color:{_score_color(fit)};margin-top:-18px;'>"
                f"{_score_grade(fit)}</p>",
                unsafe_allow_html=True,
            )

        with col_signals:
            st.subheader("JD Signals")
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Seniority", jd_info["seniority"].title())
            r1c2.metric("Education", jd_info["education"].title())
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Work Type", jd_info["work_type"].title())
            r2c2.metric("Industry", jd_info["industry"].title())
            exp = ", ".join(jd_info["years_of_experience"]) or "Not specified"
            st.metric("Experience Required", exp)

        # ── Colour-coded skill tags ────────────────────────────────────────────
        st.divider()
        st.subheader("Skill Coverage")
        if jd_info["skills_and_tools"]:
            st.markdown(
                _skill_tags_html(score_info["matched_skills"], score_info["missing_skills"]),
                unsafe_allow_html=True,
            )
            m, g = len(score_info["matched_skills"]), len(score_info["missing_skills"])
            st.caption(f"✅ {m} matched &nbsp;·&nbsp; ❌ {g} missing &nbsp;·&nbsp; {m + g} total")
        else:
            st.info("No specific tools/skills detected in the JD.")

        if jd_info.get("soft_skills"):
            st.markdown("**Soft skills in JD:**")
            st.markdown(_soft_tags_html(jd_info["soft_skills"]), unsafe_allow_html=True)

        # ── Score breakdown + classification pie ───────────────────────────────
        st.divider()
        col_breakdown, col_pie = st.columns(2)

        with col_breakdown:
            st.subheader("Score Breakdown")
            st.plotly_chart(
                _breakdown_chart(score_info["score_breakdown"]),
                use_container_width=True, config={"displayModeBar": False},
            )

        with col_pie:
            st.subheader("JD Sentence Mix")
            st.plotly_chart(
                _classification_pie(buckets),
                use_container_width=True, config={"displayModeBar": False},
            )

        # ── Recommendation ─────────────────────────────────────────────────────
        st.divider()
        st.info(f"💡 {_generate_recommendation(score_info, jd_info)}")

        # ── Sentence classification (tabbed) ───────────────────────────────────
        st.subheader("JD Sentence Classification")
        tab_req, tab_nice, tab_resp, tab_co = st.tabs(
            ["Required 🔴", "Nice-to-Have 🟡", "Responsibilities 🔵", "Company Info ⚪"]
        )
        for tab, key in [
            (tab_req, "Required"),
            (tab_nice, "Nice-to-Have"),
            (tab_resp, "Responsibility"),
            (tab_co, "Company Info"),
        ]:
            with tab:
                items = buckets.get(key, [])
                if items:
                    for item in items:
                        conf = item["confidence"]
                        badge = "🟢" if conf >= 80 else "🟡" if conf >= 60 else "🔴"
                        st.markdown(f"{badge} {item['text']} `{conf}%`")
                else:
                    st.info("Nothing classified here.")

        # ── Download ───────────────────────────────────────────────────────────
        st.divider()
        report: dict[str, Any] = {
            "fit_score": fit,
            "grade": _score_grade(fit),
            "recommendation": _generate_recommendation(score_info, jd_info),
            "score_breakdown": score_info["score_breakdown"],
            "jd_signals": {
                "seniority": jd_info["seniority"],
                "education": jd_info["education"],
                "work_type": jd_info["work_type"],
                "industry": jd_info["industry"],
                "years_of_experience": jd_info["years_of_experience"],
                "hard_skills": jd_info["skills_and_tools"],
                "soft_skills": jd_info.get("soft_skills", []),
            },
            "skill_analysis": {
                "matched": score_info["matched_skills"],
                "missing": score_info["missing_skills"],
            },
            "keyword_analysis": {
                "matched": score_info["matched_keywords"],
                "gaps": score_info["gap_keywords"],
            },
            "classification": {
                bucket: [i["text"] for i in items]
                for bucket, items in buckets.items()
            },
            "models_used": {"embedding": EMBEDDING_MODEL, "zero_shot": ZERO_SHOT_MODEL},
        }
        st.download_button(
            label="Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="jd_lens_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE CANDIDATES
# ════════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.subheader("Rank Multiple Candidates Against One JD")
    st.caption(
        "Uses semantic embedding + TF-IDF + skill match scoring. "
        "Zero-shot classification skipped here for speed."
    )

    jd_compare = st.text_area(
        "jd_compare", height=200,
        placeholder="Paste the job description here…",
        label_visibility="collapsed",
    )

    n_candidates = st.slider("Number of candidates", min_value=2, max_value=3, value=2)
    cand_cols = st.columns(n_candidates)
    candidate_names: list[str] = []
    candidate_texts: list[str] = []

    for i, col in enumerate(cand_cols):
        with col:
            name = st.text_input("Name", value=f"Candidate {i + 1}", key=f"cname_{i}")
            text = st.text_area(
                "Profile",
                height=220,
                placeholder=f"Paste {name}'s resume or profile…",
                key=f"ctext_{i}",
                label_visibility="collapsed",
            )
            candidate_names.append(name)
            candidate_texts.append(text)

    if st.button("Compare ✨", type="primary", use_container_width=True, key="compare_btn"):
        if not jd_compare.strip():
            st.error("Please provide a job description.")
            st.stop()

        filled = [
            (name, text)
            for name, text in zip(candidate_names, candidate_texts)
            if text.strip()
        ]
        if len(filled) < 2:
            st.error("Provide at least 2 candidate profiles to compare.")
            st.stop()

        with st.spinner("Loading embedding model…"):
            embedder = load_embedder()

        jd_info_compare = extract_jd(jd_compare)

        with st.spinner(f"Scoring {len(filled)} candidates…"):
            results = []
            for name, text in filled:
                s = compute_fit_score(
                    jd_compare, text,
                    jd_skills=jd_info_compare["skills_and_tools"],
                    embedding_model=embedder,
                )
                results.append({
                    "name": name,
                    "fit_score": s["fit_score"],
                    "matched_count": len(s["matched_skills"]),
                    "missing_count": len(s["missing_skills"]),
                    "top_missing": s["missing_skills"][:3],
                    "breakdown": s["score_breakdown"],
                })

        results.sort(key=lambda r: r["fit_score"], reverse=True)

        st.divider()
        medals = ["🥇", "🥈", "🥉"]

        for rank, r in enumerate(results):
            color = _score_color(r["fit_score"])
            c1, c2, c3, c4, c5 = st.columns([0.5, 2.5, 1.5, 2, 2.5])
            c1.markdown(f"## {medals[rank]}")
            c2.markdown(f"### {r['name']}")
            c3.markdown(
                f"<span style='color:{color};font-size:2.2rem;font-weight:800'>"
                f"{r['fit_score']}%</span>",
                unsafe_allow_html=True,
            )
            c4.markdown(
                f"✅ **{r['matched_count']}** matched  \n"
                f"❌ **{r['missing_count']}** missing"
            )
            gaps = ", ".join(f"`{g}`" for g in r["top_missing"]) if r["top_missing"] else "—"
            c5.markdown(f"**Key gaps:** {gaps}")

        # Comparison bar chart
        st.subheader("Score Comparison")
        fig = go.Figure(go.Bar(
            x=[r["name"] for r in results],
            y=[r["fit_score"] for r in results],
            marker_color=[_score_color(r["fit_score"]) for r in results],
            text=[f"{r['fit_score']}%" for r in results],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 110], title="Fit Score (%)"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem;'>"
    "Built by <a href='https://github.com/Dhyey2901' target='_blank' style='color:#2563EB;'>Dhyey Vyas</a> &nbsp;·&nbsp; "
    "<a href='https://github.com/Dhyey2901/jd-lens' target='_blank' style='color:#2563EB;'>Source Code</a> &nbsp;·&nbsp; "
    "Powered by HuggingFace Transformers &amp; Sentence Transformers"
    "</div>",
    unsafe_allow_html=True,
)
