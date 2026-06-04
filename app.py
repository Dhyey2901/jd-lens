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

# ── Page config ────────────────────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _score_color(score: float) -> str:
    if score >= 65:
        return "#2ecc71"
    if score >= 40:
        return "#f39c12"
    return "#e74c3c"


def _score_grade(score: float) -> str:
    if score >= 65:
        return "Strong Fit ✅"
    if score >= 40:
        return "Partial Fit ⚠️"
    return "Weak Fit ❌"


def _breakdown_chart(breakdown: dict[str, float]) -> go.Figure:
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    colors = [_score_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=10),
        height=160,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def _skill_chart(matched: list[str], missing: list[str]) -> go.Figure:
    fig = go.Figure()
    if matched:
        fig.add_trace(go.Bar(
            name="Matched",
            x=matched[:20],
            marker_color="#2ecc71",
            text=matched[:20],
        ))
    if missing:
        fig.add_trace(go.Bar(
            name="Missing",
            x=missing[:20],
            marker_color="#e74c3c",
            text=missing[:20],
        ))
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-35,
        margin=dict(l=10, r=10, t=10, b=80),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔍 JD Lens")
st.caption("NLP-powered Job Description Analyser & Role-Fit Scorer")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
        | Step | Model | What it does |
        |---|---|---|
        | Extraction | Regex | Skills, tools, seniority, education, work type |
        | Classification | `facebook/bart-large-mnli` | Buckets each JD sentence |
        | Fit Score | `all-MiniLM-L6-v2` + TF-IDF | Semantic + keyword + skill match |
        """
    )

col_jd, col_candidate = st.columns(2)
with col_jd:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "jd",
        height=320,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )
with col_candidate:
    st.subheader("Candidate Profile")
    candidate_text = st.text_area(
        "candidate",
        height=320,
        placeholder="Paste your resume / profile summary here…",
        label_visibility="collapsed",
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

    # ── Fit Score Banner ───────────────────────────────────────────────────────
    st.divider()
    fit = score_info["fit_score"]
    color = _score_color(fit)
    grade = _score_grade(fit)

    st.markdown(
        f"""
        <div style="text-align:center;padding:1.5rem;border-radius:12px;
                    background:{color}22;border:2px solid {color};">
            <p style="font-size:.9rem;color:{color};margin:0;letter-spacing:.08em;
                      text-transform:uppercase;">Role Fit Score</p>
            <p style="font-size:3.8rem;font-weight:800;color:{color};margin:0;
                      line-height:1.1;">{fit}%</p>
            <p style="font-size:1.1rem;color:{color};margin:0;">{grade}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Score Breakdown Chart ──────────────────────────────────────────────────
    st.subheader("Score Breakdown")
    st.plotly_chart(
        _breakdown_chart(score_info["score_breakdown"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── JD Signal Metrics ──────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Seniority", jd_info["seniority"].title())
    c2.metric("Education", jd_info["education"].title())
    c3.metric("Work Type", jd_info["work_type"].title())
    exp = ", ".join(jd_info["years_of_experience"]) or "Not specified"
    c4.metric("Experience", exp)

    # ── Skill Chart ────────────────────────────────────────────────────────────
    st.subheader("Skill Coverage")
    if jd_info["skills_and_tools"]:
        st.plotly_chart(
            _skill_chart(score_info["matched_skills"], score_info["missing_skills"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        mc, gc = st.columns(2)
        with mc:
            st.caption(f"**Matched ({len(score_info['matched_skills'])})**")
            st.write(" ".join(f"`{s}`" for s in score_info["matched_skills"]) or "—")
        with gc:
            st.caption(f"**Missing ({len(score_info['missing_skills'])})**")
            st.write(" ".join(f"`{s}`" for s in score_info["missing_skills"]) or "—")
    else:
        st.info("No recognised tools/skills detected in the JD.")

    # ── Keyword Analysis ───────────────────────────────────────────────────────
    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        st.subheader("Keyword Matches ✅")
        kw = score_info["matched_keywords"]
        st.write(" ".join(f"`{k}`" for k in kw[:40]) if kw else "No overlap found.")
        if len(kw) > 40:
            st.caption(f"…and {len(kw) - 40} more")
    with kc2:
        st.subheader("Keyword Gaps ⚠️")
        gw = score_info["gap_keywords"]
        st.write(" ".join(f"`{k}`" for k in gw[:40]) if gw else "No significant gaps.")
        if len(gw) > 40:
            st.caption(f"…and {len(gw) - 40} more")

    # ── Sentence Classification ────────────────────────────────────────────────
    st.divider()
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

    # ── Download Report ────────────────────────────────────────────────────────
    st.divider()
    report: dict[str, Any] = {
        "fit_score": score_info["fit_score"],
        "grade": grade,
        "score_breakdown": score_info["score_breakdown"],
        "jd_signals": {
            "seniority": jd_info["seniority"],
            "education": jd_info["education"],
            "work_type": jd_info["work_type"],
            "years_of_experience": jd_info["years_of_experience"],
            "skills_detected": jd_info["skills_and_tools"],
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
        "models_used": {
            "embedding": EMBEDDING_MODEL,
            "zero_shot": ZERO_SHOT_MODEL,
        },
    }
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="jd_lens_report.json",
        mime="application/json",
        use_container_width=True,
    )
