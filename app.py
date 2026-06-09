"""JD Lens — Streamlit UI entry point."""
from __future__ import annotations

import json
import re
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from classifier import classify_jd, get_pipeline
from config import EMBEDDING_MODEL, MAX_CHARS, MAX_FILE_SIZE_MB, MIN_RESUME_WORDS, ZERO_SHOT_MODEL
from extractor import extract_jd, extract_skills_and_tools, extract_soft_skills, extract_unknown_tools
from scorer import compute_fit_score, get_cross_encoder, get_embedding_model
from signals import compute_hiring_signals, generate_prediction
from utils import SUPPORTED_LABEL, SUPPORTED_TYPES, extract_text_from_file

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

st.html('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">')

st.markdown("""<style>
html,body,[class*="css"],.stMarkdown,.stTextArea textarea,.stTextInput input,.stSelectbox,button{font-family:'Inter',sans-serif!important}
h1{font-weight:700!important;letter-spacing:-.5px!important}
h2,h3{font-weight:600!important;letter-spacing:-.3px!important}
hr{border-color:#E2E8F0!important;margin:20px 0!important}
[data-testid="stMetricLabel"]{font-size:.78rem!important;color:#64748B!important}
[data-testid="stMetricValue"]{font-size:1.15rem!important;font-weight:600!important}
.stButton>button[kind="primary"]{font-weight:600!important;letter-spacing:.01em!important;border-radius:8px!important}
[data-baseweb="tab"]{font-weight:500!important}
.block-container{padding-top:2rem!important}
</style>""", unsafe_allow_html=True)

# ── Model loading (cached across sessions) ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifier():
    return get_pipeline()


@st.cache_resource(show_spinner=False)
def load_embedder():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    return get_cross_encoder()


# ── Visual helpers ─────────────────────────────────────────────────────────────

# Abbreviations that str.title() would mangle (e.g. "ai" → "Ai")
_INDUSTRY_ABBREV = {"ai", "saas", "b2b", "ehr", "fhir", "d2c", "wfh"}

def _display_industry(raw: str) -> str:
    """Title-case an industry string while preserving uppercase abbreviations."""
    def _cap(word: str) -> str:
        return word.upper() if word.lower() in _INDUSTRY_ABBREV else word.capitalize()
    return " ".join(_cap(w) for w in raw.replace("/", "/ ").split())


def _char_counter(text: str, limit: int = MAX_CHARS) -> None:
    """Render a subtle character counter below a text area."""
    used = len(text)
    pct = used / limit
    color = "#e74c3c" if pct > 0.9 else "#f39c12" if pct > 0.7 else "#888"
    st.markdown(
        f"<p style='font-size:.72rem;color:{color};text-align:right;"
        f"margin-top:-14px;'>{used:,} / {limit:,} chars</p>",
        unsafe_allow_html=True,
    )


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
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "tickfont": {"color": "rgba(0,0,0,0)"},
                "showticklabels": False,
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(231,76,60,0.10)"},
                {"range": [40, 65], "color": "rgba(243,156,18,0.10)"},
                {"range": [65, 100],"color": "rgba(46,204,113,0.10)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _skill_tags_html(
    matched: list[str],
    semantic: list[str],
    missing: list[str],
) -> str:
    parts = []
    for s in matched:
        parts.append(
            f'<span style="background:#2ecc7118;color:#2ecc71;border:1px solid #2ecc7144;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
            f'✓ {s}</span>'
        )
    for s in semantic:
        parts.append(
            f'<span style="background:#f39c1218;color:#e67e22;border:1px solid #f39c1244;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
            f'≈ {s}</span>'
        )
    for s in missing:
        parts.append(
            f'<span style="background:#e74c3c18;color:#e74c3c;border:1px solid #e74c3c44;'
            f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
            f'✗ {s}</span>'
        )
    return "".join(parts)


def _candidate_skill_tags_html(skills: list[str], extra: bool = False) -> str:
    color = "#10b981" if not extra else "#6366f1"
    return "".join(
        f'<span style="background:{color}18;color:{color};border:1px solid {color}44;'
        f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;font-size:.82rem;">'
        f'{s}</span>'
        for s in skills
    )


def _soft_skill_coverage_html(
    jd_soft: list[str],
    candidate_text: str,
) -> str:
    """Render JD soft skills with ✓/✗ indicating presence in candidate text."""
    parts = []
    for skill in jd_soft:
        found = bool(re.search(r"\b" + re.escape(skill) + r"\b", candidate_text, re.IGNORECASE))
        if found:
            parts.append(
                f'<span style="background:#2ecc7118;color:#2ecc71;border:1px solid #2ecc7144;'
                f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;'
                f'font-size:.82rem;">✓ {skill}</span>'
            )
        else:
            parts.append(
                f'<span style="background:#e74c3c18;color:#e74c3c;border:1px solid #e74c3c44;'
                f'padding:4px 12px;border-radius:20px;margin:3px;display:inline-block;'
                f'font-size:.82rem;">✗ {skill}</span>'
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, family="Inter"),
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
        font=dict(family="Inter"),
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

    semantic = score_info.get("semantic_skills", [])
    skill_line = (
        f"Exact match on {len(matched)}/{total} tools"
        + (f", semantic match on {len(semantic)} more." if semantic else ".")
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


# ── Shared upload helper ───────────────────────────────────────────────────────

def _handle_upload(
    label: str,
    session_key: str,
    uploader_key: str,
) -> None:
    """Render a file uploader, extract text, and store it in session_state."""
    uploaded = st.file_uploader(
        label,
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
        key=uploader_key,
    )
    if uploaded is not None:
        file_bytes = uploaded.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"File is {size_mb:.1f} MB — maximum allowed is {MAX_FILE_SIZE_MB} MB. "
                "Try a smaller file or paste the text directly."
            )
            return
        try:
            text = extract_text_from_file(file_bytes, uploaded.name)
            word_count = len(text.split())
            if not text or word_count < MIN_RESUME_WORDS:
                st.warning(
                    f"Only {word_count} word(s) extracted. "
                    "PDFs with scanned images need OCR — try a DOCX or TXT version, "
                    "or paste the text directly."
                )
            else:
                st.session_state[session_key] = text
                ext = uploaded.name.rsplit(".", 1)[-1].upper()
                st.success(f"Extracted {word_count:,} words from **{uploaded.name}** ({ext})")
                with st.expander("Preview extracted text", expanded=False):
                    st.text(text[:1500] + ("…" if len(text) > 1500 else ""))
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not read file: {e}")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:28px 0 8px;">
  <h1 style="font-size:2rem;font-weight:700;letter-spacing:-.5px;margin:0;">🔍 JD Lens</h1>
  <p style="font-size:1rem;color:#64748B;margin:6px 0 0;font-weight:400;">
    AI-powered resume scorer for IT &amp; tech hiring —
    JD-match score, hiring signals, skill gaps, and keyword analysis in seconds.
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
| Step | Model | What it does |
| --- | --- | --- |
| **JD Extraction** | Regex | Skills, soft skills, seniority, education, industry, work type, section weights |
| **JD-Match Score** | `all-MiniLM-L6-v2` + TF-IDF | 30% semantic · 30% keyword overlap (bigrams) · 40% skill match rate |
| **Hiring Signal Score** | Rule-based | 7 dimensions: impact, depth, deployment, verbs, trajectory, soft skills, skill gap |
| **Four-Quadrant Verdict** | Score axes | Strong Candidate / Overlooked Gem / Keyword Match–Verify Depth / Significant Gaps |
| **Sentence Classification** | `facebook/bart-large-mnli` | Required / Nice-to-Have / Responsibility / Company Info |
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
        _char_counter(jd_text)

    with col_cand:
        st.subheader("Candidate Profile")
        st.caption(f"Upload a {SUPPORTED_LABEL} resume, or paste / edit text below.")
        _handle_upload("Upload resume", "candidate_text", "resume_upload_analyse")
        candidate_text = st.text_area(
            "candidate", height=200,
            placeholder="e.g. 6 years of Python, built REST APIs with FastAPI, daily AWS and Docker usage…",
            label_visibility="collapsed",
            key="candidate_text",
        )
        _char_counter(candidate_text)

    analyse = st.button("Analyse ✨", type="primary", use_container_width=True)

    if analyse:
        if not jd_text.strip() or not candidate_text.strip():
            st.error("Please provide both a job description and a candidate profile.")
            st.stop()
        if len(jd_text) > MAX_CHARS:
            st.error(f"Job description exceeds {MAX_CHARS:,} characters. Please trim it.")
            st.stop()
        if len(candidate_text) > MAX_CHARS:
            st.error(f"Candidate profile exceeds {MAX_CHARS:,} characters. Please trim it.")
            st.stop()

        with st.spinner("Loading models…"):
            embedder = load_embedder()

        with st.spinner("Extracting JD signals…"):
            jd_info = extract_jd(jd_text)

        with st.spinner("Scoring candidate fit…"):
            score_info = compute_fit_score(
                jd_text, candidate_text,
                jd_skills=jd_info["skills_and_tools"],
                embedding_model=embedder,
            )

        # Extract skills/soft-skills directly from the candidate resume.
        # unknown_tools catches context-pattern mentions ("proficiency in Workday")
        # that aren't in KNOWN_TOOLS — critical for non-tech domains.
        candidate_hard_skills = extract_skills_and_tools(candidate_text)
        candidate_unknown_skills = extract_unknown_tools(candidate_text)
        candidate_soft_skills = extract_soft_skills(candidate_text)

        # Hiring signals — rule-based, zero model overhead
        fit = score_info["fit_score"]
        hiring = compute_hiring_signals(
            candidate_text,
            jd_soft_skills=jd_info.get("soft_skills", []),
            jd_text=jd_text,
            jd_skills=jd_info.get("skills_and_tools", []),
            missing_skills=score_info["missing_skills"],
            skill_weights=jd_info.get("skill_weights", {}),
        )
        prediction = generate_prediction(
            jd_match=fit,
            signal_score=hiring["hiring_signal_score"],
            missing_skills=score_info["missing_skills"],
        )

        st.divider()

        # ── Dual score header ──────────────────────────────────────────────────
        col_jd_gauge, col_sig_gauge = st.columns(2)

        with col_jd_gauge:
            st.markdown(
                "<p style='text-align:center;font-size:.85rem;"
                "color:#888;margin-bottom:-8px;'>JD Match Score</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _gauge_chart(fit), use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f"<p style='text-align:center;font-size:1.05rem;"
                f"color:{_score_color(fit)};margin-top:-18px;'>{_score_grade(fit)}</p>",
                unsafe_allow_html=True,
            )

        with col_sig_gauge:
            sig = hiring["hiring_signal_score"]
            sig_color = "#2ecc71" if sig >= 70 else "#f39c12" if sig >= 50 else "#e74c3c"
            st.markdown(
                "<p style='text-align:center;font-size:.85rem;"
                "color:#888;margin-bottom:-8px;'>Hiring Signal Score</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _gauge_chart(sig), use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f"<p style='text-align:center;font-size:1.05rem;"
                f"color:{sig_color};margin-top:-18px;'>{hiring['grade']}</p>",
                unsafe_allow_html=True,
            )
            role_type = hiring.get("role_type", "general")
            if role_type != "general":
                role_label = role_type.replace("_", " ").title()
                st.markdown(
                    f"<p style='text-align:center;font-size:.75rem;color:#888;"
                    f"margin-top:-6px;'>Weights calibrated for: <em>{role_label}</em></p>",
                    unsafe_allow_html=True,
                )

        # ── Prediction verdict ─────────────────────────────────────────────────
        p_color = prediction["color"]
        st.markdown(
            f"""<div style="border-left:5px solid {p_color};padding:14px 20px;
            border-radius:6px;background:{p_color}12;margin:12px 0;">
            <span style="font-size:1.4rem;">{prediction['icon']}</span>&nbsp;
            <strong style="font-size:1.05rem;color:{p_color};">{prediction['verdict']}</strong>
            <p style="margin:6px 0 0;font-size:.9rem;color:#ccc;">{prediction['explanation']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Hiring signal breakdown ────────────────────────────────────────────
        with st.expander("Hiring Signal breakdown", expanded=False):
            st.caption(
                "Measures quality signals that experienced recruiters screen for, "
                "independent of JD keyword matching."
            )
            for dim, val in hiring["breakdown"].items():
                bar_color = "#2ecc71" if val >= 70 else "#f39c12" if val >= 45 else "#e74c3c"
                st.markdown(
                    f"<div style='font-size:.83rem;margin-bottom:3px;'>"
                    f"<span style='color:#aaa;'>{dim}</span>"
                    f"<span style='float:right;font-weight:600;color:{bar_color};'>{val}%</span>"
                    f"</div>"
                    f"<div style='background:#333;border-radius:4px;height:6px;margin-bottom:10px;'>"
                    f"<div style='background:{bar_color};width:{min(val,100)}%;height:6px;border-radius:4px;'>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            if hiring["quantified_lines"] < 3:
                st.info(
                    f"💡 Only **{hiring['quantified_lines']} bullet point(s)** contain measurable numbers. "
                    "Adding metrics (e.g. team size, scale, % improvement) is the single highest-ROI resume change."
                )

            top_bullets = score_info.get("top_bullets", [])
            if top_bullets:
                st.markdown("**Top resume lines for this JD:**")
                for b in top_bullets:
                    jd_req = b.get("jd_req", "")
                    req_html = (
                        f"<div style='font-size:.75rem;color:#888;margin-top:3px;'>"
                        f"↔ &nbsp;<em>{jd_req}</em></div>"
                        if jd_req else ""
                    )
                    st.markdown(
                        f"<div style='font-size:.84rem;padding:8px 12px;margin:5px 0;"
                        f"border-left:3px solid #2563eb;background:#2563eb08;"
                        f"border-radius:4px;'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:flex-start;'>"
                        f"<span>{b['text']}</span>"
                        f"<span style='font-size:.75rem;color:#888;white-space:nowrap;"
                        f"margin-left:10px;'>{b['score']}%</span></div>"
                        f"{req_html}</div>",
                        unsafe_allow_html=True,
                    )

        # ── JD signals ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("JD Signals")
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Seniority", jd_info["seniority"].title())
        r1c2.metric("Education", jd_info["education"].title())
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Work Type", jd_info["work_type"].title())
        r2c2.metric("Industry", _display_industry(jd_info["industry"]))
        exp = ", ".join(jd_info["years_of_experience"]) or "Not specified"
        st.metric("Experience Required", exp)

        # ── Skill Coverage ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Skill Coverage")

        col_jd_sk, col_cand_sk = st.columns(2)

        with col_jd_sk:
            st.markdown("**Required by JD**")
            if jd_info["skills_and_tools"]:
                st.markdown(
                    _skill_tags_html(
                        score_info["matched_skills"],
                        score_info.get("semantic_skills", []),
                        score_info["missing_skills"],
                    ),
                    unsafe_allow_html=True,
                )
                m = len(score_info["matched_skills"])
                s = len(score_info.get("semantic_skills", []))
                g = len(score_info["missing_skills"])
                st.caption(
                    f"✅ {m} exact &nbsp;·&nbsp; ≈ {s} semantic &nbsp;·&nbsp; "
                    f"❌ {g} missing"
                )
            else:
                st.caption("No specific tools detected in JD.")
            if jd_info.get("unknown_tools"):
                st.markdown(
                    "<span style='font-size:.78rem;color:#888;'>Also in JD (unlisted):</span>",
                    unsafe_allow_html=True,
                )
                cand_text_lower = candidate_text.lower()
                unk_tags = []
                for t in jd_info["unknown_tools"]:
                    has = bool(re.search(r"\b" + re.escape(t) + r"\b", cand_text_lower))
                    bg = "#10b98118" if has else "#ef444418"
                    fg = "#059669" if has else "#dc2626"
                    border = "#10b98144" if has else "#ef444444"
                    icon = "✓" if has else "✗"
                    unk_tags.append(
                        f'<span style="background:{bg};color:{fg};border:1px solid {border};'
                        f'padding:3px 9px;border-radius:20px;margin:2px;'
                        f'display:inline-block;font-size:.78rem;">{icon} {t}</span>'
                    )
                st.markdown(" ".join(unk_tags), unsafe_allow_html=True)

        with col_cand_sk:
            st.markdown("**Detected in candidate**")
            total_cand = len(candidate_hard_skills) + len(candidate_unknown_skills)
            if candidate_hard_skills or candidate_unknown_skills:
                jd_set = set(jd_info["skills_and_tools"])
                jd_unk_set = set(jd_info.get("unknown_tools", []))
                in_jd = [s for s in candidate_hard_skills if s in jd_set]
                extra = [s for s in candidate_hard_skills if s not in jd_set]
                if in_jd:
                    st.markdown(_candidate_skill_tags_html(in_jd), unsafe_allow_html=True)
                if extra:
                    st.markdown(
                        "<span style='font-size:.78rem;color:#888;'>Also brings:</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _candidate_skill_tags_html(extra, extra=True),
                        unsafe_allow_html=True,
                    )
                if candidate_unknown_skills:
                    st.markdown(
                        "<span style='font-size:.78rem;color:#888;'>Other tools mentioned:</span>",
                        unsafe_allow_html=True,
                    )
                    unk_in_jd = [t for t in candidate_unknown_skills if t in jd_unk_set]
                    unk_extra = [t for t in candidate_unknown_skills if t not in jd_unk_set]
                    if unk_in_jd:
                        st.markdown(_candidate_skill_tags_html(unk_in_jd), unsafe_allow_html=True)
                    if unk_extra:
                        st.markdown(
                            _candidate_skill_tags_html(unk_extra, extra=True),
                            unsafe_allow_html=True,
                        )
                st.caption(f"{total_cand} skills detected in resume")
            else:
                st.caption("No skills detected in resume.")

        # ── Soft Skills ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Soft Skills")
        col_jd_soft, col_cand_soft = st.columns(2)

        with col_jd_soft:
            jd_soft = jd_info.get("soft_skills", [])
            st.markdown("**Required by JD**")
            if jd_soft:
                st.markdown(
                    _soft_skill_coverage_html(jd_soft, candidate_text),
                    unsafe_allow_html=True,
                )
                covered = sum(
                    1 for sk in jd_soft
                    if re.search(r"\b" + re.escape(sk) + r"\b",
                                 candidate_text, re.IGNORECASE)
                )
                st.caption(f"✅ {covered} of {len(jd_soft)} JD soft skills found in resume")
            else:
                st.caption("No soft skills detected in JD.")

        with col_cand_soft:
            st.markdown("**Detected in candidate**")
            if candidate_soft_skills:
                jd_soft_set = set(jd_info.get("soft_skills", []))
                soft_in_jd  = [s for s in candidate_soft_skills if s in jd_soft_set]
                soft_extra  = [s for s in candidate_soft_skills if s not in jd_soft_set]
                if soft_in_jd:
                    st.markdown(_candidate_skill_tags_html(soft_in_jd), unsafe_allow_html=True)
                if soft_extra:
                    st.markdown(
                        "<span style='font-size:.78rem;color:#888;'>Additional strengths:</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _candidate_skill_tags_html(soft_extra, extra=True),
                        unsafe_allow_html=True,
                    )
                st.caption(f"{len(candidate_soft_skills)} soft skills detected in resume")
            else:
                st.caption("No soft skills from vocabulary detected in resume.")

        # ── Keyword Analysis ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Keyword Analysis")
        matched_kw = score_info.get("matched_keywords", [])
        gap_kw = score_info.get("gap_keywords", [])
        if matched_kw or gap_kw:
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**✅ Found in resume**")
                if matched_kw:
                    st.markdown(
                        " ".join(
                            f'<span style="background:#10b98118;color:#059669;'
                            f'border:1px solid #10b98144;padding:3px 9px;border-radius:20px;'
                            f'margin:2px;display:inline-block;font-size:.78rem;">{kw}</span>'
                            for kw in matched_kw
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No JD keywords found in resume.")
            with kw_col2:
                st.markdown("**❌ Missing from resume**")
                if gap_kw:
                    st.markdown(
                        " ".join(
                            f'<span style="background:#ef444418;color:#dc2626;'
                            f'border:1px solid #ef444444;padding:3px 9px;border-radius:20px;'
                            f'margin:2px;display:inline-block;font-size:.78rem;">{kw}</span>'
                            for kw in gap_kw
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("All key JD terms found in resume.")
        else:
            st.caption("Keyword analysis not available.")

        # ── Score breakdown ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Score Breakdown")
        st.plotly_chart(
            _breakdown_chart(score_info["score_breakdown"]),
            use_container_width=True, config={"displayModeBar": False},
        )

        # ── Recommendation ─────────────────────────────────────────────────────
        st.divider()
        st.info(f"💡 {_generate_recommendation(score_info, jd_info)}")

        # ── Sentence classification — runs after score is visible ──────────────
        st.divider()
        with st.spinner("Classifying JD sentences (first run downloads ~1.6 GB model)…"):
            clf_pipeline = load_classifier()
            buckets = classify_jd(jd_text, pipeline=clf_pipeline)

        col_pie, col_blank = st.columns([2, 3])
        with col_pie:
            st.subheader("JD Sentence Mix")
            st.plotly_chart(
                _classification_pie(buckets),
                use_container_width=True, config={"displayModeBar": False},
            )

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
                "unknown_tools": jd_info.get("unknown_tools", []),
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
            st.caption(f"Upload {SUPPORTED_LABEL} or paste text.")
            _handle_upload(
                f"Upload {name}'s resume",
                f"ctext_{i}",
                f"resume_upload_compare_{i}",
            )
            text = st.text_area(
                "Profile",
                height=180,
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
                    "grade": _score_grade(s["fit_score"]),
                    "matched": s["matched_skills"],
                    "semantic": s.get("semantic_skills", []),
                    "missing": s["missing_skills"],
                    "breakdown": s["score_breakdown"],
                })

        results.sort(key=lambda r: r["fit_score"], reverse=True)
        winner = results[0]
        all_jd_skills = jd_info_compare["skills_and_tools"]

        st.divider()
        medals = ["🥇", "🥈", "🥉"]
        decision_colors = ["#2ecc71", "#e74c3c", "#e74c3c"]
        decision_labels = ["SELECTED", "NOT SELECTED", "NOT SELECTED"]

        # ── Per-candidate cards ────────────────────────────────────────────────
        for rank, r in enumerate(results):
            score_color = _score_color(r["fit_score"])
            dec_color = decision_colors[rank]
            dec_label = decision_labels[rank]
            gap_from_winner = winner["fit_score"] - r["fit_score"]

            # Decision banner
            st.markdown(
                f"""<div style="border-left:5px solid {dec_color};
                    padding:14px 18px;border-radius:6px;
                    background:{dec_color}0d;margin-bottom:4px;">
                  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <span style="font-size:1.6rem;">{medals[rank]}</span>
                    <span style="font-size:1.25rem;font-weight:700;">{r['name']}</span>
                    <span style="font-size:1.8rem;font-weight:800;color:{score_color};">{r['fit_score']}%</span>
                    <span style="font-size:.85rem;color:{score_color};">{r['grade']}</span>
                    <span style="margin-left:auto;font-size:.8rem;font-weight:700;
                        color:{dec_color};text-transform:uppercase;letter-spacing:.06em;">
                        {dec_label}</span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Detail columns
            dc1, dc2, dc3 = st.columns([2, 2, 3])

            with dc1:
                st.markdown("**Score breakdown**")
                for dim, val in r["breakdown"].items():
                    bar_color = _score_color(val)
                    st.markdown(
                        f"<div style='font-size:.8rem;margin-bottom:4px;'>"
                        f"<span style='color:#666;'>{dim}</span>"
                        f"<span style='float:right;font-weight:600;color:{bar_color};'>{val}%</span>"
                        f"</div>"
                        f"<div style='background:#eee;border-radius:4px;height:6px;margin-bottom:8px;'>"
                        f"<div style='background:{bar_color};width:{min(val,100)}%;height:6px;border-radius:4px;'>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            with dc2:
                st.markdown("**Skill coverage**")
                m, s_sem, miss = len(r["matched"]), len(r["semantic"]), len(r["missing"])
                total = max(len(all_jd_skills), 1)
                st.markdown(
                    f"✅ **{m}** exact &nbsp; ≈ **{s_sem}** semantic &nbsp; ❌ **{miss}** missing  \n"
                    f"Coverage: **{round((m + 0.5*s_sem)/total*100)}%** of {total} JD skills"
                )
                if r["missing"]:
                    st.markdown(
                        "**Missing skills:**  \n"
                        + "  \n".join(f"&nbsp;&nbsp;`{sk}`" for sk in r["missing"])
                    )

            with dc3:
                st.markdown("**Selection reasoning**")
                if rank == 0:
                    strengths = r["matched"][:3]
                    st.success(
                        f"Highest overall fit. Covers {m}/{total} required skills. "
                        + (f"Key strengths: {', '.join(strengths)}." if strengths else "")
                    )
                else:
                    reasons = []
                    if gap_from_winner >= 20:
                        reasons.append(f"score is {gap_from_winner:.0f} pts below top pick")
                    extra_missing = [sk for sk in r["missing"] if sk in winner["matched"]]
                    if extra_missing:
                        reasons.append(
                            f"missing {len(extra_missing)} skill(s) the top pick has: "
                            f"{', '.join(extra_missing[:3])}"
                        )
                    if r["breakdown"]["Skill Match Rate"] < winner["breakdown"]["Skill Match Rate"] - 10:
                        reasons.append("lower skill match rate")
                    if r["breakdown"]["Semantic Similarity"] < winner["breakdown"]["Semantic Similarity"] - 10:
                        reasons.append("weaker alignment to JD language")
                    if reasons:
                        st.error("Not selected — " + "; ".join(reasons) + ".")
                    else:
                        st.warning(
                            f"Close call — only {gap_from_winner:.1f} pts behind. "
                            "Consider interviewing if top pick declines."
                        )

            st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:12px 0;'>",
                        unsafe_allow_html=True)

        # ── Skill comparison matrix ────────────────────────────────────────────
        if all_jd_skills:
            st.subheader("Skill Comparison Matrix")
            st.caption("Shows how each candidate covers every required skill from the JD.")

            header = "| Skill | " + " | ".join(r["name"] for r in results) + " |"
            sep    = "|---|" + "---|" * len(results)
            rows = []
            for skill in all_jd_skills:
                cells = []
                for r in results:
                    if skill in r["matched"]:
                        cells.append("✅ Exact")
                    elif skill in r["semantic"]:
                        cells.append("≈ Semantic")
                    else:
                        cells.append("❌ Missing")
                rows.append(f"| `{skill}` | " + " | ".join(cells) + " |")

            st.markdown("\n".join([header, sep] + rows))

        # ── Score comparison chart ─────────────────────────────────────────────
        st.subheader("Score Comparison")

        dims = list(results[0]["breakdown"].keys())
        fig = go.Figure()
        dim_colors = ["#3498db", "#e67e22", "#2ecc71"]
        for di, dim in enumerate(dims):
            fig.add_trace(go.Bar(
                name=dim,
                x=[r["name"] for r in results],
                y=[r["breakdown"][dim] for r in results],
                marker_color=dim_colors[di],
                text=[f"{r['breakdown'][dim]}%" for r in results],
                textposition="inside",
            ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 110], title="Score (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
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
