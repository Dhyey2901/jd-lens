import streamlit as st
from extractor import extract_jd
from classifier import classify_jd
from scorer import compute_fit_score

st.set_page_config(page_title="JD Lens", page_icon="🔍", layout="wide")

st.title("🔍 JD Lens — Job Description Analyser & Role-Fit Scorer")
st.caption("Paste a job description and your candidate profile to get an instant fit analysis.")

col_jd, col_candidate = st.columns(2)

with col_jd:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder="e.g. We are looking for a Senior Python Engineer with 5+ years of experience...",
        label_visibility="collapsed",
    )

with col_candidate:
    st.subheader("Candidate Profile")
    candidate_text = st.text_area(
        "Paste your resume / profile summary here",
        height=320,
        placeholder="e.g. I have 6 years of Python experience, worked with AWS, Docker, and FastAPI...",
        label_visibility="collapsed",
    )

analyse = st.button("Analyse ✨", type="primary", use_container_width=True)

if analyse:
    if not jd_text.strip() or not candidate_text.strip():
        st.error("Please provide both a job description and a candidate profile.")
        st.stop()

    with st.spinner("Extracting signals from JD…"):
        jd_info = extract_jd(jd_text)

    with st.spinner("Scoring candidate fit…"):
        score_info = compute_fit_score(jd_text, candidate_text)

    with st.spinner("Classifying JD sentences (this may take ~30s on first run)…"):
        buckets = classify_jd(jd_text)

    # ── Fit Score ──────────────────────────────────────────────────────────────
    st.divider()
    fit = score_info["fit_score"]
    color = "#2ecc71" if fit >= 65 else "#f39c12" if fit >= 40 else "#e74c3c"
    grade = "Strong Fit ✅" if fit >= 65 else "Partial Fit ⚠️" if fit >= 40 else "Weak Fit ❌"

    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem; border-radius: 12px;
                    background: {color}22; border: 2px solid {color};">
            <p style="font-size:1rem; color:{color}; margin:0;">Role Fit Score</p>
            <p style="font-size:3.5rem; font-weight:700; color:{color}; margin:0;">{fit}%</p>
            <p style="font-size:1.1rem; color:{color}; margin:0;">{grade}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── JD Signals ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Seniority Level", jd_info["seniority"].title())
    with c2:
        exp = ", ".join(jd_info["years_of_experience"]) or "Not specified"
        st.metric("Experience Required", exp)
    with c3:
        st.metric("Tools/Skills Found", len(jd_info["skills_and_tools"]))

    st.subheader("Skills & Tools Detected in JD")
    if jd_info["skills_and_tools"]:
        st.write(" ".join(f"`{s}`" for s in jd_info["skills_and_tools"]))
    else:
        st.info("No specific tools detected.")

    st.divider()

    # ── Keyword Match / Gap ────────────────────────────────────────────────────
    col_match, col_gap = st.columns(2)

    with col_match:
        st.subheader("Matched Keywords ✅")
        matched = score_info["matched_keywords"]
        if matched:
            st.write(" ".join(f"`{k}`" for k in matched[:40]))
            if len(matched) > 40:
                st.caption(f"…and {len(matched) - 40} more")
        else:
            st.info("No keyword overlap found.")

    with col_gap:
        st.subheader("Skill Gaps ⚠️")
        gaps = score_info["gap_keywords"]
        if gaps:
            st.write(" ".join(f"`{k}`" for k in gaps[:40]))
            if len(gaps) > 40:
                st.caption(f"…and {len(gaps) - 40} more")
        else:
            st.success("No significant gaps detected!")

    st.divider()

    # ── Sentence Classification ────────────────────────────────────────────────
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
                    st.markdown(f"- {item}")
            else:
                st.info("Nothing classified here.")
