"""FastAPI REST layer — exposes JD Lens analysis as a documented HTTP API."""
from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extractor import extract_jd
from scorer import compute_fit_score

logger = logging.getLogger(__name__)

app = FastAPI(
    title="JD Lens API",
    description=(
        "NLP-powered job description analysis and role-fit scoring. "
        "Extracts skills, seniority, and education signals from a JD, "
        "then scores a candidate profile using semantic embeddings + TF-IDF."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    jd_text: str = Field(
        ...,
        min_length=50,
        max_length=12_000,
        description="Full text of the job description.",
        examples=["We are hiring a Senior Python Engineer with 5+ years of experience in AWS and Docker."],
    )
    candidate_text: str = Field(
        ...,
        min_length=20,
        max_length=12_000,
        description="Candidate resume or profile summary.",
        examples=["6 years of Python, deployed microservices on AWS ECS, daily Docker and Kubernetes usage."],
    )


class JdSignals(BaseModel):
    seniority: str
    education: str
    work_type: str
    industry: str
    years_of_experience: list[str]
    skills_detected: list[str]
    soft_skills: list[str]


class SkillAnalysis(BaseModel):
    matched: list[str]
    semantic: list[str]
    missing: list[str]
    match_rate_pct: float


class KeywordAnalysis(BaseModel):
    matched: list[str]
    gaps: list[str]


class AnalyseResponse(BaseModel):
    fit_score: float = Field(..., description="Composite fit score (0–100).")
    grade: str = Field(..., description="Strong Fit / Partial Fit / Weak Fit")
    score_breakdown: dict[str, float] = Field(
        ..., description="Per-dimension scores: Semantic Similarity, Keyword Overlap, Skill Match Rate."
    )
    jd_signals: JdSignals
    skill_analysis: SkillAnalysis
    keyword_analysis: KeywordAnalysis


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health() -> dict[str, str]:
    """Liveness probe — returns API version and status."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyse", response_model=AnalyseResponse, tags=["Analysis"])
def analyse(req: AnalyseRequest) -> Any:
    """
    Analyse a job description against a candidate profile.

    Returns a composite fit score (0–100), per-dimension score breakdown,
    extracted JD signals, matched/missing skills, and keyword gap analysis.
    """
    try:
        jd_info = extract_jd(req.jd_text)
        score_info = compute_fit_score(
            req.jd_text,
            req.candidate_text,
            jd_skills=jd_info["skills_and_tools"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    fit = score_info["fit_score"]
    grade = "Strong Fit" if fit >= 65 else "Partial Fit" if fit >= 40 else "Weak Fit"
    matched = score_info["matched_skills"]
    semantic = score_info.get("semantic_skills", [])
    missing = score_info["missing_skills"]
    total = max(len(jd_info["skills_and_tools"]), 1)
    match_rate = round((len(matched) + 0.5 * len(semantic)) / total * 100, 1)
    logger.info("API /analyse: fit=%.1f grade=%s", fit, grade)

    return AnalyseResponse(
        fit_score=fit,
        grade=grade,
        score_breakdown=score_info["score_breakdown"],
        jd_signals=JdSignals(
            seniority=jd_info["seniority"],
            education=jd_info["education"],
            work_type=jd_info["work_type"],
            industry=jd_info["industry"],
            years_of_experience=jd_info["years_of_experience"],
            skills_detected=jd_info["skills_and_tools"],
            soft_skills=jd_info.get("soft_skills", []),
        ),
        skill_analysis=SkillAnalysis(
            matched=matched,
            semantic=semantic,
            missing=missing,
            match_rate_pct=match_rate,
        ),
        keyword_analysis=KeywordAnalysis(
            matched=score_info["matched_keywords"],
            gaps=score_info["gap_keywords"],
        ),
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
