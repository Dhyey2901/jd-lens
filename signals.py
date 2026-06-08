"""Hiring signal scorer — measures quality signals that predict candidate selection.

Operates entirely on resume text, independently of JD keyword overlap.
Captures the patterns that experienced recruiters recognise:
impact quantification, project depth, deployment evidence, action-verb
strength, learning trajectory, and soft-skill coverage.

This is the insight behind the tool: a candidate can score low on JD
keyword match and still be selected because hiring signals matter more —
especially for graduate and early-career roles where tools are teachable
but depth of thinking is not.
"""
from __future__ import annotations

import re
from typing import Any

# ── Impact quantification ──────────────────────────────────────────────────────
# Lines with numbers signal real, measurable work rather than vague claims.
_METRIC_RE = re.compile(
    r"\b\d+(?:[,.\s]\d+)*\s*(?:[KkMBb%+x]|million|billion|thousand|hundred|"
    r"percent|users|docs|records|factors|models|datasets|hours|minutes|days|"
    r"members|engineers|scientists|clients|projects)?\b",
    re.IGNORECASE,
)
_SCALE_BOOST_RE = re.compile(
    r"\b(\d+[KkMBb+]\b|\d+\s*(?:million|billion|thousand)|\d+%|\d+x\b|\d{4,})",
    re.IGNORECASE,
)

# ── Deployment & production evidence ──────────────────────────────────────────
_DEPLOYMENT_SIGNALS: frozenset[str] = frozenset({
    "deployed", "production", "shipped", "released", "launched", "live",
    "web application", "api endpoint", "hosted", "serving", "real-world",
    "end users", "in production", "available", "accessible",
})

# ── Action verbs — strong vs passive ─────────────────────────────────────────
_STRONG_VERBS: frozenset[str] = frozenset({
    "led", "spearheaded", "architected", "drove", "launched", "deployed",
    "shipped", "reduced", "increased", "improved", "achieved", "delivered",
    "optimised", "optimized", "designed", "built", "developed", "implemented",
    "created", "established", "mentored", "managed", "supervised",
    "coordinated", "directed", "analysed", "analyzed", "engineered",
    "automated", "integrated", "generated", "evaluated", "performed",
    "constructed", "formulated", "strategised", "strategized",
})
_WEAK_VERBS: frozenset[str] = frozenset({
    "helped", "assisted", "supported", "participated", "was involved",
    "contributed to", "aided", "shadowed", "observed",
})

# ── Project complexity markers ─────────────────────────────────────────────────
_COMPLEXITY_SIGNALS: frozenset[str] = frozenset({
    "end-to-end", "end to end", "pipeline", "architecture", "scalable",
    "distributed", "real-time", "real time", "hybrid", "ensemble",
    "integration", "microservice", "automated", "optimized", "optimised",
    "validation", "evaluation", "cross-validation", "fine-tuning",
    "retrieval", "inference", "embedding", "vector", "semantic",
})

# ── Learning trajectory ────────────────────────────────────────────────────────
_DEGREE_LEVEL: dict[str, int] = {
    "phd": 4, "doctorate": 4, "doctoral": 4,
    "master": 3, "msc": 3, "m.s": 3, "postgraduate": 3, "graduate": 2,
    "bachelor": 2, "bsc": 2, "b.s": 2, "undergraduate": 2,
    "diploma": 1, "certificate": 1,
}
_ENROLLMENT_SIGNALS: frozenset[str] = frozenset({
    "present", "current", "ongoing", "in progress", "expected",
    "graduating", "pursuing", "enrolled",
})

# ── Role detection ─────────────────────────────────────────────────────────────

_ROLE_SIGNALS: dict[str, frozenset[str]] = {
    "consulting": frozenset({
        "stakeholder", "client", "strategy", "advisory", "engagement",
        "transformation", "management consulting", "change management",
        "deliverable", "consulting", "consultant",
    }),
    "data_engineering": frozenset({
        "pipeline", "etl", "spark", "airflow", "kafka", "databricks",
        "dbt", "orchestration", "data warehouse", "snowflake", "ingestion",
        "streaming", "data lake", "batch processing",
    }),
    "data_analyst": frozenset({
        "dashboard", "reporting", "insights", "tableau", "power bi",
        "looker", "business intelligence", "kpi", "excel",
        "visualis", "bi tool",
    }),
    "research_ml": frozenset({
        "research", "publication", "paper", "experiment", "benchmark",
        "fine-tuning", "pretraining", "arxiv", "phd", "llm",
    }),
    "software_engineering": frozenset({
        "api", "backend", "microservice", "kubernetes", "infrastructure",
        "ci/cd", "devops", "system design", "distributed systems",
        "containeris",
    }),
}

# Per-role dimension weights — all rows sum to 1.0.
# skill_gap is held at ~0.17-0.18 across all roles since required-skill
# coverage matters regardless of role type.
_ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "consulting": {
        "impact": 0.18, "depth": 0.10, "deployment": 0.05,
        "verbs": 0.22, "trajectory": 0.12, "soft_cov": 0.15, "skill_gap": 0.18,
    },
    "data_engineering": {
        "impact": 0.18, "depth": 0.22, "deployment": 0.22,
        "verbs": 0.10, "trajectory": 0.05, "soft_cov": 0.05, "skill_gap": 0.18,
    },
    "data_analyst": {
        "impact": 0.25, "depth": 0.17, "deployment": 0.08,
        "verbs": 0.13, "trajectory": 0.10, "soft_cov": 0.10, "skill_gap": 0.17,
    },
    "research_ml": {
        "impact": 0.18, "depth": 0.27, "deployment": 0.08,
        "verbs": 0.12, "trajectory": 0.12, "soft_cov": 0.05, "skill_gap": 0.18,
    },
    "software_engineering": {
        "impact": 0.18, "depth": 0.22, "deployment": 0.17,
        "verbs": 0.12, "trajectory": 0.07, "soft_cov": 0.07, "skill_gap": 0.17,
    },
    "general": {
        "impact": 0.21, "depth": 0.16, "deployment": 0.10,
        "verbs": 0.16, "trajectory": 0.09, "soft_cov": 0.10, "skill_gap": 0.18,
    },
}


# ── Individual scorers ─────────────────────────────────────────────────────────

def _score_impact(text: str) -> tuple[float, int]:
    """Fraction of non-trivial lines that contain a measurable number."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 15]
    if not lines:
        return 0.0, 0
    quantified = sum(1 for ln in lines if _METRIC_RE.search(ln))
    scale_hits = len(_SCALE_BOOST_RE.findall(text))
    base = quantified / len(lines)
    boost = min(scale_hits * 0.06, 0.25)
    return min(base + boost, 1.0), quantified


def _score_depth(text: str) -> float:
    """Project complexity: multi-technology stacks and architectural thinking."""
    lower = text.lower()
    complexity_hits = sum(1 for s in _COMPLEXITY_SIGNALS if s in lower)
    # Count "|" separators in project headers — each signals a multi-tool stack
    pipe_count = text.count("|")
    # Multi-component bullet heuristic: bullets mentioning "and" between tech terms
    multi_component = len(re.findall(
        r"(?:using|with|including|via)\s+\w+\s+and\s+\w+", lower
    ))
    depth = (
        min(complexity_hits / 7.0, 0.45)
        + min(pipe_count / 5.0, 0.30)
        + min(multi_component / 4.0, 0.25)
    )
    return min(depth, 1.0)


def _score_deployment(text: str) -> float:
    """Evidence that the candidate shipped something real."""
    lower = text.lower()
    hits = sum(1 for s in _DEPLOYMENT_SIGNALS if s in lower)
    return min(hits / 3.0, 1.0)


def _score_verbs(text: str) -> float:
    """Ratio of strong action verbs to weak/passive ones."""
    lower = text.lower()
    strong = sum(
        1 for v in _STRONG_VERBS
        if re.search(r"\b" + re.escape(v) + r"\b", lower)
    )
    weak = sum(
        1 for v in _WEAK_VERBS
        if re.search(r"\b" + re.escape(v) + r"\b", lower)
    )
    total = strong + weak
    return (strong / total) if total else 0.5


def _score_trajectory(text: str) -> float:
    """Degree progression and evidence of continuous learning."""
    lower = text.lower()
    highest = max(
        (lvl for kw, lvl in _DEGREE_LEVEL.items() if kw in lower),
        default=0,
    )
    currently_enrolled = any(s in lower for s in _ENROLLMENT_SIGNALS)
    degree_score = highest / 4.0
    trajectory_bonus = 0.20 if (currently_enrolled and highest >= 2) else 0.0
    return min(degree_score + trajectory_bonus, 1.0)


def _score_soft_coverage(candidate_text: str, jd_soft_skills: list[str]) -> float:
    """Fraction of JD soft skills present in the resume."""
    if not jd_soft_skills:
        return 0.5
    lower = candidate_text.lower()
    matched = sum(
        1 for skill in jd_soft_skills
        if re.search(r"\b" + re.escape(skill) + r"\b", lower)
    )
    return matched / len(jd_soft_skills)


def detect_role_type(jd_text: str) -> str:
    """Detect role category from JD text using keyword frequency.

    Counts signal hits per category and returns the winner when it reaches
    a minimum of 2 hits (to avoid noise from incidental word matches).
    Falls back to 'general' when no category clears the threshold.
    """
    lower = jd_text.lower()
    scores = {
        role: sum(1 for sig in signals if sig in lower)
        for role, signals in _ROLE_SIGNALS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def _score_skill_gap(
    jd_skills: list[str],
    missing_skills: list[str] | None,
    skill_weights: dict[str, float] | None = None,
) -> float:
    """Weighted fraction of JD-required skills the candidate covers.

    When skill_weights are provided (from parse_jd_sections), missing a
    Required-section skill penalises more than missing a Nice-to-have.
    Falls back to uniform weighting when weights are unavailable.
    Returns 1.0 (neutral) when no JD skills or missing_skills data.
    """
    if not jd_skills or missing_skills is None:
        return 1.0
    if skill_weights:
        total_w = sum(skill_weights.get(s, 1.0) for s in jd_skills)
        missing_w = sum(skill_weights.get(s, 1.0) for s in missing_skills)
        return max(1.0 - missing_w / total_w, 0.0)
    present = max(len(jd_skills) - len(missing_skills), 0)
    return present / len(jd_skills)


# ── Prediction engine ──────────────────────────────────────────────────────────

def generate_prediction(
    jd_match: float,
    signal_score: float,
    missing_skills: list[str],
) -> dict[str, str]:
    """Four-quadrant hiring prediction from JD match + signal score.

    This is the core insight: jd_match measures keyword alignment,
    signal_score measures the quality patterns that experienced recruiters
    actually screen for.  The combination predicts outcome better than
    either dimension alone.
    """
    missing_preview = ", ".join(missing_skills[:3])

    if jd_match >= 65 and signal_score >= 70:
        return {
            "verdict": "Strong Candidate",
            "icon": "🎯",
            "color": "#2ecc71",
            "explanation": (
                "Meets JD requirements and demonstrates strong quality signals. "
                "High probability of advancing through both ATS and human screening."
            ),
        }

    if jd_match < 65 and signal_score >= 70:
        gap_note = f" Missing: {missing_preview}." if missing_preview else ""
        return {
            "verdict": "Overlooked Gem",
            "icon": "💎",
            "color": "#3498db",
            "explanation": (
                f"Strong quality signals suggest high potential beyond the keyword gap.{gap_note} "
                "Experienced recruiters typically advance profiles like this — "
                "depth of work shown here is harder to teach than the missing tools."
            ),
        }

    if jd_match >= 65 and signal_score < 50:
        return {
            "verdict": "Keyword Match — Verify Depth",
            "icon": "🔎",
            "color": "#f39c12",
            "explanation": (
                "Resume aligns with JD requirements but quality signals are limited. "
                "Recommend screening for real project depth and measurable outcomes."
            ),
        }

    if jd_match < 40 and signal_score < 50:
        return {
            "verdict": "Significant Gaps",
            "icon": "❌",
            "color": "#e74c3c",
            "explanation": (
                "Both keyword coverage and quality signals are below threshold. "
                "Consider targeting a different role or strengthening the resume significantly."
            ),
        }

    return {
        "verdict": "Moderate Fit",
        "icon": "📊",
        "color": "#f39c12",
        "explanation": (
            "Reasonable alignment with some gaps. Review the missing skills and "
            "quality signal breakdown to identify the highest-impact improvements."
        ),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_hiring_signals(
    candidate_text: str,
    jd_soft_skills: list[str] | None = None,
    jd_text: str | None = None,
    jd_skills: list[str] | None = None,
    missing_skills: list[str] | None = None,
    skill_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute hiring signal scores, weighted for the detected role type.

    When jd_text is supplied the role category is detected and dimension
    weights are adjusted accordingly — so the same resume scores differently
    against a consulting JD vs a data-engineering JD.  jd_skills +
    missing_skills feed the skill-gap dimension (7th).

    All JD params are optional; omitting them falls back to 'general' weights
    with neutral skill-gap scoring, preserving backward compatibility.
    """
    jd_soft_skills = jd_soft_skills or []
    jd_skills = jd_skills or []

    role_type = detect_role_type(jd_text) if jd_text else "general"
    weights = _ROLE_WEIGHTS[role_type]

    impact, quantified_count = _score_impact(candidate_text)
    depth = _score_depth(candidate_text)
    deployment = _score_deployment(candidate_text)
    verbs = _score_verbs(candidate_text)
    trajectory = _score_trajectory(candidate_text)
    soft_cov = _score_soft_coverage(candidate_text, jd_soft_skills)
    skill_gap = _score_skill_gap(jd_skills, missing_skills, skill_weights)

    composite = (
        weights["impact"]     * impact
        + weights["depth"]      * depth
        + weights["deployment"] * deployment
        + weights["verbs"]      * verbs
        + weights["trajectory"] * trajectory
        + weights["soft_cov"]   * soft_cov
        + weights["skill_gap"]  * skill_gap
    )

    if composite >= 0.70:
        grade = "Strong Signals"
    elif composite >= 0.50:
        grade = "Good Signals"
    else:
        grade = "Weak Signals"

    return {
        "hiring_signal_score": round(composite * 100, 1),
        "grade": grade,
        "role_type": role_type,
        "breakdown": {
            "Impact & Metrics":     round(impact * 100, 1),
            "Project Depth":        round(depth * 100, 1),
            "Deployment Evidence":  round(deployment * 100, 1),
            "Action Verb Strength": round(verbs * 100, 1),
            "Learning Trajectory":  round(trajectory * 100, 1),
            "Soft Skill Coverage":  round(soft_cov * 100, 1),
            "Skill Gap Coverage":   round(skill_gap * 100, 1),
        },
        "quantified_lines": quantified_count,
    }
