"""JD signal extraction using pure regex — no spaCy dependency."""
from __future__ import annotations

import logging
import re

from config import (
    EDUCATION_SIGNALS,
    INDUSTRY_SIGNALS,
    KNOWN_TOOLS,
    SENIORITY_SIGNALS,
    SOFT_SKILLS,
    WORK_TYPE_SIGNALS,
)

logger = logging.getLogger(__name__)

# Phrases that introduce technology requirements in JDs.
_TECH_CONTEXT_RE = re.compile(
    r"(?:proficiency|experience|expertise|familiarity|knowledge|skills?|background)\s+"
    r"(?:in|with|using|of|on|across)\s+"
    r"([A-Za-z][A-Za-z0-9+#.\-]{2,})",
    re.IGNORECASE,
)

# Common English words and JD boilerplate that appear after context phrases
# but are not tool names.
_CONTEXT_NOISE: frozenset[str] = frozenset({
    "the", "and", "with", "using", "all", "our", "your", "their", "its",
    "any", "some", "this", "these", "those", "other", "various", "multiple",
    "several", "modern", "latest", "current", "new", "existing", "large",
    "fast", "high", "low", "good", "great", "best", "top", "strong", "solid",
    "deep", "broad", "wide", "extensive", "hands",
    "experience", "knowledge", "skills", "skill", "background",
    "understanding", "familiarity", "proficiency", "expertise", "ability",
    "minimum", "least", "years", "year", "months", "month", "degree", "level",
    "core", "key", "main", "primary", "advanced", "basic", "scripting",
    "programming", "development", "engineering", "computing",
    "technologies", "technology", "frameworks", "framework", "libraries",
    "languages", "language", "databases", "database", "tools", "tool",
    "platforms", "platform", "systems", "system", "services", "service",
    "environments", "environment", "solutions", "solution", "applications",
    "distributed", "cloud", "backend", "frontend", "fullstack", "full",
    "stack", "data", "enterprise", "production", "open", "source", "based",
    "driven", "oriented", "native",
})


def extract_years_of_experience(text: str) -> list[str]:
    pattern = re.compile(
        r"(\d+\+?\s*(?:to|-)\s*\d+\s*years?|\d+\+\s*years?|\d+\s*years?\s*of\s*experience)",
        re.IGNORECASE,
    )
    return sorted({m.group().strip() for m in pattern.finditer(text)})


def extract_seniority(text: str) -> str:
    lower = text.lower()
    for level, signals in SENIORITY_SIGNALS.items():
        if any(s in lower for s in signals):
            return level
    return "unspecified"


def extract_education(text: str) -> str:
    lower = text.lower()
    for level, signals in EDUCATION_SIGNALS.items():
        if any(s in lower for s in signals):
            return level
    return "unspecified"


def extract_work_type(text: str) -> str:
    lower = text.lower()
    for work_type, signals in WORK_TYPE_SIGNALS.items():
        if any(s in lower for s in signals):
            return work_type
    return "unspecified"


def extract_skills_and_tools(text: str) -> list[str]:
    lower = text.lower()
    return sorted(
        tool for tool in KNOWN_TOOLS
        if re.search(r"\b" + re.escape(tool) + r"\b", lower)
    )


def extract_soft_skills(text: str) -> list[str]:
    """Return soft skills and competencies mentioned in the text."""
    lower = text.lower()
    return sorted(
        skill for skill in SOFT_SKILLS
        if re.search(r"\b" + re.escape(skill) + r"\b", lower)
    )


def extract_industry(text: str) -> str:
    """Infer target industry from vocabulary signals; defaults to 'general tech'."""
    lower = text.lower()
    for industry, signals in INDUSTRY_SIGNALS.items():
        if any(s in lower for s in signals):
            return industry
    return "general tech"


def extract_unknown_tools(text: str) -> list[str]:
    """Detect tech tool names in the JD not present in KNOWN_TOOLS.

    Extracts the object of context phrases like 'experience with X' or
    'proficiency in X', then filters against KNOWN_TOOLS and common English
    words.  Useful for catching tools not yet in the hardcoded vocabulary
    (e.g. Temporal, Polars, Prefect, Databricks).
    """
    known_lower = {t.lower() for t in KNOWN_TOOLS}
    detected: set[str] = set()
    for match in _TECH_CONTEXT_RE.finditer(text):
        term = match.group(1).strip().lower()
        if (
            len(term) >= 3
            and term not in known_lower
            and term not in _CONTEXT_NOISE
            and not term.isdigit()
        ):
            detected.add(term)
    result = sorted(detected)
    if result:
        logger.debug("Unknown tools detected: %s", result)
    return result


def extract_jd(jd_text: str) -> dict[str, object]:
    """Return all structured signals parsed from a raw job description."""
    skills_and_tools = extract_skills_and_tools(jd_text)
    unknown_tools = extract_unknown_tools(jd_text)
    result = {
        "skills_and_tools": skills_and_tools,
        "unknown_tools": unknown_tools,
        "soft_skills": extract_soft_skills(jd_text),
        "years_of_experience": extract_years_of_experience(jd_text),
        "seniority": extract_seniority(jd_text),
        "education": extract_education(jd_text),
        "work_type": extract_work_type(jd_text),
        "industry": extract_industry(jd_text),
    }
    logger.info(
        "Extracted JD: seniority=%s industry=%s skills=%d unknown=%d soft=%d",
        result["seniority"], result["industry"],
        len(skills_and_tools), len(unknown_tools), len(result["soft_skills"]),
    )
    return result
