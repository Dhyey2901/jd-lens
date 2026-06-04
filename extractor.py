"""JD signal extraction using pure regex — no spaCy dependency."""
from __future__ import annotations

import re

from config import (
    EDUCATION_SIGNALS,
    INDUSTRY_SIGNALS,
    KNOWN_TOOLS,
    SENIORITY_SIGNALS,
    SOFT_SKILLS,
    WORK_TYPE_SIGNALS,
)


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


def extract_jd(jd_text: str) -> dict[str, object]:
    """Return all structured signals parsed from a raw job description."""
    return {
        "skills_and_tools": extract_skills_and_tools(jd_text),
        "soft_skills": extract_soft_skills(jd_text),
        "years_of_experience": extract_years_of_experience(jd_text),
        "seniority": extract_seniority(jd_text),
        "education": extract_education(jd_text),
        "work_type": extract_work_type(jd_text),
        "industry": extract_industry(jd_text),
    }
