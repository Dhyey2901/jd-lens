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
    # Office suite / soft-skill words that appear after context phrases
    "microsoft", "office", "excel", "word", "powerpoint", "suite",
    "attention", "detail", "written", "verbal", "interpersonal",
    "cross", "functional", "analytical", "critical", "detail",
    # Generic adjectives / nouns that aren't tools
    "complex", "technical", "business", "industry", "domain", "sector",
    "relevant", "related", "similar", "equivalent", "comparable",
    "hands", "real", "world", "practical", "working",
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


# Sentence-level signals that strongly indicate a "Required" sentence.
_REQUIRED_SIGNALS: list[str] = [
    "required", "must have", "you must", "we require", "essential",
    "mandatory", "need to have", "you need", "you will need",
    "bachelor", "master", "degree in", "years of experience",
    "proficiency in", "experience with", "knowledge of", "familiarity with",
    "expertise in", "background in", "skilled in",
    "citizen", "permanent resident", "eligible to work",
]


# ── Section-aware JD parsing ───────────────────────────────────────────────────

_SECTION_WEIGHT_MAP: list[tuple[list[str], float]] = [
    (["required", "must have", "essential", "mandatory", "minimum"], 2.0),
    (["skill", "qualification", "technical requirement", "core competenc"], 1.5),
    (["preferred", "desirable", "desired", "ideal candidate"],          1.0),
    (["nice to have", "nice-to-have", "bonus", "advantageous", "optional"], 0.5),
    (["responsibilit", "duties", "you will", "what you'll"],            1.0),
]


def _section_multiplier(header: str) -> float:
    lower = header.lower()
    for keywords, mult in _SECTION_WEIGHT_MAP:
        if any(kw in lower for kw in keywords):
            return mult
    return 1.0


def _is_section_header(line: str) -> bool:
    s = line.strip().rstrip(":").strip()
    return (
        3 <= len(s) <= 45
        and "," not in s
        and "." not in s
        and len(s.split()) <= 5
        and bool(s) and s[0].isupper()
    )


def parse_jd_sections(jd_text: str) -> list[tuple[str, float]]:
    """Split a JD into (section_text, weight_multiplier) pairs.

    The first block before any detected section header is treated as the
    title / overview and given a 3× multiplier.  Each subsequent section
    is weighted by its header (Required → 2×, Nice-to-have → 0.5×, etc.).
    Falls back to [(jd_text, 1.0)] when no headers are detected.
    """
    lines = jd_text.splitlines()
    split_points: list[int] = []  # line indices of section headers
    for i, line in enumerate(lines):
        if _is_section_header(line) and line.strip():
            split_points.append(i)

    if not split_points:
        return [(jd_text, 1.0)]

    sections: list[tuple[str, float]] = []

    # Title block
    title_text = "\n".join(lines[:split_points[0]]).strip()
    if title_text:
        sections.append((title_text, 3.0))

    for idx, start_line in enumerate(split_points):
        header = lines[start_line].strip().rstrip(":")
        end_line = split_points[idx + 1] if idx + 1 < len(split_points) else len(lines)
        content = "\n".join(lines[start_line + 1:end_line]).strip()
        if content:
            sections.append((content, _section_multiplier(header)))

    return sections or [(jd_text, 1.0)]


def weighted_skill_frequencies(
    jd_text: str,
    skills: list[str],
) -> dict[str, float]:
    """Return a normalised importance weight [0.5, 1.0] for each skill.

    Skills mentioned in Required sections score closer to 1.0; skills
    only appearing in Nice-to-have score closer to 0.5.  Skills absent
    from the JD get no entry — callers should use .get(skill, 1.0).
    """
    sections = parse_jd_sections(jd_text)
    raw: dict[str, float] = {}
    for skill in skills:
        pat = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
        total = sum(len(pat.findall(text)) * mult for text, mult in sections)
        if total > 0:
            raw[skill] = total

    if not raw:
        return {}

    max_w = max(raw.values())
    return {s: 0.5 + 0.5 * (w / max_w) for s, w in raw.items()}


def extract_required_sentences(text: str) -> str:
    """Return sentences that look like hard requirements — no model needed.

    Used to build required_jd_text for focused semantic scoring without
    waiting for the zero-shot classifier (~1.6 GB bart-large-mnli).
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    matched = [
        s.strip() for s in sentences
        if len(s.strip()) > 15
        and any(sig in s.lower() for sig in _REQUIRED_SIGNALS)
    ]
    return " ".join(matched)


def extract_jd(jd_text: str) -> dict[str, object]:
    """Return all structured signals parsed from a raw job description."""
    skills_and_tools = extract_skills_and_tools(jd_text)
    unknown_tools = extract_unknown_tools(jd_text)
    skill_weights = weighted_skill_frequencies(jd_text, skills_and_tools)
    result = {
        "skills_and_tools": skills_and_tools,
        "unknown_tools": unknown_tools,
        "skill_weights": skill_weights,
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
