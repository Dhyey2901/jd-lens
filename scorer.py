"""Hybrid fit scorer: semantic embeddings (30%) + keyword overlap (30%) + skill match (40%)."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SCORE_WEIGHTS, SKILL_ALIASES

logger = logging.getLogger(__name__)

# Semantic match threshold for skill embedding similarity.
_SEMANTIC_SKILL_THRESHOLD = 0.52

# Top N JD keywords to check for presence in candidate text.
_KEYWORD_TOP_N = 20

# Minimum words per line when cleaning resume noise.
_RESUME_MIN_WORDS = 4


# ── Text pre-processing ────────────────────────────────────────────────────────

def _clean_resume(text: str) -> str:
    """Remove noise lines from PDF-extracted resumes.

    Strips headers, dates, addresses, and phone numbers — lines with fewer
    than _RESUME_MIN_WORDS words that add no signal to scoring.
    """
    lines = text.splitlines()
    meaningful = [line for line in lines if len(line.split()) >= _RESUME_MIN_WORDS]
    return "\n".join(meaningful).strip()


def _normalize_aliases(text: str) -> str:
    """Replace common tech abbreviations with their canonical KNOWN_TOOLS names.

    Applied to candidate text before skill matching so e.g. 'sklearn' registers
    as 'scikit-learn', 'postgres' as 'postgresql'.
    """
    result = text.lower()
    for alias, canonical in SKILL_ALIASES.items():
        result = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, result)
    return result


# ── Similarity functions ───────────────────────────────────────────────────────

def get_embedding_model() -> Any:
    """Lazy-load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


_embedding_model: Any = None


def _model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


def _semantic_similarity(text_a: str, text_b: str, model: Any) -> float:
    import torch
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True, normalize_embeddings=True)
    score = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    )
    return float(score.item())


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


_JD_BOILERPLATE = frozenset({
    "nice", "required", "requirements", "responsibilities", "responsibility",
    "experience", "years", "degree", "remote", "hybrid", "position", "role",
    "team", "company", "join", "looking", "seeking", "opportunity",
    "excellent", "strong", "ability", "skills", "skill", "knowledge",
    "understanding", "familiarity", "plus", "bonus", "preferred", "desired",
    "ideal", "candidate", "candidates", "minimum", "least", "year", "month",
    "mentor", "mentoring", "build", "builds", "building", "develop", "design",
    "lead", "leading", "bachelor", "master", "science", "scientist", "scientists",
    "senior", "junior", "engineer", "engineers", "platform", "presentations",
    "production", "models",
})


def _keyword_overlap_score(jd_text: str, candidate_text: str, top_n: int = _KEYWORD_TOP_N) -> float:
    """Extract top N technical JD keywords and check presence in candidate.

    Uses unigrams + custom boilerplate stop-words so CountVectorizer surfaces
    actual skill/tool terms rather than JD template language.
    Both texts should already have alias normalisation applied before calling.
    """
    jd_clean = re.sub(r"[-•–\n]+", " ", jd_text).strip()
    stop_words = list(_JD_BOILERPLATE)

    try:
        vec = CountVectorizer(
            ngram_range=(1, 1),
            stop_words=stop_words,
            max_features=top_n,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{2,}\b",
        )
        vec.fit([jd_clean])
        keywords = vec.get_feature_names_out()
    except ValueError:
        return 0.0

    candidate_lower = candidate_text.lower()
    hits = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", candidate_lower))
    score = hits / max(len(keywords), 1)
    logger.debug(
        "Keyword overlap: %d/%d hit — matched=%s",
        hits, len(keywords),
        [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", candidate_lower)],
    )
    return score


def _tokenize(text: str) -> set[str]:
    # (?<!\w) instead of \b so special-char tokens like c++ and node.js match correctly.
    return set(re.findall(r"(?<!\w)[a-z][a-z0-9+#.\-]+", text.lower()))


# ── Skill classification ───────────────────────────────────────────────────────

def _classify_skills(
    jd_skills: list[str],
    candidate_text: str,
    model: Any,
    threshold: float = _SEMANTIC_SKILL_THRESHOLD,
) -> tuple[list[str], list[str], list[str]]:
    """Split JD skills into exact matches, semantic matches, and missing.

    - Exact: regex word-boundary match after alias normalisation.
    - Semantic: embedding similarity to a candidate sentence ≥ threshold.
    - Missing: neither.
    """
    if not jd_skills:
        return [], [], []

    candidate_lower = candidate_text.lower()
    exact, semantic, missing = [], [], []

    chunks = [s.strip() for s in re.split(r"[.\n,;]", candidate_text) if len(s.strip()) > 8]
    if not chunks:
        chunks = [candidate_text]

    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    skill_embs = model.encode(jd_skills, normalize_embeddings=True)
    sim_matrix: np.ndarray = cosine_similarity(skill_embs, chunk_embs)

    for i, skill in enumerate(jd_skills):
        if re.search(r"\b" + re.escape(skill) + r"\b", candidate_lower):
            exact.append(skill)
        elif sim_matrix[i].max() >= threshold:
            semantic.append(skill)
            logger.debug("Semantic match: '%s' (score %.2f)", skill, sim_matrix[i].max())
        else:
            missing.append(skill)

    return exact, semantic, missing


def _tokenize_meaningful(tokens: set[str]) -> list[str]:
    return sorted(t for t in tokens if len(t) > 2 and not t.isdigit())


# ── Main scoring function ──────────────────────────────────────────────────────

def compute_fit_score(
    jd_text: str,
    candidate_text: str,
    jd_skills: list[str],
    embedding_model: Any = None,
) -> dict[str, Any]:
    """Return composite fit score with per-dimension breakdown and keyword analysis.

    Pipeline:
      1. Clean candidate text (strip PDF noise lines < 4 words).
      2. Normalise candidate aliases (sklearn → scikit-learn, etc.) for skill matching.
      3. Semantic similarity on cleaned texts.
      4. Keyword score = 0.6 × TF-IDF cosine + 0.4 × CountVectorizer top-20 overlap.
      5. Skill classification → exact (1pt) + semantic (0.5pt) / total.
      6. Composite = 0.30 sem + 0.30 keyword + 0.40 skill.
    """
    model = embedding_model or _model()

    # Step 1 — clean resume noise from candidate text
    candidate_clean = _clean_resume(candidate_text)
    if not candidate_clean:
        candidate_clean = candidate_text  # fallback if everything stripped

    # Step 2 — alias normalisation for skill matching only
    candidate_for_skills = _normalize_aliases(candidate_clean)

    # Step 3 — semantic similarity (cleaned texts, no alias substitution)
    sem_score = _semantic_similarity(jd_text, candidate_clean, model)

    # Step 4 — combined keyword score (use alias-normalised candidate so
    # "sklearn" counts toward "scikit-learn", "postgres" toward "postgresql")
    tfidf_score = _tfidf_similarity(jd_text, candidate_for_skills)
    kw_score = _keyword_overlap_score(jd_text, candidate_for_skills)
    combined_kw = 0.60 * tfidf_score + 0.40 * kw_score

    # Step 5 — skill matching on alias-normalised candidate
    exact, semantic_match, missing = _classify_skills(jd_skills, candidate_for_skills, model)
    skill_rate = (len(exact) + 0.5 * len(semantic_match)) / max(len(jd_skills), 1)

    # Step 6 — composite
    composite = (
        SCORE_WEIGHTS["semantic"] * sem_score
        + SCORE_WEIGHTS["tfidf"] * combined_kw
        + SCORE_WEIGHTS["skill"] * skill_rate
    )

    jd_tokens = _tokenize(jd_text)
    candidate_tokens = _tokenize(candidate_clean)

    logger.info(
        "Fit score: %.1f%% (sem=%.1f kw=%.1f skill=%.1f) | "
        "exact=%d semantic=%d missing=%d",
        composite * 100,
        sem_score * 100, combined_kw * 100, skill_rate * 100,
        len(exact), len(semantic_match), len(missing),
    )

    return {
        "fit_score": round(composite * 100, 1),
        "score_breakdown": {
            "Semantic Similarity": round(sem_score * 100, 1),
            "Keyword Overlap": round(combined_kw * 100, 1),
            "Skill Match Rate": round(skill_rate * 100, 1),
        },
        "matched_skills": exact,
        "semantic_skills": semantic_match,
        "missing_skills": missing,
        "matched_keywords": _tokenize_meaningful(jd_tokens & candidate_tokens)[:60],
        "gap_keywords": _tokenize_meaningful(jd_tokens - candidate_tokens)[:60],
    }
