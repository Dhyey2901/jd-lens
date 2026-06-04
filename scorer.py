"""Hybrid fit scorer: semantic embeddings (50%) + TF-IDF (30%) + skill match (20%)."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SCORE_WEIGHTS

logger = logging.getLogger(__name__)

# Semantic match threshold: skill embedding vs candidate text chunks.
# Tuned so common synonyms (e.g. "PyTorch" ↔ "deep learning") score ≥ threshold
# while unrelated terms stay below it.
_SEMANTIC_SKILL_THRESHOLD = 0.52


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


def _tokenize(text: str) -> set[str]:
    # (?<!\w) instead of \b so special-char tokens like c++ and node.js match correctly.
    return set(re.findall(r"(?<!\w)[a-z][a-z0-9+#.\-]+", text.lower()))


def _classify_skills(
    jd_skills: list[str],
    candidate_text: str,
    model: Any,
    threshold: float = _SEMANTIC_SKILL_THRESHOLD,
) -> tuple[list[str], list[str], list[str]]:
    """Split JD skills into exact matches, semantic matches, and missing.

    - Exact: regex word-boundary match in candidate text.
    - Semantic: no exact match, but embedding similarity to a candidate
      sentence exceeds *threshold* (e.g. 'PyTorch' ↔ 'deep learning').
    - Missing: neither.
    """
    if not jd_skills:
        return [], [], []

    candidate_lower = candidate_text.lower()
    exact, semantic, missing = [], [], []

    # Chunk candidate text at sentence/clause boundaries for finer matching
    chunks = [s.strip() for s in re.split(r"[.\n,;]", candidate_text) if len(s.strip()) > 8]
    if not chunks:
        chunks = [candidate_text]

    chunk_embs = model.encode(chunks, normalize_embeddings=True)
    skill_embs = model.encode(jd_skills, normalize_embeddings=True)
    # Shape: (n_skills, n_chunks)
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


def compute_fit_score(
    jd_text: str,
    candidate_text: str,
    jd_skills: list[str],
    embedding_model: Any = None,
) -> dict[str, Any]:
    """Return composite fit score with per-dimension breakdown and keyword analysis."""
    model = embedding_model or _model()

    sem_score = _semantic_similarity(jd_text, candidate_text, model)
    tfidf_score = _tfidf_similarity(jd_text, candidate_text)

    exact, semantic, missing = _classify_skills(jd_skills, candidate_text, model)

    # Exact match = 1.0 point, semantic match = 0.5 points
    skill_rate = (len(exact) + 0.5 * len(semantic)) / max(len(jd_skills), 1)

    composite = (
        SCORE_WEIGHTS["semantic"] * sem_score
        + SCORE_WEIGHTS["tfidf"] * tfidf_score
        + SCORE_WEIGHTS["skill"] * skill_rate
    )

    jd_tokens = _tokenize(jd_text)
    candidate_tokens = _tokenize(candidate_text)

    logger.info(
        "Fit score: %.1f%% (sem=%.1f tfidf=%.1f skill=%.1f) | "
        "exact=%d semantic=%d missing=%d",
        composite * 100, sem_score * 100, tfidf_score * 100, skill_rate * 100,
        len(exact), len(semantic), len(missing),
    )

    return {
        "fit_score": round(composite * 100, 1),
        "score_breakdown": {
            "Semantic Similarity": round(sem_score * 100, 1),
            "Keyword Overlap": round(tfidf_score * 100, 1),
            "Skill Match Rate": round(skill_rate * 100, 1),
        },
        "matched_skills": exact,
        "semantic_skills": semantic,
        "missing_skills": missing,
        "matched_keywords": _tokenize_meaningful(jd_tokens & candidate_tokens)[:60],
        "gap_keywords": _tokenize_meaningful(jd_tokens - candidate_tokens)[:60],
    }
