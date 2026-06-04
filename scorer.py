"""Hybrid fit scorer: semantic embeddings (50%) + TF-IDF (30%) + skill match (20%)."""
from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SCORE_WEIGHTS


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
    return set(re.findall(r"\b[a-z][a-z0-9+#.\-]{2,}\b", text.lower()))


def compute_fit_score(
    jd_text: str,
    candidate_text: str,
    jd_skills: list[str],
    embedding_model: Any = None,
) -> dict[str, Any]:
    """Return composite fit score with per-dimension breakdown and keyword analysis."""
    model = embedding_model or _model()

    semantic = _semantic_similarity(jd_text, candidate_text, model)
    tfidf = _tfidf_similarity(jd_text, candidate_text)

    candidate_lower = candidate_text.lower()
    matched_skills = [
        s for s in jd_skills
        if re.search(r"\b" + re.escape(s) + r"\b", candidate_lower)
    ]
    missing_skills = [s for s in jd_skills if s not in matched_skills]
    skill_rate = len(matched_skills) / max(len(jd_skills), 1)

    composite = (
        SCORE_WEIGHTS["semantic"] * semantic
        + SCORE_WEIGHTS["tfidf"] * tfidf
        + SCORE_WEIGHTS["skill"] * skill_rate
    )

    jd_tokens = _tokenize(jd_text)
    candidate_tokens = _tokenize(candidate_text)
    is_meaningful = lambda t: len(t) > 2 and not t.isdigit()

    return {
        "fit_score": round(composite * 100, 1),
        "score_breakdown": {
            "Semantic Similarity": round(semantic * 100, 1),
            "Keyword Overlap": round(tfidf * 100, 1),
            "Skill Match Rate": round(skill_rate * 100, 1),
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_keywords": [t for t in sorted(jd_tokens & candidate_tokens) if is_meaningful(t)][:60],
        "gap_keywords": [t for t in sorted(jd_tokens - candidate_tokens) if is_meaningful(t)][:60],
    }
