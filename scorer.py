"""Hybrid fit scorer: semantic embeddings (30%) + keyword overlap (30%) + skill match (40%)."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CROSS_ENCODER_MODEL, SCORE_WEIGHTS, SKILL_ALIASES

logger = logging.getLogger(__name__)

# Semantic match threshold for skill embedding similarity.
_SEMANTIC_SKILL_THRESHOLD = 0.52

# Top N JD keywords to check for presence in candidate text.
_KEYWORD_TOP_N = 20

# Common resume section header patterns.
_SECTION_HEADER_RE = re.compile(
    r"^(TECHNICAL\s+SKILLS?|SKILLS?|CORE\s+SKILLS?|COMPETENCIES|TECHNOLOGIES|"
    r"KEY\s+SKILLS?|TOOLS?\s*[&]+\s*TECHNOLOGIES?|TECH\s+STACK|"
    r"WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT\s+HISTORY|"
    r"EDUCATION|PROJECTS?|PERSONAL\s+PROJECTS?|"
    r"CERTIFICATIONS?|CERTIFICATES?|TRAINING|COURSES?|"
    r"SUMMARY|PROFESSIONAL\s+SUMMARY|PROFILE|OBJECTIVE|ABOUT\s*ME?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Keys that identify the skills section when scanning section names.
_SKILLS_SECTION_KEYS: frozenset[str] = frozenset({
    "skills", "technical skills", "core skills", "competencies",
    "technologies", "key skills", "tools & technologies",
    "tools and technologies", "tech stack", "technical competencies",
})

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


# ── Section detection ─────────────────────────────────────────────────────────

def _detect_resume_sections(text: str) -> dict[str, str]:
    """Split a resume into named sections using common header patterns.

    Returns {normalised_section_name: section_content}.
    Returns an empty dict when no recognisable headers are found (blob text).
    """
    headers = list(_SECTION_HEADER_RE.finditer(text))
    if not headers:
        return {}

    sections: dict[str, str] = {}
    for i, match in enumerate(headers):
        name = match.group(0).strip().rstrip(":").strip().lower()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[start:end].strip()
        if content:
            sections[name] = content

    return sections


def _get_skills_section(sections: dict[str, str]) -> str | None:
    """Return the content of the skills/technologies section, or None."""
    for key, content in sections.items():
        if any(sk in key for sk in _SKILLS_SECTION_KEYS):
            return content
    return None


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


def get_cross_encoder() -> Any | None:
    """Lazy-load the CrossEncoder model (~22 MB).

    Returns None if the package or model is unavailable so callers can
    fall back to the bi-encoder gracefully.
    """
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception as exc:
        logger.warning("CrossEncoder unavailable (%s) — falling back to bi-encoder", exc)
        return None


def _cross_encoder_score(
    required_sentences: list[str],
    candidate_text: str,
    model: Any,
) -> float:
    """Score candidate fit against required JD sentences using a cross-encoder.

    Each required sentence is scored independently against a truncated slice
    of the candidate text to stay within the 512-token limit.  The mean of
    the top-5 sigmoid-normalised logits is returned.
    """
    if not required_sentences:
        return 0.0
    candidate_trunc = candidate_text[:2500]
    pairs = [(sent, candidate_trunc) for sent in required_sentences[:20]]
    raw_scores = np.array(model.predict(pairs), dtype=float)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))
    k = min(5, len(sigmoid_scores))
    score = float(np.sort(sigmoid_scores)[-k:].mean())
    logger.debug("Cross-encoder semantic score: %.1f%%", score * 100)
    return score


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

    # Primary split: punctuation / newlines
    chunks = [s.strip() for s in re.split(r"[.\n,;]", candidate_text) if len(s.strip()) > 8]

    # Fallback for PDF blobs — text with stripped punctuation gives ≤ 2 chunks,
    # making skill-level semantic matching unreliable.  When a SKILLS section is
    # detectable, prepend its fine-grained splits as priority chunks so skill terms
    # get their own embeddings.  Otherwise fall back to a sliding word window.
    if len(chunks) < 3:
        sections = _detect_resume_sections(candidate_text)
        skills_text = _get_skills_section(sections)
        words = candidate_text.split()
        chunk_size, step = 40, 25
        window_chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), step)
            if " ".join(words[i: i + chunk_size]).strip()
        ]
        if skills_text:
            skill_chunks = [
                s.strip() for s in re.split(r"[.\n,;|•\-]", skills_text)
                if len(s.strip()) > 4
            ]
            chunks = skill_chunks + window_chunks
            logger.debug(
                "Blob fallback: %d skills-section chunks + %d windows",
                len(skill_chunks), len(window_chunks),
            )
        else:
            chunks = window_chunks
            logger.debug(
                "Blob fallback: %d word-window chunks from %d words",
                len(chunks), len(words),
            )

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


def _bullet_alignment_scores(
    jd_text: str,
    candidate_text: str,
    model: Any,
    top_n: int = 3,
) -> list[dict]:
    """Return top N resume lines with their best-matching JD requirement.

    For each top resume bullet, finds the single JD requirement sentence it
    aligns with most closely — so HR can see "this bullet satisfies that
    requirement" rather than just a raw similarity number.
    """
    bullets = [
        s.strip()
        for s in re.split(r"\n|(?<=[.!?])\s+", candidate_text)
        if len(s.strip()) > 25
    ]
    if not bullets:
        return []

    # JD requirements: individual sentences, filtered to a useful length range
    jd_reqs = [
        s.strip()
        for s in re.split(r"\n|(?<=[.!?])\s+", jd_text)
        if 20 < len(s.strip()) < 300
    ] or [jd_text]

    bullet_embs = model.encode(bullets, normalize_embeddings=True)
    req_embs   = model.encode(jd_reqs,  normalize_embeddings=True)

    # Rank bullets against JD centroid for selection
    jd_centroid = req_embs.mean(axis=0, keepdims=True)
    jd_scores   = cosine_similarity(bullet_embs, jd_centroid).flatten()

    # Per-bullet × per-requirement similarity for best-match lookup
    req_sim = cosine_similarity(bullet_embs, req_embs)  # (n_bullets, n_reqs)

    top_idx = np.argsort(jd_scores)[::-1][:top_n]
    return [
        {
            "text":    bullets[i],
            "jd_req":  jd_reqs[int(np.argmax(req_sim[i]))],
            "score":   round(float(req_sim[i].max()) * 100, 1),
        }
        for i in top_idx
    ]


# ── Result cache ──────────────────────────────────────────────────────────────
# Keyed by SHA-256 of (jd_text, candidate_text, sorted jd_skills).
# Model instances and required_jd_text are NOT part of the key — same inputs
# always produce the same score regardless of which model object is passed.

_score_cache: dict[str, dict[str, Any]] = {}
_SCORE_CACHE_MAX = 128


def _cache_key(jd_text: str, candidate_text: str, jd_skills: list[str]) -> str:
    raw = f"{jd_text}\x00{candidate_text}\x00{'|'.join(sorted(jd_skills))}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Main scoring function ──────────────────────────────────────────────────────

def compute_fit_score(
    jd_text: str,
    candidate_text: str,
    jd_skills: list[str],
    embedding_model: Any = None,
    required_jd_text: str | None = None,
    cross_encoder: Any = None,
) -> dict[str, Any]:
    """Return composite fit score with per-dimension breakdown and keyword analysis.

    Pipeline:
      1. Clean candidate text (strip PDF noise lines < 4 words).
      2. Normalise candidate aliases (sklearn → scikit-learn, etc.) for skill matching.
      3. Semantic similarity — cross-encoder on required sentences when available,
         otherwise bi-encoder on required_jd_text or full JD text.
      4. Keyword score = 0.6 × TF-IDF cosine + 0.4 × CountVectorizer top-20 overlap.
      5. Skill classification → exact (1pt) + semantic (0.5pt) / total.
      6. Composite = 0.30 sem + 0.30 keyword + 0.40 skill.
    """
    model = embedding_model or _model()

    # Cache check — skip recomputation for repeated (jd, candidate, skills) triples.
    _key = _cache_key(jd_text, candidate_text, jd_skills)
    if _key in _score_cache:
        logger.debug("Score cache hit — returning cached result")
        return _score_cache[_key]

    # Diagnose early: empty jd_skills means the 40% skill weight contributes 0
    # and the total score is capped at ~60% even for a perfect candidate.
    logger.info(
        "JD skills extracted: %d — %s",
        len(jd_skills), jd_skills[:10] if jd_skills else "EMPTY — check JD formatting",
    )
    if not jd_skills:
        logger.warning(
            "No skills detected in JD. Skill dimension will be skipped and weights "
            "redistributed to semantic + keyword. Ensure the JD lists tools/technologies."
        )

    # Step 1 — clean resume noise from candidate text
    candidate_clean = _clean_resume(candidate_text)
    if not candidate_clean:
        candidate_clean = candidate_text  # fallback if everything stripped

    # Step 2 — alias normalisation for skill matching only
    candidate_for_skills = _normalize_aliases(candidate_clean)

    # Step 3 — semantic similarity.
    # Cross-encoder (when provided) scores each required sentence against the
    # candidate independently, avoiding the averaging-out of a single document
    # embedding.  Falls back to bi-encoder on required_jd_text or full JD text.
    if cross_encoder is not None and required_jd_text:
        required_sents = [
            s.strip() for s in re.split(r"[.!?\n]", required_jd_text)
            if len(s.strip()) > 10
        ]
        sem_score = _cross_encoder_score(
            required_sents or [required_jd_text], candidate_clean, cross_encoder
        )
    else:
        sem_score = _semantic_similarity(
            required_jd_text if required_jd_text else jd_text,
            candidate_clean,
            model,
        )

    # Step 4 — combined keyword score (use alias-normalised candidate so
    # "sklearn" counts toward "scikit-learn", "postgres" toward "postgresql")
    tfidf_score = _tfidf_similarity(jd_text, candidate_for_skills)
    kw_score = _keyword_overlap_score(jd_text, candidate_for_skills)
    combined_kw = 0.60 * tfidf_score + 0.40 * kw_score

    # Step 5 — skill matching on alias-normalised candidate
    exact, semantic_match, missing = _classify_skills(jd_skills, candidate_for_skills, model)
    skill_rate = (len(exact) + 0.5 * len(semantic_match)) / max(len(jd_skills), 1)

    # Step 6 — composite
    # If no JD skills were detected, redistribute the 40% skill weight to
    # semantic + keyword so the score isn't artificially floored at ~60%.
    if not jd_skills:
        sem_w = SCORE_WEIGHTS["semantic"] / (SCORE_WEIGHTS["semantic"] + SCORE_WEIGHTS["tfidf"])
        kw_w  = SCORE_WEIGHTS["tfidf"]    / (SCORE_WEIGHTS["semantic"] + SCORE_WEIGHTS["tfidf"])
        composite = sem_w * sem_score + kw_w * combined_kw
    else:
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

    top_bullets = _bullet_alignment_scores(jd_text, candidate_clean, model)

    result = {
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
        "top_bullets": top_bullets,
    }

    # Evict oldest entry when the cache is full (dict insertion order = LRU).
    if len(_score_cache) >= _SCORE_CACHE_MAX:
        _score_cache.pop(next(iter(_score_cache)))
    _score_cache[_key] = result
    return result
