import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9+#.\-]{1,}\b", text.lower()))


def compute_fit_score(jd_text: str, candidate_text: str) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform([jd_text, candidate_text])
    score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    jd_tokens = _tokenize(jd_text)
    candidate_tokens = _tokenize(candidate_text)

    matched = sorted(jd_tokens & candidate_tokens)
    gaps = sorted(jd_tokens - candidate_tokens)

    # keep only meaningful tokens (length > 2, not pure numbers)
    meaningful = lambda tokens: [t for t in tokens if len(t) > 2 and not t.isdigit()]

    return {
        "fit_score": round(score * 100, 1),
        "matched_keywords": meaningful(matched),
        "gap_keywords": meaningful(gaps),
    }
