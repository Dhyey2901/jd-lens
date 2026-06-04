import re
from transformers import pipeline

_classifier = None

CANDIDATE_LABELS = ["required skill", "nice to have", "responsibility", "company info"]

LABEL_MAP = {
    "required skill": "Required",
    "nice to have": "Nice-to-Have",
    "responsibility": "Responsibility",
    "company info": "Company Info",
}


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def classify_jd(jd_text: str, batch_size: int = 8) -> dict[str, list[str]]:
    sentences = split_sentences(jd_text)
    if not sentences:
        return {label: [] for label in LABEL_MAP.values()}

    clf = _get_classifier()
    results = clf(sentences, CANDIDATE_LABELS, batch_size=batch_size)

    buckets: dict[str, list[str]] = {label: [] for label in LABEL_MAP.values()}
    for sentence, result in zip(sentences, results):
        top_label = result["labels"][0]
        bucket = LABEL_MAP.get(top_label, "Company Info")
        buckets[bucket].append(sentence)

    return buckets
