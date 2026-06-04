"""Zero-shot JD sentence classification using facebook/bart-large-mnli."""
from __future__ import annotations

import re
from typing import Any

from config import CANDIDATE_LABELS, LABEL_DISPLAY, ZERO_SHOT_MODEL

_pipeline: Any = None


def get_pipeline() -> Any:
    """Lazy-load and cache the zero-shot classification pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    return _pipeline


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def classify_jd(
    jd_text: str,
    pipeline: Any = None,
    batch_size: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Classify each JD sentence into a bucket, returning text + confidence."""
    clf = pipeline or get_pipeline()
    sentences = split_sentences(jd_text)
    if not sentences:
        return {label: [] for label in LABEL_DISPLAY.values()}

    results = clf(sentences, CANDIDATE_LABELS, batch_size=batch_size)

    buckets: dict[str, list[dict[str, Any]]] = {
        label: [] for label in LABEL_DISPLAY.values()
    }
    for sentence, result in zip(sentences, results):
        top_label = result["labels"][0]
        confidence = round(result["scores"][0] * 100, 1)
        bucket = LABEL_DISPLAY.get(top_label, "Company Info")
        buckets[bucket].append({"text": sentence, "confidence": confidence})

    return buckets
