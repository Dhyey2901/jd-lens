"""Utility helpers — PDF text extraction."""
from __future__ import annotations

import io
import re


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract and clean plain text from a PDF file's bytes."""
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    raw = "\n\n".join(pages)

    # Collapse runs of blank lines and strip trailing whitespace per line
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    return cleaned.strip()
