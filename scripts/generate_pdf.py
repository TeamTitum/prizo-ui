"""Utility to generate a simple PDF from plain text using PyMuPDF (fitz).

Provides a single function `generate_pdf(text, pdf_path='quote.pdf')` which
creates a multi-page PDF if the text is long. This file lives under `scripts/`
so it can be imported from the Streamlit app.
"""
from __future__ import annotations

import math
import fitz


def _chunk_text(text: str, max_chars: int = 3000):
    """Yield successive chunks of `text` of up to `max_chars` characters.

    This is a simple heuristic to avoid overflowing a single PDF page.
    It prefers splitting at newline boundaries when possible.  
    """
    if not text:
        return [""]

    parts = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # Try to split at last newline within the window
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start:
                end = nl + 1
        parts.append(text[start:end])
        start = end
    return parts


def generate_pdf(text: str, pdf_path: str = "quote.pdf") -> None:
    """Generate a PDF file from `text` and save it to `pdf_path`.

    Raises exceptions from PyMuPDF upward  so callers can handle/display
    errors.
    """
    # Split into chunks to place across pages
    chunks = _chunk_text(text or "")

    doc = fitz.open()

    for chunk in chunks:
        page = doc.new_page()
        # 1 inch margins on 612x792 (default points) page -> rect uses points
        rect = fitz.Rect(72, 72, 540, 720)
        # Basic insert; PyMuPDF will wrap the text inside the textbox
        page.insert_textbox(rect, chunk, fontsize=11, fontname="helv")

    # Save to path (overwrites existing file)
    doc.save(pdf_path)
    doc.close()
