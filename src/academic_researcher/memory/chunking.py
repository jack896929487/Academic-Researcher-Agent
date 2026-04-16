"""Utilities for splitting long research artifacts into retrieval-friendly chunks."""

from __future__ import annotations

import re
from typing import List


_BOUNDARY_MARKERS = ("\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", "; ", "；")


def normalize_text(text: str) -> str:
    """Collapse noisy whitespace while preserving paragraph breaks."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Split a long string into overlapping chunks.

    The splitter is intentionally lightweight:
    - prefers paragraph/sentence boundaries near the limit
    - falls back to character windows when no good boundary is found
    - adds overlap so adjacent chunks retain context continuity
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        hard_end = min(start + chunk_size, text_len)
        end = hard_end

        if hard_end < text_len:
            window = cleaned[start:hard_end]
            boundary = _find_boundary(window)
            if boundary >= chunk_size // 2:
                end = start + boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)
        while start < text_len and cleaned[start].isspace():
            start += 1

    return chunks


def _find_boundary(window: str) -> int:
    """Return the preferred split position inside the current window."""
    boundary = -1
    for marker in _BOUNDARY_MARKERS:
        boundary = max(boundary, window.rfind(marker))
    if boundary == -1:
        return len(window)
    return boundary + 1
