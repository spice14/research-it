# src/ingest/chunking.py
from __future__ import annotations
from typing import Iterator

def chunk_text(text: str, size: int = 600, overlap: int = 80) -> Iterator[str]:
    if size <= 0:
        yield text
        return
    i = 0
    n = len(text)
    step = max(1, size - overlap)
    while i < n:
        yield text[i:i+size]
        i += step
