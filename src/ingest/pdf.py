# src/ingest/pdf.py
from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path
import io, re, requests
import fitz  # PyMuPDF

UA = "leann-rag/0.1 (+local)"

def pdf_bytes_from(url_or_path: str, timeout: int = 60) -> bytes:
    if re.match(r"^https?://", url_or_path, re.I):
        r = requests.get(url_or_path, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.content
    else:
        return Path(url_or_path).read_bytes()

def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    text = "\n".join(text_parts)
    # light cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fetch_pdf_text(url_or_path: str) -> Tuple[str, str]:
    """
    Return (title, text). Title falls back to filename or URL basename.
    """
    data = pdf_bytes_from(url_or_path)
    text = extract_pdf_text(data)
    base = url_or_path.rstrip("/").split("/")[-1]
    title = base.replace(".pdf", "").replace("_", " ")
    return title, text
