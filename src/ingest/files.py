# src/ingest/files.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Dict
from .pdf import fetch_pdf_text

def iter_files(root: Path, exts={".txt", ".md", ".pdf", ".docx"}) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def load_local_file(p: Path) -> Dict[str, str]:
    """
    Return {"title","text"} for a single file path.
    """
    ext = p.suffix.lower()
    if ext in {".txt", ".md"}:
        return {"title": p.stem, "text": p.read_text(encoding="utf-8", errors="ignore")}
    if ext == ".pdf":
        title, text = fetch_pdf_text(str(p))
        return {"title": title or p.stem, "text": text}
    if ext == ".docx":
        try:
            import docx  # python-docx
        except Exception as e:
            raise RuntimeError("Install python-docx to read .docx") from e
        doc = docx.Document(str(p))
        text = "\n".join(para.text for para in doc.paragraphs)
        return {"title": p.stem, "text": text}
    raise ValueError(f"Unsupported file type: {p}")
