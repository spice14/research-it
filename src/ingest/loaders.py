from pathlib import Path
from typing import Iterable, Iterator, Dict

# Super-simple text loader + chunker for .txt / .md to keep things minimal.
def iter_text_files(root: Path, exts={".txt", ".md"}) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p

def chunk(text: str, size: int = 600, overlap: int = 80) -> Iterator[str]:
    if size <= 0:
        yield text
        return
    i = 0
    n = len(text)
    while i < n:
        yield text[i : i + size]
        i += max(1, size - overlap)

def load_and_chunk(root: Path, size: int = 600, overlap: int = 80) -> Iterator[Dict]:
    for path in iter_text_files(root):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for piece in chunk(txt, size=size, overlap=overlap):
            yield {
                "text": piece,
                "metadata": {"path": str(path), "filename": path.name, "ext": path.suffix.lower()}
            }
