# src/ingest/html.py
from __future__ import annotations
import re, requests
from typing import Tuple
from readability import Document
from bs4 import BeautifulSoup

UA = "leann-rag/0.1 (+local)"

def fetch_url_text(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Return (title, cleaned_text) from a webpage using readability + bs4.
    Works broadly (arXiv HTML, ResearchGate pages, NeurIPS pages, blog posts, etc.).
    """
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    html = r.text

    # Let readability isolate main content
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    content_html = doc.summary(html_partial=True)

    soup = BeautifulSoup(content_html, "html.parser")
    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        bad.extract()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\u00A0", " ", text)              # nbsp
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces
    text = re.sub(r"\n{2,}", "\n\n", text).strip()   # collapse blank lines
    return title or "", text
