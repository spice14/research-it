from __future__ import annotations
import re
from typing import Tuple
import requests
from bs4 import BeautifulSoup
from readability import Document

def fetch_url_text(url: str, timeout: int = 30) -> Tuple[str, str]:
    """
    Returns (title, cleaned_text) from a web page.
    Uses readability to grab main content, strips boilerplate, collapses whitespace.
    """
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "leann-rag/0.1"})
    r.raise_for_status()

    html = r.text
    doc = Document(html)
    title = doc.short_title() or ""
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "html.parser")

    # remove script/style/nav elements
    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        bad.extract()

    text = soup.get_text(separator="\n")
    # collapse whitespace
    text = re.sub(r"\u00A0", " ", text)      # nbsp
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()

    return title.strip(), text
