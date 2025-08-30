# app_streamlit.py
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from leann import LeannChat, LeannBuilder

# ---------------- Defaults ----------------
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = str(APP_ROOT / "indexes" / "arxiv-2307-09218.index")
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TOP_K = 6
DEFAULT_NUM_CTX = 2048
SYSTEM = (
    "You are a helpful assistant. Use the retrieved CONTEXT faithfully. "
    "If the answer is not in the context, say you don't know."
)

# ---------------- Helpers ----------------
def resolve(path: str) -> Path:
    try:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    except Exception:
        return Path(path)

def list_indexes(root: Path) -> List[str]:
    return sorted(glob.glob(str(root / "**/*.index"), recursive=True))

def expected_sidecars_for(index_path: Path) -> Tuple[Path, Path, Path]:
    """LEANN (in your env) looks for .index-based sidecars: .index.meta.json/.index.passages.*"""
    base = index_path.name  # e.g., "foo.index"
    parent = index_path.parent
    meta     = parent / f"{base}.meta.json"
    pass_idx = parent / f"{base}.passages.idx"
    pass_jsl = parent / f"{base}.passages.jsonl"
    return meta, pass_idx, pass_jsl

def candidate_sources_for(index_path: Path) -> Dict[str, List[Path]]:
    """Possible existing sidecars we can copy from (plain and .leann.* variants)."""
    parent = index_path.parent
    stem = index_path.stem  # "foo" for "foo.index"
    return {
        "meta": [
            parent / f"{stem}.meta.json",
            parent / f"{stem}.leann.meta.json",
        ],
        "pass_idx": [
            parent / f"{stem}.passages.idx",
            parent / f"{stem}.leann.passages.idx",
        ],
        "pass_jsl": [
            parent / f"{stem}.passages.jsonl",
            parent / f"{stem}.leann.passages.jsonl",
        ],
    }

def ensure_sidecars(index_path: Path) -> Dict[str, bool]:
    """Make sure .index.* sidecars exist; if not, copy from best available variant."""
    expected_meta, expected_idx, expected_jsl = expected_sidecars_for(index_path)
    status = {
        "expected_meta_exists": expected_meta.exists(),
        "expected_idx_exists": expected_idx.exists(),
        "expected_jsl_exists": expected_jsl.exists(),
    }
    if all(status.values()):
        return status

    cands = candidate_sources_for(index_path)

    def create_if_missing(expected: Path, sources: List[Path]):
        if expected.exists():
            return
        for src in sources:
            if src.exists():
                shutil.copy2(src, expected)
                break

    create_if_missing(expected_meta, cands["meta"])
    create_if_missing(expected_idx,  cands["pass_idx"])
    create_if_missing(expected_jsl,  cands["pass_jsl"])

    status["expected_meta_exists"] = expected_meta.exists()
    status["expected_idx_exists"]  = expected_idx.exists()
    status["expected_jsl_exists"]  = expected_jsl.exists()
    return status

def normalize_index_prefix(user_path: str) -> Path:
    """Accept .leann/.index/bare; return a base prefix Path (no suffix)."""
    p = Path(user_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix in {".leann", ".index"}:
        return p.with_suffix("")
    return p

@st.cache_resource(show_spinner=False)
def get_chat(resolved_index_path: str, model: str, num_ctx: int) -> LeannChat:
    ix = Path(resolved_index_path)
    if ix.suffix != ".index":
        raise ValueError(f"Expected a .index file, got: {ix}")
    status = ensure_sidecars(ix)
    if not all(status.values()):
        missing = [k for k, ok in status.items() if not ok]
        raise FileNotFoundError(
            f"Missing required LEANN sidecar(s) for {ix.name}: {', '.join(missing)}."
        )
    return LeannChat(
        resolved_index_path,
        llm_config={"type": "ollama", "model": model, "num_ctx": num_ctx},
        system_prompt=SYSTEM,
    )

def run_rag(chat: LeannChat, question: str, top_k: int) -> Dict[str, Any]:
    out = chat.ask(question, top_k=top_k)
    if isinstance(out, dict):
        return {"answer": out.get("text", ""), "sources": out.get("sources", [])}
    return {"answer": str(out), "sources": []}

def source_title(meta: Dict) -> str:
    return meta.get("filename") or meta.get("title") or meta.get("path") or meta.get("url") or "source"

def trim_text(t: str, limit: int = 800) -> str:
    if not isinstance(t, str):
        return ""
    return (t if len(t) <= limit else t[:limit] + "…").strip()

# ---------- URL indexing (built-in) ----------
def fetch_url_text(url: str) -> tuple[str, str]:
    """Return (title, cleaned_text) using readability + bs4."""
    import re, requests
    from readability import Document
    from bs4 import BeautifulSoup

    r = requests.get(url, timeout=30, headers={"User-Agent": "leann-rag/0.1"})
    r.raise_for_status()
    html = r.text
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "html.parser")
    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        bad.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return title, text

def chunk_text(text: str, size: int = 600, overlap: int = 80):
    if size <= 0:
        yield text
        return
    n = len(text)
    i = 0
    while i < n:
        yield text[i:i+size]
        i += max(1, size - overlap)

def build_index_from_url(url: str, index_path_input: str, chunk_size: int, chunk_overlap: int, save_copy=True) -> Path:
    """
    Build a LEANN index (graph-only, recompute=True) from URL content.
    Returns the absolute path to the .index file to use with LeannChat.
    """
    # 1) fetch
    title, text = fetch_url_text(url)
    if not text:
        raise RuntimeError("No text extracted from the URL.")

    # 2) normalize index base prefix (no suffix)
    index_prefix = normalize_index_prefix(index_path_input)
    # ensure parent dir
    index_prefix.parent.mkdir(parents=True, exist_ok=True)

    # optional: save cleaned text for audit
    if save_copy:
        data_dir = APP_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe = (title or "page").replace("/", "_")[:120] or "doc"
        (data_dir / f"{safe}.txt").write_text(text, encoding="utf-8")

    # 3) build compact graph index with on-demand embeddings
    builder = LeannBuilder(backend_name="hnsw", recompute=True)
    for piece in chunk_text(text, size=chunk_size, overlap=chunk_overlap):
        builder.add_text(piece, metadata={"source": "web", "url": url, "title": title})

    builder.build_index(str(index_prefix))  # writes <prefix>.index + sidecars (variant names possible)

    # 4) the .index path we want to open
    index_file = index_prefix.with_suffix(".index")

    # 5) reconcile names if builder used a different sidecar suffix variant
    ensure_sidecars(index_file)

    return index_file.resolve()

# ---------------- UI ----------------
st.set_page_config(page_title="LEANN RAG (Local)", page_icon="🧠", layout="wide")
st.title("🧠 LEANN RAG — Local (Ollama)")

with st.sidebar:
    st.header("Settings")
    # --- Existing index picker ---
    indexes_root = APP_ROOT / "indexes"
    found = list_indexes(indexes_root) if indexes_root.exists() else []
    picked = st.selectbox(
        "Pick an existing LEANN index (.index)",
        options=["(type path manually)"] + found,
        index=0,
    )
    raw_index_path = DEFAULT_INDEX if picked == "(type path manually)" else picked
    user_index_path = st.text_input("Index (.index) path", value=raw_index_path)

    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    top_k = st.slider("Top-K (retrieval)", 1, 20, value=DEFAULT_TOP_K, step=1)
    num_ctx = st.slider("LLM context (num_ctx)", 512, 8192, value=DEFAULT_NUM_CTX, step=256)

    # --- New: Build index from URL ---
    st.subheader("Index a URL")
    url_to_index = st.text_input("URL to index", value="https://arxiv.org/html/2307.09218v3")
    new_index_out = st.text_input("Output index path (prefix/.leann/.index)", value=str(APP_ROOT / "indexes" / "arxiv-2307-09218.leann"))
    new_chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=700, step=50)
    new_chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=10)

    build_btn = st.button("Build / Rebuild Index from URL", type="primary")
    cold_start = st.button("Re-init pipeline")

    # Diagnostics
    st.divider()
    st.caption("Diagnostics")
    resolved = resolve(user_index_path)
    exists = resolved.exists()
    size = resolved.stat().st_size if exists else 0
    st.write(f"**CWD:** `{os.getcwd()}`")
    st.write(f"**App root:** `{APP_ROOT}`")
    st.write("**Resolved index path:**")
    st.code(str(resolved))
    st.write(f"**Exists:** {'✅' if exists else '❌'}")
    st.write(f"**Size:** {size} bytes")

    if exists and resolved.suffix == ".index":
        em, ei, ej = expected_sidecars_for(resolved)
        st.write("**Expected sidecars:**")
        st.write(f"- {em.name}: {'✅' if em.exists() else '❌'}")
        st.write(f"- {ei.name}: {'✅' if ei.exists() else '❌'}")
        st.write(f"- {ej.name}: {'✅' if ej.exists() else '❌'}")

    if indexes_root.exists():
        st.write("**Discovered under ./indexes:**")
        if found:
            for p in found[:20]:
                st.code(p)
            if len(found) > 20:
                st.caption(f"... and {len(found)-20} more")
        else:
            st.caption("(no .index files found)")

# Perform build (blocking) and update selection to the new index
if build_btn:
    with st.spinner("Fetching & indexing… (embeddings computed on-demand, no big GPU spike)"):
        try:
            new_ix = build_index_from_url(
                url=url_to_index.strip(),
                index_path_input=new_index_out.strip(),
                chunk_size=int(new_chunk_size),
                chunk_overlap=int(new_chunk_overlap),
                save_copy=True,
            )
            st.success(f"Index built: {new_ix}")
            # auto-select the new index & clear cache so LeannChat re-inits
            user_index_path = str(new_ix)
            resolved = new_ix
            if "messages" in st.session_state:
                st.session_state["messages"].clear()
            get_chat.clear()
        except Exception as e:
            st.error("Failed to build index from URL:")
            st.exception(e)

# (Re)initialize pipeline
chat = None
init_error: Exception | None = None
if resolved.exists():
    try:
        if cold_start:
            get_chat.clear()
        chat = get_chat(str(resolved), model, num_ctx)
    except Exception as e:
        init_error = e
else:
    st.sidebar.warning("Index not found at the resolved path above. Use a .index file path or build one.")

# show init errors
if init_error:
    st.error("Failed to initialize LeannChat. See details below.")
    st.exception(init_error)

# keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# show history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(m["sources"], 1):
                    meta = s.get("metadata", {}) if isinstance(s, dict) else {}
                    title = source_title(meta)
                    st.markdown(f"**{i}. {title}**")
                    if "url" in meta:
                        st.markdown(f"[link]({meta['url']})")
                    if "path" in meta:
                        st.code(meta["path"])
                    snippet = s.get("text", "")
                    if snippet:
                        st.write(trim_text(snippet))
                    st.divider()

# input & inference
user_q = st.chat_input("Ask a question about your indexed documents…")
if user_q:
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if chat is None:
        with st.chat_message("assistant"):
            if init_error:
                st.error("Pipeline failed to initialize:")
                st.exception(init_error)
            else:
                st.error("Index not found / invalid. Pick a .index file or build one in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    out = run_rag(chat, user_q, top_k=DEFAULT_TOP_K if top_k is None else top_k)
                    answer = out["answer"] or "_(No answer returned)_"
                    sources = out.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for i, s in enumerate(sources, 1):
                                meta = s.get("metadata", {}) if isinstance(s, dict) else {}
                                title = source_title(meta)
                                st.markdown(f"**{i}. {title}**")
                                if "url" in meta:
                                    st.markdown(f"[link]({meta['url']})")
                                if "path" in meta:
                                    st.code(meta["path"])
                                snippet = s.get("text", "")
                                if snippet:
                                    st.write(trim_text(snippet))
                                st.divider()
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error("Error during RAG:")
                    st.exception(e)
