# app_streamlit.py
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from leann import LeannChat

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
    """
    LEANN (in your env) expects sidecars named with the .index base:
      <name>.index.meta.json
      <name>.index.passages.idx
      <name>.index.passages.jsonl
    """
    base = index_path.name  # e.g., arxiv-2307-09218.index
    parent = index_path.parent
    meta     = parent / f"{base}.meta.json"
    pass_idx = parent / f"{base}.passages.idx"
    pass_jsl = parent / f"{base}.passages.jsonl"
    return meta, pass_idx, pass_jsl

def candidate_sources_for(index_path: Path) -> Dict[str, List[Path]]:
    """
    Return lists of candidate source files we might copy FROM
    to create the expected .index.* sidecars.
    We check both the plain and `.leann.` variants.
    """
    parent = index_path.parent
    stem   = index_path.stem  # 'arxiv-2307-09218' (since name is '.index')
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
    """
    Make sure the expected .index.* sidecars exist.
    If missing, copy from first available candidate source.
    Return dict indicating existence of each expected file after reconciliation.
    """
    expected_meta, expected_idx, expected_jsl = expected_sidecars_for(index_path)
    status = {
        "expected_meta_exists": expected_meta.exists(),
        "expected_idx_exists": expected_idx.exists(),
        "expected_jsl_exists": expected_jsl.exists(),
    }

    # If already good, nothing to do
    if all(status.values()):
        return status

    candidates = candidate_sources_for(index_path)

    def create_if_missing(expected: Path, sources: List[Path]):
        if expected.exists():
            return
        for src in sources:
            if src.exists():
                shutil.copy2(src, expected)
                break

    # Try to create missing files by copying from any viable source variant
    create_if_missing(expected_meta, candidates["meta"])
    create_if_missing(expected_idx,  candidates["pass_idx"])
    create_if_missing(expected_jsl,  candidates["pass_jsl"])

    # Update status after attempts
    status["expected_meta_exists"] = expected_meta.exists()
    status["expected_idx_exists"]  = expected_idx.exists()
    status["expected_jsl_exists"]  = expected_jsl.exists()
    return status

@st.cache_resource(show_spinner=False)
def get_chat(resolved_index_path: str, model: str, num_ctx: int) -> LeannChat:
    """
    Cache LeannChat keyed by absolute index path + model + ctx.
    Reconcile sidecars to the .index.* naming LEANN expects in your environment.
    """
    ix = Path(resolved_index_path)
    if ix.suffix != ".index":
        raise ValueError(f"Expected a .index file, got: {ix}")

    # Ensure the required sidecars exist with the exact names LEANN will look for.
    status = ensure_sidecars(ix)
    if not all(status.values()):
        missing = [k for k, ok in status.items() if not ok]
        raise FileNotFoundError(
            f"Missing required LEANN sidecar(s) for {ix.name}: {', '.join(missing)}. "
            "Check that either the plain or `.leann.` variants exist so the app can copy them."
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

# ---------------- UI ----------------
st.set_page_config(page_title="LEANN RAG (Local)", page_icon="🧠", layout="wide")
st.title("🧠 LEANN RAG — Local (Ollama)")

with st.sidebar:
    st.header("Settings")

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

# (Re)initialize pipeline
chat = None
init_error: Exception | None = None
if exists:
    try:
        if cold_start:
            get_chat.clear()
        chat = get_chat(str(resolved), model, num_ctx)
    except Exception as e:
        init_error = e
else:
    st.sidebar.warning("Index not found at the resolved path above. Use a .index file path.")

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
                st.error("Index not found / invalid. Update the path in the sidebar (must be a .index file).")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    out = run_rag(chat, user_q, top_k=top_k)
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
