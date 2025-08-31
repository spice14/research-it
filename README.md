
# 🧠 Research-It: Fully Local RAG for Research Papers

**Research-It** is a **fully local Retrieval-Augmented Generation (RAG) system** for research papers.  
It lets you **ingest PDFs, URLs, or directories of papers** → build vector indexes → and **chat with them** via a **Streamlit UI** backed by **Ollama LLMs**.  

Run everything **offline** on your own machine — no API keys, no cloud costs.  

---

## ✨ Features

- 📄 **Index Any Research Source**
  - Direct PDF files (local or from the web)
  - HTML pages (arXiv, NeurIPS, ResearchGate, etc.)
  - Entire folders of papers (`./data`)
- 🧩 **Chunking + Embeddings**
  - Smart text chunking with overlap
  - `sentence-transformers` for dense embeddings
  - `hnsw` index backend (FAISS-compatible, efficient)
- 🔍 **Fast Local Retrieval**
  - LEANN (`leann`) powers vector search with pruning + quantization
- 🤖 **Local LLM**
  - [Ollama](https://ollama.com) for running quantized models (`llama3.2:1b`, `3b`, etc.)
  - No GPU? Use CPU or small quantized models
- 🎨 **Streamlit UI**
  - Upload a paper or enter a URL → index instantly
  - Chat about your documents with sources and context shown
  - Sidebar lets you pick indexes, tweak chunk size, overlap, retrieval Top-K, context length, and model
- 🔒 **Privacy by Design**
  - No data leaves your machine
  - No API keys, no cloud dependencies

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **[LEANN](https://github.com/leann-ai/leann)** → embeddings, HNSW/CSR vector indexes
- **[sentence-transformers](https://www.sbert.net/)** → embeddings (default: `facebook/contriever`)
- **[Ollama](https://ollama.com)** → local quantized LLMs
- **Streamlit** → simple and responsive chat UI
- **BeautifulSoup + readability** → web text cleaning
- **PyMuPDF** → PDF parsing
- **Typer** → CLI utilities

---

## 🚀 Setup & Installation

We provide a **one-command setup script** for Linux/macOS and Windows.  
It installs Python deps, Ollama, pulls models, and sets up project folders.

### Linux / macOS

```bash
git clone https://github.com/spice14/research-it.git
cd research-it

# Run setup
bash scripts/setup.sh

# Activate virtualenv
source .venv/bin/activate

# Launch the app
streamlit run app_streamlit.py
````

### Windows (PowerShell)

```powershell
git clone https://github.com/spice14/research-it.git
cd research-it

# First time only
Set-ExecutionPolicy -Scope CurrentUser Bypass -Force

# Run setup
.\scripts\setup.ps1

# Activate venv
.\.venv\Scripts\Activate.ps1

# Launch the app
streamlit run app_streamlit.py

---

## 📚 Usage

### 1. Build an Index (CLI)

* From a PDF:

```bash
python -m src.index.build_any \
  --index-path ./indexes/my-paper.leann \
  --file ~/Downloads/some_paper.pdf
```

* From a URL:

```bash
python -m src.index.build_any \
  --index-path ./indexes/my-paper.leann \
  --url "https://arxiv.org/pdf/2307.09218v3.pdf"
```

* From a folder of papers:

```bash
python -m src.index.build_any \
  --index-path ./indexes/my-library.leann \
  --dir ./data
```

This generates:

```
./indexes/<name>.index
./indexes/<name>.meta.json
./indexes/<name>.passages.idx
./indexes/<name>.passages.jsonl
```

---

### 2. Chat via Streamlit UI

Launch the app:

```bash
streamlit run app_streamlit.py
```

Features in the sidebar:

* Pick an existing index or type a path
* Build a new index from URL or PDF upload
* Select Ollama model (`llama3.2:1b` recommended for small GPUs)
* Adjust retrieval params (chunk size, overlap, Top-K, num\_ctx)

Ask questions in the chat box — you’ll get an answer with **retrieved sources** shown below.

---

## ⚡ Performance Tips

* **Low VRAM (≤4GB GPU)?**

  * Use `llama3.2:1b` or another quantized small model
  * Reduce `num_ctx` (e.g., 1024–1536)
  * Lower `Top-K` (3–4)
* **No GPU?**

  * Ollama runs CPU-only models too, just slower
* **Avoid CUDA OOM**:
  Export before running:

  ```bash
  export LEANN_EMBED_BATCH=4
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=64,expandable_segments:True"
  ```

---

## 📦 Project Layout

```
research-it/
├── app_streamlit.py       # Streamlit UI
├── scripts/
│   ├── setup.sh           # Linux/macOS setup
│   └── setup.ps1          # Windows setup
├── src/
│   ├── index/             # build_any, build_from_url, etc.
│   ├── ingest/            # PDF, HTML, file loaders
│   └── rag/               # CLI chat pipeline
├── indexes/               # saved vector indexes
├── data/                  # cleaned paper text dumps
└── uploads/               # optional file uploads
```

---

## 🔮 Roadmap

* [ ] Support more embedding backends (bge-m3, InstructorXL, etc.)
* [ ] Add incremental indexing / updates
* [ ] Hybrid retrieval (BM25 + dense)
* [ ] Docker image for one-shot deploy
* [ ] Offline model bundling for air-gapped use

---

## ❤️ Credits

* [LEANN](https://github.com/leann-ai/leann) — blazing fast vector store
* [Ollama](https://ollama.com) — local quantized LLMs
* [SentenceTransformers](https://www.sbert.net/) — embeddings
* [Streamlit](https://streamlit.io) — UI
* Inspired by the need to read and **truly understand papers locally** ✍️
