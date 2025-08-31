#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL1="${MODEL1:-llama3.2:1b}"      # tiny & safe on 4GB
MODEL2="${MODEL2:-llama3.2:3b}"      # optional, a bit larger
APP_ENTRY="${APP_ENTRY:-app_streamlit.py}"

echo "==> repo: $(pwd)"
echo "==> using python: ${PYTHON_BIN}"
echo "==> venv: ${VENV_DIR}"

# ---------- prerequisites ----------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found. Install curl and retry."
  exit 1
fi

# ---------- virtualenv ----------
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> creating venv"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---------- python deps ----------
echo "==> installing Python dependencies"
# Note: torch is pulled by sentence-transformers where available; avoid pinning here to keep it simple.
pip install \
  leann streamlit typer rich \
  requests readability-lxml beautifulsoup4 \
  pymupdf python-docx \
  sentence-transformers

# optional: keep embedding batches small (less VRAM)
export LEANN_EMBED_BATCH="${LEANN_EMBED_BATCH:-4}"

# ---------- project layout (safe if exists) ----------
mkdir -p data indexes uploads src/ingest src/index src/rag
touch .env.example README.md pyproject.toml \
      src/__init__.py src/ingest/__init__.py src/index/__init__.py src/rag/__init__.py

# ---------- ollama install ----------
if ! command -v ollama >/dev/null 2>&1; then
  echo "==> installing Ollama (Linux/macOS)"
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "==> ollama already installed"
fi

# ---------- start ollama daemon ----------
echo "==> starting ollama daemon (if not already running)"
# try to start in background; ignore error if already running
nohup ollama serve >/dev/null 2>&1 & disown || true

# wait for API to respond (up to ~20s)
echo -n "==> waiting for ollama API"
for i in {1..40}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo " ...ready"
    break
  fi
  echo -n "."
  sleep 0.5
done

# ---------- pull at least one small model ----------
echo "==> pulling models: ${MODEL1} (and ${MODEL2})"
ollama pull "${MODEL1}" || true
ollama pull "${MODEL2}" || true

cat <<'EONEXT'
----------------------------------------------------------------
✅ Setup complete.

Next steps:
1) Activate venv:
   source .venv/bin/activate

2) Run the app:
   streamlit run research-it.py

Tips to avoid OOM:
- In the sidebar, set Model to `llama3.2:1b` (or your custom quantized alias)
- Set LLM context to ~1024–1536 (or your custom value with the slider)
- Set Top-K to 3–4 (or your custom value with the slider)
----------------------------------------------------------------
EONEXT
