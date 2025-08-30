import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "indexes" / "local_docs.leann")).resolve()
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Embedding backend & model for LEANN
# LEANN supports: sentence-transformers / openai / mlx / ollama  (we'll use sbert by default)
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "facebook/contriever")  # LEANN default example
# tip: try "sentence-transformers/all-MiniLM-L6-v2" on very tight memory

# Graph/index params
GRAPH_DEGREE = int(os.getenv("GRAPH_DEGREE", "32"))
BUILD_COMPLEXITY = int(os.getenv("BUILD_COMPLEXITY", "64"))
RECOMPUTE = os.getenv("RECOMPUTE", "true").lower() != "false"  # on-demand embeddings

# Ollama local LLM
LLM_PROVIDER = "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")  # or qwen2.5:3b / phi3:mini / gemma2:2b
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "4096"))  # keep conservative for 8GB RAM
