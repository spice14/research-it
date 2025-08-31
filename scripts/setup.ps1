#Requires -Version 7
param(
  [string]$PythonBin = "py",
  [string]$VenvDir   = ".venv",
  [string]$Model1    = "llama3.2:1b",
  [string]$Model2    = "llama3.2:3b",
  [string]$AppEntry  = "app_streamlit.py"
)

Write-Host "==> repo: $(Get-Location)"
Write-Host "==> using python: $PythonBin"
Write-Host "==> venv: $VenvDir"

function Test-Command($name) {
  $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# ---------- prerequisites ----------
if (-not (Test-Command $PythonBin)) {
  Write-Error "Python not found. Install Python 3.10+ and retry." ; exit 1
}

# ---------- virtualenv ----------
if (-not (Test-Path $VenvDir)) {
  Write-Host "==> creating venv"
  & $PythonBin -3 -m venv $VenvDir
}
$venvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
. $venvActivate
python -m pip install --upgrade pip wheel setuptools

# ---------- python deps ----------
Write-Host "==> installing Python dependencies"
pip install `
  leann streamlit typer rich `
  requests readability-lxml beautifulsoup4 `
  pymupdf python-docx `
  sentence-transformers

# optional: smaller embedding batches
$env:LEANN_EMBED_BATCH = $env:LEANN_EMBED_BATCH ?? "4"

# ---------- project layout ----------
New-Item -ItemType Directory -Force -Path data, indexes, uploads, src\ingest, src\index, src\rag | Out-Null
New-Item -ItemType File -Force -Path .env.example, README.md, pyproject.toml | Out-Null
New-Item -ItemType File -Force -Path src\__init__.py, src\ingest\__init__.py, src\index\__init__.py, src\rag\__init__.py | Out-Null

# ---------- ollama install ----------
if (-not (Test-Command "ollama")) {
  Write-Host "==> installing Ollama (Windows via winget)"
  if (Test-Command "winget") {
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  } else {
    Write-Warning "winget not found. Please install Ollama manually from https://ollama.com/download and re-run."
  }
} else {
  Write-Host "==> ollama already installed"
}

# ---------- start ollama daemon ----------
Write-Host "==> starting ollama daemon (if not already running)"
try {
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
} catch {}

# wait for API
Write-Host -NoNewline "==> waiting for ollama API"
for ($i=0; $i -lt 40; $i++) {
  try {
    Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 1 | Out-Null
    Write-Host " ...ready"
    break
  } catch {
    Start-Sleep -Milliseconds 500
    Write-Host -NoNewline "."
  }
}

# ---------- pull models ----------
Write-Host "==> pulling models: $Model1 (and $Model2)"
try { ollama pull $Model1 } catch {}
try { ollama pull $Model2 } catch {}

@"
----------------------------------------------------------------
✅ Setup complete (Windows).

Next steps (PowerShell):
1) Activate venv:
   .\.venv\Scripts\Activate.ps1

2) Run the app:
   streamlit run $AppEntry

Tips to avoid OOM:
- In the sidebar, set Model to `llama3.2:1b`
- Set LLM context to ~1024–1536
- Set Top-K to 3–4
----------------------------------------------------------------
"@ | Write-Host
