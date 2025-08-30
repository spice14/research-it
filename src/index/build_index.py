from pathlib import Path
import typer
from rich import print
from leann import LeannBuilder  # documented in README/demo
from src.config import (
    DATA_DIR, INDEX_PATH, EMBEDDING_MODE, EMBEDDING_MODEL,
    GRAPH_DEGREE, BUILD_COMPLEXITY, RECOMPUTE
)
from src.ingest.loaders import load_and_chunk

app = typer.Typer(add_completion=False)

@app.command()
def build(
    data_dir: Path = typer.Option(DATA_DIR, help="Folder with .txt/.md to index"),
    index_path: Path = typer.Option(INDEX_PATH, help="Where to write the .leann index"),
    chunk_size: int = typer.Option(600, help="Chars per chunk"),
    chunk_overlap: int = typer.Option(80, help="Overlap in chars"),
):
    print(f"[bold]Building LEANN index[/] → {index_path}")
    builder = LeannBuilder(
        index_path=str(index_path),
        backend="hnsw",
        embedding_mode=EMBEDDING_MODE,
        embedding_model=EMBEDDING_MODEL,
        graph_degree=GRAPH_DEGREE,
        build_complexity=BUILD_COMPLEXITY,
        recompute=RECOMPUTE,
        compact=True,
    )

    count = 0
    for item in load_and_chunk(data_dir, size=chunk_size, overlap=chunk_overlap):
        builder.add_text(item["text"], metadata=item["metadata"])  # README shows add_text & metadata
        count += 1

    builder.build()
    print(f"[green]Done.[/] Chunks indexed: {count}\nIndex: {index_path}")

if __name__ == "__main__":
    app()
