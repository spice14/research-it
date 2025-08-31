from pathlib import Path
import typer
from rich import print
from leann import LeannBuilder
from src.config import INDEX_PATH, DATA_DIR
from src.ingest.fetchers import fetch_url_text
from src.ingest.loaders import chunk

app = typer.Typer(add_completion=False)

@app.command()
def build_from_url(
    url: str,
    index_path: Path = typer.Option(INDEX_PATH, help="Where to write the .leann index"),
    save_copy: bool = typer.Option(True, help="Save cleaned text under data/"),
    chunk_size: int = typer.Option(600),
    chunk_overlap: int = typer.Option(80),
):
    print(f"[bold]Fetching[/] {url}")
    title, text = fetch_url_text(url)
    if not text:
        print("[red]No text extracted.[/]")
        raise typer.Exit(code=1)

    if save_copy:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        safe = (title or "page").replace("/", "_")[:120]
        out_txt = DATA_DIR / f"{safe or 'doc'}.txt"
        out_txt.write_text(text, encoding="utf-8")
        print(f"[green]Saved cleaned text[/] → {out_txt}")

    # ensure the indexes dir exists
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bold]Building LEANN index[/] → {index_path}")

    builder = LeannBuilder(backend_name="hnsw")
    for piece in chunk(text, size=chunk_size, overlap=chunk_overlap):
        builder.add_text(piece, metadata={"source": "web", "url": url, "title": title})

    builder.build_index(str(index_path))
    print(f"[green]Done.[/] Index: {index_path}")

if __name__ == "__main__":
    app()  # ✅ this was missing
