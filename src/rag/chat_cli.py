import typer
from rich import print
from src.rag.pipeline import RAGPipeline

app = typer.Typer(add_completion=False)

@app.command()
def chat():
    rag = RAGPipeline()
    print("[bold]Local RAG chat (LEANN + Ollama). Type 'exit' to quit.[/]")
    while True:
        q = input("\n> ")
        if not q or q.strip().lower() in {"exit", "quit"}:
            break
        out = rag.ask(q)
        print(f"\n[cyan]Answer:[/] {out['answer']}")
        if out["sources"]:
            print("[dim]Sources:[/]")
            for i, s in enumerate(out["sources"], 1):
                meta = s.get("metadata", {})
                print(f"  {i}. {meta.get('filename','?')}  [{meta.get('path','')}]")

if __name__ == "__main__":
    app()
