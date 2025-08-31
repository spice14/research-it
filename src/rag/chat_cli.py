# src/rag/chat_cli.py
import os
import typer
from rich import print
from src.rag.pipeline import RAGPipeline

app = typer.Typer(add_completion=False)

@app.command()
def chat(
    index_path: str = typer.Option("./indexes/arxiv-2307-09218.leann", help="Path to .leann index"),
    model: str = typer.Option("llama3.2:3b", help="Ollama model id"),
    top_k: int = typer.Option(6),
    num_ctx: int = typer.Option(2048),
):
    rag = RAGPipeline(index_path=index_path, llm_model=model, num_ctx=num_ctx)
    print(f"[bold]Local RAG[/] (index: {index_path}, model: {model}). Type 'exit' to quit.")

    while True:
        q = input("\n> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        out = rag.ask(q, top_k=top_k)
        print(f"\n[cyan]Answer:[/] {out['answer']}")
        if out["sources"]:
            print("[dim]Sources:[/]")
            for i, s in enumerate(out["sources"], 1):
                meta = s.get("metadata", {})
                fname = meta.get("filename") or meta.get("title") or meta.get("path") or ""
                print(f"  {i}. {fname}")

if __name__ == "__main__":
    app()
