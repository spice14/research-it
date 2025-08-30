# src/rag/pipeline.py
from typing import Dict, List
from leann import LeannChat

SYSTEM = (
    "You are a helpful assistant. Use the retrieved CONTEXT faithfully. "
    "If the answer is not in the context, say you don't know."
)

class RAGPipeline:
    def __init__(self, index_path: str, llm_model: str = "llama3.2:3b", num_ctx: int = 2048):
        self.chat = LeannChat(
            index_path,
            llm_config={"type": "ollama", "model": llm_model, "num_ctx": num_ctx},
            system_prompt=SYSTEM,
        )

    def ask(self, question: str, top_k: int = 6) -> Dict:
        """
        Returns:
          {
            "answer": "...",
            "sources": [{"text": "...", "metadata": {...}}, ...]
          }
        """
        out = self.chat.ask(question, top_k=top_k)
        if isinstance(out, dict):
            return {"answer": out.get("text", ""), "sources": out.get("sources", [])}
        return {"answer": str(out), "sources": []}
