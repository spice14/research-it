from typing import List, Dict
from leann import LeannChat
from src.config import INDEX_PATH, LLM_PROVIDER, LLM_MODEL, TOP_K, MAX_CONTEXT

SYSTEM = """You are a helpful assistant. Use the retrieved CONTEXT faithfully.
If the answer is not in context, say you don't know and suggest where to look in the files."""

class RAGPipeline:
    def __init__(self, index_path=str(INDEX_PATH)):
        self.chat = LeannChat(
            index_path,
            llm_config={"type": LLM_PROVIDER, "model": LLM_MODEL, "num_ctx": MAX_CONTEXT},
            system_prompt=SYSTEM,
        )

    def ask(self, question: str, top_k: int = TOP_K) -> Dict:
        # LeannChat internally retrieves (using the same index) and calls the LLM.
        # CLI analog: `leann ask <index> --llm ollama --model llama3.2:3b`
        answer = self.chat.ask(question, top_k=top_k)
        return {
            "answer": answer.get("text") if isinstance(answer, dict) else str(answer),
            "sources": answer.get("sources", []) if isinstance(answer, dict) else [],
        }
