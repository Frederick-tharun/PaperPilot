"""Find the chunks most relevant to a question, optionally re-ranked.

Two-stage retrieval:
1. Vector search returns `fetch_k` candidates quickly (bi-encoder).
2. A cross-encoder reads each (question, chunk) pair together and re-scores
   them; we keep the best `top_k`. Slower, but usually more precise.
"""

from __future__ import annotations

from .config import Settings
from .embeddings import Embedder
from .store import Hit, VectorStore


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(question, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [h for h, _ in ranked]


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, reranker=None):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, question: str, k: int | None = None, topic: str | None = None) -> list[Hit]:
        k = k or self.settings.top_k
        fetch = max(self.settings.fetch_k, k) if self.reranker else k
        vector = self.embedder.embed([question])[0]
        hits = self.store.query(vector, fetch, topic=topic)
        if self.reranker:
            hits = self.reranker.rerank(question, hits)
        return hits[:k]
