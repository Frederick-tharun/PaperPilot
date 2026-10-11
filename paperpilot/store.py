"""The vector database (ChromaDB, stored on disk in data/chroma)."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .chunking import Chunk
from .config import Settings


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    score: float  # cosine similarity, higher is better
    topic: str = ""
    url: str = ""


def _hit(cid: str, doc: str, meta: dict, score: float) -> Hit:
    return Hit(chunk_id=cid, paper_id=meta["paper_id"], title=meta["title"], page=int(meta["page"]),
               text=doc, score=score, topic=meta.get("topic", ""), url=meta.get("url", ""))


class VectorStore:
    def __init__(self, settings: Settings):
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.db_dir))
        self.name = settings.collection_name
        self.collection = self._open()

    def _open(self):
        # embedding_function=None: we always pass our own vectors.
        return self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self._open()

    def count(self) -> int:
        return self.collection.count()

    def add(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 256) -> None:
        for start in range(0, len(chunks), batch):
            part = chunks[start:start + batch]
            self.collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[start:start + batch],
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )

    def query(self, vector: list[float], k: int, topic: str | None = None) -> list[Hit]:
        """Nearest chunks; `topic` restricts the search to papers downloaded for that topic."""
        if self.count() == 0:
            return []
        where = {"topic": topic} if topic else None
        res = self.collection.query(query_embeddings=[vector], n_results=min(k, self.count()), where=where)
        return [_hit(cid, doc, meta, 1.0 - float(dist))
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]

    def topics(self) -> dict[str, int]:
        """Indexed research topics and how many chunks each has."""
        counts: dict[str, int] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            if meta.get("topic"):
                counts[meta["topic"]] = counts.get(meta["topic"], 0) + 1
        return dict(sorted(counts.items()))

    def sample(self, n: int, seed: int = 42) -> list[Hit]:
        """Random chunks, used to auto-generate evaluation questions."""
        import random

        data = self.collection.get(include=["documents", "metadatas"])
        rows = list(zip(data["ids"], data["documents"], data["metadatas"]))
        random.Random(seed).shuffle(rows)
        return [_hit(cid, doc, m, 1.0) for cid, doc, m in rows[:n]]
