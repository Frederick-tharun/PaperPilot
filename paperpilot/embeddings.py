"""Turn text into vectors.

* SentenceTransformerEmbedder: the real one (runs locally, free).
* HashEmbedder: a tiny bag-of-words stand-in with no downloads. It is only
  for tests and offline demos; its retrieval quality is much worse.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings

TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "hash":
        return HashEmbedder()
    if settings.embed_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embed_model)
    raise ValueError(f"Unknown PP_EMBED_BACKEND: {settings.embed_backend!r}")
