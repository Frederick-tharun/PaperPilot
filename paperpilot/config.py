"""Central configuration.

Every setting can be changed with an environment variable (or a line in the
.env file), so you never have to edit code to try a different model,
chunk size or LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Where things live on disk
    papers_dir: Path
    db_dir: Path
    eval_dir: Path
    research_dir: Path

    # Embeddings: "sentence-transformers" (real) or "hash" (offline, for tests)
    embed_backend: str
    embed_model: str

    # Chunking (sizes are in words)
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    top_k: int
    fetch_k: int
    rerank: bool
    rerank_model: str
    min_score: float

    # Answer generation: "anthropic", "ollama" or "extractive" (offline)
    llm_backend: str
    anthropic_model: str
    ollama_url: str
    ollama_model: str
    max_tokens: int

    # Topic research (multi-source paper search)
    s2_api_key: str            # optional Semantic Scholar key (higher rate limit)
    contact_email: str         # optional, sent to OpenAlex/arXiv as a polite contact
    research_per_query: int    # results requested per query and source
    rank_sim_weight: float     # weight of topic/abstract embedding similarity
    rank_cite_weight: float    # weight of log(1 + citation count)
    download_delay: float      # seconds between PDF downloads

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            papers_dir=Path(os.getenv("PP_PAPERS_DIR", "data/papers")),
            db_dir=Path(os.getenv("PP_DB_DIR", "data/chroma")),
            eval_dir=Path(os.getenv("PP_EVAL_DIR", "eval")),
            research_dir=Path(os.getenv("PP_RESEARCH_DIR", "data/research")),
            embed_backend=os.getenv("PP_EMBED_BACKEND", "sentence-transformers"),
            embed_model=os.getenv("PP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("PP_CHUNK_SIZE", "300")),
            chunk_overlap=int(os.getenv("PP_CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("PP_TOP_K", "5")),
            fetch_k=int(os.getenv("PP_FETCH_K", "20")),
            rerank=_bool(os.getenv("PP_RERANK", "false")),
            rerank_model=os.getenv("PP_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            min_score=float(os.getenv("PP_MIN_SCORE", "0.0")),
            llm_backend=os.getenv("PP_LLM", "anthropic"),
            anthropic_model=os.getenv("PP_ANTHROPIC_MODEL", "claude-haiku-5-5"),
            ollama_url=os.getenv("PP_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("PP_OLLAMA_MODEL", "llama3.1"),
            max_tokens=int(os.getenv("PP_MAX_TOKENS", "600")),
            s2_api_key=os.getenv("PP_S2_API_KEY", ""),
            contact_email=os.getenv("PP_CONTACT_EMAIL", ""),
            research_per_query=int(os.getenv("PP_RESEARCH_PER_QUERY", "25")),
            rank_sim_weight=float(os.getenv("PP_RANK_SIM_WEIGHT", "1.0")),
            rank_cite_weight=float(os.getenv("PP_RANK_CITE_WEIGHT", "0.03")),
            download_delay=float(os.getenv("PP_DOWNLOAD_DELAY", "1.0")),
        )

    def with_(self, **changes) -> "Settings":
        """Return a copy with some settings changed (used by the evaluation)."""
        return replace(self, **changes)

    @property
    def collection_name(self) -> str:
        """One vector collection per (embedding model, chunk size, overlap).

        This lets the evaluation compare chunk sizes without the indexes
        overwriting each other.
        """
        if self.embed_backend == "hash":
            model = "hash"
        else:
            model = re.sub(r"[^a-z0-9]+", "-", self.embed_model.split("/")[-1].lower()).strip("-")[:40]
        return f"cs{self.chunk_size}-ov{self.chunk_overlap}-{model}"[:63]


def get_settings(**overrides) -> Settings:
    settings = Settings.from_env()
    return settings.with_(**overrides) if overrides else settings
