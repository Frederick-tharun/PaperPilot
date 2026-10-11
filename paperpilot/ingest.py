"""Step 2: PDFs -> chunks -> embeddings -> vector database.

Usage:
    python -m paperpilot.ingest
    python -m paperpilot.ingest --chunk-size 500 --overlap 80 --reset
"""

from __future__ import annotations

import argparse
import time

from .chunking import chunk_paper
from .config import Settings, get_settings
from .embeddings import get_embedder
from .loaders import load_papers
from .store import VectorStore


def ingest(settings: Settings, reset: bool = False, embedder=None,
           paper_ids: set[str] | None = None) -> int:
    """Chunk, embed and store papers. `paper_ids` limits it to those papers (used by research)."""
    started = time.time()
    store = VectorStore(settings)
    if reset:
        store.reset()

    papers = load_papers(settings.papers_dir, only=paper_ids)
    if not papers:
        print(f"No PDFs found in {settings.papers_dir}. Run `python -m paperpilot.download` first.")
        return 0

    chunks = []
    for paper in papers:
        chunks.extend(chunk_paper(paper.paper_id, paper.title, paper.pages,
                                  settings.chunk_size, settings.chunk_overlap,
                                  topic=paper.topic, url=paper.url))
    print(f"{len(papers)} papers -> {len(chunks)} chunks "
          f"(size={settings.chunk_size} words, overlap={settings.chunk_overlap})")

    embedder = embedder or get_embedder(settings)
    vectors = []
    batch = 64
    for start in range(0, len(chunks), batch):
        vectors.extend(embedder.embed([c.text for c in chunks[start:start + batch]]))
        print(f"  embedded {min(start + batch, len(chunks))}/{len(chunks)}", end="\r")
    print()

    store.add(chunks, vectors)
    print(f"Collection '{store.name}' now holds {store.count()} chunks ({time.time() - started:.1f}s)")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index downloaded PDFs into the vector database.")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--reset", action="store_true", help="delete this collection first")
    args = parser.parse_args()

    overrides = {}
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        overrides["chunk_overlap"] = args.overlap
    ingest(get_settings(**overrides), reset=args.reset)


if __name__ == "__main__":
    main()
