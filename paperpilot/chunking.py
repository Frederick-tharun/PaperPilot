"""Step 2b: split each page into overlapping chunks.

Design choices (good interview talking points):
* Chunks never cross a page boundary, so every chunk has one exact page
  number to cite.
* Overlap keeps a sentence that falls on a boundary visible in both chunks.
* Everything after a "References" heading is dropped: reference lists match
  almost any query and pollute retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
MIN_WORDS = 25


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    topic: str = ""   # research topic the paper was downloaded for ("" = none)
    url: str = ""     # landing page of the paper

    def metadata(self) -> dict:
        meta = {"paper_id": self.paper_id, "title": self.title, "page": self.page}
        # Chroma rejects empty/None metadata values, so only store what we have.
        if self.topic:
            meta["topic"] = self.topic
        if self.url:
            meta["url"] = self.url
        return meta


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)             # collapse whitespace and newlines
    return text.strip()


def split_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")
    step = size - overlap
    windows = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        windows.append(window)
        if start + size >= len(words):
            break
    return windows


def chunk_paper(paper_id: str, title: str, pages: list[tuple[int, str]],
                size: int = 300, overlap: int = 50, topic: str = "", url: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, raw in pages:
        match = REFERENCES_HEADING.search(raw)
        stop_after = match is not None
        if match:
            raw = raw[:match.start()]
        words = clean_text(raw).split()
        windows = split_words(words, size, overlap) if words else []
        for i, window in enumerate(windows):
            if len(window) < MIN_WORDS and (len(windows) > 1 or len(words) < MIN_WORDS):
                continue  # skip tiny fragments (headers, captions, page numbers)
            chunks.append(Chunk(
                chunk_id=f"{paper_id}-p{page_no}-c{i}",
                paper_id=paper_id,
                title=title,
                page=page_no,
                text=" ".join(window),
                topic=topic,
                url=url,
            ))
        if stop_after:
            break
    return chunks
