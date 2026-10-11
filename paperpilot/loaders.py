"""Step 2a: read PDFs into (page number, text) pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Paper:
    paper_id: str
    title: str
    pages: list[tuple[int, str]]  # (1-based page number, text)
    topic: str = ""
    url: str = ""


def load_metadata(papers_dir: Path) -> dict[str, dict]:
    path = papers_dir / "metadata.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["paper_id"]: row for row in rows}


def load_pdf(path: Path, title: str | None = None) -> Paper:
    with pymupdf.open(path) as doc:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
        pdf_title = (doc.metadata or {}).get("title") or ""
    return Paper(paper_id=path.stem, title=title or pdf_title.strip() or path.stem, pages=pages)


def load_papers(papers_dir: Path, only: set[str] | None = None) -> list[Paper]:
    """Load every PDF (or only the given paper ids) with title/topic/url from metadata.jsonl.

    Older metadata lines without "topic" or "url" still load; those fields are just empty.
    """
    meta = load_metadata(papers_dir)
    papers = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        if only is not None and pdf.stem not in only:
            continue
        row = meta.get(pdf.stem, {})
        try:
            paper = load_pdf(pdf, row.get("title"))
        except Exception as exc:  # a broken PDF should not stop ingestion
            print(f"  could not read {pdf.name}: {exc}")
            continue
        paper.topic = row.get("topic") or ""
        paper.url = row.get("url") or ""
        papers.append(paper)
    return papers
