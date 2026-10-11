"""Step 1: download papers from arXiv.

Usage:
    python -m paperpilot.download --max 50
    python -m paperpilot.download --query "abs:\"time series forecasting\"" --max 30
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import arxiv
import requests

from .config import get_settings

DEFAULT_QUERY = 'abs:"reservoir computing" OR abs:"echo state network"'


def pdf_url(result) -> str:
    """Works across arxiv library versions (v4 removed Result.download_pdf)."""
    url = getattr(result, "pdf_url", None)
    if not url:
        url = result.entry_id.replace("/abs/", "/pdf/")
    return url.replace("http://", "https://")


def save_pdf(url: str, path: Path) -> None:
    response = requests.get(url, timeout=60, headers={"User-Agent": "PaperPilot/0.1 (research project)"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    path.write_bytes(response.content)


def safe_id(entry_id: str) -> str:
    """'http://arxiv.org/abs/2101.01234v2' -> '2101.01234v2' (safe as a filename)."""
    short = entry_id.rstrip("/").split("/abs/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", short)


def download(query: str = DEFAULT_QUERY, max_results: int = 50) -> int:
    settings = get_settings()
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    meta_path = settings.papers_dir / "metadata.jsonl"

    known = set()
    if meta_path.exists():
        known = {json.loads(line)["paper_id"] for line in meta_path.read_text(encoding="utf-8").splitlines() if line}

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    new = 0
    with meta_path.open("a", encoding="utf-8") as meta:
        for result in client.results(search):
            paper_id = safe_id(result.entry_id)
            if paper_id in known:
                continue
            filename = f"{paper_id}.pdf"
            try:
                save_pdf(pdf_url(result), settings.papers_dir / filename)
            except Exception as exc:  # network hiccups should not stop the whole run
                print(f"  skipped {paper_id}: {exc}")
                continue
            time.sleep(1)  # be polite to arXiv's servers
            meta.write(json.dumps({
                "paper_id": paper_id,
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "published": result.published.date().isoformat(),
                "url": result.entry_id,
                "file": filename,
            }) + "\n")
            known.add(paper_id)
            new += 1
            print(f"  [{new}] {result.title.strip()[:90]}")

    print(f"Downloaded {new} new papers into {settings.papers_dir}")
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv papers as PDFs.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max", type=int, default=50)
    args = parser.parse_args()
    download(args.query, args.max)


if __name__ == "__main__":
    main()
