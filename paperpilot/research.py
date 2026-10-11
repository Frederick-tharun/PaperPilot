"""Topic research: type a topic, get an indexed library of papers about it.

Pipeline:
  1. expand   the LLM turns the topic into 4-6 search queries (keyword fallback)
  2. search   every query on every source (arXiv, Semantic Scholar, OpenAlex)
  3. dedup    merge the same paper found twice (DOI -> arXiv id -> title)
  4. rank     embedding similarity(topic, title + abstract) + small citation bonus
  5. download open-access PDFs of the top N (skips papers already in the library)
  6. ingest   chunk + embed only the new papers, tagged with the topic
  7. report   data/research/<topic-slug>.md

Usage:
    python -m paperpilot.research "graph neural networks for drug discovery" --max 40
    python -m paperpilot.research "spiking neural networks" --sources arxiv,openalex --no-ingest
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings, get_settings
from .download import save_pdf
from .embeddings import get_embedder
from .ingest import ingest
from .llm import ExtractiveLLM, get_llm
from .loaders import load_metadata
from .prompts import QUERY_EXPANSION_PROMPT
from .sources import PaperCandidate, PaperSource, get_sources, normalise_doi, normalise_title

# progress(stage, message, done, total) - used by the CLI, the API job runner and Streamlit
Progress = Callable[[str, str, "int | None", "int | None"], None]

STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "using", "via", "by", "at",
    "from", "into", "about", "towards", "toward", "based", "how", "what", "is", "are", "its",
}
MIN_QUERIES, MAX_QUERIES = 2, 6


def _print_progress(stage: str, message: str, done: int | None = None, total: int | None = None) -> None:
    print(f"[{stage}] {message}")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


# ---------------------------------------------------------------- 1. query expansion
def keyword_query(topic: str) -> str:
    """Fallback query: the topic without filler words."""
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", topic) if w.lower() not in STOPWORDS]
    return " ".join(words) or topic.strip()


def parse_queries(text: str) -> list[str] | None:
    """Read {"queries": [...]} (or a bare JSON list) from LLM output and clean it.

    Returns None when the output is unusable, so the caller can fall back.
    """
    match = re.search(r"\{.*\}|\[.*\]", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    items = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    queries, seen = [], set()
    for item in items:
        if not isinstance(item, str):
            continue
        q = re.sub(r"\b(?:ti|abs|all|au|title|abstract|author)\s*:", " ", item, flags=re.I)  # field prefixes
        q = re.sub(r"[\"'():]|\b(AND|OR|NOT)\b", " ", q)
        q = " ".join(q.split())
        words = frozenset(q.lower().split())  # word order does not change search results
        if not 1 <= len(q.split()) <= 10 or words in seen:
            continue
        seen.add(words)
        queries.append(q)
    return queries[:MAX_QUERIES] if len(queries) >= MIN_QUERIES else None


def expand_queries(topic: str, llm=None, n: int = 5) -> tuple[list[str], bool]:
    """Return (queries, used_llm). Falls back to one keyword query if the LLM can't help."""
    if llm is not None and not isinstance(llm, ExtractiveLLM):
        try:
            raw = llm.generate("You output only valid JSON.", QUERY_EXPANSION_PROMPT.format(n=n, topic=topic))
            queries = parse_queries(raw)
            if queries:
                return queries, True
        except Exception as exc:  # network/API problems should not stop the research run
            print(f"  query expansion failed ({exc}); using keyword fallback")
    return [keyword_query(topic)], False


# ---------------------------------------------------------------- 3. deduplication
def _keys(c: PaperCandidate) -> list[str]:
    keys = []
    if c.doi:
        keys.append("doi:" + normalise_doi(c.doi))
    if c.arxiv_base:
        keys.append("arxiv:" + c.arxiv_base.lower())
    title = normalise_title(c.title)
    if len(title) > 10:
        keys.append("title:" + title)
    return keys


def _merge(a: PaperCandidate, b: PaperCandidate) -> PaperCandidate:
    """Combine two records of the same paper, preferring the one with a PDF link."""
    keep, other = (a, b) if a.pdf_url or not b.pdf_url else (b, a)
    keep.abstract = max(keep.abstract, other.abstract, key=len)
    keep.doi = keep.doi or other.doi
    keep.arxiv_id = keep.arxiv_id or other.arxiv_id
    keep.url = keep.url or other.url
    keep.year = keep.year or other.year
    keep.authors = keep.authors or other.authors
    keep.citation_count = max(keep.citation_count, other.citation_count)
    keep.sources = sorted(set(keep.sources) | set(other.sources))
    return keep


def deduplicate(candidates: list[PaperCandidate]) -> tuple[list[PaperCandidate], int]:
    """Merge candidates that share a DOI, arXiv id or normalised title.

    Returns (unique papers in first-seen order, number of duplicates removed).
    """
    unique: list[PaperCandidate] = []
    index: dict[str, int] = {}
    removed = 0
    for cand in candidates:
        keys = _keys(cand)
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            index.update({k: len(unique) for k in keys})
            unique.append(cand)
            continue
        removed += 1
        unique[hit] = _merge(unique[hit], cand)
        index.update({k: hit for k in _keys(unique[hit])})
    return unique, removed


# ---------------------------------------------------------------- 4. ranking
def rank(topic: str, candidates: list[PaperCandidate], embedder, settings: Settings) -> list[PaperCandidate]:
    """Score = w_sim * cos(topic, title+abstract) + w_cite * log(1 + citations), best first."""
    if not candidates:
        return []
    texts = [f"{c.title}. {c.abstract}"[:2000] for c in candidates]
    topic_vec, *doc_vecs = embedder.embed([topic, *texts])
    for cand, vec in zip(candidates, doc_vecs):
        similarity = sum(x * y for x, y in zip(topic_vec, vec))  # vectors are unit length
        cand.relevance_score = (settings.rank_sim_weight * similarity
                                + settings.rank_cite_weight * math.log1p(max(cand.citation_count, 0)))
    return sorted(candidates, key=lambda c: c.relevance_score, reverse=True)


# ---------------------------------------------------------------- 5. download
def known_papers(papers_dir: Path) -> dict[str, set[str]]:
    """Ids, DOIs and titles already in metadata.jsonl, so nothing is downloaded twice."""
    meta = load_metadata(papers_dir)
    known = {"ids": set(), "arxiv": set(), "doi": set(), "title": set()}
    for pid, row in meta.items():
        known["ids"].add(pid)
        known["arxiv"].add(re.sub(r"v\d+$", "", pid).lower())
        if row.get("doi"):
            known["doi"].add(normalise_doi(row["doi"]))
        if row.get("title"):
            known["title"].add(normalise_title(row["title"]))
    return known


def is_known(c: PaperCandidate, known: dict[str, set[str]]) -> bool:
    return (c.paper_id in known["ids"]
            or (c.arxiv_base and c.arxiv_base.lower() in known["arxiv"])
            or (c.doi and c.doi in known["doi"])
            or normalise_title(c.title) in known["title"])


def metadata_row(c: PaperCandidate, topic: str) -> dict:
    return {
        "paper_id": c.paper_id,
        "title": c.title,
        "authors": c.authors,
        "published": str(c.year or ""),
        "url": c.url or c.pdf_url,
        "file": f"{c.paper_id}.pdf",
        "source": ",".join(c.sources),
        "topic": topic,
        "abstract": c.abstract,
        "doi": c.doi,
        "relevance_score": round(c.relevance_score, 4),
    }


# ---------------------------------------------------------------- the whole run
@dataclass
class ResearchResult:
    topic: str
    queries: list[str]
    used_llm: bool
    per_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    candidates: int = 0
    duplicates_removed: int = 0
    without_pdf: int = 0
    already_known: int = 0
    selected: list[PaperCandidate] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    chunks_indexed: int = 0
    report_path: str = ""

    def summary(self) -> dict:
        return {
            "topic": self.topic, "queries": self.queries, "used_llm": self.used_llm,
            "per_source": self.per_source, "errors": self.errors, "candidates": self.candidates,
            "duplicates_removed": self.duplicates_removed, "without_pdf": self.without_pdf,
            "already_known": self.already_known, "downloaded": len(self.downloaded),
            "failed": len(self.failed), "chunks_indexed": self.chunks_indexed,
            "report_path": self.report_path,
            "papers": [{"paper_id": c.paper_id, "title": c.title, "url": c.url,
                        "score": round(c.relevance_score, 3), "downloaded": c.paper_id in self.downloaded}
                       for c in self.selected],
        }


def research(topic: str, max_papers: int = 40, settings: Settings | None = None,
             sources: list[PaperSource] | None = None, llm=None, embedder=None,
             progress: Progress = _print_progress, do_ingest: bool = True,
             sleep: Callable[[float], None] = time.sleep) -> ResearchResult:
    """Run the full topic-research pipeline and return what happened."""
    topic = " ".join(topic.split())
    if len(topic) < 3:
        raise ValueError("Please give a topic of at least 3 characters.")
    settings = settings or get_settings()
    sources = sources if sources is not None else get_sources(settings)
    if llm is None:
        try:
            llm = get_llm(settings)
        except Exception as exc:  # e.g. missing API key: research still works with keyword queries
            print(f"  LLM unavailable ({exc}); using keyword fallback for queries")

    # 1. expand
    progress("expand", f"Expanding '{topic}' into search queries", None, None)
    queries, used_llm = expand_queries(topic, llm)
    result = ResearchResult(topic=topic, queries=queries, used_llm=used_llm)
    for q in queries:
        progress("expand", f"  query: {q}", None, None)

    # 2. search
    found: list[PaperCandidate] = []
    for source in sources:
        progress("search", f"Searching {source.name}", None, None)
        count = 0
        for q in queries:
            try:
                hits = source.search(q, settings.research_per_query)
            except Exception as exc:  # one failing source/query must not stop the run
                result.errors.append(f"{source.name} / '{q}': {exc}")
                progress("search", f"  {source.name} failed for '{q}': {exc}", None, None)
                continue
            found.extend(hits)
            count += len(hits)
        result.per_source[source.name] = count
        progress("search", f"  {source.name}: {count} results", None, None)
    result.candidates = len(found)

    # 3. dedup
    unique, result.duplicates_removed = deduplicate(found)
    with_pdf = [c for c in unique if c.pdf_url]
    result.without_pdf = len(unique) - len(with_pdf)
    known = known_papers(settings.papers_dir)
    fresh = [c for c in with_pdf if not is_known(c, known)]
    result.already_known = len(with_pdf) - len(fresh)
    progress("rank", f"{len(unique)} unique papers ({result.duplicates_removed} duplicates removed, "
                     f"{result.without_pdf} without open-access PDF, {result.already_known} already in library)",
             None, None)

    # 4. rank
    embedder = embedder or get_embedder(settings)
    result.selected = rank(topic, fresh, embedder, settings)[:max_papers]
    progress("rank", f"Selected the top {len(result.selected)} papers", None, None)

    # 5. download
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    meta_path = settings.papers_dir / "metadata.jsonl"
    total = len(result.selected)
    with meta_path.open("a", encoding="utf-8") as meta:
        for i, cand in enumerate(result.selected, start=1):
            progress("download", f"Downloading {i}/{total}: {cand.title[:80]}", i, total)
            try:
                save_pdf(cand.pdf_url, settings.papers_dir / f"{cand.paper_id}.pdf")
            except Exception as exc:  # paywalls, HTML instead of PDF, timeouts ...
                result.failed.append(cand.paper_id)
                progress("download", f"  skipped: {exc}", i, total)
                continue
            meta.write(json.dumps(metadata_row(cand, topic), ensure_ascii=False) + "\n")
            meta.flush()
            result.downloaded.append(cand.paper_id)
            if i < total:
                sleep(settings.download_delay)  # be polite to the servers

    # 6. ingest (only the new papers)
    if do_ingest and result.downloaded:
        progress("index", f"Indexing {len(result.downloaded)} new papers", None, None)
        result.chunks_indexed = ingest(settings, embedder=embedder, paper_ids=set(result.downloaded))

    # 7. report
    result.report_path = str(write_report(result, settings))
    progress("done", f"Downloaded {len(result.downloaded)} papers, indexed {result.chunks_indexed} chunks. "
                     f"Report: {result.report_path}", total, total)
    return result


def write_report(result: ResearchResult, settings: Settings) -> Path:
    lines = [
        f"# Topic research: {result.topic}", "",
        f"*Queries {'expanded by the LLM' if result.used_llm else 'from keyword fallback'}:*", "",
        *[f"- `{q}`" for q in result.queries], "",
        "## Search", "",
        "| Source | Results |", "|---|---|",
        *[f"| {name} | {n} |" for name, n in result.per_source.items()], "",
        f"- Candidates found: {result.candidates}",
        f"- Duplicates removed: {result.duplicates_removed}",
        f"- Without open-access PDF (skipped): {result.without_pdf}",
        f"- Already in the library (skipped): {result.already_known}",
        f"- Downloaded: {len(result.downloaded)} · failed downloads: {len(result.failed)}",
        f"- Chunks indexed: {result.chunks_indexed}", "",
    ]
    if result.errors:
        lines += ["## Errors", "", *[f"- {e}" for e in result.errors], ""]
    lines += ["## Papers", "", "| # | Score | Title | Year | Sources | Status |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(result.selected, start=1):
        status = "downloaded" if c.paper_id in result.downloaded else "failed"
        title = c.title.replace("|", "/")
        link = f"[{title}]({c.url or c.pdf_url})"
        lines.append(f"| {i} | {c.relevance_score:.3f} | {link} | {c.year or ''} | {', '.join(c.sources)} | {status} |")
    path = settings.research_dir / f"{slugify(result.topic)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find, download and index papers on any topic.")
    parser.add_argument("topic", help='e.g. "graph neural networks for drug discovery"')
    parser.add_argument("--max", type=int, default=40, help="number of papers to download")
    parser.add_argument("--sources", default="arxiv,semanticscholar,openalex",
                        help="comma-separated: arxiv, semanticscholar, openalex")
    parser.add_argument("--per-query", type=int, default=None, help="results per query and source")
    parser.add_argument("--no-ingest", action="store_true", help="download only, do not index")
    args = parser.parse_args()

    settings = get_settings(**({"research_per_query": args.per_query} if args.per_query else {}))
    result = research(args.topic, args.max, settings,
                      sources=get_sources(settings, [s.strip() for s in args.sources.split(",") if s.strip()]),
                      do_ingest=not args.no_ingest)
    print()
    print(f"Topic:      {result.topic}")
    print(f"Queries:    {', '.join(result.queries)}")
    print(f"Found:      {result.candidates} results " + str(result.per_source))
    print(f"Unique:     {result.candidates - result.duplicates_removed} "
          f"({result.duplicates_removed} duplicates, {result.without_pdf} without PDF, "
          f"{result.already_known} already in library)")
    print(f"Downloaded: {len(result.downloaded)} (failed: {len(result.failed)})")
    print(f"Indexed:    {result.chunks_indexed} chunks")
    print(f"Report:     {result.report_path}")
    if result.downloaded and not args.no_ingest:
        print(f'\nTry: python -m paperpilot.rag "your question" --topic "{result.topic}"')


if __name__ == "__main__":
    main()
