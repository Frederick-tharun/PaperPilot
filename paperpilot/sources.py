"""Paper search sources for topic research.

Every source turns a search query into a list of `PaperCandidate`s through
the same small interface, so adding another source (e.g. PubMed) means
writing one class:

    class MySource:
        name = "mysource"
        def search(self, query: str, limit: int) -> list[PaperCandidate]: ...

Sources:
* ArxivSource            - the arXiv API (reuses helpers from download.py)
* SemanticScholarSource  - Semantic Scholar Graph API (free; key optional)
* OpenAlexSource         - OpenAlex works search (free, open-access PDFs)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import requests

from .config import Settings
from .download import pdf_url as arxiv_pdf_url
from .download import safe_id

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)
VERSION = re.compile(r"v\d+$")


@dataclass
class PaperCandidate:
    """One search result, normalised across sources."""

    title: str
    source: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    arxiv_id: str = ""            # may include a version suffix, e.g. 2101.01234v2
    pdf_url: str = ""             # open-access PDF link ("" = none known)
    url: str = ""                 # landing page shown to the user
    citation_count: int = 0
    sources: list[str] = field(default_factory=list)  # every source that returned it
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        self.doi = normalise_doi(self.doi)
        if not self.sources:
            self.sources = [self.source]
        if self.arxiv_id and not self.pdf_url:
            self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}"
        if self.arxiv_id and not self.url:
            self.url = f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def arxiv_base(self) -> str:
        """arXiv id without version, used for deduplication."""
        return VERSION.sub("", self.arxiv_id)

    @property
    def paper_id(self) -> str:
        """A file-name-safe id: the arXiv id if known, else the DOI, else a title hash."""
        if self.arxiv_id:
            return safe_id(self.arxiv_id)
        if self.doi:
            return "doi-" + re.sub(r"[^A-Za-z0-9._-]", "_", self.doi)[:80]
        import hashlib

        return "t-" + hashlib.sha1(normalise_title(self.title).encode()).hexdigest()[:12]


class PaperSource(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[PaperCandidate]: ...


# ---------------------------------------------------------------- helpers
def normalise_doi(doi: str | None) -> str:
    doi = (doi or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)


def normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def arxiv_id_from_url(url: str | None) -> str:
    match = ARXIV_ID.search(url or "")
    return match.group(1).removesuffix(".pdf") if match else ""


def user_agent(settings: Settings) -> str:
    contact = f"; mailto:{settings.contact_email}" if settings.contact_email else ""
    return f"PaperPilot/0.2 (research project{contact})"


def get_json(url: str, params: dict, headers: dict, retries: int = 4,
             sleep: Callable[[float], None] = time.sleep, timeout: int = 30) -> dict:
    """GET a JSON API, backing off on 429 (rate limit) and 5xx errors."""
    for attempt in range(retries + 1):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After", "")
            sleep(float(retry_after) if retry_after.isdigit() else 2.0 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- arXiv
class ArxivSource:
    name = "arxiv"

    def __init__(self, settings: Settings):
        self.settings = settings

    def _results(self, query: str, limit: int):
        import arxiv

        client = arxiv.Client(page_size=min(limit, 100), delay_seconds=3, num_retries=3)
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
        return client.results(search)

    def search(self, query: str, limit: int) -> list[PaperCandidate]:
        out = []
        for r in self._results(query, limit):
            arxiv_id = r.entry_id.rstrip("/").split("/abs/")[-1]
            published = getattr(r, "published", None)
            out.append(PaperCandidate(
                title=" ".join(r.title.split()),
                source=self.name,
                abstract=" ".join((r.summary or "").split()),
                year=published.year if published else None,
                authors=[a.name for a in r.authors],
                doi=getattr(r, "doi", None) or "",
                arxiv_id=arxiv_id,
                pdf_url=arxiv_pdf_url(r),
                url=r.entry_id.replace("http://", "https://"),
            ))
        return out


# ---------------------------------------------------------------- Semantic Scholar
class SemanticScholarSource:
    name = "semanticscholar"
    URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,citationCount,externalIds,openAccessPdf,url"
    MIN_INTERVAL = 1.0  # S2 allows about one request per second, with or without a key

    def __init__(self, settings: Settings, sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.sleep = sleep
        self._last_call: float | None = None

    def search(self, query: str, limit: int) -> list[PaperCandidate]:
        headers = {"User-Agent": user_agent(self.settings)}
        if self.settings.s2_api_key:
            headers["x-api-key"] = self.settings.s2_api_key
        if self._last_call is not None:
            wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        try:
            data = get_json(self.URL, {"query": query, "limit": min(limit, 100), "fields": self.FIELDS},
                            headers, sleep=self.sleep)
        finally:
            self._last_call = time.monotonic()
        out = []
        for p in data.get("data") or []:
            if not p.get("title"):
                continue
            ext = p.get("externalIds") or {}
            oa = p.get("openAccessPdf") or {}
            out.append(PaperCandidate(
                title=" ".join(p["title"].split()),
                source=self.name,
                abstract=p.get("abstract") or "",
                year=p.get("year"),
                authors=[a.get("name", "") for a in p.get("authors") or []],
                doi=ext.get("DOI") or "",
                arxiv_id=ext.get("ArXiv") or "",
                pdf_url=oa.get("url") or "",
                url=p.get("url") or "",
                citation_count=int(p.get("citationCount") or 0),
            ))
        return out


# ---------------------------------------------------------------- OpenAlex
def rebuild_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; turn it back into text."""
    if not inverted:
        return ""
    positions = [(pos, word) for word, places in inverted.items() for pos in places]
    return " ".join(word for _, word in sorted(positions))


class OpenAlexSource:
    name = "openalex"
    URL = "https://api.openalex.org/works"

    def __init__(self, settings: Settings, sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.sleep = sleep

    @staticmethod
    def _pdf(work: dict) -> str:
        locations = [work.get("best_oa_location"), work.get("primary_location"), *(work.get("locations") or [])]
        for loc in locations:
            if loc and loc.get("pdf_url"):
                return loc["pdf_url"]
        return ""

    def search(self, query: str, limit: int) -> list[PaperCandidate]:
        params = {"search": query, "per-page": min(limit, 200), "filter": "open_access.is_oa:true"}
        if self.settings.contact_email:
            params["mailto"] = self.settings.contact_email
        data = get_json(self.URL, params, {"User-Agent": user_agent(self.settings)}, sleep=self.sleep)
        out = []
        for w in data.get("results") or []:
            title = w.get("display_name") or w.get("title")
            if not title:
                continue
            pdf = self._pdf(w)
            landing = [(loc or {}).get("landing_page_url") for loc in w.get("locations") or []]
            arxiv_id = next((a for a in map(arxiv_id_from_url, [pdf, *landing]) if a), "")
            out.append(PaperCandidate(
                title=" ".join(title.split()),
                source=self.name,
                abstract=rebuild_abstract(w.get("abstract_inverted_index")),
                year=w.get("publication_year"),
                authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
                doi=w.get("doi") or "",
                arxiv_id=arxiv_id,
                pdf_url=pdf,
                url=w.get("doi") or w.get("id") or "",
                citation_count=int(w.get("cited_by_count") or 0),
            ))
        return out


SOURCES = {"arxiv": ArxivSource, "semanticscholar": SemanticScholarSource, "openalex": OpenAlexSource}


def get_sources(settings: Settings, names: list[str] | None = None) -> list[PaperSource]:
    names = names or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown source(s): {unknown}. Choose from {list(SOURCES)}")
    return [SOURCES[n](settings) for n in names]
