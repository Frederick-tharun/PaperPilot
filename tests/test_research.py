"""Topic research tests. Every HTTP call is mocked, so they run offline."""

import json
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paperpilot import api, research as research_mod, sources as sources_mod
from paperpilot.embeddings import HashEmbedder
from paperpilot.llm import ExtractiveLLM
from paperpilot.loaders import load_papers
from paperpilot.rag import RAGPipeline
from paperpilot.research import (deduplicate, expand_queries, keyword_query, parse_queries, rank,
                                 research)
from paperpilot.sources import (ArxivSource, OpenAlexSource, PaperCandidate, SemanticScholarSource,
                                rebuild_abstract)
from paperpilot.store import VectorStore
from tests.conftest import make_pdf

TOPIC = "graph neural networks for drug discovery"
GNN_TEXT = ("Graph neural networks represent molecules as graphs of atoms and bonds. Message passing "
            "layers learn molecular properties for drug discovery, such as binding affinity and toxicity. "
            "These models outperform fingerprint baselines on several molecular property prediction "
            "benchmarks and help screen drug candidates before expensive laboratory experiments.")


# ---------------------------------------------------------------- fakes
class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


S2_JSON = {"data": [
    {"title": "Molecular Graph Neural Networks for Drug Discovery", "abstract": "GNNs for molecules and drugs.",
     "year": 2021, "authors": [{"name": "A. Author"}], "citationCount": 120,
     "externalIds": {"DOI": "10.1000/GNN.1", "ArXiv": "2101.00001"},
     "openAccessPdf": {"url": "https://arxiv.org/pdf/2101.00001"}, "url": "https://s2/1"},
    {"title": "A Paywalled Drug Paper", "abstract": "Closed access.", "year": 2020, "authors": [],
     "citationCount": 5, "externalIds": {"DOI": "10.1000/closed"}, "openAccessPdf": None, "url": "https://s2/2"},
]}

OPENALEX_JSON = {"results": [
    {"display_name": "Molecular graph neural networks for drug discovery", "doi": "https://doi.org/10.1000/gnn.1",
     "publication_year": 2021, "cited_by_count": 130,
     "abstract_inverted_index": {"Graph": [0], "networks": [2], "neural": [1], "for": [3], "drugs": [4]},
     "authorships": [{"author": {"display_name": "A. Author"}}],
     "best_oa_location": {"pdf_url": "https://journal.example/gnn.pdf"}, "locations": []},
    {"display_name": "Message Passing for Molecular Property Prediction", "doi": "https://doi.org/10.1000/mp.2",
     "publication_year": 2019, "cited_by_count": 800,
     "abstract_inverted_index": {"Message": [0], "passing": [1], "molecules": [2]},
     "authorships": [], "best_oa_location": None,
     "locations": [{"landing_page_url": "https://arxiv.org/abs/1904.00002v2", "pdf_url": None}],
     "primary_location": {"pdf_url": "https://arxiv.org/pdf/1904.00002v2"}},
]}


def arxiv_result(arxiv_id, title, summary):
    return SimpleNamespace(entry_id=f"http://arxiv.org/abs/{arxiv_id}", title=title, summary=summary,
                           published=datetime(2022, 5, 1), authors=[SimpleNamespace(name="B. Author")],
                           doi=None, pdf_url=f"http://arxiv.org/pdf/{arxiv_id}")


ARXIV_RESULTS = [
    arxiv_result("2101.00001v3", "Molecular Graph Neural Networks for Drug Discovery", GNN_TEXT),
    arxiv_result("2203.00003v1", "Equivariant Graph Networks for Protein Ligand Binding", GNN_TEXT),
    arxiv_result("2204.00004v1", "A Survey of Reinforcement Learning for Robot Walking",
                 "Robots learn to walk with reinforcement learning in simulation."),
]


@pytest.fixture()
def fake_http(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers))
        if "semanticscholar" in url:
            return FakeResponse(S2_JSON)
        if "openalex" in url:
            return FakeResponse(OPENALEX_JSON)
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(sources_mod.requests, "get", fake_get)
    monkeypatch.setattr(ArxivSource, "_results", lambda self, q, limit: ARXIV_RESULTS)
    return calls


@pytest.fixture()
def fake_pdfs(monkeypatch):
    """save_pdf writes a small real PDF, except for one URL that 'fails'."""
    def fake_save(url, path):
        if "journal.example" in url:
            raise ValueError("response is not a PDF")
        make_pdf(path, [GNN_TEXT, GNN_TEXT.replace("Graph", "Molecular graph")])
    monkeypatch.setattr(research_mod, "save_pdf", fake_save)


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, system, user):
        return self.text


# ---------------------------------------------------------------- 1. query expansion
def test_parse_queries_cleans_and_validates():
    raw = 'Sure! {"queries": ["GNN drug discovery", "molecular graph \\"learning\\"", "gnn drug discovery", ' \
          '"title:message passing AND molecules", 42]}'
    assert parse_queries(raw) == ["GNN drug discovery", "molecular graph learning", "message passing molecules"]
    assert parse_queries("no json here") is None
    assert parse_queries('{"queries": ["only one"]}') is None      # too few to be useful


def test_parse_queries_drops_reordered_duplicates():
    raw = '{"queries": ["reservoir computing neural networks", "neural networks reservoir computing", ' \
          '"echo state networks"]}'
    assert parse_queries(raw) == ["reservoir computing neural networks", "echo state networks"]


def test_expand_queries_uses_llm_and_falls_back():
    queries, used = expand_queries(TOPIC, FakeLLM('{"queries": ["a b", "c d", "e f", "g h"]}'))
    assert used and queries == ["a b", "c d", "e f", "g h"]

    fallback = [keyword_query(TOPIC)]
    assert expand_queries(TOPIC, ExtractiveLLM()) == (fallback, False)
    assert expand_queries(TOPIC, FakeLLM("I cannot help")) == (fallback, False)
    assert keyword_query(TOPIC) == "graph neural networks drug discovery"


# ---------------------------------------------------------------- 2. sources
def test_semantic_scholar_parses_and_backs_off_on_429(settings, monkeypatch):
    responses = [FakeResponse(status=429, headers={"Retry-After": "2"}), FakeResponse(S2_JSON)]
    sent_headers = []
    monkeypatch.setattr(sources_mod.requests, "get",
                        lambda url, params=None, headers=None, timeout=None: sent_headers.append(headers)
                        or responses.pop(0))
    slept = []
    found = SemanticScholarSource(settings.with_(s2_api_key="secret"), sleep=slept.append).search("gnn", 10)
    assert slept == [2.0]
    assert sent_headers[0]["x-api-key"] == "secret"
    assert found[0].doi == "10.1000/gnn.1" and found[0].arxiv_id == "2101.00001"
    assert found[0].citation_count == 120 and found[1].pdf_url == ""


def test_semantic_scholar_spaces_consecutive_calls(settings, monkeypatch):
    monkeypatch.setattr(sources_mod.requests, "get", lambda *a, **k: FakeResponse(S2_JSON))
    slept = []
    source = SemanticScholarSource(settings, sleep=slept.append)
    source.search("first", 10)
    assert slept == []                       # the first call goes out immediately
    source.search("second", 10)
    assert len(slept) == 1 and 0 < slept[0] <= source.MIN_INTERVAL


def test_openalex_parses_abstract_pdf_and_arxiv_id(settings, fake_http):
    found = OpenAlexSource(settings.with_(contact_email="me@example.com")).search("gnn", 10)
    assert found[0].abstract == "Graph neural networks for drugs"
    assert found[0].pdf_url == "https://journal.example/gnn.pdf"
    assert found[1].arxiv_id == "1904.00002v2" and found[1].pdf_url.endswith("1904.00002v2")
    assert fake_http[0][1]["mailto"] == "me@example.com"
    assert "me@example.com" in fake_http[0][2]["User-Agent"]
    assert rebuild_abstract(None) == ""


# ---------------------------------------------------------------- 3. dedup
def test_deduplicates_across_sources_and_keeps_pdf_version():
    cands = [
        PaperCandidate("Molecular GNNs", "openalex", doi="https://doi.org/10.1000/GNN.1", citation_count=130),
        PaperCandidate("Molecular GNNs", "semanticscholar", doi="10.1000/gnn.1", arxiv_id="2101.00001",
                       citation_count=120),
        PaperCandidate("Molecular GNNs (v3)", "arxiv", arxiv_id="2101.00001v3", abstract="longer abstract text"),
        PaperCandidate("Message Passing for Molecular Property Prediction", "openalex"),
        PaperCandidate("Message passing for molecular property prediction!", "arxiv", arxiv_id="1904.00002"),
    ]
    cands[0].pdf_url = ""  # the OpenAlex record has no PDF
    unique, removed = deduplicate(cands)
    assert removed == 3 and len(unique) == 2
    gnn = unique[0]
    assert gnn.pdf_url and gnn.arxiv_id.startswith("2101.00001")
    assert gnn.citation_count == 130 and gnn.abstract == "longer abstract text"
    assert set(gnn.sources) == {"openalex", "semanticscholar", "arxiv"}


# ---------------------------------------------------------------- 4. ranking
def test_rank_orders_by_relevance_with_citation_bonus(settings):
    cands = [
        PaperCandidate("Robots learning to walk", "arxiv", abstract="reinforcement learning locomotion"),
        PaperCandidate("Graph neural networks for drug discovery", "arxiv", abstract=GNN_TEXT, citation_count=10),
        PaperCandidate("Graph neural networks for drug discovery", "openalex", abstract=GNN_TEXT, citation_count=500),
    ]
    ranked = rank(TOPIC, cands, HashEmbedder(), settings)
    assert ranked[0].citation_count == 500           # same text, more citations wins
    assert ranked[-1].title == "Robots learning to walk"
    assert ranked[0].relevance_score > ranked[-1].relevance_score


# ---------------------------------------------------------------- 5-8. full run
def test_full_research_run(settings, fake_http, fake_pdfs):
    settings = settings.with_(research_dir=settings.papers_dir.parent / "research", download_delay=0)
    result = research(TOPIC, max_papers=3, settings=settings, sources=[ArxivSource(settings),
                      SemanticScholarSource(settings), OpenAlexSource(settings)],
                      llm=ExtractiveLLM(), embedder=HashEmbedder(), progress=lambda *a: None)

    assert result.per_source == {"arxiv": 3, "semanticscholar": 2, "openalex": 2}
    assert result.duplicates_removed == 2            # the GNN paper came from all three sources
    assert result.without_pdf == 1                   # the paywalled paper is skipped
    assert len(result.selected) == 3
    assert all("Robot Walking" not in c.title for c in result.selected)  # least relevant is cut
    gnn = next(c for c in result.selected if c.title.startswith("Molecular Graph"))
    assert set(gnn.sources) == {"arxiv", "semanticscholar", "openalex"} and gnn.citation_count == 130
    assert result.chunks_indexed > 0

    rows = [json.loads(line) for line in (settings.papers_dir / "metadata.jsonl").read_text().splitlines()]
    new = [r for r in rows if r.get("topic") == TOPIC]
    assert {"source", "topic", "abstract", "doi", "relevance_score"} <= set(new[0])
    assert len(new) == len(result.downloaded)

    report = (settings.research_dir / "graph-neural-networks-for-drug-discovery.md").read_text()
    assert "Duplicates removed: 2" in report and "| arxiv | 3 |" in report

    # Running again never re-downloads a paper that is already in the library.
    again = research(TOPIC, max_papers=3, settings=settings, sources=[ArxivSource(settings)],
                     llm=ExtractiveLLM(), embedder=HashEmbedder(), progress=lambda *a: None)
    assert again.already_known == 2
    assert set(again.downloaded).isdisjoint(result.downloaded)


def test_topic_filter_in_retrieval(settings, fake_http, fake_pdfs):
    settings = settings.with_(research_dir=settings.papers_dir.parent / "research", download_delay=0)
    from paperpilot.ingest import ingest
    ingest(settings, embedder=HashEmbedder())        # the two old papers, no topic
    research(TOPIC, max_papers=2, settings=settings, sources=[ArxivSource(settings)],
             llm=ExtractiveLLM(), embedder=HashEmbedder(), progress=lambda *a: None)

    pipeline = RAGPipeline.from_settings(settings)
    assert list(pipeline.retriever.store.topics()) == [TOPIC]
    question = "What does the spectral radius control?"
    unfiltered = pipeline.retriever.retrieve(question, 3)
    filtered = pipeline.retriever.retrieve(question, 3, topic=TOPIC)
    assert unfiltered[0].paper_id == "paperA"         # the ESN paper wins without a filter
    assert filtered and all(h.topic == TOPIC for h in filtered)
    ans = pipeline.ask("How do graph neural networks represent molecules?", topic=TOPIC)
    assert ans.citations and ans.citations[0].url.startswith("https://arxiv.org/abs/")


def test_old_metadata_lines_still_load(settings):
    meta = settings.papers_dir / "metadata.jsonl"
    meta.write_text(meta.read_text() + json.dumps({"paper_id": "x", "title": "Not on disk"}) + "\n")
    papers = {p.paper_id: p for p in load_papers(settings.papers_dir)}
    assert papers["paperA"].title == "Echo State Networks Explained" and papers["paperA"].topic == ""
    assert set(papers) == {"paperA", "paperB"}


# ---------------------------------------------------------------- API
def test_topics_and_research_endpoints(indexed, monkeypatch):
    pipeline = RAGPipeline.from_settings(indexed)
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)

    def fake_research(topic, max_papers, settings, llm=None, embedder=None, progress=None):
        progress("search", "Searching arxiv", None, None)
        progress("download", "Downloading 1/1", 1, 1)
        return research_mod.ResearchResult(topic=topic, queries=[topic], used_llm=False, candidates=1)

    monkeypatch.setattr(research_mod, "research", fake_research)
    client = TestClient(api.app)

    assert client.get("/topics").json() == []        # the test papers have no topic

    started = client.post("/research", json={"topic": "spiking neural networks", "max_papers": 5})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/research/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["result"]["topic"] == "spiking neural networks"
    assert "[download] Downloading 1/1" in job["log"]
    assert client.get("/research/doesnotexist").status_code == 404
    assert client.post("/research", json={"topic": "x"}).status_code == 422


def test_ask_accepts_topic(indexed, monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", lambda: RAGPipeline.from_settings(indexed))
    body = TestClient(api.app).post("/ask", json={"question": "What is LoRA?", "topic": "no such topic"}).json()
    assert body["refused"] and body["sources"] == []
