from paperpilot.embeddings import HashEmbedder
from paperpilot.evaluate import evaluate_retrieval
from paperpilot.rag import RAGPipeline, extract_citations
from paperpilot.store import Hit, VectorStore


def test_ingest_creates_chunks(indexed):
    assert VectorStore(indexed).count() > 0


def test_retrieves_the_right_page(indexed):
    pipeline = RAGPipeline.from_settings(indexed)
    hits = pipeline.retriever.retrieve("What spectral radius keeps the echo state property?", 3)
    assert (hits[0].paper_id, hits[0].page) == ("paperA", 2)


def test_answer_has_valid_citation(indexed):
    ans = RAGPipeline.from_settings(indexed).ask("What does LoRA learn during fine-tuning?")
    assert not ans.refused
    assert ans.citations and ans.citations[0].paper_id == "paperB"


def test_refuses_when_nothing_relevant(indexed):
    pipeline = RAGPipeline.from_settings(indexed.with_(min_score=0.9))
    assert pipeline.ask("Who won the FIFA World Cup in 2014?").refused


def test_invented_citation_tags_are_ignored():
    sources = [Hit("c1", "paperA", "T", 1, "text", 0.9)]
    citations = extract_citations("Claim [S1]. Another claim [S7].", sources)
    assert [c.tag for c in citations] == ["S1"]


def test_retrieval_evaluation_runs(indexed):
    questions = [
        {"question": "Which matrix property controls reservoir memory?", "paper_id": "paperA", "page": 2},
        {"question": "How do vision transformers process image patches?", "paper_id": "paperB", "page": 1},
    ]
    results = evaluate_retrieval(indexed, [(40, 10)], [False], k=3, questions=questions, embedder=HashEmbedder())
    assert results[0].recall_at_k == 1.0
    assert (indexed.eval_dir / "results_retrieval.md").exists()
