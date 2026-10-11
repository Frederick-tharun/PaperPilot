"""Step 3: the RAG pipeline - retrieve, then generate a cited answer.

Usage:
    python -m paperpilot.rag "How is the spectral radius of an echo state network chosen?"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .embeddings import get_embedder
from .llm import get_llm
from .prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt
from .retriever import CrossEncoderReranker, Retriever
from .store import Hit, VectorStore

TAG = re.compile(r"\[S(\d+)\]")


@dataclass
class Citation:
    tag: str
    paper_id: str
    title: str
    page: int
    url: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_citations(text: str, sources: list[Hit]) -> list[Citation]:
    """Map [S1], [S2] tags in the model output back to papers and pages.

    Tags that point to a source that does not exist are ignored, so the model
    cannot invent a citation.
    """
    seen, citations = set(), []
    for number in TAG.findall(text):
        i = int(number)
        if 1 <= i <= len(sources) and i not in seen:
            seen.add(i)
            hit = sources[i - 1]
            citations.append(Citation(f"S{i}", hit.paper_id, hit.title, hit.page, hit.url))
    return citations


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: Retriever, llm):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        settings = settings or get_settings()
        store = VectorStore(settings)
        embedder = get_embedder(settings)
        reranker = CrossEncoderReranker(settings.rerank_model) if settings.rerank else None
        return cls(settings, Retriever(settings, store, embedder, reranker), get_llm(settings))

    def ask(self, question: str, top_k: int | None = None, topic: str | None = None) -> Answer:
        """Answer a question; `topic` limits retrieval to papers from one research topic."""
        question = question.strip()
        hits = self.retriever.retrieve(question, top_k, topic=topic)

        # Guardrail: if nothing relevant was found, refuse without calling the LLM.
        if not hits or hits[0].score < self.settings.min_score:
            return Answer(question, REFUSAL, refused=True, sources=hits)

        text = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
        refused = REFUSAL.lower().rstrip(".") in text.lower()
        citations = [] if refused else extract_citations(text, hits)
        return Answer(question, text, refused=refused, citations=citations, sources=hits)


def format_answer(ans: Answer) -> str:
    lines = [ans.answer, ""]
    for c in ans.citations:
        lines.append(f"  [{c.tag}] {c.title} - page {c.page}  ({c.url or 'arXiv:' + c.paper_id})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask PaperPilot a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--topic", default=None, help="only search papers downloaded for this topic")
    args = parser.parse_args()
    print(format_answer(RAGPipeline.from_settings().ask(args.question, args.top_k, args.topic)))


if __name__ == "__main__":
    main()
