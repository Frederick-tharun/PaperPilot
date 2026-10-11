"""Step 4: measure how well PaperPilot works.

Three commands:

  generate   Create evaluation questions from random chunks with the LLM.
             The chunk a question came from is the "gold" (paper, page).
             -> eval/questions.jsonl   (read through it and delete bad ones!)

  retrieval  For several chunking settings (and with/without re-ranking),
             check whether the gold page is in the top-k results.
             Metrics: Recall@1, Recall@k, MRR.  -> eval/results_retrieval.md

  answers    Run the full pipeline and check the answers.
             Metrics: citation accuracy (cites the gold page), answer rate,
             and refusal rate on out-of-scope questions.  -> eval/results_answers.md

Usage:
    python -m paperpilot.evaluate generate --n 30
    python -m paperpilot.evaluate retrieval --configs 200:40,300:50,500:80 --rerank both
    python -m paperpilot.evaluate answers
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import get_embedder
from .ingest import ingest
from .llm import get_llm
from .prompts import QUESTION_GEN_PROMPT
from .rag import RAGPipeline
from .retriever import CrossEncoderReranker, Retriever
from .store import VectorStore


# ---------------------------------------------------------------- helpers
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def parse_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_gold(hit, item: dict) -> bool:
    return hit.paper_id == item["paper_id"] and int(hit.page) == int(item["page"])


# ---------------------------------------------------------------- generate
def generate_questions(settings: Settings, n: int = 30, seed: int = 42) -> list[dict]:
    store = VectorStore(settings)
    if store.count() == 0:
        raise SystemExit("The index is empty. Run `python -m paperpilot.ingest` first.")
    llm = get_llm(settings)
    rows = []
    for hit in store.sample(n * 2, seed=seed):  # oversample: some passages make bad questions
        if len(rows) >= n:
            break
        prompt = QUESTION_GEN_PROMPT.format(title=hit.title, page=hit.page, text=hit.text)
        data = parse_json_object(llm.generate("You output only valid JSON.", prompt))
        if not data or not data.get("question"):
            continue
        rows.append({
            "id": f"q{len(rows) + 1:03d}",
            "question": data["question"].strip(),
            "reference_answer": str(data.get("answer", "")).strip(),
            "paper_id": hit.paper_id,
            "page": hit.page,
            "title": hit.title,
        })
        print(f"  {rows[-1]['id']}: {rows[-1]['question'][:90]}")
    out = settings.eval_dir / "questions.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} questions to {out}. Read them and delete any that are vague or wrong.")
    return rows


# ---------------------------------------------------------------- retrieval
@dataclass
class RetrievalResult:
    config: str
    rerank: bool
    recall_at_1: float
    recall_at_k: float
    mrr: float
    n: int


def score_retrieval(retriever: Retriever, questions: list[dict], k: int) -> tuple[float, float, float]:
    r1 = rk = rr = 0.0
    for item in questions:
        hits = retriever.retrieve(item["question"], k)
        ranks = [i for i, h in enumerate(hits, start=1) if is_gold(h, item)]
        if ranks:
            first = ranks[0]
            r1 += first == 1
            rk += 1
            rr += 1 / first
    n = max(len(questions), 1)
    return r1 / n, rk / n, rr / n


def evaluate_retrieval(base: Settings, configs: list[tuple[int, int]], rerank_modes: list[bool],
                       k: int = 5, questions: list[dict] | None = None, embedder=None) -> list[RetrievalResult]:
    questions = questions if questions is not None else read_jsonl(base.eval_dir / "questions.jsonl")
    if not questions:
        raise SystemExit("No questions found. Run `python -m paperpilot.evaluate generate` first.")
    embedder = embedder or get_embedder(base)
    reranker = CrossEncoderReranker(base.rerank_model) if any(rerank_modes) else None

    results = []
    for size, overlap in configs:
        settings = base.with_(chunk_size=size, chunk_overlap=overlap)
        store = VectorStore(settings)
        if store.count() == 0:
            print(f"Indexing chunk size {size}/{overlap} ...")
            ingest(settings, embedder=embedder)
            store = VectorStore(settings)
        for use_rerank in rerank_modes:
            retriever = Retriever(settings, store, embedder, reranker if use_rerank else None)
            r1, rk, mrr = score_retrieval(retriever, questions, k)
            results.append(RetrievalResult(f"{size}/{overlap}", use_rerank, r1, rk, mrr, len(questions)))
            print(f"  chunks {size}/{overlap} rerank={use_rerank}: R@1={r1:.2f} R@{k}={rk:.2f} MRR={mrr:.2f}")

    lines = [
        f"# Retrieval evaluation ({len(questions)} questions, k={k})", "",
        f"| Chunk size / overlap (words) | Re-ranking | Recall@1 | Recall@{k} | MRR |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.config} | {'yes' if r.rerank else 'no'} | {r.recall_at_1:.0%} | {r.recall_at_k:.0%} | {r.mrr:.2f} |")
    out = base.eval_dir / "results_retrieval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}")
    return results


# ---------------------------------------------------------------- answers
def evaluate_answers(settings: Settings, limit: int | None = None, pipeline: RAGPipeline | None = None) -> dict:
    questions = read_jsonl(settings.eval_dir / "questions.jsonl")[:limit]
    out_of_scope = read_jsonl(settings.eval_dir / "out_of_scope.jsonl")
    pipeline = pipeline or RAGPipeline.from_settings(settings)

    answered = cited_gold = 0
    rows = []
    for item in questions:
        ans = pipeline.ask(item["question"])
        hit = any(c.paper_id == item["paper_id"] and c.page == int(item["page"]) for c in ans.citations)
        answered += not ans.refused
        cited_gold += hit
        rows.append({"id": item["id"], "refused": ans.refused, "cites_gold_page": hit, "answer": ans.answer})

    refused_oos = sum(pipeline.ask(item["question"]).refused for item in out_of_scope)

    n, m = max(len(questions), 1), max(len(out_of_scope), 1)
    metrics = {
        "in_scope_questions": len(questions),
        "answer_rate": answered / n,
        "citation_accuracy": cited_gold / n,
        "out_of_scope_questions": len(out_of_scope),
        "out_of_scope_refusal_rate": refused_oos / m if out_of_scope else None,
    }
    lines = [
        f"# Answer evaluation (LLM: {settings.llm_backend}, chunks {settings.chunk_size}/{settings.chunk_overlap}, "
        f"re-ranking: {'yes' if settings.rerank else 'no'})", "",
        "| Metric | Value |", "|---|---|",
        f"| In-scope questions answered | {metrics['answer_rate']:.0%} |",
        f"| Answers citing the correct page | {metrics['citation_accuracy']:.0%} |",
    ]
    if out_of_scope:
        lines.append(f"| Out-of-scope questions correctly refused | {metrics['out_of_scope_refusal_rate']:.0%} |")
    out = settings.eval_dir / "results_answers.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_jsonl(settings.eval_dir / "answers_detail.jsonl", rows)
    print("\n".join(lines))
    return metrics


# ---------------------------------------------------------------- CLI
def parse_configs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for part in text.split(","):
        size, overlap = part.split(":")
        pairs.append((int(size), int(overlap)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaperPilot.")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="auto-generate evaluation questions")
    g.add_argument("--n", type=int, default=30)
    g.add_argument("--seed", type=int, default=42)

    r = sub.add_parser("retrieval", help="compare chunk sizes and re-ranking")
    r.add_argument("--configs", default="200:40,300:50,500:80")
    r.add_argument("--rerank", choices=["off", "on", "both"], default="both")
    r.add_argument("--k", type=int, default=5)

    a = sub.add_parser("answers", help="evaluate full answers and refusals")
    a.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    settings = get_settings()
    if args.command == "generate":
        generate_questions(settings, args.n, args.seed)
    elif args.command == "retrieval":
        modes = {"off": [False], "on": [True], "both": [False, True]}[args.rerank]
        evaluate_retrieval(settings, parse_configs(args.configs), modes, args.k)
    else:
        evaluate_answers(settings, args.limit)


if __name__ == "__main__":
    main()
