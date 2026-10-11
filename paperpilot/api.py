"""Step 5a: a small REST API.

Run:   uvicorn paperpilot.api:app --reload
Docs:  http://localhost:8000/docs

Endpoints:
  GET  /health               index size, collection and LLM in use
  POST /ask                  question (+ optional topic) -> cited answer
  GET  /topics               research topics that have been indexed
  POST /research             start a topic-research job in the background
  GET  /research/{job_id}    poll a job's progress and result
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from . import research as research_module
from .rag import RAGPipeline

app = FastAPI(title="PaperPilot", version=__version__,
              description="Ask questions about research papers and get answers with page-level citations.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)
    topic: str | None = Field(None, description="Only search papers downloaded for this research topic")


class CitationOut(BaseModel):
    tag: str
    paper_id: str
    title: str
    page: int
    url: str = ""


class SourceOut(BaseModel):
    paper_id: str
    title: str
    page: int
    score: float
    text: str
    topic: str = ""
    url: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[CitationOut]
    sources: list[SourceOut]


class TopicOut(BaseModel):
    topic: str
    chunks: int


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    max_papers: int = Field(40, ge=1, le=200)


class JobOut(BaseModel):
    job_id: str
    topic: str
    status: str                 # queued | running | done | failed
    stage: str = ""             # expand | search | rank | download | index | done
    message: str = ""
    done: int | None = None     # download progress
    total: int | None = None
    log: list[str] = []
    result: dict | None = None
    error: str | None = None


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings()


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    return {"status": "ok", "indexed_chunks": pipeline.retriever.store.count(),
            "collection": pipeline.retriever.store.name, "llm": pipeline.settings.llm_backend}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    if pipeline.retriever.store.count() == 0:
        raise HTTPException(status_code=503, detail="The index is empty. Run the ingestion step first.")
    ans = pipeline.ask(request.question, request.top_k, topic=request.topic or None)
    return AskResponse(
        question=ans.question,
        answer=ans.answer,
        refused=ans.refused,
        citations=[CitationOut(**vars(c)) for c in ans.citations],
        sources=[SourceOut(paper_id=h.paper_id, title=h.title, page=h.page, score=round(h.score, 4),
                           text=h.text, topic=h.topic, url=h.url)
                 for h in ans.sources],
    )


@app.get("/topics", response_model=list[TopicOut])
def topics() -> list[TopicOut]:
    store = get_pipeline().retriever.store
    return [TopicOut(topic=t, chunks=n) for t, n in store.topics().items()]


# ---------------------------------------------------------------- background research jobs
# One worker: jobs run one after another, so two jobs never write to the same
# metadata file / vector collection at the same time. The API stays responsive
# because the work happens off the request thread.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
MAX_LOG = 200


def _update(job_id: str, **changes) -> None:
    with _lock:
        _jobs[job_id].update(changes)


def _run_job(job_id: str, topic: str, max_papers: int) -> None:
    pipeline = get_pipeline()

    def progress(stage: str, message: str, done: int | None = None, total: int | None = None) -> None:
        with _lock:
            job = _jobs[job_id]
            job.update(stage=stage, message=message, done=done, total=total)
            job["log"] = (job["log"] + [f"[{stage}] {message}"])[-MAX_LOG:]

    _update(job_id, status="running", started=time.time())
    try:
        # Reuse the API's embedder and LLM so models are not loaded twice.
        result = research_module.research(topic, max_papers, pipeline.settings,
                                          llm=pipeline.llm, embedder=pipeline.retriever.embedder,
                                          progress=progress)
        _update(job_id, status="done", stage="done", result=result.summary())
    except Exception as exc:
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")


@app.post("/research", response_model=JobOut, status_code=202)
def start_research(request: ResearchRequest) -> JobOut:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "topic": request.topic.strip(), "status": "queued",
                         "stage": "", "message": "Waiting to start", "done": None, "total": None,
                         "log": [], "result": None, "error": None}
    _executor.submit(_run_job, job_id, request.topic.strip(), request.max_papers)
    return get_research(job_id)


@app.get("/research/{job_id}", response_model=JobOut)
def get_research(job_id: str) -> JobOut:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return JobOut(**{k: v for k, v in job.items() if k in JobOut.model_fields})
