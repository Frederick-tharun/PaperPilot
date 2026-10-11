"""Shared test fixtures.

Tests run fully offline: they create two small PDFs, use the hash embedder
and the extractive "LLM", so no model download or API key is needed.
"""

import pymupdf
import pytest

from paperpilot.config import get_settings
from paperpilot.ingest import ingest

PAPER_A = [
    "Echo state networks are recurrent neural networks in which only the output weights are trained. "
    "The reservoir weights are random and fixed. Training reduces to a linear ridge regression problem, "
    "which makes echo state networks very fast to train compared with backpropagation through time. "
    "Reservoir computing has been applied to chaotic time series prediction with good results.",
    "The spectral radius of the reservoir matrix controls the memory of the network. "
    "A spectral radius slightly below one is a common choice because it keeps the echo state property. "
    "The leak rate determines how quickly the reservoir state forgets earlier inputs. "
    "Bayesian optimization can tune the spectral radius, leak rate and input scaling together.",
]
PAPER_B = [
    "Vision transformers split an image into patches and process them with self-attention layers. "
    "Low-rank adaptation, called LoRA, fine-tunes a model by learning small low-rank update matrices "
    "while the pretrained weights stay frozen. This greatly reduces the number of trainable parameters "
    "and the memory needed during fine-tuning on small datasets.",
]


def make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path)
    doc.close()


@pytest.fixture()
def settings(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    make_pdf(papers / "paperA.pdf", PAPER_A)
    make_pdf(papers / "paperB.pdf", PAPER_B)
    (papers / "metadata.jsonl").write_text(
        '{"paper_id": "paperA", "title": "Echo State Networks Explained"}\n'
        '{"paper_id": "paperB", "title": "LoRA for Vision Transformers"}\n',
        encoding="utf-8",
    )
    return get_settings(
        papers_dir=papers,
        db_dir=tmp_path / "chroma",
        eval_dir=tmp_path / "eval",
        embed_backend="hash",
        llm_backend="extractive",
        chunk_size=40,
        chunk_overlap=10,
        rerank=False,
        min_score=0.05,
    )


@pytest.fixture()
def indexed(settings):
    ingest(settings, reset=True)
    return settings
