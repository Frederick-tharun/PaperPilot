"""Language-model backends that write the final answer.

* AnthropicLLM  - Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
* OllamaLLM     - a free local model via Ollama (https://ollama.com)
* ExtractiveLLM - no model at all: returns the top source's opening sentences.
                  Used for tests and offline demos.
"""

from __future__ import annotations

import re
from typing import Protocol

import requests

from .config import Settings


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaLLM:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def generate(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class ExtractiveLLM:
    """Offline stand-in: quotes the first two sentences of source [S1]."""

    SOURCE = re.compile(r"\[S1\][^\n]*\n(.+?)(?:\n\n\[S\d+\]|\n\nQuestion:|\Z)", re.DOTALL)

    def generate(self, system: str, user: str) -> str:
        match = self.SOURCE.search(user)
        if not match:
            return "I don't know based on the provided papers."
        sentences = re.split(r"(?<=[.!?])\s+", match.group(1).strip())
        return " ".join(sentences[:2]) + " [S1]"


def get_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "anthropic":
        return AnthropicLLM(settings.anthropic_model, settings.max_tokens)
    if settings.llm_backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if settings.llm_backend == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown PP_LLM: {settings.llm_backend!r}")
