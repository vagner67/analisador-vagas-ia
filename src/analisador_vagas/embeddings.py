"""Embeddings para o RAG.

- Com chave: Gemini (`gemini-embedding-001`) ou OpenAI (`text-embedding-3-small`) via REST.
- Sem chave: **modo offline TF-IDF** (Python puro, sem IA). É busca lexical, não semântica,
  e é rotulado como tal em todas as respostas.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from typing import Protocol

import httpx

from .llm import LLMError

STOPWORDS = set(
    "a o e de da do das dos em para com por que quais qual um uma os as no na nos nas ao "
    "se mais ou the and of to in for with on is are what which vaga vagas pede pedem".split()
)


def tokenizar(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9#+.]+", t) if w.strip(".") and w not in STOPWORDS]


def cosseno(a: list[float] | dict, b: list[float] | dict) -> float:
    if isinstance(a, dict):
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na, nb = (
            math.sqrt(sum(v * v for v in a.values())),
            math.sqrt(sum(v * v for v in b.values())),
        )
    else:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class TfidfIndex:
    """Índice TF-IDF em Python puro (modo offline, sem IA)."""

    nome = "tfidf-offline"

    def __init__(self, documentos: list[str]) -> None:
        self.docs_tokens = [tokenizar(d) for d in documentos]
        n = len(documentos)
        df = Counter(t for toks in self.docs_tokens for t in set(toks))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vetores = [self._vetor(toks) for toks in self.docs_tokens]

    def _vetor(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}

    def consultar(self, texto: str) -> list[float]:
        q = self._vetor(tokenizar(texto))
        return [cosseno(q, v) for v in self.vetores]


class Embedder(Protocol):
    nome: str

    def embed(self, textos: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", timeout: float = 60):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.nome = f"gemini:{model}"

    def embed(self, textos: list[str]) -> list[list[float]]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:batchEmbedContents"
        )
        body = {
            "requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t[:8000]}]}}
                for t in textos
            ]
        }
        r = httpx.post(
            url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout
        )
        if r.status_code != 200:
            raise LLMError(f"Gemini embeddings HTTP {r.status_code}: {r.text[:300]}")
        return [e["values"] for e in r.json()["embeddings"]]


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", timeout: float = 60):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.nome = f"openai:{model}"

    def embed(self, textos: list[str]) -> list[list[float]]:
        r = httpx.post(
            "https://api.openai.com/v1/embeddings",
            json={"model": self.model, "input": [t[:8000] for t in textos]},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise LLMError(f"OpenAI embeddings HTTP {r.status_code}: {r.text[:300]}")
        dados = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in dados]


def embedder_do_ambiente() -> Embedder | None:
    """Retorna embedder de API se houver chave; None => modo offline TF-IDF."""
    provedor = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provedor == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder(os.environ["OPENAI_API_KEY"])
    if os.getenv("GEMINI_API_KEY") and provedor != "openai":
        return GeminiEmbedder(os.environ["GEMINI_API_KEY"])
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder(os.environ["OPENAI_API_KEY"])
    return None
