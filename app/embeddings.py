"""
Embedding backend abstraction.

Two implementations:
  - SentenceTransformerBackend: real embeddings (all-MiniLM-L6-v2 by default).
    Requires network access to download the model the first time, and the
    `sentence-transformers` package.
  - MockEmbeddingBackend: deterministic, dependency-free embeddings built from
    hashed n-grams. Not semantically meaningful in a deep sense, but two
    similar strings *will* land closer together than two unrelated ones
    (shared words -> shared hashed buckets), which is enough to exercise and
    unit-test the fixation-detection math without any downloads.

Swap via FIXATION_EMBEDDING_BACKEND=sentence-transformers in your .env once
you have network access.
"""
from __future__ import annotations
import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.config import settings


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class MockEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


_backend_instance: EmbeddingBackend | None = None


def get_embedding_backend() -> EmbeddingBackend:
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    if settings.embedding_backend == "sentence-transformers":
        _backend_instance = SentenceTransformerBackend(settings.embedding_model_name)
    else:
        _backend_instance = MockEmbeddingBackend(dim=settings.embedding_dim)
    return _backend_instance
