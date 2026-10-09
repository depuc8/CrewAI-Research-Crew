"""Text embedders: Vertex AI (Gemini embeddings) for production, hashing for local/tests."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from research_crew.config import Settings


class Embedder(Protocol):
    dimensions: int
    tokens_used: int  # running total of embedded tokens, for cost tracking

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    # Gemini embeddings are only pre-normalized at the full 3072 dims; truncated
    # (output_dimensionality < 3072) vectors need normalising for dot-product search.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class GeminiEmbedder:
    """Vertex AI embeddings through the google-genai SDK (`gemini-embedding-001`)."""

    def __init__(self, settings: Settings, max_workers: int = 8):
        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project or None,
            location=settings.google_cloud_location,
        )
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        # The Gemini embedding model accepts a single input text per request; other
        # Vertex embedding models accept up to 250.
        self._batch = 1 if "gemini-embedding" in self.model else 100
        self._workers = max_workers
        self._lock = threading.Lock()
        self.tokens_used = 0

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig

        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensions),
        )
        tokens = sum(
            int(getattr(e.statistics, "token_count", 0) or 0) for e in response.embeddings if e.statistics
        )
        with self._lock:
            self.tokens_used += tokens
        return [_normalize(list(e.values)) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self._batch] for i in range(0, len(texts), self._batch)]
        with ThreadPoolExecutor(self._workers) as pool:
            results = list(pool.map(lambda b: self._embed(b, "RETRIEVAL_DOCUMENT"), batches))
        return [vec for batch in results for vec in batch]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "which with we our can not but also than then these those their they them such into over".split()
)


class HashingEmbedder:
    """Deterministic bag-of-words embedder (feature hashing, sublinear TF, L2-normalised).

    Purely lexical: it lets the whole pipeline run offline for local development and
    tests, but retrieval quality is that of keyword matching, not semantic search.
    """

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
        self.tokens_used = 0

    def _vector(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for tok in _TOKEN.findall(text.lower()):
            if tok in _STOP or len(tok) < 2:
                continue
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            counts[h % self.dimensions] = counts.get(h % self.dimensions, 0.0) + 1.0
        vec = [0.0] * self.dimensions
        for idx, c in counts.items():
            vec[idx] = 1.0 + math.log(c)
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.tokens_used += sum(len(t.split()) for t in texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "gcp":
        return GeminiEmbedder(settings)
    return HashingEmbedder(settings.embedding_dimensions)
