"""Question -> relevant chunks, plus a log of everything retrieved during a job.

The log is the ground truth for citation checking: a claim may only cite pages of chunks
that were actually shown to the agents.
"""

from __future__ import annotations

import threading
from typing import Sequence

from research_crew.models import Chunk, RetrievedChunk
from research_crew.services import Services


class RetrievalLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self.queries = 0

    def record(self, hits: Sequence[RetrievedChunk]) -> None:
        with self._lock:
            self.queries += 1
            for h in hits:
                self._chunks[h.chunk.chunk_id] = h.chunk

    @property
    def chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def supports(self, paper_id: str, page: int) -> bool:
        """True if some retrieved chunk of `paper_id` covers `page`."""
        return any(c.paper_id == paper_id and c.covers_page(page) for c in self.chunks)

    def paper_ids(self) -> set[str]:
        return {c.paper_id for c in self.chunks}


class Retriever:
    def __init__(self, services: Services, log: RetrievalLog | None = None):
        self._s = services
        self.log = log or RetrievalLog()

    def search(
        self, query: str, paper_ids: Sequence[str] | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        k = top_k or self._s.settings.retrieval_top_k
        vector = self._s.embedder.embed_query(query)
        scored = self._s.vectors.query(vector, k, paper_ids)
        chunks = self._s.metadata.get_chunks([cid for cid, _ in scored])
        hits = [RetrievedChunk(chunk=chunks[cid], score=score) for cid, score in scored if cid in chunks]
        self.log.record(hits)
        return hits
