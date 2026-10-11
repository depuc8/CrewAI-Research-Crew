"""Vector stores.

Both backends store only ids + vectors (+ a `paper_id` restrict for filtering); the chunk
text lives in the metadata store and is joined back in by the retriever. That mirrors how
Vertex AI Vector Search works: it returns datapoint ids and distances, not documents.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from research_crew.config import Settings
from research_crew.models import Chunk

PAPER_NAMESPACE = "paper_id"
UPSERT_BATCH = 100  # Vector Search accepts up to 1000 per upsert; stay well below it


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def query(
        self, vector: Sequence[float], top_k: int, paper_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return [(chunk_id, similarity)] best-first. Higher similarity is better."""

    def delete(self, chunk_ids: Sequence[str]) -> None: ...


class LocalVectorStore:
    """Exact cosine search over unit vectors, persisted as JSONL. For dev, tests and eval."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._papers: list[str] = []
        self._rows: list[list[float]] = []
        if path and path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                self._append(rec["id"], rec["paper_id"], rec["vector"])

    def _append(self, cid: str, paper_id: str, vector: Sequence[float]) -> None:
        self._ids.append(cid)
        self._papers.append(paper_id)
        self._rows.append(list(vector))

    def upsert(self, chunks, vectors) -> None:
        with self._lock:
            replace = {c.chunk_id for c in chunks}
            keep = [i for i, cid in enumerate(self._ids) if cid not in replace]
            self._ids = [self._ids[i] for i in keep]
            self._papers = [self._papers[i] for i in keep]
            self._rows = [self._rows[i] for i in keep]
            for chunk, vec in zip(chunks, vectors, strict=True):
                self._append(chunk.chunk_id, chunk.paper_id, vec)
            self._flush()

    def _flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as fh:
            for cid, pid, vec in zip(self._ids, self._papers, self._rows):
                fh.write(json.dumps({"id": cid, "paper_id": pid, "vector": vec}) + "\n")

    def query(self, vector, top_k, paper_ids=None):
        with self._lock:
            if not self._ids:
                return []
            mask = np.ones(len(self._ids), dtype=bool)
            if paper_ids is not None:
                allowed = set(paper_ids)
                mask = np.array([p in allowed for p in self._papers])
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return []
            matrix = np.asarray(self._rows, dtype=np.float32)[idx]
            scores = matrix @ np.asarray(vector, dtype=np.float32)
            order = np.argsort(-scores)[:top_k]
            return [(self._ids[idx[i]], float(scores[i])) for i in order]

    def delete(self, chunk_ids) -> None:
        with self._lock:
            drop = set(chunk_ids)
            keep = [i for i, cid in enumerate(self._ids) if cid not in drop]
            self._ids = [self._ids[i] for i in keep]
            self._papers = [self._papers[i] for i in keep]
            self._rows = [self._rows[i] for i in keep]
            self._flush()


class VertexVectorStore:
    """Vertex AI Vector Search. Needs a STREAM_UPDATE index deployed to an endpoint.

    The index is created with DOT_PRODUCT_DISTANCE over normalised vectors, so the
    returned `distance` is already a similarity (larger = closer).
    """

    def __init__(self, settings: Settings, index=None, endpoint=None):
        from google.cloud import aiplatform

        if index is None or endpoint is None:
            aiplatform.init(project=settings.google_cloud_project, location=settings.google_cloud_location)
        self._index = index or aiplatform.MatchingEngineIndex(index_name=settings.vector_index_id)
        self._endpoint = endpoint or aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=settings.vector_index_endpoint_id
        )
        self._deployed_id = settings.vector_deployed_index_id

    def upsert(self, chunks, vectors) -> None:
        from google.cloud.aiplatform_v1.types import IndexDatapoint

        points = [
            IndexDatapoint(
                datapoint_id=c.chunk_id,
                feature_vector=list(v),
                restricts=[IndexDatapoint.Restriction(namespace=PAPER_NAMESPACE, allow_list=[c.paper_id])],
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            self._index.upsert_datapoints(datapoints=points[i : i + UPSERT_BATCH])

    def query(self, vector, top_k, paper_ids=None):
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

        kwargs = {}
        if paper_ids is not None:
            if not paper_ids:
                return []
            kwargs["filter"] = [Namespace(name=PAPER_NAMESPACE, allow_tokens=list(paper_ids))]
        response = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_id,
            queries=[list(vector)],
            num_neighbors=top_k,
            **kwargs,
        )
        return [(n.id, float(n.distance)) for n in (response[0] if response else [])]

    def delete(self, chunk_ids) -> None:
        ids = list(chunk_ids)
        for i in range(0, len(ids), UPSERT_BATCH):
            self._index.remove_datapoints(datapoint_ids=ids[i : i + UPSERT_BATCH])


def get_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "gcp":
        return VertexVectorStore(settings)
    return LocalVectorStore(settings.local_data_dir / "vectors.jsonl")
