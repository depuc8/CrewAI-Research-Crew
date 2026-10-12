"""Paper metadata, full text and chunk text.

GCP layout:  GCS  papers/<paper_id>/{paper.pdf, parsed.json}   (original PDF + full parse)
             Firestore  papers/<paper_id>                      (PaperMeta, for listing)
             Firestore  chunks/<chunk_id>                      (chunk text, joined to vector hits)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from research_crew.config import Settings
from research_crew.models import Chunk, PaperMeta, ParsedPaper

FIRESTORE_BATCH = 400  # hard limit is 500 writes per batch


class MetadataStore(Protocol):
    def save_paper(self, paper: ParsedPaper, chunks: Sequence[Chunk], pdf: bytes | None = None) -> None: ...

    def get_meta(self, paper_id: str) -> PaperMeta | None: ...

    def get_paper(self, paper_id: str) -> ParsedPaper | None: ...

    def list_papers(self) -> list[PaperMeta]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

    def chunk_ids_for(self, paper_id: str) -> list[str]: ...


class LocalMetadataStore:
    """Plain files under `<root>/papers/<paper_id>/`."""

    def __init__(self, root: Path):
        self._root = root / "papers"

    def _dir(self, paper_id: str) -> Path:
        return self._root / paper_id

    def save_paper(self, paper, chunks, pdf=None) -> None:
        d = self._dir(paper.meta.paper_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "parsed.json").write_text(paper.model_dump_json())
        (d / "chunks.json").write_text(json.dumps([c.model_dump() for c in chunks]))
        if pdf is not None:
            (d / "paper.pdf").write_bytes(pdf)

    def get_meta(self, paper_id):
        paper = self.get_paper(paper_id)
        return paper.meta if paper else None

    def get_paper(self, paper_id):
        f = self._dir(paper_id) / "parsed.json"
        return ParsedPaper.model_validate_json(f.read_text()) if f.exists() else None

    def list_papers(self):
        if not self._root.exists():
            return []
        metas = [self.get_meta(d.name) for d in sorted(self._root.iterdir()) if d.is_dir()]
        return [m for m in metas if m]

    def _chunks(self, paper_id: str) -> list[Chunk]:
        f = self._dir(paper_id) / "chunks.json"
        return [Chunk(**c) for c in json.loads(f.read_text())] if f.exists() else []

    def chunk_ids_for(self, paper_id):
        return [c.chunk_id for c in self._chunks(paper_id)]

    def get_chunks(self, chunk_ids):
        out: dict[str, Chunk] = {}
        cache: dict[str, dict[str, Chunk]] = {}
        for cid in chunk_ids:
            pid = cid.rsplit(":", 1)[0]
            if pid not in cache:
                cache[pid] = {c.chunk_id: c for c in self._chunks(pid)}
            if cid in cache[pid]:
                out[cid] = cache[pid][cid]
        return out


class GcpMetadataStore:
    def __init__(self, settings: Settings, bucket=None, firestore_client=None):
        from google.cloud import firestore, storage

        self._bucket = bucket or storage.Client(project=settings.google_cloud_project or None).bucket(
            settings.gcs_bucket
        )
        self._db = firestore_client or firestore.Client(project=settings.google_cloud_project or None)

    def save_paper(self, paper, chunks, pdf=None) -> None:
        pid = paper.meta.paper_id
        if pdf is not None:
            self._bucket.blob(f"papers/{pid}/paper.pdf").upload_from_string(pdf, content_type="application/pdf")
        self._bucket.blob(f"papers/{pid}/parsed.json").upload_from_string(
            paper.model_dump_json(), content_type="application/json"
        )
        self._db.collection("papers").document(pid).set(paper.meta.model_dump())
        for i in range(0, len(chunks), FIRESTORE_BATCH):
            batch = self._db.batch()
            for c in chunks[i : i + FIRESTORE_BATCH]:
                batch.set(self._db.collection("chunks").document(c.chunk_id), c.model_dump())
            batch.commit()

    def get_meta(self, paper_id):
        snap = self._db.collection("papers").document(paper_id).get()
        return PaperMeta(**snap.to_dict()) if snap.exists else None

    def get_paper(self, paper_id):
        blob = self._bucket.blob(f"papers/{paper_id}/parsed.json")
        return ParsedPaper.model_validate_json(blob.download_as_bytes()) if blob.exists() else None

    def list_papers(self):
        return [PaperMeta(**d.to_dict()) for d in self._db.collection("papers").stream()]

    def chunk_ids_for(self, paper_id):
        query = self._db.collection("chunks").where("paper_id", "==", paper_id)
        return [d.id for d in query.stream()]

    def get_chunks(self, chunk_ids):
        ids = list(chunk_ids)
        out: dict[str, Chunk] = {}
        for i in range(0, len(ids), 100):
            refs = [self._db.collection("chunks").document(cid) for cid in ids[i : i + 100]]
            for snap in self._db.get_all(refs):
                if snap.exists:
                    out[snap.id] = Chunk(**snap.to_dict())
        return out


def get_metadata_store(settings: Settings) -> MetadataStore:
    if settings.storage_backend == "gcp":
        return GcpMetadataStore(settings)
    return LocalMetadataStore(settings.local_data_dir)
