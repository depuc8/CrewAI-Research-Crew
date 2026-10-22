"""Review job records and where they are stored (local files or Firestore)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from research_crew.citations import CitationReport
from research_crew.config import Settings
from research_crew.models import PaperMeta
from research_crew.schemas import AnalysisOutput, ReaderOutput
from research_crew.usage import JobMetrics


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    paper_ids: list[str] = Field(default_factory=list, description="Previously uploaded papers to use.")
    use_arxiv: bool = Field(False, description="Also fetch papers from arXiv. Implied when no paper_ids given.")
    max_papers: int | None = Field(None, ge=1, le=20)


class ReviewResult(BaseModel):
    markdown: str  # after the citation check
    raw_markdown: str  # exactly what the writer produced
    citation_report: CitationReport
    citation_accuracy_raw: float  # share of the writer's citations that were valid before cleanup
    papers: list[PaperMeta]
    summaries: ReaderOutput
    analysis: AnalysisOutput


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    request: JobRequest
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    error: str | None = None
    metrics: JobMetrics | None = None
    result: ReviewResult | None = None


class JobStore(Protocol):
    def save(self, job: Job) -> None: ...

    def get(self, job_id: str) -> Job | None: ...


class LocalJobStore:
    def __init__(self, root: Path | None = None):
        self._root = root / "jobs" if root else None
        self._lock = threading.Lock()
        self._jobs: dict[str, str] = {}

    def save(self, job: Job) -> None:
        job.updated_at = _now()
        payload = job.model_dump_json()
        with self._lock:
            self._jobs[job.job_id] = payload
            if self._root:
                self._root.mkdir(parents=True, exist_ok=True)
                (self._root / f"{job.job_id}.json").write_text(payload)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            payload = self._jobs.get(job_id)
            if payload is None and self._root and (self._root / f"{job_id}.json").exists():
                payload = (self._root / f"{job_id}.json").read_text()
        return Job.model_validate_json(payload) if payload else None


class FirestoreJobStore:
    def __init__(self, settings: Settings, client=None):
        from google.cloud import firestore

        self._db = client or firestore.Client(project=settings.google_cloud_project or None)

    def save(self, job: Job) -> None:
        job.updated_at = _now()
        self._db.collection("jobs").document(job.job_id).set(json.loads(job.model_dump_json()))

    def get(self, job_id: str) -> Job | None:
        snap = self._db.collection("jobs").document(job_id).get()
        return Job.model_validate(snap.to_dict()) if snap.exists else None


def get_job_store(settings: Settings) -> JobStore:
    if settings.job_backend == "gcp":
        return FirestoreJobStore(settings)
    return LocalJobStore(settings.local_data_dir)
