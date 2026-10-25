"""FastAPI backend.

POST /papers                    upload PDFs (multipart) -> parsed, chunked, embedded, indexed
GET  /papers                    list ingested papers
POST /reviews                   start a review job (returns immediately, 202)
GET  /reviews/{id}              job status, stage and metrics
GET  /reviews/{id}/result       the finished review (JSON, or ?format=markdown)
POST /internal/jobs/{id}/run    worker entrypoint, called by Cloud Tasks (or the local pool)

Run with: uvicorn research_crew.api:create_app --factory
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from research_crew import __version__
from research_crew.dispatch import JobDispatcher, get_dispatcher
from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.jobs import Job, JobRequest, JobStatus, JobStore, ReviewResult, get_job_store
from research_crew.models import PaperMeta
from research_crew.runner import run_review_job
from research_crew.services import Services, build_services
from research_crew.usage import JobMetrics

log = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    error: str | None = None
    metrics: JobMetrics | None = None


class ResultView(BaseModel):
    job_id: str
    markdown: str
    result: ReviewResult
    metrics: JobMetrics | None


def create_app(
    services: Services | None = None,
    store: JobStore | None = None,
    dispatcher: JobDispatcher | None = None,
    llm_factory=None,
) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    services = services or build_services()
    store = store or get_job_store(services.settings)

    def run(job_id: str):
        return run_review_job(job_id, services, store, llm_factory=llm_factory)

    dispatcher = dispatcher or get_dispatcher(services.settings, run)
    app = FastAPI(title="Research Crew", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/papers", response_model=list[PaperMeta])
    def upload_papers(files: Annotated[list[UploadFile], File(description="PDF files")]) -> list[PaperMeta]:
        metas = []
        for f in files:
            data = f.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_BYTES // 2**20} MB")
            if not data.startswith(b"%PDF"):
                raise HTTPException(400, f"{f.filename}: not a PDF")
            try:
                metas.append(ingest_pdf(services, data))
            except ValueError as exc:  # e.g. scanned PDF with no text layer
                raise HTTPException(422, f"{f.filename}: {exc}") from exc
        return metas

    @app.get("/papers", response_model=list[PaperMeta])
    def list_papers() -> list[PaperMeta]:
        return services.metadata.list_papers()

    @app.post("/reviews", status_code=202, response_model=JobCreated)
    def start_review(req: JobRequest) -> JobCreated:
        unknown = [p for p in req.paper_ids if services.metadata.get_meta(p) is None]
        if unknown:
            raise HTTPException(422, f"unknown paper_ids: {', '.join(unknown)}")
        job = Job(request=req)
        store.save(job)
        dispatcher.dispatch(job.job_id)
        return JobCreated(job_id=job.job_id, status=job.status)

    def _job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/reviews/{job_id}", response_model=JobView)
    def job_status(job_id: str) -> JobView:
        job = _job(job_id)
        return JobView(job_id=job.job_id, status=job.status, stage=job.stage, error=job.error, metrics=job.metrics)

    @app.get("/reviews/{job_id}/result", response_model=ResultView)
    def job_result(job_id: str, format: str = "json"):
        job = _job(job_id)
        if job.status == JobStatus.failed:
            raise HTTPException(409, f"job failed: {job.error}")
        if job.status != JobStatus.succeeded or job.result is None:
            raise HTTPException(409, f"job is {job.status.value} (stage: {job.stage})")
        if format == "markdown":
            return PlainTextResponse(job.result.markdown, media_type="text/markdown")
        return ResultView(job_id=job.job_id, markdown=job.result.markdown, result=job.result, metrics=job.metrics)

    @app.post("/internal/jobs/{job_id}/run", response_model=JobView)
    def run_job(job_id: str, request: Request) -> JobView:
        s = services.settings
        if s.job_backend == "gcp" and request.headers.get("x-cloudtasks-queuename") != s.tasks_queue:
            raise HTTPException(403, "only the job queue may call this endpoint")
        job = run(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        # Always 200 once the job exists: a failed job is recorded on the job itself, and a
        # non-2xx here would make Cloud Tasks re-deliver and re-run it.
        return JobView(job_id=job.job_id, status=job.status, stage=job.stage, error=job.error, metrics=job.metrics)

    return app
