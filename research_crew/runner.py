"""Runs one review job: gather papers -> Reader/Analyst/Writer crew -> citation check."""

from __future__ import annotations

import logging
import time
from functools import partial

from research_crew.citations import check_citations, extract_citations
from research_crew.crew import build_crew
from research_crew.ingestion.pipeline import fetch_for_question
from research_crew.jobs import Job, JobStatus, JobStore, ReviewResult
from research_crew.models import PaperMeta
from research_crew.retrieval import Retriever
from research_crew.schemas import AnalysisOutput, LiteratureReview, ReaderOutput, render_markdown
from research_crew.services import Services
from research_crew.usage import MetricsRecorder

log = logging.getLogger(__name__)

AGENT_STAGES = ("reader", "analyst", "writer")


def _mark_done(finished: list[float], _task_output) -> None:
    finished.append(time.monotonic())


def gather_papers(services: Services, job: Job) -> list[PaperMeta]:
    req = job.request
    limit = min(req.max_papers or services.settings.max_papers_per_job, services.settings.max_papers_per_job)
    papers: list[PaperMeta] = []
    for pid in dict.fromkeys(req.paper_ids):
        meta = services.metadata.get_meta(pid)
        if meta is None:
            raise ValueError(f"unknown paper_id '{pid}': upload it first")
        papers.append(meta)
    papers = papers[:limit]
    if (req.use_arxiv or not papers) and len(papers) < limit:
        have = {p.paper_id for p in papers}
        fetched = fetch_for_question(services, req.question, limit - len(papers))
        papers += [p for p in fetched if p.paper_id not in have]
    if not papers:
        raise ValueError("no papers available: upload PDFs or enable arXiv search")
    return papers[:limit]


def _set(store: JobStore, job: Job, **fields) -> None:
    for k, v in fields.items():
        setattr(job, k, v)
    store.save(job)


def run_review_job(job_id: str, services: Services, store: JobStore, llm_factory=None) -> Job | None:
    """Execute a queued job. Safe to call twice: only a `queued` job is ever started."""
    job = store.get(job_id)
    if job is None or job.status != JobStatus.queued:
        return job
    _set(store, job, status=JobStatus.running, stage="ingest")
    rec = MetricsRecorder(job_id, services.settings)
    embedded_before = services.embedder.tokens_used
    try:
        with rec.stage("ingest"):
            papers = gather_papers(services, job)

        _set(store, job, stage="crew")
        retriever = Retriever(services)
        finished: list[float] = []
        review_crew = build_crew(
            services, retriever, papers, job.request.question, llm_factory=llm_factory,
            task_callback=partial(_mark_done, finished),
        )
        started = time.monotonic()
        output = review_crew.crew.kickoff()
        previous = started
        for name, t in zip(AGENT_STAGES, finished):
            rec.add_stage_time(name, t - previous)
            previous = t
        rec.set_llm_usage(output.token_usage)

        reader, analysis, review = (t.pydantic for t in output.tasks_output)
        for name, value, kind in (("reader", reader, ReaderOutput), ("analyst", analysis, AnalysisOutput),
                                  ("writer", review, LiteratureReview)):
            if not isinstance(value, kind):
                raise RuntimeError(f"the {name} agent did not return valid structured output")

        _set(store, job, stage="verify")
        with rec.stage("verify"):
            by_id = {p.paper_id: p for p in papers}
            raw = render_markdown(review, by_id)
            cited = extract_citations(raw.split("## References")[0])
            clean, report = check_citations(raw, retriever.log, set(by_id))
        raw_accuracy = report.valid_citations / len(cited) if cited else 0.0

        result = ReviewResult(
            markdown=clean, raw_markdown=raw, citation_report=report, citation_accuracy_raw=raw_accuracy,
            papers=papers, summaries=reader, analysis=analysis,
        )
        rec.add_embedding_tokens(services.embedder.tokens_used - embedded_before)  # ingest + query embeddings
        _set(store, job, status=JobStatus.succeeded, stage="done", result=result, metrics=rec.finish(len(papers)))
    except Exception as exc:  # the job record is the error channel; never let it escape the worker
        log.exception("job %s failed", job_id)
        rec.add_embedding_tokens(services.embedder.tokens_used - embedded_before)
        _set(store, job, status=JobStatus.failed, stage="failed", error=f"{type(exc).__name__}: {exc}"[:2000],
             metrics=rec.finish(0))
    return job
