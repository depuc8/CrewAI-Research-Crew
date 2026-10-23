import pytest

from research_crew.ingestion import arxiv
from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.jobs import Job, JobRequest, JobStatus, LocalJobStore
from research_crew.runner import gather_papers, run_review_job

from .fake_llm import FakeLLM


def _job(**kw) -> Job:
    return Job(request=JobRequest(question="What reduces hallucination in RAG?", **kw))


@pytest.fixture
def store(tmp_path):
    return LocalJobStore(tmp_path)


def test_successful_job_produces_checked_review_and_metrics(services, two_papers, store):
    ingest_pdf(services, two_papers["rag-paper"], paper_id="rag-paper")
    job = _job(paper_ids=["rag-paper"])
    store.save(job)

    done = run_review_job(job.job_id, services, store, llm_factory=FakeLLM)

    assert done.status == JobStatus.succeeded and done.stage == "done" and done.error is None
    res = done.result
    assert "[rag-paper, 999]" in res.raw_markdown and "[rag-paper, 999]" not in res.markdown
    assert res.citation_report.dropped_claims and 0 < res.citation_accuracy_raw < 1
    assert [p.paper_id for p in res.papers] == ["rag-paper"]
    assert "## References" in res.markdown and "Self Verification For Retrieval Augmented Generation" in res.markdown

    m = done.metrics
    assert (m.llm_prompt_tokens, m.llm_completion_tokens) == (400, 200)  # 4 scripted calls, counted once (not once per agent)
    assert {"ingest", "reader", "analyst", "writer", "verify"} <= set(m.stage_seconds)
    assert m.cost_usd > 0 and m.papers == 1
    assert store.get(job.job_id).status == JobStatus.succeeded  # persisted


def test_only_queued_jobs_start(services, store):
    job = _job()
    job.status = JobStatus.succeeded
    store.save(job)
    assert run_review_job(job.job_id, services, store, llm_factory=FakeLLM).status == JobStatus.succeeded
    assert run_review_job("missing", services, store) is None


def test_unknown_paper_fails_the_job_with_a_message(services, store):
    job = _job(paper_ids=["nope"])
    store.save(job)
    done = run_review_job(job.job_id, services, store, llm_factory=FakeLLM)
    assert done.status == JobStatus.failed
    assert "unknown paper_id 'nope'" in done.error and done.result is None and done.metrics is not None


def test_arxiv_mode_fetches_papers(services, two_papers, monkeypatch):
    found = [arxiv.ArxivPaper(paper_id="rag-paper", title="t", url="u", pdf_url="p")]
    monkeypatch.setattr(arxiv, "search", lambda q, max_results=5, client=None: found)
    monkeypatch.setattr(arxiv, "download_pdf", lambda p, client=None: two_papers["rag-paper"])
    papers = gather_papers(services, _job(use_arxiv=True))
    assert [p.paper_id for p in papers] == ["rag-paper"]


def test_uploads_are_used_alone_unless_arxiv_requested(services, two_papers, monkeypatch):
    ingest_pdf(services, two_papers["bio-paper"], paper_id="bio-paper")
    monkeypatch.setattr(arxiv, "search", lambda *a, **k: pytest.fail("arXiv must not be queried"))
    assert [p.paper_id for p in gather_papers(services, _job(paper_ids=["bio-paper"]))] == ["bio-paper"]


def test_paper_count_is_capped(services, two_papers):
    for pid, data in two_papers.items():
        ingest_pdf(services, data, paper_id=pid)
    papers = gather_papers(services, _job(paper_ids=list(two_papers), max_papers=1))
    assert len(papers) == 1
