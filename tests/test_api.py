import time

import pytest
from fastapi.testclient import TestClient

from research_crew.api import create_app
from research_crew.dispatch import LocalDispatcher
from research_crew.jobs import LocalJobStore
from research_crew.runner import run_review_job

from .fake_llm import FakeLLM


class InlineDispatcher:
    """Runs the job synchronously so tests need no polling."""

    def __init__(self, run):
        self.run = run

    def dispatch(self, job_id):
        self.run(job_id)


@pytest.fixture
def client(services, tmp_path):
    store = LocalJobStore(tmp_path)
    dispatcher = InlineDispatcher(lambda jid: run_review_job(jid, services, store, llm_factory=FakeLLM))
    return TestClient(create_app(services, store, dispatcher, llm_factory=FakeLLM))


def _upload(client, two_papers, *names):
    files = [("files", (f"{n}.pdf", two_papers[n], "application/pdf")) for n in names]
    return client.post("/papers", files=files)


def test_health(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_upload_lists_and_rejects_bad_files(client, two_papers):
    r = _upload(client, two_papers, "rag-paper", "bio-paper")
    assert r.status_code == 200
    metas = r.json()
    assert len(metas) == 2 and metas[0]["title"].startswith("Self Verification")
    assert {m["paper_id"] for m in client.get("/papers").json()} == {m["paper_id"] for m in metas}

    bad = client.post("/papers", files=[("files", ("x.pdf", b"not a pdf", "application/pdf"))])
    assert bad.status_code == 400 and "not a PDF" in bad.json()["detail"]


def test_full_review_flow(client, two_papers):
    pid = _upload(client, two_papers, "rag-paper").json()[0]["paper_id"]

    created = client.post("/reviews", json={"question": "What reduces hallucination in RAG?", "paper_ids": [pid]})
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    status = client.get(f"/reviews/{job_id}").json()
    assert status["status"] == "succeeded" and status["stage"] == "done"
    assert status["metrics"]["llm_prompt_tokens"] == 400 and status["metrics"]["cost_usd"] > 0

    body = client.get(f"/reviews/{job_id}/result").json()
    assert "## Introduction" in body["markdown"]
    assert f"[{pid}, 999]" not in body["markdown"]  # the fabricated page was removed by the check
    assert body["result"]["citation_report"]["invalid"]
    md = client.get(f"/reviews/{job_id}/result", params={"format": "markdown"})
    assert md.headers["content-type"].startswith("text/markdown") and md.text == body["markdown"]


def test_start_review_validates_input(client):
    assert client.post("/reviews", json={"question": "hi"}).status_code == 422  # too short
    r = client.post("/reviews", json={"question": "What reduces hallucination?", "paper_ids": ["ghost"]})
    assert r.status_code == 422 and "ghost" in r.json()["detail"]


def test_unknown_job_and_unfinished_result(client, services, tmp_path):
    assert client.get("/reviews/nope").status_code == 404
    assert client.get("/reviews/nope/result").status_code == 404


def test_result_of_pending_and_failed_jobs_is_409(services, tmp_path, monkeypatch):
    from research_crew.ingestion import arxiv

    monkeypatch.setattr(arxiv, "search", lambda *a, **k: [])  # keep the test offline
    store = LocalJobStore(tmp_path)

    class Never:
        def dispatch(self, job_id):
            pass

    c = TestClient(create_app(services, store, Never()))
    job_id = c.post("/reviews", json={"question": "What reduces hallucination?", "use_arxiv": True}).json()["job_id"]
    r = c.get(f"/reviews/{job_id}/result")
    assert r.status_code == 409 and "queued" in r.json()["detail"]

    run_review_job(job_id, services, store, llm_factory=FakeLLM)  # arXiv finds nothing -> job fails
    assert c.get(f"/reviews/{job_id}").json()["status"] == "failed"
    failed = c.get(f"/reviews/{job_id}/result")
    assert failed.status_code == 409 and "no papers available" in failed.json()["detail"]


def test_internal_run_endpoint_runs_once_and_is_idempotent(services, two_papers, tmp_path):
    store = LocalJobStore(tmp_path)

    class Never:
        def dispatch(self, job_id):
            pass

    created: list[FakeLLM] = []

    def counting_factory():
        created.append(FakeLLM())
        return created[-1]

    c = TestClient(create_app(services, store, Never(), llm_factory=counting_factory))
    pid = _upload(c, two_papers, "rag-paper").json()[0]["paper_id"]
    job_id = c.post("/reviews", json={"question": "What reduces hallucination?", "paper_ids": [pid]}).json()["job_id"]

    first = c.post(f"/internal/jobs/{job_id}/run")
    assert first.status_code == 200 and first.json()["status"] == "succeeded"
    assert len(created) == 3  # one LLM per agent
    again = c.post(f"/internal/jobs/{job_id}/run")  # Cloud Tasks re-delivery must not re-run the crew
    assert again.status_code == 200 and again.json()["status"] == "succeeded"
    assert len(created) == 3
    assert c.post("/internal/jobs/nope/run").status_code == 404


def test_internal_endpoint_requires_queue_header_on_gcp(services, tmp_path):
    services.settings.job_backend = "gcp"
    services.settings.tasks_queue = "q1"
    store = LocalJobStore(tmp_path)

    class Never:
        def dispatch(self, job_id):
            pass

    c = TestClient(create_app(services, store, Never()))
    assert c.post("/internal/jobs/x/run").status_code == 403
    assert c.post("/internal/jobs/x/run", headers={"X-CloudTasks-QueueName": "other"}).status_code == 403
    assert c.post("/internal/jobs/x/run", headers={"X-CloudTasks-QueueName": "q1"}).status_code == 404  # passed auth


def test_local_dispatcher_integration(services, two_papers, tmp_path):
    store = LocalJobStore(tmp_path)
    runner = lambda jid: run_review_job(jid, services, store, llm_factory=FakeLLM)
    c = TestClient(create_app(services, store, LocalDispatcher(runner)))
    pid = _upload(c, two_papers, "rag-paper").json()[0]["paper_id"]
    job_id = c.post("/reviews", json={"question": "What reduces hallucination?", "paper_ids": [pid]}).json()["job_id"]
    for _ in range(100):
        if c.get(f"/reviews/{job_id}").json()["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert c.get(f"/reviews/{job_id}").json()["status"] == "succeeded"
