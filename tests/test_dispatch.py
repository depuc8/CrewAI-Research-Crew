from research_crew.config import Settings
from research_crew.dispatch import CloudTasksDispatcher, LocalDispatcher
from research_crew.jobs import FirestoreJobStore, Job, JobRequest, JobStatus, LocalJobStore


def _job() -> Job:
    return Job(request=JobRequest(question="What reduces hallucination in RAG?"))


def test_local_job_store_roundtrip_and_persistence(tmp_path):
    store = LocalJobStore(tmp_path)
    job = _job()
    store.save(job)
    assert store.get(job.job_id).status == JobStatus.queued
    assert LocalJobStore(tmp_path).get(job.job_id).request.question == job.request.question  # from disk
    assert store.get("missing") is None


def test_local_dispatcher_runs_the_job():
    import threading

    seen, done = [], threading.Event()
    LocalDispatcher(lambda jid: (seen.append(jid), done.set())).dispatch("abc")
    assert done.wait(5) and seen == ["abc"]


def test_cloud_tasks_dispatcher_builds_authenticated_task():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.queue_path.return_value = "projects/p/locations/l/queues/q"
    settings = Settings(
        google_cloud_project="p", google_cloud_location="l", tasks_queue="q",
        service_url="https://svc.run.app/", tasks_service_account="sa@p.iam.gserviceaccount.com",
    )
    CloudTasksDispatcher(settings, client=client).dispatch("job9")
    req = client.create_task.call_args.kwargs["request"]
    http = req["task"].http_request
    assert req["parent"] == "projects/p/locations/l/queues/q"
    assert http.url == "https://svc.run.app/internal/jobs/job9/run"
    assert http.oidc_token.service_account_email == "sa@p.iam.gserviceaccount.com"
    assert http.oidc_token.audience == "https://svc.run.app"
    assert req["task"].dispatch_deadline.seconds == 1800


def test_firestore_job_store_roundtrip():
    from unittest.mock import MagicMock

    db = MagicMock()
    saved = {}
    db.collection.return_value.document.return_value.set.side_effect = lambda d: saved.update(d)
    store = FirestoreJobStore(Settings(), client=db)
    job = _job()
    store.save(job)
    snap = MagicMock(exists=True)
    snap.to_dict.return_value = saved
    db.collection.return_value.document.return_value.get.return_value = snap
    assert store.get(job.job_id).request.question == job.request.question
