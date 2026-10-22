"""How a queued job gets executed: in-process worker threads locally, Cloud Tasks on GCP."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

from research_crew.config import Settings

RUN_PATH = "/internal/jobs/{job_id}/run"
DISPATCH_DEADLINE_SECONDS = 1800  # Cloud Tasks maximum for HTTP targets


class JobDispatcher(Protocol):
    def dispatch(self, job_id: str) -> None: ...


class LocalDispatcher:
    """Runs jobs on a small thread pool inside the API process (dev and tests)."""

    def __init__(self, runner: Callable[[str], object], max_workers: int = 2):
        self._runner = runner
        self._pool = ThreadPoolExecutor(max_workers, thread_name_prefix="review-job")

    def dispatch(self, job_id: str) -> None:
        self._pool.submit(self._runner, job_id)


class CloudTasksDispatcher:
    """Enqueues an authenticated HTTP call back to this service's /internal run endpoint.

    The task carries an OIDC token for `tasks_service_account`, which Cloud Run verifies
    (the service is not publicly invokable), so only the queue can start jobs.
    """

    def __init__(self, settings: Settings, client=None):
        from google.cloud import tasks_v2

        self._tasks = tasks_v2
        self._client = client or tasks_v2.CloudTasksClient()
        self._queue = self._client.queue_path(
            settings.google_cloud_project, settings.google_cloud_location, settings.tasks_queue
        )
        self._base_url = settings.service_url.rstrip("/")
        self._service_account = settings.tasks_service_account

    def build_task(self, job_id: str):
        from google.protobuf import duration_pb2

        t = self._tasks
        return t.Task(
            http_request=t.HttpRequest(
                http_method=t.HttpMethod.POST,
                url=self._base_url + RUN_PATH.format(job_id=job_id),
                headers={"Content-Type": "application/json"},
                body=json.dumps({"job_id": job_id}).encode(),
                oidc_token=t.OidcToken(service_account_email=self._service_account, audience=self._base_url),
            ),
            dispatch_deadline=duration_pb2.Duration(seconds=DISPATCH_DEADLINE_SECONDS),
        )

    def dispatch(self, job_id: str) -> None:
        self._client.create_task(request={"parent": self._queue, "task": self.build_task(job_id)})


def get_dispatcher(settings: Settings, runner: Callable[[str], object]) -> JobDispatcher:
    if settings.job_backend == "gcp":
        return CloudTasksDispatcher(settings)
    return LocalDispatcher(runner)
