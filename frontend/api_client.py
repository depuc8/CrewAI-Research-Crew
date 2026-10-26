"""Thin HTTP client for the Research Crew API (used by the Streamlit app)."""

from __future__ import annotations

import os
import time
from typing import Callable, Iterable

import httpx

TERMINAL = {"succeeded", "failed"}


def _id_token_headers(audience: str | None) -> dict[str, str]:
    """On Cloud Run the API is private: authenticate as this service's account with an ID token."""
    if not audience:
        return {}
    import google.auth.transport.requests
    import google.oauth2.id_token

    token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
    return {"Authorization": f"Bearer {token}"}


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str | None = None, audience: str | None = None, client: httpx.Client | None = None):
        self.base_url = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.audience = audience if audience is not None else os.getenv("API_AUDIENCE")
        self._client = client or httpx.Client(timeout=120)

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        headers = {**_id_token_headers(self.audience), **kw.pop("headers", {})}
        resp = self._client.request(method, self.base_url + path, headers=headers, **kw)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ApiError(f"{resp.status_code}: {detail}")
        return resp

    def upload(self, files: Iterable[tuple[str, bytes]]) -> list[dict]:
        payload = [("files", (name, data, "application/pdf")) for name, data in files]
        return self._request("POST", "/papers", files=payload).json()

    def start_review(self, question: str, paper_ids: list[str], use_arxiv: bool, max_papers: int | None) -> str:
        body = {"question": question, "paper_ids": paper_ids, "use_arxiv": use_arxiv, "max_papers": max_papers}
        return self._request("POST", "/reviews", json=body).json()["job_id"]

    def status(self, job_id: str) -> dict:
        return self._request("GET", f"/reviews/{job_id}").json()

    def result(self, job_id: str) -> dict:
        return self._request("GET", f"/reviews/{job_id}/result").json()

    def wait(
        self,
        job_id: str,
        on_update: Callable[[dict], None] | None = None,
        interval: float = 3.0,
        timeout: float = 3600.0,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            st = self.status(job_id)
            if on_update:
                on_update(st)
            if st["status"] in TERMINAL:
                return st
            if time.monotonic() > deadline:
                raise ApiError("timed out waiting for the review")
            time.sleep(interval)
