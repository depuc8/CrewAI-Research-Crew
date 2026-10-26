import json

import httpx
import pytest

from frontend.api_client import ApiClient, ApiError


def _client(handler, **kw) -> ApiClient:
    return ApiClient("http://api", audience="", client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_start_review_and_status():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            seen["body"] = json.loads(req.content)
            return httpx.Response(202, json={"job_id": "j1", "status": "queued"})
        return httpx.Response(200, json={"job_id": "j1", "status": "running", "stage": "crew"})

    c = _client(handler)
    assert c.start_review("What reduces hallucination?", ["p1"], False, 6) == "j1"
    assert seen["body"] == {"question": "What reduces hallucination?", "paper_ids": ["p1"], "use_arxiv": False, "max_papers": 6}
    assert c.status("j1")["stage"] == "crew"


def test_upload_sends_multipart_pdfs():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["content-type"].startswith("multipart/form-data")
        assert b'filename="a.pdf"' in req.content and b"%PDF-data" in req.content
        return httpx.Response(200, json=[{"paper_id": "up-1"}])

    assert _client(handler).upload([("a.pdf", b"%PDF-data")]) == [{"paper_id": "up-1"}]


def test_errors_surface_the_api_detail():
    c = _client(lambda req: httpx.Response(422, json={"detail": "unknown paper_ids: ghost"}))
    with pytest.raises(ApiError, match="422: unknown paper_ids: ghost"):
        c.status("x")
    c = _client(lambda req: httpx.Response(502, text="bad gateway"))
    with pytest.raises(ApiError, match="502: bad gateway"):
        c.status("x")


def test_wait_polls_until_terminal():
    states = iter(["queued", "running", "succeeded"])
    updates = []

    def handler(req):
        return httpx.Response(200, json={"status": next(states), "stage": "x"})

    final = _client(handler).wait("j1", on_update=updates.append, interval=0)
    assert final["status"] == "succeeded" and len(updates) == 3


def test_wait_times_out():
    c = _client(lambda req: httpx.Response(200, json={"status": "running", "stage": "crew"}))
    with pytest.raises(ApiError, match="timed out"):
        c.wait("j1", interval=0, timeout=-1)


def test_streamlit_app_renders_without_backend_calls():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).parents[1] / "frontend" / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.endswith("Research Crew")
    assert at.button[0].disabled  # no question yet
    at.text_area[0].set_value("What methods reduce hallucination in RAG systems?").run()
    assert not at.button[0].disabled
