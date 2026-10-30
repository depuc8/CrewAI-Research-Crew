"""Minimal arXiv API client: keyword search and PDF download.

arXiv asks API users to make at most one request every 3 seconds and to identify
themselves, so requests are serialised through a shared lock with a minimum spacing.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{id}"
MIN_INTERVAL = 3.0
USER_AGENT = "research-crew/0.1 (literature review tool)"
_NS = {"a": "http://www.w3.org/2005/Atom"}
_ID = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v\d+)?$")

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in into is it of on or that the this to "
    "what when which who why with methods method approaches approach techniques technique "
    "reduce reduces reducing improve improves improving using use used".split()
)

_gate = threading.Lock()
_last_call = 0.0


class ArxivPaper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str
    pdf_url: str


def normalize_id(raw: str) -> str:
    """'http://arxiv.org/abs/hep-th/9901001v2' -> 'hep-th_9901001' (safe as a path / doc id)."""
    m = _ID.search(raw)
    ident = m.group("id") if m else re.sub(r"v\d+$", "", raw.strip())
    return ident.replace("/", "_")


def build_query(question: str, max_terms: int = 6) -> str:
    """Turn a natural-language question into an arXiv `all:` keyword query."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", question.lower()) if w not in _STOPWORDS]
    terms = list(dict.fromkeys(words))[:max_terms]
    return " AND ".join(f"all:{t}" for t in terms) if terms else "all:" + question.strip()


def _throttle() -> None:
    global _last_call
    with _gate:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _get(url: str, params: dict | None = None, client: httpx.Client | None = None) -> httpx.Response:
    owns = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    try:
        for attempt in range(4):
            _throttle()
            resp = client.get(url, params=params)
            if resp.status_code in (429, 503) and attempt < 3:
                time.sleep(2**attempt * 2)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")
    finally:
        if owns:
            client.close()


def parse_feed(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", _NS):
        raw_id = entry.findtext("a:id", default="", namespaces=_NS)
        if "arxiv.org/abs/" not in raw_id:
            continue  # error entries
        pid = normalize_id(raw_id)
        published = entry.findtext("a:published", default="", namespaces=_NS)
        papers.append(
            ArxivPaper(
                paper_id=pid,
                title=" ".join(entry.findtext("a:title", default="", namespaces=_NS).split()),
                authors=[a.findtext("a:name", default="", namespaces=_NS) for a in entry.findall("a:author", _NS)],
                year=datetime.fromisoformat(published.replace("Z", "+00:00")).year if published else None,
                abstract=" ".join(entry.findtext("a:summary", default="", namespaces=_NS).split()),
                url=raw_id,
                pdf_url=PDF_URL.format(id=pid.replace("_", "/", 1) if re.match(r"^[a-z\-]+_\d", pid) else pid),
            )
        )
    return papers


def search(query: str, max_results: int = 5, client: httpx.Client | None = None) -> list[ArxivPaper]:
    params = {"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance"}
    return parse_feed(_get(API_URL, params, client).text)


def get_by_id(paper_id: str, client: httpx.Client | None = None) -> ArxivPaper | None:
    ident = paper_id.replace("_", "/", 1) if re.match(r"^[a-z\-]+_\d", paper_id) else paper_id
    results = parse_feed(_get(API_URL, {"id_list": ident}, client).text)
    return results[0] if results else None


def download_pdf(paper: ArxivPaper, client: httpx.Client | None = None) -> bytes:
    return _get(paper.pdf_url, client=client).content


def get_many(paper_ids: list[str], client: httpx.Client | None = None) -> list[ArxivPaper]:
    """Metadata for several papers in one request (arXiv returns them in id_list order)."""
    ids = [i.replace("_", "/", 1) if re.match(r"^[a-z\-]+_\d", i) else i for i in paper_ids]
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    return parse_feed(_get(API_URL, params, client).text)
