import pytest

from research_crew.ingestion import arxiv, pipeline
from research_crew.ingestion.pipeline import ingest_pdf


def test_ingest_stores_meta_chunks_and_vectors(services, sample_pdf):
    meta = ingest_pdf(services, sample_pdf, paper_id="p1")
    assert meta.paper_id == "p1" and meta.title == "Grounded Generation With Dense Retrieval"
    ids = services.metadata.chunk_ids_for("p1")
    assert ids and services.metadata.get_paper("p1").meta.num_pages == meta.num_pages
    hits = services.vectors.query(services.embedder.embed_query("dense retriever encoder"), 3)
    assert hits and all(cid in ids for cid, _ in hits)


def test_ingest_is_idempotent(services, sample_pdf):
    ingest_pdf(services, sample_pdf, paper_id="p1")
    before = services.metadata.chunk_ids_for("p1")
    ingest_pdf(services, sample_pdf, paper_id="p1")
    assert services.metadata.chunk_ids_for("p1") == before
    assert len(services.vectors.query([0.1] * services.embedder.dimensions, 1000)) == len(before)


def test_known_arxiv_metadata_overrides_parsed(services, sample_pdf):
    known = arxiv.ArxivPaper(
        paper_id="2005.11401", title="Official Title", authors=["P. Lewis"], year=2020,
        abstract="Official abstract.", url="http://arxiv.org/abs/2005.11401", pdf_url="x",
    )
    meta = ingest_pdf(services, sample_pdf, "2005.11401", "arxiv", known.url, known=known)
    assert (meta.title, meta.authors, meta.year, meta.abstract) == (
        "Official Title", ["P. Lewis"], 2020, "Official abstract."
    )
    assert meta.source == "arxiv"


def test_pdf_without_text_is_rejected(services):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(ValueError, match="no extractable text"):
        ingest_pdf(services, doc.tobytes(), paper_id="blank")


def test_fetch_for_question_ingests_search_results(services, two_papers, monkeypatch):
    found = [
        arxiv.ArxivPaper(paper_id=pid, title=pid, year=2024, url=f"u/{pid}", pdf_url=pid)
        for pid in two_papers
    ]
    monkeypatch.setattr(arxiv, "search", lambda q, max_results=5, client=None: found[:max_results])
    monkeypatch.setattr(arxiv, "download_pdf", lambda p, client=None: two_papers[p.paper_id])
    metas = pipeline.fetch_for_question(services, "hallucination in RAG", max_papers=2)
    assert sorted(m.paper_id for m in metas) == ["bio-paper", "rag-paper"]


def test_failed_download_is_skipped_not_fatal(services, two_papers, monkeypatch):
    papers = [
        arxiv.ArxivPaper(paper_id=pid, title=pid, url=f"u/{pid}", pdf_url=pid) for pid in ("bad", "rag-paper")
    ]

    def download(p, client=None):
        if p.paper_id == "bad":
            raise RuntimeError("404")
        return two_papers[p.paper_id]

    monkeypatch.setattr(arxiv, "download_pdf", download)
    metas = pipeline.ingest_arxiv_papers(services, papers)
    assert [m.paper_id for m in metas] == ["rag-paper"]
