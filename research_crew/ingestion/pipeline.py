"""Ingestion: PDF bytes (uploaded or fetched from arXiv) -> chunks -> embeddings -> stores."""

from __future__ import annotations

import logging

from research_crew.ingestion import arxiv
from research_crew.ingestion.chunking import chunk_paper
from research_crew.ingestion.pdf_parser import parse_pdf
from research_crew.models import PaperMeta
from research_crew.services import Services

log = logging.getLogger(__name__)


def ingest_pdf(
    services: Services,
    data: bytes,
    paper_id: str | None = None,
    source: str = "upload",
    url: str | None = None,
    known: arxiv.ArxivPaper | None = None,
    force: bool = False,
) -> PaperMeta:
    """Parse, chunk, embed and store one paper. Idempotent per paper_id unless `force`."""
    paper = parse_pdf(data, paper_id=paper_id, source_kind=source, url=url)
    if known:  # the arXiv API's metadata is more reliable than layout heuristics
        paper.meta = paper.meta.model_copy(
            update={
                "title": known.title,
                "authors": known.authors or paper.meta.authors,
                "year": known.year or paper.meta.year,
                "abstract": known.abstract or paper.meta.abstract,
            }
        )
    pid = paper.meta.paper_id
    if not force and services.metadata.chunk_ids_for(pid):
        log.info("paper %s already ingested; skipping", pid)
        return services.metadata.get_meta(pid) or paper.meta

    s = services.settings
    chunks = chunk_paper(paper, s.chunk_words, s.chunk_overlap_words)
    if not chunks:
        raise ValueError(f"no extractable text in {pid} (scanned PDF?)")
    # Prefix each chunk with the title/section so short chunks keep their context.
    inputs = [f"{paper.meta.title}\n{c.section}\n{c.text}" for c in chunks]
    vectors = services.embedder.embed_documents(inputs)
    services.metadata.save_paper(paper, chunks, data)
    services.vectors.upsert(chunks, vectors)
    log.info("ingested %s: %d pages, %d chunks", pid, paper.meta.num_pages, len(chunks))
    return paper.meta


def ingest_arxiv_papers(services: Services, papers: list[arxiv.ArxivPaper]) -> list[PaperMeta]:
    """Download and ingest the given arXiv papers; papers whose PDF fails are skipped."""
    metas = []
    for p in papers:
        try:
            if services.metadata.chunk_ids_for(p.paper_id):
                metas.append(services.metadata.get_meta(p.paper_id))
                continue
            data = arxiv.download_pdf(p)
            metas.append(ingest_pdf(services, data, p.paper_id, "arxiv", p.url, known=p))
        except Exception:  # one bad PDF must not sink the whole job
            log.exception("skipping arXiv paper %s", p.paper_id)
    return [m for m in metas if m]


def fetch_for_question(services: Services, question: str, max_papers: int) -> list[PaperMeta]:
    """arXiv mode: search for the question, then ingest the top results."""
    found = arxiv.search(arxiv.build_query(question), max_results=max_papers)
    if len(found) < max_papers:  # AND-query too strict; widen to the plain question text
        extra = arxiv.search(f"all:{question}", max_results=max_papers)
        found += [p for p in extra if p.paper_id not in {f.paper_id for f in found}]
    return ingest_arxiv_papers(services, found[:max_papers])
