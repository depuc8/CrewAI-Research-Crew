from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.retrieval import RetrievalLog, Retriever


def _ingest_both(services, two_papers):
    for pid, data in two_papers.items():
        ingest_pdf(services, data, paper_id=pid)


def test_search_finds_the_relevant_paper(services, two_papers):
    _ingest_both(services, two_papers)
    hits = Retriever(services).search("verification reduces hallucination in retrieval augmented generation")
    assert hits and hits[0].chunk.paper_id == "rag-paper"
    assert hits[0].score >= hits[-1].score


def test_search_can_be_restricted_to_papers(services, two_papers):
    _ingest_both(services, two_papers)
    hits = Retriever(services).search("hallucination", paper_ids=["bio-paper"])
    assert {h.chunk.paper_id for h in hits} == {"bio-paper"}


def test_hits_carry_section_and_pages(services, two_papers):
    _ingest_both(services, two_papers)
    hit = Retriever(services).search("protein folding intermediates microsecond")[0]
    assert hit.chunk.paper_id == "bio-paper"
    assert hit.chunk.section and 1 <= hit.chunk.page_start <= hit.chunk.page_end


def test_log_records_retrieved_chunks_for_citation_support(services, two_papers):
    _ingest_both(services, two_papers)
    retriever = Retriever(services, RetrievalLog())
    hit = retriever.search("verifier checks generated claims", paper_ids=["rag-paper"])[0]
    log = retriever.log
    assert log.queries == 1 and "rag-paper" in log.paper_ids()
    assert log.supports("rag-paper", hit.chunk.page_start)
    assert not log.supports("rag-paper", 999)
    assert not log.supports("bio-paper", hit.chunk.page_start)


def test_empty_index_returns_no_hits(services):
    assert Retriever(services).search("anything") == []
