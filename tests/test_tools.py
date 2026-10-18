from research_crew.ingestion import arxiv
from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.retrieval import Retriever
from research_crew.tools import ArxivSearchTool, PDFParserTool, VectorSearchTool


def _setup(services, two_papers):
    for pid, data in two_papers.items():
        ingest_pdf(services, data, paper_id=pid)
    retriever = Retriever(services)
    return retriever, VectorSearchTool(retriever=retriever, paper_ids=list(two_papers))


def test_vector_search_tool_formats_chunks_with_citation_labels(services, two_papers):
    retriever, tool = _setup(services, two_papers)
    out = tool.run(query="verifier checks claims against passages")
    assert out.startswith("[rag-paper, pages ")
    assert "section:" in out
    assert retriever.log.queries == 1  # the tool feeds the retrieval log


def test_vector_search_tool_scopes_to_paper_and_rejects_unknown(services, two_papers):
    _, tool = _setup(services, two_papers)
    out = tool.run(query="folding", paper_id="bio-paper")
    assert "rag-paper" not in out
    assert "Unknown paper_id" in tool.run(query="folding", paper_id="nope")


def test_vector_search_tool_schema_is_exposed_to_the_llm(services, two_papers):
    _, tool = _setup(services, two_papers)
    props = tool.args_schema.model_json_schema()["properties"]
    assert set(props) == {"query", "paper_id", "top_k"}


def test_pdf_tool_outline_and_section(services, two_papers):
    _setup(services, two_papers)
    tool = PDFParserTool(metadata=services.metadata)
    outline = tool.run(paper_id="rag-paper")
    assert "Self Verification" in outline and "Sections:" in outline and "- Method (pages" in outline
    section = tool.run(paper_id="rag-paper", section="results")
    assert section.startswith("[rag-paper, pages") and "thirty percent" in section
    assert "Available:" in tool.run(paper_id="rag-paper", section="Appendix Z")
    assert "Unknown paper_id" in tool.run(paper_id="zzz")


def test_arxiv_tool_lists_candidates(monkeypatch):
    monkeypatch.setattr(
        arxiv, "search",
        lambda q, max_results=5, client=None: [
            arxiv.ArxivPaper(paper_id="2310.11511", title="Self-RAG", authors=["A. Asai"], year=2023,
                             abstract="Adaptive retrieval.", url="u", pdf_url="p")
        ],
    )
    out = ArxivSearchTool().run(query="self reflective retrieval")
    assert "2310.11511 (2023) Self-RAG" in out
