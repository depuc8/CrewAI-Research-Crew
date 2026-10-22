import pytest
from crewai import Process

from research_crew.citations import check_citations
from research_crew.crew import build_crew
from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.retrieval import Retriever
from research_crew.schemas import AnalysisOutput, LiteratureReview, ReaderOutput, render_markdown

from .fake_llm import FakeLLM


@pytest.fixture
def ingested(services, two_papers):
    metas = [ingest_pdf(services, data, paper_id=pid) for pid, data in two_papers.items() if pid == "rag-paper"]
    return metas


def test_crew_structure(services, ingested):
    rc = build_crew(services, Retriever(services), ingested, "What reduces hallucination?", llm_factory=FakeLLM)
    assert rc.crew.process == Process.sequential
    assert [a.role for a in rc.crew.agents] == [
        "Research Paper Reader", "Research Analyst", "Literature Review Writer",
    ]
    assert [t.output_pydantic for t in rc.tasks] == [ReaderOutput, AnalysisOutput, LiteratureReview]
    assert rc.analyst_task.context == [rc.reader_task]
    assert rc.writer_task.context == [rc.reader_task, rc.analyst_task]
    tool_names = {t.name for t in rc.reader_task.agent.tools}
    assert tool_names == {"search_paper_chunks", "read_paper_structure", "search_arxiv"}
    assert "rag-paper" in rc.reader_task.description


def test_crew_runs_end_to_end_and_citation_check_removes_fabricated_page(services, ingested):
    retriever = Retriever(services)
    rc = build_crew(services, retriever, ingested, "What reduces hallucination?", llm_factory=FakeLLM)
    result = rc.crew.kickoff()

    reader, analysis, review = (t.pydantic for t in result.tasks_output)
    assert isinstance(reader, ReaderOutput) and reader.summaries[0].paper_id == "rag-paper"
    assert isinstance(analysis, AnalysisOutput) and analysis.comparison_table
    assert isinstance(review, LiteratureReview)
    assert retriever.log.queries >= 1  # the reader really went through the retrieval tool

    md = render_markdown(review, {m.paper_id: m for m in ingested})
    assert "[rag-paper, 999]" in md  # the scripted writer fabricated this page
    clean, report = check_citations(md, retriever.log, {m.paper_id for m in ingested})
    assert "[rag-paper, 999]" not in clean and "halves latency" not in clean
    assert "A verifier reduces hallucination" in clean
    assert report.invalid and report.invalid[0].page == 999
    assert 0 < report.citation_accuracy < 1
