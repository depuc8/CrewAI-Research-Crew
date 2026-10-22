"""The Reader -> Analyst -> Writer crew (CrewAI, sequential process, Gemini on Vertex AI)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from crewai import LLM, Agent, Crew, Process, Task

from research_crew.config import Settings
from research_crew.models import PaperMeta
from research_crew.retrieval import Retriever
from research_crew.schemas import AnalysisOutput, LiteratureReview, ReaderOutput
from research_crew.services import Services
from research_crew.tools import ArxivSearchTool, PDFParserTool, VectorSearchTool

CITATION_RULE = (
    "Cite as [paper_id, page] using the exact paper_id and a page number that lies inside the page range "
    "of a chunk returned by search_paper_chunks. Never cite a page you have not seen in a retrieved chunk."
)


def build_llm(settings: Settings) -> LLM:
    """Gemini via CrewAI's native Google provider. With no API key it uses Vertex AI + ADC."""
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        return LLM(
            model=f"gemini/{settings.gemini_model}",
            project=settings.google_cloud_project or None,
            location=settings.google_cloud_location,
            temperature=0.2,
        )
    return LLM(model=f"gemini/{settings.gemini_model}", temperature=0.2)


@dataclass
class ReviewCrew:
    crew: Crew
    reader_task: Task
    analyst_task: Task
    writer_task: Task
    tasks: list[Task] = field(default_factory=list)


def build_crew(
    services: Services,
    retriever: Retriever,
    papers: list[PaperMeta],
    question: str,
    llm_factory: Callable[[], LLM] | None = None,
    task_callback: Callable | None = None,
) -> ReviewCrew:
    # One LLM object per agent: CrewAI sums each agent's LLM usage, so sharing a single
    # instance across the three agents would count its tokens three times.
    make_llm = llm_factory or (lambda: build_llm(services.settings))
    paper_ids = [p.paper_id for p in papers]
    search = VectorSearchTool(retriever=retriever, paper_ids=paper_ids)
    structure = PDFParserTool(metadata=services.metadata)
    paper_list = "\n".join(f"- {p.paper_id}: {p.title} ({p.year or 'n.d.'})" for p in papers)

    reader = Agent(
        role="Research Paper Reader",
        goal="Extract an accurate, evidence-grounded structured summary of each paper relevant to the research question.",
        backstory=(
            "You are a meticulous research assistant. You only report what the retrieved text of a paper "
            "supports and say 'Not reported' when it does not."
        ),
        tools=[search, structure, ArxivSearchTool()],
        llm=make_llm(),
        max_iter=50,
        allow_delegation=False,
    )
    analyst = Agent(
        role="Research Analyst",
        goal="Compare methods and results across papers and identify agreements, contradictions and gaps.",
        backstory=(
            "You are a senior researcher who reads across a literature, separates robust findings from "
            "one-off claims, and is careful never to overstate what a paper shows."
        ),
        tools=[search],
        llm=make_llm(),
        max_iter=25,
        allow_delegation=False,
    )
    writer = Agent(
        role="Literature Review Writer",
        goal="Write a clear, well-structured literature review in which every claim is cited.",
        backstory=(
            "You are an academic writer. You never state a fact without a citation and you never invent "
            "papers, numbers or pages."
        ),
        tools=[search],
        llm=make_llm(),
        max_iter=25,
        allow_delegation=False,
    )

    reader_task = Task(
        description=(
            f"Research question: {question}\n\nPapers to read (use these exact paper_ids):\n{paper_list}\n\n"
            "For EACH paper, call search_paper_chunks with paper_id set to that paper and queries about the "
            "problem it addresses, its method, datasets/benchmarks, quantitative results, and limitations, "
            "all in relation to the research question. Then write a PaperSummary with the fields problem, "
            "method, dataset, results and limitations. Each field is a Statement: 1-2 sentences grounded in "
            "retrieved chunks, with `page` set to a page inside the range of the chunk that supports it. If "
            "the retrieved text does not say, use text 'Not reported' and page null. "
            "Include a summary for every paper listed."
        ),
        expected_output="A ReaderOutput with one PaperSummary per paper.",
        agent=reader,
        output_pydantic=ReaderOutput,
    )
    analyst_task = Task(
        description=(
            f"Research question: {question}\n\nUsing the paper summaries, produce:\n"
            "1. key_findings: the most important findings relevant to the question (paper_id, finding, page).\n"
            "2. comparison_table: one row per paper (paper_id, method, dataset, key_result, limitation, pages "
            "supporting key_result). Keep cells short.\n"
            "3. agreements: findings that several papers support (statement, paper_ids, one page per paper).\n"
            "4. contradictions: conflicting findings or results, same structure.\n"
            "5. gaps: open problems the papers leave unaddressed.\n"
            "Use search_paper_chunks to verify any claim you are unsure about. " + CITATION_RULE
        ),
        expected_output="An AnalysisOutput.",
        agent=analyst,
        context=[reader_task],
        output_pydantic=AnalysisOutput,
    )
    writer_task = Task(
        description=(
            f"Write a literature review answering: {question}\n\n"
            "Fields: title; introduction (scope and why the question matters); themes (3-5 themes, each a "
            "title and a body of 1-3 paragraphs organised by idea, not paper-by-paper); comparison_table "
            "(reuse the analyst's rows); gaps; future_directions; reference_ids (every paper cited). "
            "Every factual claim must end with a citation. " + CITATION_RULE + " Prefer pages from the "
            "reader and analyst evidence; call search_paper_chunks if you need to confirm a page. "
            "Do not write a references list; it is generated from metadata."
        ),
        expected_output="A LiteratureReview.",
        agent=writer,
        context=[reader_task, analyst_task],
        output_pydantic=LiteratureReview,
    )
    tasks = [reader_task, analyst_task, writer_task]
    crew = Crew(
        agents=[reader, analyst, writer],
        tasks=tasks,
        process=Process.sequential,
        task_callback=task_callback,
        verbose=False,
    )
    return ReviewCrew(crew, reader_task, analyst_task, writer_task, tasks)
