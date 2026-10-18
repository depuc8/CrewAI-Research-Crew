"""arXiv search as a CrewAI tool (read-only: lists candidate papers, ingests nothing)."""

from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_crew.ingestion import arxiv


class ArxivSearchInput(BaseModel):
    query: str = Field(description="Keywords, e.g. 'hallucination retrieval augmented generation'.")
    max_results: int = Field(5, ge=1, le=10)


class ArxivSearchTool(BaseTool):
    name: str = "search_arxiv"
    description: str = (
        "Search arXiv for papers. Returns paper_id, year, title, authors and the start of the abstract. "
        "Use it to look up metadata or related work; it does not add papers to the review."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        papers = arxiv.search(arxiv.build_query(query), max_results=max_results)
        if not papers:
            return "No papers found."
        return "\n\n".join(
            f"{p.paper_id} ({p.year}) {p.title}\n  authors: {', '.join(p.authors[:5])}\n  {p.abstract[:300]}..."
            for p in papers
        )
