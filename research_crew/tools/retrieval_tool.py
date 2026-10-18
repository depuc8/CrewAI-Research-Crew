"""Vector Search retrieval as a CrewAI tool."""

from __future__ import annotations

from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from research_crew.retrieval import Retriever


class VectorSearchInput(BaseModel):
    query: str = Field(description="What to look for, phrased as a short search query.")
    paper_id: str | None = Field(
        None, description="Restrict the search to this paper_id. Omit to search all papers in the review."
    )
    top_k: int = Field(6, ge=1, le=12, description="Number of chunks to return.")


class VectorSearchTool(BaseTool):
    name: str = "search_paper_chunks"
    description: str = (
        "Semantic search over the ingested papers (Vertex AI Vector Search). Returns the most relevant "
        "text chunks, each labelled [paper_id, pages] with its section. Use the paper_id and a page inside "
        "the shown range when citing. Only claims supported by returned chunks may be cited."
    )
    args_schema: Type[BaseModel] = VectorSearchInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any = Field(exclude=True)
    paper_ids: list[str] = Field(default_factory=list, exclude=True)

    def _run(self, query: str, paper_id: str | None = None, top_k: int = 6) -> str:
        retriever: Retriever = self.retriever
        scope = [paper_id] if paper_id else (self.paper_ids or None)
        if paper_id and self.paper_ids and paper_id not in self.paper_ids:
            return f"Unknown paper_id '{paper_id}'. Valid ids: {', '.join(self.paper_ids)}"
        hits = retriever.search(query, scope, top_k)
        if not hits:
            return "No relevant chunks found."
        blocks = []
        for h in hits:
            c = h.chunk
            pages = str(c.page_start) if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
            blocks.append(f"[{c.paper_id}, pages {pages}] section: {c.section} (score {h.score:.2f})\n{c.text}")
        return "\n\n---\n\n".join(blocks)
