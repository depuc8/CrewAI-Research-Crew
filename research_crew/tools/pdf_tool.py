"""Parsed-PDF access as a CrewAI tool: paper outline and full section text."""

from __future__ import annotations

from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

MAX_SECTION_CHARS = 6000


class PDFParserInput(BaseModel):
    paper_id: str = Field(description="paper_id of an ingested paper.")
    section: str | None = Field(
        None, description="Section name (e.g. 'Results') to read in full. Omit for the paper outline."
    )


class PDFParserTool(BaseTool):
    name: str = "read_paper_structure"
    description: str = (
        "Read the parsed structure of an ingested PDF: title, authors, year, abstract and the list of "
        "sections with page ranges. Pass `section` to read one section's text (with its page range)."
    )
    args_schema: Type[BaseModel] = PDFParserInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: Any = Field(exclude=True)

    def _run(self, paper_id: str, section: str | None = None) -> str:
        paper = self.metadata.get_paper(paper_id)
        if paper is None:
            return f"Unknown paper_id '{paper_id}'."
        if section is None:
            m = paper.meta
            outline = "\n".join(f"- {s.name} (pages {s.page_start}-{s.page_end})" for s in paper.sections)
            return (
                f"{m.title}\nauthors: {', '.join(m.authors) or 'unknown'}\nyear: {m.year or 'unknown'}\n"
                f"pages: {m.num_pages}\n\nAbstract: {m.abstract or 'n/a'}\n\nSections:\n{outline}"
            )
        wanted = section.strip().lower()
        match = next((s for s in paper.sections if s.name.lower() == wanted), None) or next(
            (s for s in paper.sections if wanted in s.name.lower()), None
        )
        if match is None:
            return f"No section '{section}'. Available: {', '.join(s.name for s in paper.sections)}"
        text = match.text if len(match.text) <= MAX_SECTION_CHARS else match.text[:MAX_SECTION_CHARS] + " ..."
        return f"[{paper_id}, pages {match.page_start}-{match.page_end}] {match.name}\n{text}"
