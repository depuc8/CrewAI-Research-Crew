"""Domain models shared by ingestion, storage and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A contiguous section of a paper. Pages are 1-indexed."""

    name: str
    text: str
    page_start: int
    page_end: int
    # (page, text) segments in reading order; lets the chunker keep page numbers.
    pages: list[tuple[int, str]] = Field(default_factory=list)


class PaperMeta(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    source: str = "upload"  # "upload" | "arxiv"
    url: str | None = None
    num_pages: int = 0


class ParsedPaper(BaseModel):
    meta: PaperMeta
    sections: list[Section]
    full_text: str


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    page_start: int
    page_end: int
    text: str

    def covers_page(self, page: int) -> bool:
        return self.page_start <= page <= self.page_end


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
