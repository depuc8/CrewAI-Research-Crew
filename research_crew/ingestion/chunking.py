"""Section-aware chunking with overlap.

Chunks never cross a section boundary, so a retrieved chunk always has a single
`section` label, and each chunk records the page range it spans so that citations of the
form [paper_id, page] can be verified against it later.
"""

from __future__ import annotations

import re

from research_crew.models import Chunk, ParsedPaper, Section

SKIP_SECTIONS = re.compile(r"^(references|bibliography|acknowledg\w*)$", re.IGNORECASE)


def chunk_paper(paper: ParsedPaper, chunk_words: int = 350, overlap_words: int = 60) -> list[Chunk]:
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("need chunk_words > 0 and 0 <= overlap_words < chunk_words")
    chunks: list[Chunk] = []
    for section in paper.sections:
        if SKIP_SECTIONS.match(section.name.strip()):
            continue
        for window, first, last in _windows(section, chunk_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.meta.paper_id}:{len(chunks):04d}",
                    paper_id=paper.meta.paper_id,
                    section=section.name,
                    page_start=first,
                    page_end=last,
                    text=" ".join(window),
                )
            )
    return chunks


def _windows(section: Section, size: int, overlap: int):
    words: list[str] = []
    pages: list[int] = []
    for page, text in section.pages or [(section.page_start, section.text)]:
        for w in text.split():
            words.append(w)
            pages.append(page)
    if not words:
        return
    step = size - overlap
    start = 0
    while True:
        end = min(start + size, len(words))
        yield words[start:end], pages[start], pages[end - 1]
        if end == len(words):
            return
        start += step
