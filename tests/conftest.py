"""Shared fixtures: synthetic PDFs built on the fly (no binary fixtures in the repo)."""

from __future__ import annotations

import textwrap

import pymupdf
import pytest

LOREM = (
    "Retrieval augmented generation grounds language model outputs in retrieved passages, "
    "which reduces hallucination on knowledge intensive tasks. "
)


def make_pdf(
    title: str,
    authors: str,
    abstract: str,
    sections: list[tuple[str, str]],
    footer: bool = True,
) -> bytes:
    """Render a small paper: title (18pt), authors, abstract, bold 13pt headings, 10pt body."""
    doc = pymupdf.open()
    state = {"page": None, "y": 0.0}

    def new_page() -> None:
        state["page"] = doc.new_page(width=595, height=842)
        state["y"] = 72.0
        if footer:
            state["page"].insert_text((72, 820), f"{len(doc)}", fontsize=9)

    def write(text: str, size: float, bold: bool = False, gap: float = 4.0, width: int = 95) -> None:
        font = "hebo" if bold else "helv"
        for piece in textwrap.wrap(text, width=int(width * 10 / size)) or [""]:
            if state["y"] > 760:
                new_page()
            state["page"].insert_text((72, state["y"]), piece, fontsize=size, fontname=font)
            state["y"] += size * 1.35
        state["y"] += gap

    new_page()
    write(title, 18, bold=True, gap=8, width=60)
    write(authors, 11, gap=2)
    write("Department of Computer Science, Example University", 9, gap=10)
    write("Abstract", 13, bold=True)
    write(abstract, 10, gap=8)
    for heading, body in sections:
        write(heading, 13, bold=True, gap=2)
        write(body, 10, gap=8)
    return doc.tobytes()


@pytest.fixture
def sample_sections() -> list[tuple[str, str]]:
    return [
        ("1 Introduction", LOREM * 12),
        ("2 Method", "We propose a retriever that scores passages with a dense encoder. " * 30),
        ("3 Experiments", "We evaluate on Natural Questions and HotpotQA with exact match. " * 40),
        ("4 Conclusion", "Retrieval reduces hallucination but adds latency. " * 10),
        ("References", "[1] Lewis et al. Retrieval-augmented generation. 2020."),
    ]


@pytest.fixture
def sample_pdf(sample_sections) -> bytes:
    return make_pdf(
        title="Grounded Generation With Dense Retrieval",
        authors="Ada Lovelace, Alan Turing and Grace Hopper",
        abstract="We study how retrieval reduces hallucination in language models. " * 4
        + "Published 2023.",
        sections=sample_sections,
    )
