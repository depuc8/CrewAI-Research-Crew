import pytest

from research_crew.ingestion.chunking import chunk_paper
from research_crew.ingestion.pdf_parser import parse_pdf
from research_crew.models import PaperMeta, ParsedPaper, Section


def _paper(sections: list[Section]) -> ParsedPaper:
    return ParsedPaper(meta=PaperMeta(paper_id="p1", title="T"), sections=sections, full_text="")


def _section(name: str, n_words: int, page: int = 1, prefix: str = "w") -> Section:
    text = " ".join(f"{prefix}{i}" for i in range(n_words))
    return Section(name=name, text=text, page_start=page, page_end=page, pages=[(page, text)])


def test_chunks_respect_size_and_overlap():
    chunks = chunk_paper(_paper([_section("Intro", 1000)]), chunk_words=300, overlap_words=50)
    sizes = [len(c.text.split()) for c in chunks]
    assert max(sizes) <= 300
    assert sizes[0] == 300
    # consecutive chunks share exactly `overlap` words
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-50:] == b.text.split()[:50] or len(b.text.split()) < 50
    # every word is covered and the last chunk ends at the final word
    assert chunks[-1].text.split()[-1] == "w999"


def test_chunks_never_cross_sections():
    chunks = chunk_paper(
        _paper([_section("Intro", 100, prefix="a"), _section("Method", 100, prefix="b")]),
        chunk_words=60,
        overlap_words=10,
    )
    for c in chunks:
        prefixes = {w[0] for w in c.text.split()}
        assert len(prefixes) == 1
        assert c.section == ("Intro" if "a" in prefixes else "Method")


def test_short_section_is_a_single_chunk():
    chunks = chunk_paper(_paper([_section("Abstract", 40)]), chunk_words=300, overlap_words=50)
    assert len(chunks) == 1


def test_references_are_not_indexed():
    chunks = chunk_paper(_paper([_section("Intro", 50), _section("References", 80)]))
    assert {c.section for c in chunks} == {"Intro"}


def test_chunk_ids_are_unique_and_prefixed_with_paper_id():
    chunks = chunk_paper(_paper([_section("A", 500), _section("B", 500)]), 200, 20)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("p1:") for i in ids)


def test_page_range_is_tracked_across_pages():
    section = Section(
        name="Results",
        text="",
        page_start=3,
        page_end=4,
        pages=[(3, " ".join(f"x{i}" for i in range(100))), (4, " ".join(f"y{i}" for i in range(100)))],
    )
    chunks = chunk_paper(_paper([section]), chunk_words=80, overlap_words=10)
    assert (chunks[0].page_start, chunks[0].page_end) == (3, 3)
    assert chunks[1].page_start == 3 and chunks[1].page_end == 4  # spans the page break
    assert (chunks[-1].page_start, chunks[-1].page_end) == (4, 4)
    assert chunks[1].covers_page(3) and chunks[1].covers_page(4) and not chunks[1].covers_page(5)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        chunk_paper(_paper([_section("A", 10)]), chunk_words=10, overlap_words=10)


def test_end_to_end_from_pdf(sample_pdf):
    chunks = chunk_paper(parse_pdf(sample_pdf, paper_id="pdf1"), chunk_words=120, overlap_words=20)
    assert chunks
    assert "References" not in {c.section for c in chunks}
    assert all(c.paper_id == "pdf1" and 1 <= c.page_start <= c.page_end for c in chunks)
