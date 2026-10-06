from research_crew.ingestion.pdf_parser import paper_id_for, parse_pdf

from .conftest import make_pdf


def test_extracts_title_authors_year_abstract(sample_pdf):
    paper = parse_pdf(sample_pdf, paper_id="p1")
    assert paper.meta.paper_id == "p1"
    assert paper.meta.title == "Grounded Generation With Dense Retrieval"
    assert paper.meta.authors == ["Ada Lovelace", "Alan Turing", "Grace Hopper"]
    assert paper.meta.year == 2023
    assert paper.meta.abstract.startswith("We study how retrieval reduces hallucination")
    assert paper.meta.num_pages >= 2


def test_splits_sections_in_reading_order(sample_pdf):
    paper = parse_pdf(sample_pdf)
    names = [s.name for s in paper.sections]
    assert names == ["Abstract", "Introduction", "Method", "Experiments", "Conclusion", "References"]


def test_sections_carry_page_numbers(sample_pdf):
    paper = parse_pdf(sample_pdf)
    by_name = {s.name: s for s in paper.sections}
    assert by_name["Abstract"].page_start == 1
    for s in paper.sections:
        assert 1 <= s.page_start <= s.page_end <= paper.meta.num_pages
        assert s.pages and s.pages[0][0] == s.page_start
    # a long section spills over onto a later page, and its segments record that
    experiments = by_name["Experiments"]
    assert experiments.page_end > experiments.page_start or experiments.page_start > 1


def test_page_number_footers_are_dropped(sample_pdf):
    paper = parse_pdf(sample_pdf)
    assert "\n1\n" not in paper.full_text.replace("[Page 1]\n", "[Page 1]\n\n")
    assert "[Page 1]" in paper.full_text


def test_hyphenated_line_breaks_are_rejoined():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "1 Introduction", fontsize=13, fontname="hebo")
    page.insert_text((72, 130), "Retrieval makes the sys-", fontsize=10)
    page.insert_text((72, 144), "tem more factual overall.", fontsize=10)
    paper = parse_pdf(doc.tobytes())
    assert "makes the system more factual overall." in paper.sections[0].text


def test_paper_without_headings_falls_back_to_body():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(10):
        page.insert_text((72, 100 + i * 14), "plain running text with no headings at all", fontsize=10)
    paper = parse_pdf(doc.tobytes())
    assert [s.name for s in paper.sections] == ["Body"]


def test_paper_id_is_stable_content_hash(sample_pdf):
    assert paper_id_for(sample_pdf) == paper_id_for(sample_pdf)
    assert paper_id_for(sample_pdf).startswith("up-")
    assert paper_id_for(sample_pdf) != paper_id_for(sample_pdf + b"x")
