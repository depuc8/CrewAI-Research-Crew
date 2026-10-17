from research_crew.models import PaperMeta
from research_crew.schemas import ComparisonRow, LiteratureReview, Theme, render_markdown, render_table

METAS = {
    "2005.11401": PaperMeta(
        paper_id="2005.11401", title="Retrieval-Augmented Generation", authors=["P. Lewis", "E. Perez"],
        year=2020, url="https://arxiv.org/abs/2005.11401",
    ),
    "2310.11511": PaperMeta(paper_id="2310.11511", title="Self-RAG", authors=[], year=None),
}


def _review(**kw) -> LiteratureReview:
    base = dict(
        title="Reducing Hallucination in RAG",
        introduction="Intro text [2005.11401, 1].",
        themes=[Theme(title="Retrieval", body="RAG helps [2005.11401, 3].")],
        comparison_table=[
            ComparisonRow(
                paper_id="2005.11401", method="RAG | seq2seq", dataset="NQ", key_result="44.5 EM",
                limitation="latency", pages=[5, 6],
            )
        ],
        gaps=["No long-context study."],
        future_directions=["Adaptive retrieval."],
        reference_ids=["2005.11401", "2310.11511", "ghost-id"],
    )
    return LiteratureReview(**{**base, **kw})


def test_table_rows_carry_citations_and_escape_pipes():
    table = render_table(_review().comparison_table)
    assert table.splitlines()[0].startswith("| Paper |")
    row = table.splitlines()[2]
    assert "[2005.11401, 5] [2005.11401, 6]" in row
    assert "RAG \\| seq2seq" in row


def test_markdown_has_all_sections_in_order():
    md = render_markdown(_review(), METAS)
    heads = [l for l in md.splitlines() if l.startswith("#")]
    assert heads == [
        "# Reducing Hallucination in RAG", "## Introduction", "## Themes", "### Retrieval",
        "## Comparison of Methods", "## Research Gaps", "## Future Directions", "## References",
    ]


def test_references_come_from_metadata_not_the_model():
    md = render_markdown(_review(), METAS)
    refs = md.split("## References")[1]
    assert "P. Lewis, E. Perez (2020). Retrieval-Augmented Generation." in refs
    assert "Unknown (n.d.). Self-RAG." in refs
    assert "ghost-id" not in refs  # unknown ids are dropped, never invented


def test_empty_table_renders_placeholder():
    assert "_No comparison available._" in render_markdown(_review(comparison_table=[]), METAS)
