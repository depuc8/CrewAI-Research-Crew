from research_crew.citations import check_citations, extract_citations, parse_group, split_sentences
from research_crew.models import Chunk, RetrievedChunk
from research_crew.retrieval import RetrievalLog


def _log() -> RetrievalLog:
    """Retrieved: paper A pages 2-3 and 7, paper B page 1."""
    log = RetrievalLog()
    mk = lambda pid, a, b, n: RetrievedChunk(
        chunk=Chunk(chunk_id=f"{pid}:{n}", paper_id=pid, section="S", page_start=a, page_end=b, text="t"), score=1.0
    )
    log.record([mk("A", 2, 3, 0), mk("A", 7, 7, 1), mk("B", 1, 1, 0)])
    return log


def test_parse_group_variants():
    assert parse_group("2005.11401, 3") == [("2005.11401", 3, 3)]
    assert parse_group("up-ab12, p. 4") == [("up-ab12", 4, 4)]
    assert parse_group("A, 2; B, pp. 3-5") == [("A", 2, 2), ("B", 3, 5)]
    assert parse_group("1") is None and parse_group("see 4") is None and parse_group("A, 2; junk") is None


def test_extract_citations_ignores_non_citations():
    assert extract_citations("Claim [A, 2] and [B, 1]; ref [12] and [link text](x)") == [("A", 2), ("B", 1)]


def test_split_sentences_keeps_abbreviations_together():
    s = split_sentences("Lewis et al. show gains. Self-RAG adds critique, e.g. reflection tokens. Done.")
    assert s == ["Lewis et al. show gains.", "Self-RAG adds critique, e.g. reflection tokens.", "Done."]


def test_valid_citations_pass_through_unchanged():
    md = "## Themes\n\n### T\n\nRAG reduces errors [A, 2]. Critique helps [B, 1].\n"
    out, rep = check_citations(md, _log())
    assert out == md
    assert (rep.total_citations, rep.valid_citations, rep.citation_accuracy) == (2, 2, 1.0)


def test_page_outside_any_retrieved_chunk_is_invalid_and_claim_dropped():
    md = "## Themes\n\nRAG reduces errors [A, 2]. Another claim [A, 5]. Third claim [A, 7].\n"
    out, rep = check_citations(md, _log())
    assert "Another claim" not in out and "RAG reduces errors [A, 2]." in out and "[A, 7]" in out
    assert rep.invalid[0].reason == "no retrieved chunk covers this page"
    assert rep.dropped_claims == ["Another claim [A, 5]."]
    assert rep.citation_accuracy == 2 / 3


def test_chunk_page_range_boundaries():
    md = "## Themes\n\nEdge low [A, 2]. Edge high [A, 3]. Past it [A, 4].\n"
    out, _ = check_citations(md, _log())
    assert "Edge low" in out and "Edge high" in out and "Past it" not in out


def test_unknown_or_unretrieved_paper_is_invalid():
    md = "## Themes\n\nGhost paper claim [Z, 1]. Excluded paper claim [B, 1].\n"
    out, rep = check_citations(md, _log(), paper_ids={"A"})
    assert out.strip() == "## Themes"
    assert {i.reason for i in rep.invalid} == {"paper is not part of this review"}
    _, rep2 = check_citations("## Themes\n\nClaim [Z, 1].\n", _log())
    assert rep2.invalid[0].reason == "paper was never retrieved"


def test_mixed_group_keeps_only_valid_parts():
    out, rep = check_citations("## Themes\n\nBoth agree [A, 2; Z, 9].\n", _log())
    assert "Both agree [A, 2]." in out and "Z" not in out
    assert rep.valid_citations == 1 and len(rep.invalid) == 1 and not rep.dropped_claims


def test_flag_mode_marks_instead_of_dropping():
    out, rep = check_citations("## Themes\n\nBad claim [A, 99].\n", _log(), on_unsupported="flag")
    assert "Bad claim" in out and "unsupported citation" in out and "[A, 99]" not in out
    assert rep.dropped_claims


def test_bullets_and_table_rows_are_checked_as_units():
    md = (
        "## Research Gaps\n\n- Gap one [A, 2]\n- Gap two [A, 50]\n\n"
        "## Comparison of Methods\n\n| Paper | Result |\n|---|---|\n| A | 44 EM [A, 2] |\n| B | 50 EM [B, 40] |\n"
    )
    out, rep = check_citations(md, _log())
    assert "- Gap one [A, 2]" in out and "Gap two" not in out
    assert "| Paper | Result |" in out and "|---|---|" in out
    assert "44 EM" in out and "50 EM" not in out
    assert len(rep.dropped_claims) == 2


def test_uncited_factual_claims_in_themes_are_flagged_not_dropped():
    md = "## Themes\n\nRetrieval improves exact match by ten points on every benchmark.\n\n## Introduction\n\nThis review improves nothing yet 12 words are enough to look like a claim.\n"
    out, rep = check_citations(md, _log())
    assert "improves exact match by ten points on every benchmark. ⚠️[uncited]" in out
    assert "⚠" not in out.split("## Introduction")[1]  # intro is not policed
    assert len(rep.flagged_claims) == 1


def test_references_section_is_untouched():
    md = "## References\n\n- **[A]** Authors (2020). Title [A, 99].\n"
    out, rep = check_citations(md, _log())
    assert out == md and rep.total_citations == 0


def test_no_citations_means_zero_accuracy_not_a_crash():
    _, rep = check_citations("## Themes\n\nNothing cited.\n", _log())
    assert rep.citation_accuracy == 0.0
