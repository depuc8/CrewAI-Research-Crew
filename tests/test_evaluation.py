from pathlib import Path

import pytest

from research_crew.evaluation import (
    EvalQuestion, contains_keywords, evaluate_retrieval, key_paper_coverage, keyword_problems,
    load_questions, normalize, paper_recall, time_saved,
)
from research_crew.ingestion.pipeline import ingest_pdf

QUESTIONS_FILE = Path(__file__).parents[1] / "eval" / "questions.json"


def _q(paper_ids_findings) -> EvalQuestion:
    return EvalQuestion(
        id="q", question="Does verification reduce hallucination in retrieval augmented generation?",
        key_papers=[
            {"arxiv_id": pid, "title": pid, "findings": [{"text": f"finding of {pid}", "keywords": kw}]}
            for pid, kw in paper_ids_findings
        ],
    )


def test_ground_truth_file_is_well_formed():
    qs = load_questions(QUESTIONS_FILE)
    assert len(qs) == 5
    assert len({q.id for q in qs}) == 5
    all_ids = [p for q in qs for p in q.paper_ids]
    assert len(all_ids) == len(set(all_ids))  # a paper is a key paper of exactly one question
    for q in qs:
        assert 3 <= len(q.key_papers) <= 5
        for p in q.key_papers:
            assert p.findings and all(f.keywords for f in p.findings)


def test_normalize_and_keyword_matching():
    assert normalize("Chain-of-Thought  Prompting") == "chain of thought prompting"
    assert contains_keywords("We use chain-of-thought prompting.", ["chain of thought"])
    assert contains_keywords("Top-20 accuracy vs BM25", ["bm25", "top-20"])
    assert not contains_keywords("only BM25 here", ["bm25", "top-20"])


def test_paper_recall():
    assert paper_recall(["a", "b", "c", "d"], ["a", "c", "x"]) == 0.5
    assert paper_recall([], ["a"]) == 0.0


def test_evaluate_retrieval_on_known_corpus(services, two_papers):
    for pid, data in two_papers.items():
        ingest_pdf(services, data, paper_id=pid)
    q = _q([("rag-paper", ["verification", "hallucination"]), ("bio-paper", ["protein folding"])])
    report = evaluate_retrieval(services, [q], top_k=50, per_paper_k=3)  # k covers the whole tiny corpus
    r = report.questions[0]
    assert r.papers_found == ["rag-paper", "bio-paper"] and r.paper_recall == 1.0
    assert r.finding_recall == 1.0 and report.mean_paper_recall == 1.0


def test_missing_paper_and_unmatched_keywords_are_reported(services, two_papers):
    ingest_pdf(services, two_papers["rag-paper"], paper_id="rag-paper")
    q = _q([("rag-paper", ["quantum chromodynamics"]), ("absent-paper", ["anything"])])
    r = evaluate_retrieval(services, [q], top_k=5, per_paper_k=3).questions[0]
    assert r.papers_missed == ["absent-paper"] and r.paper_recall == 0.5
    assert r.finding_recall == 0.0 and len(r.findings_missed) == 2


def test_keyword_problems_flags_unverifiable_ground_truth():
    q = _q([("p1", ["alpha"]), ("p2", ["beta"]), ("p3", ["gamma"])])
    problems = keyword_problems([q], {"p1": "has ALPHA in it", "p2": "nothing relevant"})
    assert problems == ["p2: keywords ['beta'] not found in paper", "p3: paper text not available"]


def test_key_paper_coverage_counts_only_body_citations():
    md = "## Themes\n\nA [p1, 3]. B [p2, 1].\n\n## References\n\n- [p3, 9]\n"
    assert key_paper_coverage(["p1", "p2", "p3", "p4"], md) == 0.5


def test_time_saved_math_and_missing_baseline():
    t = time_saved(system_seconds=300, manual_minutes=120)
    assert t.system_minutes == 5 and t.saved_minutes == 115
    assert t.saved_percent == pytest.approx(95.83, abs=0.01) and t.speedup == 24
    none = time_saved(300, None)
    assert none.manual_minutes is None and none.saved_percent is None
