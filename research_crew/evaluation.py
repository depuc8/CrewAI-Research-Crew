"""Evaluation: retrieval recall, citation accuracy and time saved.

Retrieval is scored against hand-written ground truth (`eval/questions.json`): the key
papers a good review must draw on, and for each a finding identified by keywords that
must occur in the retrieved text.

* paper recall@K  - the question is searched across the whole corpus (key papers of all
                    questions act as each other's distractors); recall is the share of the
                    question's key papers that appear among the top-K chunks.
* finding recall  - the way the Reader agent retrieves (filtered to one paper, top-k chunks);
                    a finding counts as retrieved when one chunk contains all its keywords.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from research_crew.retrieval import Retriever
from research_crew.services import Services


class ExpectedFinding(BaseModel):
    text: str
    keywords: list[str]


class KeyPaper(BaseModel):
    arxiv_id: str
    title: str
    findings: list[ExpectedFinding]


class EvalQuestion(BaseModel):
    id: str
    question: str
    key_papers: list[KeyPaper]

    @property
    def paper_ids(self) -> list[str]:
        return [p.arxiv_id for p in self.key_papers]


def load_questions(path: Path) -> list[EvalQuestion]:
    return [EvalQuestion(**q) for q in json.loads(Path(path).read_text())]


def normalize(text: str) -> str:
    """Lowercase and treat hyphens/dashes as spaces so 'chain-of-thought' matches 'chain of thought'."""
    return re.sub(r"\s+", " ", re.sub(r"[\-‐-―]", " ", text.lower()))


def contains_keywords(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return all(normalize(k) in haystack for k in keywords)


def paper_recall(expected: list[str], retrieved: list[str]) -> float:
    return len(set(expected) & set(retrieved)) / len(expected) if expected else 0.0


class QuestionRetrieval(BaseModel):
    question_id: str
    paper_recall: float
    papers_found: list[str]
    papers_missed: list[str]
    finding_recall: float
    findings_missed: list[str] = Field(default_factory=list)


class RetrievalReport(BaseModel):
    top_k: int
    per_paper_k: int
    questions: list[QuestionRetrieval]

    @property
    def mean_paper_recall(self) -> float:
        return mean(q.paper_recall for q in self.questions) if self.questions else 0.0

    @property
    def mean_finding_recall(self) -> float:
        return mean(q.finding_recall for q in self.questions) if self.questions else 0.0


def evaluate_retrieval(
    services: Services, questions: list[EvalQuestion], top_k: int = 20, per_paper_k: int = 6
) -> RetrievalReport:
    results = []
    for q in questions:
        hits = Retriever(services).search(q.question, top_k=top_k)
        found = list(dict.fromkeys(h.chunk.paper_id for h in hits))

        recalled, missed = 0, []
        total = sum(len(p.findings) for p in q.key_papers)
        for paper in q.key_papers:
            chunks = Retriever(services).search(q.question, [paper.arxiv_id], per_paper_k)
            for finding in paper.findings:
                if any(contains_keywords(h.chunk.text, finding.keywords) for h in chunks):
                    recalled += 1
                else:
                    missed.append(f"{paper.arxiv_id}: {finding.text}")
        results.append(
            QuestionRetrieval(
                question_id=q.id,
                paper_recall=paper_recall(q.paper_ids, found),
                papers_found=[p for p in q.paper_ids if p in found],
                papers_missed=[p for p in q.paper_ids if p not in found],
                finding_recall=recalled / total if total else 0.0,
                findings_missed=missed,
            )
        )
    return RetrievalReport(top_k=top_k, per_paper_k=per_paper_k, questions=results)


def keyword_problems(questions: list[EvalQuestion], full_texts: dict[str, str]) -> list[str]:
    """Ground-truth sanity check: every finding's keywords must occur in its paper's full text."""
    problems = []
    for q in questions:
        for p in q.key_papers:
            text = full_texts.get(p.arxiv_id)
            if text is None:
                problems.append(f"{p.arxiv_id}: paper text not available")
                continue
            for f in p.findings:
                if not contains_keywords(text, f.keywords):
                    problems.append(f"{p.arxiv_id}: keywords {f.keywords} not found in paper")
    return problems


# --------------------------------------------------------------------------- full-review metrics


class ReviewScore(BaseModel):
    question_id: str
    job_status: str
    seconds: float = 0.0
    citation_accuracy_raw: float = 0.0  # valid citations / all citations the writer produced
    citations_total: int = 0
    claims_dropped: int = 0
    key_papers_cited: float = 0.0  # share of expected papers cited in the final review
    tokens: int = 0
    cost_usd: float = 0.0


def key_paper_coverage(expected: list[str], markdown: str) -> float:
    body = markdown.split("## References")[0]
    return len([p for p in expected if re.search(rf"\[{re.escape(p)},", body)]) / len(expected) if expected else 0.0


class TimeSaved(BaseModel):
    system_minutes: float
    manual_minutes: float | None
    saved_minutes: float | None = None
    saved_percent: float | None = None
    speedup: float | None = None


def time_saved(system_seconds: float, manual_minutes: float | None) -> TimeSaved:
    system_minutes = system_seconds / 60
    if not manual_minutes:
        return TimeSaved(system_minutes=system_minutes, manual_minutes=None)
    return TimeSaved(
        system_minutes=system_minutes,
        manual_minutes=manual_minutes,
        saved_minutes=manual_minutes - system_minutes,
        saved_percent=100 * (manual_minutes - system_minutes) / manual_minutes,
        speedup=manual_minutes / system_minutes if system_minutes else None,
    )
