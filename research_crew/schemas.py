"""Pydantic outputs for each agent (used as CrewAI `output_pydantic`) and the review renderer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_crew.models import PaperMeta


def cite(paper_id: str, page: int) -> str:
    return f"[{paper_id}, {page}]"


# --------------------------------------------------------------------------- Reader


class Statement(BaseModel):
    text: str = Field(description="One or two sentences, grounded in retrieved chunks. 'Not reported' if absent.")
    page: int | None = Field(None, description="Page of the retrieved chunk that supports the text.")


class PaperSummary(BaseModel):
    paper_id: str
    title: str
    problem: Statement
    method: Statement
    dataset: Statement
    results: Statement
    limitations: Statement


class ReaderOutput(BaseModel):
    summaries: list[PaperSummary]


# --------------------------------------------------------------------------- Analyst


class KeyFinding(BaseModel):
    paper_id: str
    finding: str
    page: int


class ComparisonRow(BaseModel):
    paper_id: str
    method: str
    dataset: str
    key_result: str
    limitation: str
    pages: list[int] = Field(default_factory=list, description="Pages supporting key_result.")


class Relation(BaseModel):
    statement: str = Field(description="What the papers agree / disagree on.")
    paper_ids: list[str]
    pages: list[int] = Field(default_factory=list, description="One supporting page per paper_id, same order.")


class AnalysisOutput(BaseModel):
    key_findings: list[KeyFinding]
    comparison_table: list[ComparisonRow]
    agreements: list[Relation]
    contradictions: list[Relation]
    gaps: list[str]


# --------------------------------------------------------------------------- Writer


class Theme(BaseModel):
    title: str
    body: str = Field(description="Paragraphs; every factual claim ends with [paper_id, page].")


class LiteratureReview(BaseModel):
    title: str
    introduction: str
    themes: list[Theme]
    comparison_table: list[ComparisonRow]
    gaps: list[str]
    future_directions: list[str]
    reference_ids: list[str] = Field(description="paper_ids of every paper cited in the review.")


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split()) or "-"


def render_table(rows: list[ComparisonRow]) -> str:
    if not rows:
        return ""
    lines = [
        "| Paper | Method | Dataset | Key result | Limitation |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        cites = " ".join(cite(r.paper_id, p) for p in r.pages)
        lines.append(
            f"| {_cell(r.paper_id)} | {_cell(r.method)} | {_cell(r.dataset)} | "
            f"{_cell(r.key_result)} {cites}".rstrip() + f" | {_cell(r.limitation)} |"
        )
    return "\n".join(lines)


def format_reference(meta: PaperMeta) -> str:
    authors = ", ".join(meta.authors[:6]) + (" et al." if len(meta.authors) > 6 else "") if meta.authors else "Unknown"
    year = meta.year or "n.d."
    link = f" {meta.url}" if meta.url else ""
    return f"- **[{meta.paper_id}]** {authors} ({year}). {meta.title}.{link}"


def render_markdown(review: LiteratureReview, metas: dict[str, PaperMeta]) -> str:
    """Render the review. References come from stored metadata, never from model output."""
    out = [f"# {review.title}", "", "## Introduction", "", review.introduction.strip(), "", "## Themes", ""]
    for theme in review.themes:
        out += [f"### {theme.title}", "", theme.body.strip(), ""]
    out += ["## Comparison of Methods", "", render_table(review.comparison_table) or "_No comparison available._", ""]
    out += ["## Research Gaps", ""] + [f"- {g.strip()}" for g in review.gaps] + [""]
    out += ["## Future Directions", ""] + [f"- {f.strip()}" for f in review.future_directions] + [""]
    out += ["## References", ""]
    ids = list(dict.fromkeys(review.reference_ids)) or list(metas)
    out += [format_reference(metas[i]) for i in ids if i in metas]
    return "\n".join(out).strip() + "\n"
