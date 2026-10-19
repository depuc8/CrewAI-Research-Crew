"""Final verification step: every [paper_id, page] must point at a chunk that was retrieved.

The review is checked claim by claim (a sentence, bullet or table row is one claim):

* every citation valid        -> kept as is
* some citations invalid      -> the invalid ones are removed, the claim stays
* all citations invalid       -> the claim is dropped (or marked, with on_unsupported="flag")
* no citation, but it reads like a factual claim in the themes -> kept and marked "[uncited]"

Valid means: the paper was part of the job, and a chunk of that paper covering the cited
page was returned by retrieval during the job (see `RetrievalLog`).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from research_crew.retrieval import RetrievalLog

CITE_GROUP = re.compile(r"\[([^\[\]]*\d[^\[\]]*)\]")
CITE_PART = re.compile(r"^\s*(?P<id>[\w.\-]+)\s*,\s*(?:pp?\.\s*)?(?P<start>\d+)(?:\s*[-–]\s*(?P<end>\d+))?\s*$")
ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "vs.", "fig.", "eq.", "cf.", "approx.", "no.")
CLAIM_WORDS = re.compile(
    r"\d|\b(show|demonstrat|achiev|outperform|reduc|improv|propos|found|find|report|increas|decreas|"
    r"state-of-the-art|better|worse|higher|lower)\w*",
    re.IGNORECASE,
)
UNCITED_FLAG = " ⚠️[uncited]"
UNSUPPORTED_FLAG = "⚠️[unsupported citation]"


class CitationIssue(BaseModel):
    paper_id: str
    page: int
    claim: str
    reason: str


class CitationReport(BaseModel):
    total_citations: int = 0
    valid_citations: int = 0
    invalid: list[CitationIssue] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)
    flagged_claims: list[str] = Field(default_factory=list)

    @property
    def citation_accuracy(self) -> float:
        return self.valid_citations / self.total_citations if self.total_citations else 0.0


def parse_group(inner: str) -> list[tuple[str, int, int]] | None:
    """'a, 3; b, p. 4-5' -> [('a', 3, 3), ('b', 4, 5)]; None if it is not a citation."""
    parts = []
    for raw in inner.split(";"):
        m = CITE_PART.match(raw)
        if not m:
            return None
        start = int(m.group("start"))
        parts.append((m.group("id"), start, int(m.group("end") or start)))
    return parts


def extract_citations(text: str) -> list[tuple[str, int]]:
    """All (paper_id, first page) citations found in `text`."""
    out = []
    for m in CITE_GROUP.finditer(text):
        for pid, start, _ in parse_group(m.group(1)) or []:
            out.append((pid, start))
    return out


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text.strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and merged[-1].lower().endswith(ABBREVIATIONS):
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged


class _Checker:
    def __init__(self, log: RetrievalLog, allowed: set[str] | None, mode: str):
        self.log = log
        self.allowed = allowed
        self.mode = mode
        self.report = CitationReport()

    def _valid(self, pid: str, start: int, end: int) -> tuple[bool, str]:
        if self.allowed is not None and pid not in self.allowed:
            return False, "paper is not part of this review"
        if pid not in self.log.paper_ids():
            return False, "paper was never retrieved"
        if not self.log.supports(pid, start):
            return False, "no retrieved chunk covers this page"
        return True, ""

    def unit(self, text: str, claim_section: bool = False, table: bool = False) -> str | None:
        """Check one claim. Returns the (possibly rewritten) text, or None if it is dropped."""
        groups = [(m, parse_group(m.group(1))) for m in CITE_GROUP.finditer(text)]
        groups = [(m, parts) for m, parts in groups if parts]
        if not groups:
            if claim_section and CLAIM_WORDS.search(text) and len(text.split()) >= 6:
                self.report.flagged_claims.append(text.strip())
                return text if table else text.rstrip() + UNCITED_FLAG
            return text

        valid_total = invalid_total = 0
        rewritten = text
        for m, parts in reversed(groups):
            keep = []
            for pid, start, end in parts:
                ok, reason = self._valid(pid, start, end)
                self.report.total_citations += 1
                if ok:
                    self.report.valid_citations += 1
                    valid_total += 1
                    keep.append(f"{pid}, {start}" if start == end else f"{pid}, {start}-{end}")
                else:
                    invalid_total += 1
                    self.report.invalid.append(
                        CitationIssue(paper_id=pid, page=start, claim=text.strip()[:200], reason=reason)
                    )
            replacement = f"[{'; '.join(keep)}]" if keep else ""
            rewritten = rewritten[: m.start()] + replacement + rewritten[m.end() :]

        if valid_total == 0:
            self.report.dropped_claims.append(text.strip())
            if self.mode == "flag" and not table:
                return CITE_GROUP.sub("", text).rstrip() + " " + UNSUPPORTED_FLAG
            return None
        return re.sub(r"[ \t]{2,}", " ", rewritten).replace(" .", ".").replace(" ,", ",")


_BULLET = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")


def check_citations(
    markdown: str,
    log: RetrievalLog,
    paper_ids: set[str] | None = None,
    on_unsupported: Literal["drop", "flag"] = "drop",
) -> tuple[str, CitationReport]:
    """Verify the review's citations against the retrieval log; return (clean markdown, report)."""
    checker = _Checker(log, paper_ids, on_unsupported)
    lines = markdown.splitlines()
    out: list[str] = []
    section = ""
    in_references = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= 2:  # "###" theme headings stay inside their "## Themes" section
                section = stripped.lstrip("#").strip().lower()
                in_references = section.startswith("references")
            out.append(line)
            continue
        if in_references or not stripped:
            out.append(line)
            continue
        claim_section = section == "themes"
        if stripped.startswith("|"):
            is_sep = set(stripped.replace("|", "").strip()) <= set("-: ")
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            is_header = nxt.startswith("|") and set(nxt.replace("|", "").strip()) <= set("-: ")
            if is_sep or is_header:
                out.append(line)
                continue
            result = checker.unit(line, claim_section=False, table=True)
            if result is not None:
                out.append(result)
            continue
        if m := _BULLET.match(line):
            result = checker.unit(m.group(2), claim_section=claim_section)
            if result is not None:
                out.append(m.group(1) + result)
            continue
        kept = [r for s in split_sentences(line) if (r := checker.unit(s, claim_section=claim_section)) is not None]
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else ""), checker.report
