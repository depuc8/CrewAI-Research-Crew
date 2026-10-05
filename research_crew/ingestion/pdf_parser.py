"""PDF -> structured paper (title, authors, year, abstract, sections with page numbers).

Layout heuristics rather than ML: headings are recognised from font size / weight and
numbering, which holds up well for typical two-column conference and arXiv PDFs. When a
heuristic fails the parser degrades gracefully (a single "Body" section) so ingestion
never crashes on an odd PDF.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pymupdf

from research_crew.models import PaperMeta, ParsedPaper, Section

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "experiments",
    "experimental setup", "evaluation", "results", "analysis", "discussion",
    "limitations", "conclusion", "conclusions", "future work", "references",
    "bibliography", "acknowledgments", "acknowledgements", "appendix",
}
NUMBERED_HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?|[IVX]{1,5}\.|[A-Z]\.)\s+(?P<name>[A-Z][^.!?]{1,90})$"
)
ABSTRACT_INLINE = re.compile(r"^abstract[\s.:—–-]+(?P<rest>\S.*)$", re.IGNORECASE)
AFFILIATION_WORDS = (
    "university", "institute", "department", "school", "laborator", "research",
    "google", "microsoft", "meta ", "openai", "college", "inc.", "corp", "@",
)
ARXIV_STAMP = re.compile(r"arXiv:\s*\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+(?P<year>(?:19|20)\d{2})")
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?|arXiv:\S+")
YEAR = re.compile(r"\b(?P<year>(?:19[5-9]\d|20[0-4]\d))\b")


@dataclass
class _Line:
    page: int
    text: str
    size: float
    bold: bool
    y: float
    page_height: float


def paper_id_for(data: bytes) -> str:
    return "up-" + hashlib.sha1(data).hexdigest()[:10]


def parse_pdf(
    source: bytes | str | Path,
    paper_id: str | None = None,
    source_kind: str = "upload",
    url: str | None = None,
) -> ParsedPaper:
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        lines = _extract_lines(doc)
        metadata = doc.metadata or {}
        num_pages = doc.page_count
    finally:
        doc.close()

    lines = _drop_furniture(lines, num_pages)
    body_size = _body_size(lines)
    title, title_end = _find_title(lines, body_size, metadata.get("title") or "")
    sections = _split_sections(lines[title_end:], body_size)
    sections = _fold_preamble(sections)

    abstract = next((s.text for s in sections if s.name.lower() == "abstract"), "")
    first_page = [ln for ln in lines if ln.page == 1]
    meta = PaperMeta(
        paper_id=paper_id or paper_id_for(data),
        title=title or "Untitled",
        authors=_find_authors(lines, title_end, body_size, title),
        year=_find_year(first_page, metadata),
        abstract=abstract,
        source=source_kind,
        url=url,
        num_pages=num_pages,
    )
    return ParsedPaper(meta=meta, sections=sections, full_text=_render_full_text(lines))


# --------------------------------------------------------------------------- extraction


def _extract_lines(doc: pymupdf.Document) -> list[_Line]:
    out: list[_Line] = []
    for page in doc:
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                if tuple(round(v) for v in line.get("dir", (1, 0))) != (1, 0):
                    continue  # rotated text, e.g. the arXiv margin stamp
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join("".join(s["text"] for s in spans).split())
                weights = Counter()
                for s in spans:
                    weights[round(s["size"], 1)] += len(s["text"])
                size = weights.most_common(1)[0][0]
                bold = sum(len(s["text"]) for s in spans if _is_bold(s)) > len(text) / 2
                out.append(_Line(page.number + 1, text, size, bold, line["bbox"][1], height))
    return out


def _is_bold(span: dict) -> bool:
    font = span.get("font", "").lower()
    return bool(span.get("flags", 0) & 16) or "bold" in font or "bx" in font


def _drop_furniture(lines: list[_Line], num_pages: int) -> list[_Line]:
    """Remove page numbers and running headers/footers."""
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower())

    edge_counts: Counter[str] = Counter()
    for ln in lines:
        if ln.y < ln.page_height * 0.07 or ln.y > ln.page_height * 0.92:
            edge_counts[norm(ln.text)] += 1
    repeated = {t for t, n in edge_counts.items() if num_pages >= 3 and n >= max(3, num_pages // 2)}
    kept = []
    for ln in lines:
        at_edge = ln.y < ln.page_height * 0.07 or ln.y > ln.page_height * 0.92
        if re.fullmatch(r"\d{1,4}", ln.text) and at_edge:
            continue
        if at_edge and norm(ln.text) in repeated:
            continue
        kept.append(ln)
    return kept


def _body_size(lines: list[_Line]) -> float:
    counts: Counter[float] = Counter()
    for ln in lines:
        counts[ln.size] += len(ln.text)
    return counts.most_common(1)[0][0] if counts else 10.0


# --------------------------------------------------------------------------- title/authors/year


def _find_title(lines: list[_Line], body: float, meta_title: str) -> tuple[str, int]:
    """Return (title, index of the first line after the title block)."""
    first = [(i, ln) for i, ln in enumerate(lines) if ln.page == 1]
    if not first:
        return meta_title.strip(), 0
    biggest = max(ln.size for _, ln in first[:40])
    if biggest >= body * 1.25:
        idx = [i for i, ln in first if abs(ln.size - biggest) < 0.6 and ln.y < ln.page_height * 0.5]
        if idx:
            start = idx[0]
            end = start
            for i in idx:
                if i - end <= 1:
                    end = i
                else:
                    break
            title = " ".join(lines[i].text for i in range(start, end + 1))
            return title.strip(), end + 1
    return meta_title.strip(), 0


def _find_authors(lines: list[_Line], title_end: int, body: float, title: str) -> list[str]:
    """Author names sit between the title and the abstract / first heading on page 1."""
    region: list[_Line] = []
    for ln in lines[title_end:]:
        if ln.page != 1 or len(region) >= 8:
            break
        if ABSTRACT_INLINE.match(ln.text) or _heading_name(ln, body, strict=True):
            break
        if ln.text not in title:
            region.append(ln)
    names: list[str] = []
    for ln in region:
        low = ln.text.lower()
        if any(w in low for w in AFFILIATION_WORDS) or re.search(r"\d{3,}", ln.text):
            continue
        for part in re.split(r",|;|\band\b|\u00b7|\u2217|\*|\u2020|\u2021", ln.text):
            name = re.sub(r"[\d\u00b9\u00b2\u00b3\u2070-\u2079]+", "", part).strip()
            words = name.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
                names.append(name)
    return list(dict.fromkeys(names))


def _find_year(first_page: list[_Line], metadata: dict) -> int | None:
    text = " ".join(ln.text for ln in first_page)
    if m := ARXIV_STAMP.search(text):
        return int(m.group("year"))
    cleaned = ARXIV_ID.sub(" ", text)
    now = datetime.now().year
    years = [int(m.group("year")) for m in YEAR.finditer(cleaned) if int(m.group("year")) <= now + 1]
    if years:
        return max(years)
    if m := re.match(r"D:(\d{4})", metadata.get("creationDate") or ""):
        return int(m.group(1))
    return None


# --------------------------------------------------------------------------- sections


def _heading_name(ln: _Line, body: float, strict: bool = False) -> str | None:
    """`strict` only accepts numbered / well-known headings (used before the body starts)."""
    text = ln.text.strip()
    if not text or len(text) > 100 or text.endswith((".", ",", ";")) and not NUMBERED_HEADING.match(text):
        return None
    numbered = NUMBERED_HEADING.match(text)
    bare = (numbered.group("name") if numbered else text).strip().rstrip(":")
    if bare.lower() in KNOWN_HEADINGS and len(text) <= 60:
        return bare
    emphasised = ln.bold or ln.size >= body + 0.8
    if numbered and emphasised:
        return bare
    if strict:
        return None
    if emphasised and ln.size >= body + 0.8 and len(text.split()) <= 8 and not text.endswith("."):
        return bare
    if text.isupper() and ln.bold and 2 < len(text) < 50:
        return text.title()
    return None


def _split_sections(lines: list[_Line], body: float) -> list[Section]:
    sections: list[Section] = []
    current: dict | None = None
    in_front_matter = True  # authors / affiliations: styled like headings but are not

    def flush() -> None:
        nonlocal current
        pages = (
            [(p, " ".join(words).strip()) for p, words in current["segments"]] if current else []
        )
        pages = [(p, t) for p, t in pages if t]
        if current and pages:
            sections.append(
                Section(
                    name=current["name"],
                    text=" ".join(t for _, t in pages),
                    page_start=pages[0][0],
                    page_end=pages[-1][0],
                    pages=pages,
                )
            )
        current = None

    def new(name: str, page: int) -> dict:
        return {"name": name, "segments": [(page, [])]}

    def add(text: str, page: int) -> None:
        nonlocal current
        if current is None:
            current = new("Preamble", page)
        if current["segments"][-1][0] != page:
            current["segments"].append((page, []))
        words = current["segments"][-1][1]
        if words and words[-1].endswith("-") and text[:1].islower():
            words[-1] = words[-1][:-1] + text
        else:
            words.append(text)

    for ln in lines:
        inline = ABSTRACT_INLINE.match(ln.text)
        if inline and len(ln.text) > 12:
            flush()
            in_front_matter = False
            current = new("Abstract", ln.page)
            add(inline.group("rest"), ln.page)
            continue
        name = _heading_name(ln, body, strict=in_front_matter)
        if name:
            in_front_matter = False
            flush()
            current = new(name, ln.page)
        else:
            add(ln.text, ln.page)
    flush()
    return sections


def _fold_preamble(sections: list[Section]) -> list[Section]:
    """Drop front matter (affiliations etc.) when real sections were found."""
    named = [s for s in sections if s.name != "Preamble"]
    if not named:
        return [s.model_copy(update={"name": "Body"}) for s in sections]
    return named


def _render_full_text(lines: list[_Line]) -> str:
    pages: dict[int, list[str]] = {}
    for ln in lines:
        pages.setdefault(ln.page, []).append(ln.text)
    return "\n\n".join(f"[Page {n}]\n" + "\n".join(texts) for n, texts in sorted(pages.items()))
