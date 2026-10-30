"""Evaluation CLI.

  python -m eval.run_eval retrieval                 # free: arXiv + local embeddings/vector search
  python -m eval.run_eval retrieval --embeddings gcp
  python -m eval.run_eval reviews --confirm-spend   # runs full Gemini reviews (costs money)
  python -m eval.run_eval report                    # writes eval/results/summary.md

`retrieval` ingests the key papers of all questions into one corpus, so each question's
papers compete with the other questions' papers as distractors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from research_crew.config import Settings
from research_crew.evaluation import (
    ReviewScore, evaluate_retrieval, key_paper_coverage, keyword_problems, load_questions, time_saved,
)
from research_crew.ingestion import arxiv
from research_crew.ingestion.embeddings import get_embedder
from research_crew.ingestion.pipeline import ingest_pdf
from research_crew.jobs import Job, JobRequest, JobStatus, LocalJobStore
from research_crew.runner import run_review_job
from research_crew.services import Services
from research_crew.storage.metadata_store import LocalMetadataStore
from research_crew.storage.vector_store import LocalVectorStore

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PDF_CACHE = Path(".local/eval_pdfs")


def build_eval_services(embeddings: str, data_dir: Path) -> Services:
    settings = Settings(embedding_backend=embeddings, local_data_dir=data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return Services(settings, get_embedder(settings), LocalVectorStore(data_dir / "vectors.jsonl"),
                    LocalMetadataStore(data_dir))


def ingest_corpus(services: Services, questions) -> dict[str, str]:
    """Download (cached) and ingest every key paper. Returns {arxiv_id: full_text}."""
    ids = [p for q in questions for p in q.paper_ids]
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    missing = [i for i in ids if not (PDF_CACHE / f"{i}.pdf").exists()]
    meta = {p.paper_id: p for p in arxiv.get_many(ids)} if missing else {}
    texts = {}
    for i in ids:
        path = PDF_CACHE / f"{i}.pdf"
        if not path.exists():
            print(f"  downloading {i}", flush=True)
            path.write_bytes(arxiv.download_pdf(meta[i]))
        known = meta.get(i)
        ingest_pdf(services, path.read_bytes(), i, "arxiv", f"https://arxiv.org/abs/{i}", known=known)
        paper = services.metadata.get_paper(i)
        texts[i] = paper.full_text
    return texts


def cmd_retrieval(args) -> int:
    questions = load_questions(ROOT / "questions.json")
    services = build_eval_services(args.embeddings, Path(args.data_dir) / f"retrieval-{args.embeddings}")
    print(f"Ingesting {sum(len(q.key_papers) for q in questions)} papers ({args.embeddings} embeddings)…")
    texts = ingest_corpus(services, questions)

    problems = keyword_problems(questions, texts)
    for p in problems:
        print("GROUND-TRUTH PROBLEM:", p)

    report = evaluate_retrieval(services, questions, top_k=args.top_k, per_paper_k=args.per_paper_k)
    out = {
        "embeddings": args.embeddings,
        "vector_search": "exact cosine (local); Vertex AI Vector Search is approximate",
        "corpus_papers": sum(len(q.key_papers) for q in questions),
        "top_k": report.top_k, "per_paper_k": report.per_paper_k,
        "mean_paper_recall": report.mean_paper_recall, "mean_finding_recall": report.mean_finding_recall,
        "ground_truth_problems": problems,
        "questions": [q.model_dump() for q in report.questions],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"retrieval-{args.embeddings}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'question':28} paper_recall@{args.top_k}  finding_recall")
    for q in report.questions:
        print(f"{q.question_id:28} {q.paper_recall:>14.2f}  {q.finding_recall:>13.2f}")
    print(f"{'MEAN':28} {report.mean_paper_recall:>14.2f}  {report.mean_finding_recall:>13.2f}")
    return 1 if problems else 0


def cmd_reviews(args) -> int:
    if not args.confirm_spend:
        print("This runs full Gemini reviews on your GCP project and costs money. Re-run with --confirm-spend.")
        return 2
    questions = load_questions(ROOT / "questions.json")
    services = build_eval_services(args.embeddings, Path(args.data_dir) / f"reviews-{args.embeddings}")
    ingest_corpus(services, questions)
    store = LocalJobStore(Path(args.data_dir) / "eval-jobs")
    scores, reviews = [], {}
    for q in questions:
        job = Job(request=JobRequest(question=q.question, paper_ids=q.paper_ids))
        store.save(job)
        t0 = time.monotonic()
        done = run_review_job(job.job_id, services, store)
        seconds = time.monotonic() - t0
        if done.status != JobStatus.succeeded:
            scores.append(ReviewScore(question_id=q.id, job_status="failed", seconds=seconds))
            print(f"{q.id}: FAILED {done.error}")
            continue
        r, m = done.result, done.metrics
        scores.append(ReviewScore(
            question_id=q.id, job_status="succeeded", seconds=seconds,
            citation_accuracy_raw=r.citation_accuracy_raw,
            citations_total=len(r.citation_report.invalid) + r.citation_report.valid_citations,
            claims_dropped=len(r.citation_report.dropped_claims),
            key_papers_cited=key_paper_coverage(q.paper_ids, r.markdown),
            tokens=m.llm_total_tokens, cost_usd=m.cost_usd,
        ))
        reviews[q.id] = r.markdown
        print(f"{q.id}: {seconds:.0f}s, raw citation accuracy {r.citation_accuracy_raw:.0%}, ${m.cost_usd:.3f}")

    ok = [s for s in scores if s.job_status == "succeeded"]
    baseline = json.loads((ROOT / "baseline.json").read_text())
    mean_s = sum(s.seconds for s in ok) / len(ok) if ok else 0.0
    out = {
        "embeddings": args.embeddings,
        "scores": [s.model_dump() for s in scores],
        "mean_seconds": mean_s,
        "mean_citation_accuracy_raw": sum(s.citation_accuracy_raw for s in ok) / len(ok) if ok else 0.0,
        "time_saved": time_saved(mean_s, baseline["manual_minutes_per_review"]).model_dump(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reviews.json").write_text(json.dumps(out, indent=2))
    for qid, md in reviews.items():
        (RESULTS / f"review-{qid}.md").write_text(md)
    return 0


def cmd_report(_args) -> int:
    lines = ["# Evaluation results", ""]
    for f in sorted(RESULTS.glob("retrieval-*.json")):
        d = json.loads(f.read_text())
        lines += [
            f"## Retrieval ({d['embeddings']} embeddings, {d['corpus_papers']}-paper corpus)", "",
            f"Paper recall@{d['top_k']} chunks (corpus-wide search): **{d['mean_paper_recall']:.0%}**  ",
            f"Finding recall (per-paper search, top {d['per_paper_k']} chunks): **{d['mean_finding_recall']:.0%}**", "",
            "| Question | Paper recall | Finding recall | Missed papers |", "|---|---|---|---|",
        ]
        lines += [f"| {q['question_id']} | {q['paper_recall']:.0%} | {q['finding_recall']:.0%} | "
                  f"{', '.join(q['papers_missed']) or '-'} |" for q in d["questions"]]
        lines.append("")
    rv = RESULTS / "reviews.json"
    if rv.exists():
        d = json.loads(rv.read_text())
        ts = d["time_saved"]
        lines += ["## Full reviews", "", "| Question | Status | Time (s) | Raw citation accuracy | Claims dropped | Key papers cited | Cost (USD) |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {s['question_id']} | {s['job_status']} | {s['seconds']:.0f} | {s['citation_accuracy_raw']:.0%} | "
                  f"{s['claims_dropped']} | {s['key_papers_cited']:.0%} | {s['cost_usd']:.3f} |" for s in d["scores"]]
        lines += ["", f"Mean time per review: **{ts['system_minutes']:.1f} min**. "]
        if ts["manual_minutes"]:
            lines.append(f"Manual baseline: {ts['manual_minutes']:.0f} min → saved {ts['saved_percent']:.0f}% ({ts['speedup']:.1f}x).")
        else:
            lines.append("Manual baseline: not recorded yet (see eval/baseline.json), so no time-saved figure is claimed.")
    else:
        lines += ["## Full reviews", "", "_Not run yet: `python -m eval.run_eval reviews --confirm-spend`._"]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print((RESULTS / "summary.md").read_text())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("retrieval", cmd_retrieval), ("reviews", cmd_reviews)):
        s = sub.add_parser(name)
        s.add_argument("--embeddings", choices=["local", "gcp"], default="local")
        s.add_argument("--data-dir", default=".local/eval")
        s.add_argument("--top-k", type=int, default=20)
        s.add_argument("--per-paper-k", type=int, default=6)
        if name == "reviews":
            s.add_argument("--confirm-spend", action="store_true")
        s.set_defaults(fn=fn)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
