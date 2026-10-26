"""Streamlit UI: upload papers (or use arXiv), ask a question, read the cited review.

Run with: streamlit run frontend/app.py   (API_URL points at the backend)
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import ApiClient, ApiError

STAGES = {
    "queued": "Queued",
    "ingest": "Fetching and indexing papers",
    "crew": "Reader, Analyst and Writer agents working",
    "verify": "Verifying citations",
    "done": "Done",
}

st.set_page_config(page_title="Research Crew", page_icon="📚", layout="wide")
api = ApiClient()

st.title("📚 Research Crew")
st.caption("Ask a research question. Agents read the papers and draft a literature review with verified citations.")

question = st.text_area(
    "Research question",
    placeholder="What methods reduce hallucination in RAG systems?",
    height=80,
)
col_a, col_b = st.columns([3, 2])
with col_a:
    uploads = st.file_uploader("Upload PDFs (optional)", type="pdf", accept_multiple_files=True)
with col_b:
    use_arxiv = st.checkbox("Fetch papers from arXiv", value=not uploads, help="Always on when nothing is uploaded.")
    max_papers = st.slider("Max papers", 2, 12, 6)

if st.button("Write literature review", type="primary", disabled=len(question.strip()) < 5):
    try:
        paper_ids: list[str] = []
        if uploads:
            with st.spinner(f"Parsing and indexing {len(uploads)} PDF(s)…"):
                metas = api.upload([(f.name, f.getvalue()) for f in uploads])
            paper_ids = [m["paper_id"] for m in metas]
        st.session_state["job_id"] = api.start_review(question.strip(), paper_ids, use_arxiv or not uploads, max_papers)
    except ApiError as exc:
        st.error(str(exc))

job_id = st.session_state.get("job_id")
with st.sidebar:
    st.text_input("Job id", value=job_id or "", key="job_input", help="Paste an id to reopen a past review.")
    if st.session_state.get("job_input") and st.session_state["job_input"] != job_id:
        st.session_state["job_id"] = job_id = st.session_state["job_input"]

if job_id:
    try:
        with st.status("Working…", expanded=True) as box:
            final = api.wait(job_id, on_update=lambda s: box.update(label=STAGES.get(s["stage"], s["stage"])))
            box.update(label=STAGES.get(final["stage"], final["stage"]), state="complete" if final["status"] == "succeeded" else "error")
        if final["status"] == "failed":
            st.error(final.get("error") or "The job failed.")
        else:
            data = api.result(job_id)
            result, metrics = data["result"], data["metrics"] or {}
            tab_review, tab_checks, tab_usage = st.tabs(["Review", "Citation check", "Usage"])
            with tab_review:
                st.markdown(data["markdown"])
                st.download_button("Download Markdown", data["markdown"], file_name=f"review-{job_id[:8]}.md")
            with tab_checks:
                rep = result["citation_report"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Citations valid (after check)", f"{rep['valid_citations']}/{rep['total_citations']}")
                c2.metric("Writer's raw citation accuracy", f"{result['citation_accuracy_raw']:.0%}")
                c3.metric("Claims dropped", len(rep["dropped_claims"]))
                for item in rep["dropped_claims"]:
                    st.warning(f"Dropped (unsupported): {item}")
                for item in rep["flagged_claims"]:
                    st.info(f"Flagged (no citation): {item}")
            with tab_usage:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total time", f"{metrics.get('total_seconds', 0):.0f}s")
                c2.metric("LLM tokens", f"{metrics.get('llm_prompt_tokens', 0) + metrics.get('llm_completion_tokens', 0):,}")
                c3.metric("Estimated cost", f"${metrics.get('cost_usd', 0):.3f}")
                st.json(metrics.get("stage_seconds", {}))
    except ApiError as exc:
        st.error(str(exc))
