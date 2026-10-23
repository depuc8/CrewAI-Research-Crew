"""A scripted LLM so the real CrewAI loop (tools, context passing, pydantic output) runs offline.

It plays all three agents. The reader first calls the search tool, then summarises from the
observation; the writer cites one page that was retrieved and one that was not, so tests can
check the citation step removes the fabricated one.
"""

from __future__ import annotations

import json
import re

from crewai import BaseLLM

PAGE_IN_CHUNK = re.compile(r"\[(?P<pid>[^\],]+), pages (?P<a>\d+)(?:-(?P<b>\d+))?\]")


def _text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content", "")) for m in messages)


class FakeLLM(BaseLLM):
    bad_page: int = 999

    def __init__(self, **kw):
        super().__init__(model="fake-llm", **kw)
        self.calls = 0

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 100_000

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.calls += 1
        text = _text(messages)
        self._track_token_usage_internal({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        role = (from_agent.role if from_agent is not None else "") or ""
        if "Reader" in role:
            return self._reader(text)
        if "Analyst" in role:
            return self._analyst(text)
        return self._writer(text)

    @staticmethod
    def _final(payload: dict) -> str:
        return "Thought: I now know the final answer\nFinal Answer: " + json.dumps(payload)

    def _reader(self, text: str) -> str:
        m = PAGE_IN_CHUNK.search(text.split("Observation:")[-1]) if "Observation:" in text else None
        if not m:
            pid = re.search(r"^- ([^:\n]+):", text, re.M).group(1)
            return (
                "Thought: I should search the paper\nAction: search_paper_chunks\n"
                f'Action Input: {{"query": "method and results", "paper_id": "{pid}"}}'
            )
        pids = re.findall(r"^- ([^:\n]+): (.+?) \(", text, re.M)
        page = int(m.group("a"))
        st = lambda t: {"text": t, "page": page}
        return self._final({"summaries": [
            {"paper_id": pid, "title": title, "problem": st("Hallucination."), "method": st("Verification."),
             "dataset": st("Natural Questions."), "results": st("Lower hallucination."),
             "limitations": {"text": "Not reported", "page": None}}
            for pid, title in pids
        ]})

    def _analyst(self, text: str) -> str:
        pairs = re.findall(r'"paper_id":\s*"([^"]+)".*?"page":\s*(\d+)', text, re.S)
        pid, page = pairs[0][0], int(pairs[0][1])
        return self._final({
            "key_findings": [{"paper_id": pid, "finding": "Verification lowers hallucination.", "page": page}],
            "comparison_table": [{"paper_id": pid, "method": "Verifier", "dataset": "NQ",
                                  "key_result": "30% fewer hallucinations", "limitation": "latency", "pages": [page]}],
            "agreements": [], "contradictions": [], "gaps": ["No long-context evaluation."],
        })

    def _writer(self, text: str) -> str:
        pairs = re.findall(r'"paper_id":\s*"([^"]+)".*?"page":\s*(\d+)', text, re.S)
        pid, page = pairs[0][0], int(pairs[0][1])
        return self._final({
            "title": "Reducing Hallucination in RAG",
            "introduction": f"This review covers verification for retrieval augmented generation [{pid}, {page}].",
            "themes": [{"title": "Verification", "body": (
                f"A verifier reduces hallucination [{pid}, {page}]. "
                f"It also halves latency overhead [{pid}, {self.bad_page}]."
            )}],
            "comparison_table": [{"paper_id": pid, "method": "Verifier", "dataset": "NQ",
                                  "key_result": "30% fewer hallucinations", "limitation": "latency", "pages": [page]}],
            "gaps": [f"Long-context settings are unexplored [{pid}, {page}]."],
            "future_directions": ["Adaptive verification."],
            "reference_ids": [pid],
        })
