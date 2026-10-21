"""Per-job tokens, latency and cost, logged as one structured JSON line per job."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

from pydantic import BaseModel, Field

from research_crew.config import Settings

log = logging.getLogger("research_crew.metrics")


class CostBreakdown(BaseModel):
    llm_usd: float = 0.0
    embedding_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.llm_usd + self.embedding_usd


class JobMetrics(BaseModel):
    job_id: str = ""
    papers: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_requests: int = 0
    embedding_tokens: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    total_seconds: float = 0.0
    cost_usd: float = 0.0
    cost_breakdown: CostBreakdown = Field(default_factory=CostBreakdown)

    @property
    def llm_total_tokens(self) -> int:
        return self.llm_prompt_tokens + self.llm_completion_tokens


def estimate_cost(settings: Settings, prompt: int, completion: int, embedding: int) -> CostBreakdown:
    return CostBreakdown(
        llm_usd=prompt / 1e6 * settings.price_input_per_mtok + completion / 1e6 * settings.price_output_per_mtok,
        embedding_usd=embedding / 1e6 * settings.price_embedding_per_mtok,
    )


class MetricsRecorder:
    def __init__(self, job_id: str, settings: Settings):
        self._settings = settings
        self._started = time.monotonic()
        self.metrics = JobMetrics(job_id=job_id)

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.add_stage_time(name, time.monotonic() - t0)

    def add_stage_time(self, name: str, seconds: float) -> None:
        self.metrics.stage_seconds[name] = round(self.metrics.stage_seconds.get(name, 0.0) + seconds, 3)

    def set_llm_usage(self, usage) -> None:
        """`usage` is CrewAI's UsageMetrics (crew.usage_metrics / CrewOutput.token_usage)."""
        self.metrics.llm_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        self.metrics.llm_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.metrics.llm_requests = int(getattr(usage, "successful_requests", 0) or 0)

    def add_embedding_tokens(self, tokens: int) -> None:
        self.metrics.embedding_tokens += max(tokens, 0)

    def finish(self, papers: int) -> JobMetrics:
        m = self.metrics
        m.papers = papers
        m.total_seconds = round(time.monotonic() - self._started, 3)
        m.cost_breakdown = estimate_cost(
            self._settings, m.llm_prompt_tokens, m.llm_completion_tokens, m.embedding_tokens
        )
        m.cost_usd = round(m.cost_breakdown.total_usd, 6)
        log.info("job_metrics %s", json.dumps(m.model_dump(), sort_keys=True))
        return m
