import json
import logging
from types import SimpleNamespace

import pytest

from research_crew.config import Settings
from research_crew.usage import MetricsRecorder, estimate_cost


def test_estimate_cost_uses_per_million_prices():
    s = Settings(price_input_per_mtok=1.0, price_output_per_mtok=4.0, price_embedding_per_mtok=0.5)
    c = estimate_cost(s, prompt=2_000_000, completion=500_000, embedding=1_000_000)
    assert c.llm_usd == pytest.approx(2.0 + 2.0)
    assert c.embedding_usd == pytest.approx(0.5)
    assert c.total_usd == pytest.approx(4.5)


def test_recorder_collects_stages_tokens_and_cost(caplog):
    s = Settings(price_input_per_mtok=1.0, price_output_per_mtok=2.0, price_embedding_per_mtok=1.0)
    rec = MetricsRecorder("job-1", s)
    with rec.stage("ingest"):
        pass
    rec.add_stage_time("crew", 1.5)
    rec.add_stage_time("crew", 0.5)
    rec.set_llm_usage(SimpleNamespace(prompt_tokens=1000, completion_tokens=500, successful_requests=7))
    rec.add_embedding_tokens(2000)
    with caplog.at_level(logging.INFO, logger="research_crew.metrics"):
        m = rec.finish(papers=3)

    assert m.stage_seconds["crew"] == 2.0 and "ingest" in m.stage_seconds
    assert (m.llm_prompt_tokens, m.llm_completion_tokens, m.llm_total_tokens, m.llm_requests) == (1000, 500, 1500, 7)
    assert m.cost_usd == pytest.approx(0.001 + 0.001 + 0.002)
    assert m.papers == 3 and m.total_seconds >= 0

    line = next(r.message for r in caplog.records if r.message.startswith("job_metrics "))
    logged = json.loads(line.removeprefix("job_metrics "))
    assert logged["job_id"] == "job-1" and logged["llm_prompt_tokens"] == 1000


def test_stage_time_recorded_even_if_stage_raises():
    rec = MetricsRecorder("j", Settings())
    with pytest.raises(RuntimeError):
        with rec.stage("crew"):
            raise RuntimeError("boom")
    assert "crew" in rec.metrics.stage_seconds
