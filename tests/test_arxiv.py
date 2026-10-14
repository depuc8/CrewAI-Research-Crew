from unittest.mock import MagicMock

from research_crew.ingestion import arxiv

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:58:06Z</published>
    <title>Retrieval-Augmented Generation
      for Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained
      language models store knowledge.  </summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v2</id>
    <published>1999-01-04T00:00:00Z</published>
    <title>An Old-Style Identifier</title>
    <summary>Old.</summary>
    <author><name>A. Physicist</name></author>
  </entry>
  <entry><id>http://arxiv.org/api/errors#bad_request</id><title>Error</title></entry>
</feed>"""


def test_parse_feed_extracts_metadata_and_skips_errors():
    papers = arxiv.parse_feed(FEED)
    assert [p.paper_id for p in papers] == ["2005.11401", "hep-th_9901001"]
    first = papers[0]
    assert first.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert first.abstract == "Large pre-trained language models store knowledge."
    assert first.authors == ["Patrick Lewis", "Ethan Perez"]
    assert first.year == 2020
    assert first.pdf_url == "https://arxiv.org/pdf/2005.11401"
    assert papers[1].pdf_url == "https://arxiv.org/pdf/hep-th/9901001"


def test_normalize_id_strips_version_and_slashes():
    assert arxiv.normalize_id("http://arxiv.org/abs/2310.11511v1") == "2310.11511"
    assert arxiv.normalize_id("http://arxiv.org/abs/cs/0112017v1") == "cs_0112017"
    assert arxiv.normalize_id("2401.15884v2") == "2401.15884"


def test_build_query_drops_stopwords_and_caps_terms():
    q = arxiv.build_query("What methods reduce hallucination in RAG systems?")
    assert q == "all:hallucination AND all:rag AND all:systems"
    long_q = arxiv.build_query("alpha beta gamma delta epsilon zeta eta theta", max_terms=3)
    assert long_q.count("all:") == 3


def test_search_passes_params_and_parses(monkeypatch):
    monkeypatch.setattr(arxiv, "_throttle", lambda: None)
    client = MagicMock()
    client.get.return_value = MagicMock(status_code=200, text=FEED, raise_for_status=lambda: None)
    papers = arxiv.search("all:rag", max_results=2, client=client)
    assert len(papers) == 2
    params = client.get.call_args.kwargs["params"]
    assert params["search_query"] == "all:rag" and params["max_results"] == 2


def test_get_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(arxiv, "_throttle", lambda: None)
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    limited = MagicMock(status_code=429)
    ok = MagicMock(status_code=200, text=FEED, raise_for_status=lambda: None)
    client = MagicMock()
    client.get.side_effect = [limited, ok]
    assert len(arxiv.search("all:rag", client=client)) == 2
    assert client.get.call_count == 2
