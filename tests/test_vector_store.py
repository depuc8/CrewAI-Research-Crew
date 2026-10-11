from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from research_crew.config import Settings
from research_crew.ingestion.embeddings import HashingEmbedder
from research_crew.models import Chunk
from research_crew.storage.vector_store import LocalVectorStore, VertexVectorStore


def _chunk(pid: str, n: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{pid}:{n:04d}", paper_id=pid, section="S", page_start=1, page_end=1, text=text)


@pytest.fixture
def corpus():
    emb = HashingEmbedder(256)
    chunks = [
        _chunk("a", 0, "dense retrieval reduces hallucination in generation"),
        _chunk("a", 1, "we train the encoder with contrastive loss"),
        _chunk("b", 0, "self reflection tokens critique retrieved passages and hallucination"),
        _chunk("b", 1, "protein folding molecular dynamics energy landscape"),
    ]
    return emb, chunks, emb.embed_documents([c.text for c in chunks])


def test_local_query_returns_best_match_first(corpus):
    emb, chunks, vecs = corpus
    store = LocalVectorStore()
    store.upsert(chunks, vecs)
    hits = store.query(emb.embed_query("hallucination in retrieval generation"), top_k=2)
    assert hits[0][0] == "a:0000"
    assert hits[0][1] >= hits[1][1]


def test_local_query_filters_by_paper(corpus):
    emb, chunks, vecs = corpus
    store = LocalVectorStore()
    store.upsert(chunks, vecs)
    hits = store.query(emb.embed_query("hallucination"), top_k=5, paper_ids=["b"])
    assert {cid.split(":")[0] for cid, _ in hits} == {"b"}
    assert store.query(emb.embed_query("hallucination"), 5, paper_ids=[]) == []


def test_local_upsert_replaces_and_delete_removes(corpus):
    emb, chunks, vecs = corpus
    store = LocalVectorStore()
    store.upsert(chunks, vecs)
    store.upsert(chunks[:1], vecs[:1])  # same id again: no duplicates
    assert len(store.query(vecs[0], 10)) == 4
    store.delete(["a:0000"])
    assert "a:0000" not in {cid for cid, _ in store.query(vecs[0], 10)}


def test_local_store_persists_to_disk(tmp_path, corpus):
    emb, chunks, vecs = corpus
    path = tmp_path / "v.jsonl"
    LocalVectorStore(path).upsert(chunks, vecs)
    reloaded = LocalVectorStore(path)
    assert reloaded.query(vecs[2], 1)[0][0] == "b:0000"


def test_empty_store_returns_nothing():
    assert LocalVectorStore().query([0.1, 0.2], 3) == []


def _vertex():
    index, endpoint = MagicMock(), MagicMock()
    settings = Settings(vector_deployed_index_id="dep1")
    return VertexVectorStore(settings, index=index, endpoint=endpoint), index, endpoint


def test_vertex_upsert_sends_paper_restricts_in_batches(corpus):
    _, chunks, vecs = corpus
    store, index, _ = _vertex()
    many = [chunks[0].model_copy(update={"chunk_id": f"a:{i}"}) for i in range(250)]
    store.upsert(many, [vecs[0]] * 250)
    sizes = [len(c.kwargs["datapoints"]) for c in index.upsert_datapoints.call_args_list]
    assert sizes == [100, 100, 50]
    point = index.upsert_datapoints.call_args_list[0].kwargs["datapoints"][0]
    assert point.datapoint_id == "a:0"
    assert point.restricts[0].namespace == "paper_id"
    assert list(point.restricts[0].allow_list) == ["a"]


def test_vertex_query_applies_filter_and_maps_neighbors():
    store, _, endpoint = _vertex()
    endpoint.find_neighbors.return_value = [
        [SimpleNamespace(id="a:0000", distance=0.9), SimpleNamespace(id="b:0001", distance=0.4)]
    ]
    hits = store.query([0.1, 0.2], top_k=2, paper_ids=["a", "b"])
    assert hits == [("a:0000", 0.9), ("b:0001", 0.4)]
    kwargs = endpoint.find_neighbors.call_args.kwargs
    assert kwargs["deployed_index_id"] == "dep1" and kwargs["num_neighbors"] == 2
    assert kwargs["filter"][0].name == "paper_id" and kwargs["filter"][0].allow_tokens == ["a", "b"]


def test_vertex_query_without_filter_and_with_empty_filter():
    store, _, endpoint = _vertex()
    endpoint.find_neighbors.return_value = [[]]
    assert store.query([0.1], 3) == []
    assert "filter" not in endpoint.find_neighbors.call_args.kwargs
    endpoint.find_neighbors.reset_mock()
    assert store.query([0.1], 3, paper_ids=[]) == []
    endpoint.find_neighbors.assert_not_called()
