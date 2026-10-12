from types import SimpleNamespace
from unittest.mock import MagicMock

from research_crew.config import Settings
from research_crew.ingestion.chunking import chunk_paper
from research_crew.ingestion.pdf_parser import parse_pdf
from research_crew.storage.metadata_store import GcpMetadataStore, LocalMetadataStore


def test_local_roundtrip(tmp_path, sample_pdf):
    paper = parse_pdf(sample_pdf, paper_id="p1")
    chunks = chunk_paper(paper, 100, 10)
    store = LocalMetadataStore(tmp_path)
    store.save_paper(paper, chunks, sample_pdf)

    assert store.get_meta("p1").title == paper.meta.title
    assert store.get_paper("p1").full_text == paper.full_text
    assert [m.paper_id for m in store.list_papers()] == ["p1"]
    assert store.chunk_ids_for("p1") == [c.chunk_id for c in chunks]
    got = store.get_chunks([chunks[0].chunk_id, "p1:9999", "nope:0001"])
    assert list(got) == [chunks[0].chunk_id]
    assert got[chunks[0].chunk_id].text == chunks[0].text
    assert (tmp_path / "papers" / "p1" / "paper.pdf").read_bytes() == sample_pdf


def test_local_missing_paper(tmp_path):
    store = LocalMetadataStore(tmp_path)
    assert store.get_meta("x") is None and store.get_paper("x") is None and store.list_papers() == []


def test_gcp_save_writes_gcs_and_firestore_in_batches(sample_pdf):
    paper = parse_pdf(sample_pdf, paper_id="p1")
    chunks = chunk_paper(paper, 20, 5) * 4  # > 1 chunk, duplicates fine for the mock
    bucket, db = MagicMock(), MagicMock()
    GcpMetadataStore(Settings(), bucket=bucket, firestore_client=db).save_paper(paper, chunks, sample_pdf)

    blobs = [c.args[0] for c in bucket.blob.call_args_list]
    assert blobs == ["papers/p1/paper.pdf", "papers/p1/parsed.json"]
    db.collection.return_value.document.assert_any_call("p1")
    assert db.batch.return_value.set.call_count == len(chunks)
    db.batch.return_value.commit.assert_called()


def test_gcp_get_chunks_skips_missing(sample_pdf):
    paper = parse_pdf(sample_pdf, paper_id="p1")
    chunk = chunk_paper(paper, 100, 10)[0]
    db = MagicMock()
    db.get_all.return_value = [
        SimpleNamespace(exists=True, id=chunk.chunk_id, to_dict=lambda: chunk.model_dump()),
        SimpleNamespace(exists=False, id="p1:9999", to_dict=lambda: None),
    ]
    store = GcpMetadataStore(Settings(), bucket=MagicMock(), firestore_client=db)
    assert list(store.get_chunks([chunk.chunk_id, "p1:9999"])) == [chunk.chunk_id]
