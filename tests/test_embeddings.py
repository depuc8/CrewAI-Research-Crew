import math

from research_crew.ingestion.embeddings import HashingEmbedder


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_vectors_are_unit_length_and_deterministic():
    emb = HashingEmbedder(256)
    a = emb.embed_query("retrieval augmented generation reduces hallucination")
    b = emb.embed_query("retrieval augmented generation reduces hallucination")
    assert a == b
    assert math.isclose(_dot(a, a), 1.0, rel_tol=1e-9)
    assert len(a) == 256


def test_related_text_scores_higher_than_unrelated():
    emb = HashingEmbedder(512)
    query = emb.embed_query("methods that reduce hallucination in retrieval augmented generation")
    related, unrelated = emb.embed_documents(
        [
            "We reduce hallucination in retrieval augmented generation with verification.",
            "The protein folding energy landscape was sampled with molecular dynamics.",
        ]
    )
    assert _dot(query, related) > _dot(query, unrelated)


def test_empty_text_does_not_produce_nan():
    vec = HashingEmbedder(64).embed_query("")
    assert all(not math.isnan(v) for v in vec)


def test_tracks_token_usage():
    emb = HashingEmbedder(64)
    emb.embed_documents(["one two three", "four five"])
    assert emb.tokens_used == 5
