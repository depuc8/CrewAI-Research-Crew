"""Wiring: builds the embedder + stores for the configured backends."""

from __future__ import annotations

from dataclasses import dataclass

from research_crew.config import Settings, get_settings
from research_crew.ingestion.embeddings import Embedder, get_embedder
from research_crew.storage.metadata_store import MetadataStore, get_metadata_store
from research_crew.storage.vector_store import VectorStore, get_vector_store


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    vectors: VectorStore
    metadata: MetadataStore


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    return Services(
        settings=settings,
        embedder=get_embedder(settings),
        vectors=get_vector_store(settings),
        metadata=get_metadata_store(settings),
    )
