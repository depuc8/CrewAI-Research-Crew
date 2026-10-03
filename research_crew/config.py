"""Runtime settings, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Google Cloud
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    # Models
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # Backends. "local" runs everything on disk / in process with no GCP access.
    storage_backend: Literal["local", "gcp"] = "local"
    vector_backend: Literal["local", "gcp"] = "local"
    embedding_backend: Literal["local", "gcp"] = "local"
    job_backend: Literal["local", "gcp"] = "local"
    local_data_dir: Path = Path(".local")

    # GCP resources
    gcs_bucket: str = ""
    vector_index_id: str = ""
    vector_index_endpoint_id: str = ""
    vector_deployed_index_id: str = "research_crew_deployed"
    tasks_queue: str = "research-crew-jobs"
    tasks_service_account: str = ""
    service_url: str = ""

    # Ingestion
    chunk_words: int = 350
    chunk_overlap_words: int = 60

    # Review job limits
    max_papers_per_job: int = 8
    retrieval_top_k: int = 6

    # Cost model, USD per 1M tokens. These are configurable defaults for cost
    # *estimates*; set them to your model's current list price.
    price_input_per_mtok: float = 0.30
    price_output_per_mtok: float = 2.50
    price_embedding_per_mtok: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()
