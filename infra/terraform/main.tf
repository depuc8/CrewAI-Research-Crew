data "google_project" "this" {}

locals {
  apis = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtasks.googleapis.com",
    "firestore.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

# --- Storage -----------------------------------------------------------------

resource "google_storage_bucket" "papers" {
  name                        = "${var.project_id}-${var.name_prefix}-papers"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.force_destroy_bucket

  depends_on = [google_project_service.apis]
}

resource "google_firestore_database" "default" {
  count       = var.create_firestore_database ? 1 : 0
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = var.name_prefix
  location      = var.region
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# --- Service accounts --------------------------------------------------------

# Runtime identity of the API / worker service.
resource "google_service_account" "api" {
  account_id   = "${var.name_prefix}-api"
  display_name = "Research Crew API and worker"
}

# Runtime identity of the Streamlit frontend; it may only invoke the API.
resource "google_service_account" "ui" {
  account_id   = "${var.name_prefix}-ui"
  display_name = "Research Crew frontend"
}

resource "google_project_iam_member" "api_roles" {
  for_each = toset([
    "roles/aiplatform.user",     # Gemini, embeddings, Vector Search queries and upserts
    "roles/datastore.user",      # Firestore: papers, chunks, jobs
    "roles/cloudtasks.enqueuer", # create tasks
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_bucket" {
  bucket = google_storage_bucket.papers.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.api.email}"
}

# Cloud Tasks signs its OIDC token as the API account, so that account must be able to act as itself.
resource "google_service_account_iam_member" "api_acts_as_self" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}

# --- Job queue ---------------------------------------------------------------

resource "google_cloud_tasks_queue" "jobs" {
  name     = "${var.name_prefix}-jobs"
  location = var.region

  rate_limits {
    max_concurrent_dispatches = 3 # reviews are LLM-heavy; bound the parallelism
    max_dispatches_per_second = 1
  }

  retry_config {
    max_attempts = 2 # a re-delivered job is a no-op once started, so retries are cheap and safe
    min_backoff  = "30s"
    max_backoff  = "300s"
  }

  depends_on = [google_project_service.apis]
}

# --- Vertex AI Vector Search ---------------------------------------------------

# Streaming index: chunks are upserted online as papers are ingested.
# contents_delta_uri is intentionally omitted so the index starts empty.
resource "google_vertex_ai_index" "chunks" {
  display_name        = "${var.name_prefix}-chunks"
  description         = "Paper chunk embeddings (restrict namespace: paper_id)"
  region              = var.region
  index_update_method = "STREAM_UPDATE"

  metadata {
    config {
      dimensions                  = var.embedding_dimensions
      approximate_neighbors_count = 150
      distance_measure_type       = "DOT_PRODUCT_DISTANCE" # vectors are L2-normalised in the app
      shard_size                  = "SHARD_SIZE_SMALL"

      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_vertex_ai_index_endpoint" "chunks" {
  display_name            = "${var.name_prefix}-endpoint"
  region                  = var.region
  public_endpoint_enabled = true # access is IAM-controlled; no VPC peering needed

  depends_on = [google_project_service.apis]
}

resource "google_vertex_ai_index_endpoint_deployed_index" "chunks" {
  deployed_index_id = var.deployed_index_id
  display_name      = "${var.name_prefix}-deployed"
  region            = var.region
  index             = google_vertex_ai_index.chunks.id
  index_endpoint    = google_vertex_ai_index_endpoint.chunks.id

  dedicated_resources {
    machine_spec {
      machine_type = var.index_machine_type
    }
    min_replica_count = 1
    max_replica_count = 1
  }
}
