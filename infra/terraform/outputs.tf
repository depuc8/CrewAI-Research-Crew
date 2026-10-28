output "project_number" {
  value = data.google_project.this.number
}

output "bucket_name" {
  value = google_storage_bucket.papers.name
}

output "artifact_repository" {
  description = "Docker repository for the image (…/<project>/<repo>)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "vector_index_id" {
  value = google_vertex_ai_index.chunks.id
}

output "vector_index_endpoint_id" {
  value = google_vertex_ai_index_endpoint.chunks.id
}

output "vector_deployed_index_id" {
  value = google_vertex_ai_index_endpoint_deployed_index.chunks.deployed_index_id
}

output "tasks_queue" {
  value = google_cloud_tasks_queue.jobs.name
}

output "api_service_account" {
  value = google_service_account.api.email
}

output "ui_service_account" {
  value = google_service_account.ui.email
}
