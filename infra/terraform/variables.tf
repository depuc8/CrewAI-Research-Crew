variable "project_id" {
  description = "GCP project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for Vertex AI, Cloud Run, Cloud Tasks, Firestore and the bucket."
  type        = string
  default     = "us-central1"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "research-crew"
}

variable "embedding_dimensions" {
  description = "Must equal EMBEDDING_DIMENSIONS in the app (gemini-embedding-001 is truncated to this size)."
  type        = number
  default     = 768
}

variable "deployed_index_id" {
  description = "ID of the deployed index on the endpoint (letters, digits, underscores)."
  type        = string
  default     = "research_crew_deployed"
}

variable "index_machine_type" {
  description = "Machine type serving the deployed index. This is the main fixed cost of the stack."
  type        = string
  default     = "e2-standard-2"
}

variable "create_firestore_database" {
  description = "Create the (default) Firestore database. Set false if the project already has one."
  type        = bool
  default     = true
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete a non-empty papers bucket."
  type        = bool
  default     = false
}
