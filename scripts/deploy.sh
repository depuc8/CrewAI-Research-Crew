#!/usr/bin/env bash
# Build the image with Cloud Build and deploy the API and the Streamlit frontend to Cloud Run.
#
# Prerequisites: `terraform apply` has been run in infra/terraform and `gcloud auth login` is done.
#
#   PROJECT=my-project ./scripts/deploy.sh
#
# Optional: REGION (default: from terraform), PUBLIC_UI=true to let anyone open the frontend,
#           GEMINI_MODEL, TAG.
set -euo pipefail

cd "$(dirname "$0")/.."
TF="terraform -chdir=infra/terraform"
out() { $TF output -raw "$1"; }

PROJECT="${PROJECT:?set PROJECT to your GCP project id}"
REGION="${REGION:-us-central1}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.6-flash}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
PUBLIC_UI="${PUBLIC_UI:-false}"

IMAGE="$(out artifact_repository)/research-crew:${TAG}"
API_SA="$(out api_service_account)"
UI_SA="$(out ui_service_account)"
API_SERVICE="research-crew-api"
UI_SERVICE="research-crew-ui"

# Cloud Run URLs are deterministic: https://<service>-<project number>.<region>.run.app
# so the API can be told its own address (needed for Cloud Tasks) in a single deploy.
PROJECT_NUMBER="$(out project_number)"
API_URL="https://${API_SERVICE}-${PROJECT_NUMBER}.${REGION}.run.app"

echo "==> Building ${IMAGE}"
gcloud builds submit --project "$PROJECT" --tag "$IMAGE" .

echo "==> Deploying ${API_SERVICE}"
API_ENV=(
  SERVICE=api
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT="${PROJECT}"
  GOOGLE_CLOUD_LOCATION="${REGION}"
  GEMINI_MODEL="${GEMINI_MODEL}"
  EMBEDDING_MODEL=gemini-embedding-001
  EMBEDDING_DIMENSIONS=768
  STORAGE_BACKEND=gcp
  VECTOR_BACKEND=gcp
  EMBEDDING_BACKEND=gcp
  JOB_BACKEND=gcp
  GCS_BUCKET="$(out bucket_name)"
  VECTOR_INDEX_ID="$(out vector_index_id)"
  VECTOR_INDEX_ENDPOINT_ID="$(out vector_index_endpoint_id)"
  VECTOR_DEPLOYED_INDEX_ID="$(out vector_deployed_index_id)"
  TASKS_QUEUE="$(out tasks_queue)"
  TASKS_SERVICE_ACCOUNT="${API_SA}"
  SERVICE_URL="${API_URL}"
)
# Not publicly invokable: the frontend (and Cloud Tasks) authenticate with ID tokens.
# One review runs inside one request, so allow a long timeout and few concurrent requests.
gcloud run deploy "$API_SERVICE" \
  --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$API_SA" \
  --no-allow-unauthenticated \
  --cpu 2 --memory 2Gi --timeout 1800 --concurrency 4 --max-instances 5 \
  --set-env-vars "$(IFS=,; echo "${API_ENV[*]}")"

echo "==> Allowing Cloud Tasks (as the API account) and the frontend to invoke the API"
for member in "serviceAccount:${API_SA}" "serviceAccount:${UI_SA}"; do
  gcloud run services add-iam-policy-binding "$API_SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --member "$member" --role roles/run.invoker >/dev/null
done

echo "==> Deploying ${UI_SERVICE}"
UI_AUTH="--no-allow-unauthenticated"
[ "$PUBLIC_UI" = "true" ] && UI_AUTH="--allow-unauthenticated"
gcloud run deploy "$UI_SERVICE" \
  --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$UI_SA" $UI_AUTH \
  --cpu 1 --memory 1Gi --timeout 3600 --max-instances 2 \
  --set-env-vars "SERVICE=frontend,API_URL=${API_URL},API_AUDIENCE=${API_URL}"

echo
echo "API:      ${API_URL}  (private)"
echo "Frontend: $(gcloud run services describe "$UI_SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
[ "$PUBLIC_UI" = "true" ] || echo "The frontend is private too; open it with:  gcloud run services proxy ${UI_SERVICE} --project ${PROJECT} --region ${REGION}"
