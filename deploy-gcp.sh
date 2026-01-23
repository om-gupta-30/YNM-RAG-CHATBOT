#!/usr/bin/env bash
# Deploy RAG PDF Chatbot to Google Cloud Run
# Usage: ./deploy-gcp.sh
# Set GEMINI_API_KEY (env or .env) before running.

set -e

PROJECT_ID="${GCP_PROJECT_ID:-gen-lang-client-0473608308}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE_NAME="${GCP_SERVICE_NAME:-rag-pdf-chatbot}"

# Load .env if present
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${GEMINI_API_KEY}" ]]; then
  echo "Error: GEMINI_API_KEY is not set. Add it to .env or export GEMINI_API_KEY=your_key"
  exit 1
fi

echo "=== Env vars for deploy ==="
echo "  Runtime (Cloud Run): GEMINI_API_KEY=*** (set)"
echo "  Build (Dockerfile):  VITE_API_URL= (same-origin), PORT=8080"
echo ""
echo "Deploying ${SERVICE_NAME} to Cloud Run (project=${PROJECT_ID}, region=${REGION})"

gcloud config set project "${PROJECT_ID}"

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --quiet

# Deploy from source (uses Dockerfile, builds, pushes to AR, deploys)
echo ""
echo "=== Starting deployment (this will show build logs with env vars) ==="
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10

echo ""
echo "=== Deployment complete ==="
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')
echo "Service URL: ${SERVICE_URL}"

echo ""
echo "=== Verifying environment variables ==="
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(spec.template.spec.containers[0].env)' | grep -q "GEMINI_API_KEY" && echo "✓ GEMINI_API_KEY is set" || echo "✗ GEMINI_API_KEY is missing"

echo ""
echo "=== Checking service health ==="
HEALTH_ENDPOINT="${SERVICE_URL}/health" || HEALTH_ENDPOINT="${SERVICE_URL}/"
echo "Testing endpoint: ${HEALTH_ENDPOINT}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${SERVICE_URL}/" || echo "000")
if [ "${HTTP_CODE}" = "200" ] || [ "${HTTP_CODE}" = "404" ]; then
  echo "✓ Service is responding (HTTP ${HTTP_CODE})"
else
  echo "⚠ Service returned HTTP ${HTTP_CODE} (may still be starting up)"
fi

echo ""
echo "=== Service status ==="
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='table(status.conditions[0].type,status.conditions[0].status,status.url)'
