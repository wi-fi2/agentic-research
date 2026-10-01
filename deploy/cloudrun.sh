#!/usr/bin/env bash
# Deploy the Research Agent to Google Cloud Run (builds from the Dockerfile with Cloud Build).
# Usage: PROJECT_ID=my-project ./deploy/cloudrun.sh
# Needs: gcloud CLI logged in, billing enabled, and at least one LLM key (see .env.example).
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-agentic-research}"
REPO="${REPO:-agentic-research}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:$(git rev-parse --short HEAD)"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com --project "$PROJECT_ID"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --project "$PROJECT_ID"

gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" --timeout 40m

# Laya (local decision model) needs >= 2.5 GB RAM, so use 4 GiB. Set ACCESS_TOKEN so strangers can't spend your free LLM quota.
# Secrets (e.g. GROQ_API_KEY=groq-key:latest) go in Secret Manager: pass SECRETS="GROQ_API_KEY=groq-key:latest,ACCESS_TOKEN=access-token:latest"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --project "$PROJECT_ID" \
  --memory 4Gi --cpu 2 --concurrency 4 --timeout 900 --min-instances 0 --max-instances 2 \
  --cpu-boost --allow-unauthenticated \
  ${SECRETS:+--set-secrets "$SECRETS"}

gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)'
