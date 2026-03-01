#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# deploy-all.sh — Deploy all Diageno services to K8s
# ──────────────────────────────────────────────────────
set -euo pipefail

K8S_DIR="$(cd "$(dirname "$0")/../k8s" && pwd)"

echo "=== Deploying Diageno to Kubernetes ==="

# 1. Namespace
kubectl apply -f "${K8S_DIR}/namespace.yaml"

# 2. Infrastructure
echo "--- Deploying Postgres ---"
kubectl apply -f "${K8S_DIR}/postgres.yaml"

echo "--- Deploying Redis ---"
kubectl apply -f "${K8S_DIR}/redis.yaml"

echo "--- Deploying MinIO ---"
kubectl apply -f "${K8S_DIR}/minio.yaml"

# Wait for Postgres to be ready
echo "--- Waiting for Postgres readiness ---"
kubectl -n diageno rollout status deployment/postgres --timeout=120s

echo "--- Deploying MLflow ---"
kubectl apply -f "${K8S_DIR}/mlflow.yaml"

# 3. Run ETL pipeline
echo "--- Running ETL pipeline ---"
kubectl apply -f "${K8S_DIR}/etl-jobs.yaml"

echo "--- Waiting for download-bronze ---"
kubectl -n diageno wait --for=condition=complete job/download-bronze --timeout=600s || echo "WARNING: download-bronze may still be running"

echo "--- Waiting for parse-silver ---"
kubectl -n diageno wait --for=condition=complete job/parse-silver --timeout=600s || echo "WARNING: parse-silver may still be running"

echo "--- Waiting for load-gold ---"
kubectl -n diageno wait --for=condition=complete job/load-gold --timeout=300s || echo "WARNING: load-gold may still be running"

echo "--- Waiting for train-model ---"
kubectl -n diageno wait --for=condition=complete job/train-model --timeout=600s || echo "WARNING: train-model may still be running"

# 4. Application
echo "--- Deploying API ---"
kubectl apply -f "${K8S_DIR}/api.yaml"

echo "--- Deploying UI ---"
kubectl apply -f "${K8S_DIR}/ui.yaml"

# 5. Monitoring
echo "--- Deploying Prometheus + Grafana + Alertmanager ---"
kubectl apply -f "${K8S_DIR}/monitoring.yaml"
kubectl apply -f "${K8S_DIR}/grafana-alertmanager.yaml"

# 6. n8n
echo "--- Deploying n8n ---"
kubectl apply -f "${K8S_DIR}/n8n.yaml"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Run './scripts/port-forward.sh' to access services locally."
