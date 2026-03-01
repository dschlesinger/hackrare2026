#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# setup-kind.sh — Create a kind cluster for Diageno
# ──────────────────────────────────────────────────────
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-diageno}"

echo "=== Creating kind cluster: ${CLUSTER_NAME} ==="

# Check prerequisites
for cmd in kind kubectl docker; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: $cmd is required but not installed."
    exit 1
  fi
done

# Create kind config
cat <<EOF >/tmp/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${CLUSTER_NAME}
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 8501
        protocol: TCP
      - containerPort: 30081
        hostPort: 8000
        protocol: TCP
      - containerPort: 30090
        hostPort: 9090
        protocol: TCP
      - containerPort: 30030
        hostPort: 3000
        protocol: TCP
      - containerPort: 30056
        hostPort: 5678
        protocol: TCP
  - role: worker
EOF

# Delete existing cluster if present
kind delete cluster --name "${CLUSTER_NAME}" 2>/dev/null || true

# Create cluster
kind create cluster --config /tmp/kind-config.yaml

echo "=== kind cluster '${CLUSTER_NAME}' created ==="
echo "=== Building and loading images ==="

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Build images
docker build -t diageno-api:latest   -f "${SCRIPT_DIR}/Dockerfile.api"   "${SCRIPT_DIR}"
docker build -t diageno-ui:latest    -f "${SCRIPT_DIR}/Dockerfile.ui"    "${SCRIPT_DIR}"
docker build -t diageno-etl:latest   -f "${SCRIPT_DIR}/Dockerfile.etl"   "${SCRIPT_DIR}"

# Load into kind
kind load docker-image diageno-api:latest   --name "${CLUSTER_NAME}"
kind load docker-image diageno-ui:latest    --name "${CLUSTER_NAME}"
kind load docker-image diageno-etl:latest   --name "${CLUSTER_NAME}"

echo "=== Images loaded into kind ==="
echo "Run './scripts/deploy-all.sh' next."
