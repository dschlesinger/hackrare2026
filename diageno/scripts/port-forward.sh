#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# port-forward.sh — Forward Diageno service ports
# ──────────────────────────────────────────────────────
set -euo pipefail

echo "=== Port forwarding Diageno services ==="
echo "Services will be available at:"
echo "  Streamlit UI   → http://localhost:8501"
echo "  FastAPI         → http://localhost:8000"
echo "  Prometheus      → http://localhost:9090"
echo "  Grafana         → http://localhost:3000"
echo "  MLflow          → http://localhost:5000"
echo "  MinIO Console   → http://localhost:9001"
echo "  n8n             → http://localhost:5678"
echo ""
echo "Press Ctrl+C to stop all port-forwards."
echo ""

# Start all port-forwards in background
kubectl -n diageno port-forward svc/ui           8501:8501 &
kubectl -n diageno port-forward svc/api          8000:8000 &
kubectl -n diageno port-forward svc/prometheus   9090:9090 &
kubectl -n diageno port-forward svc/grafana      3000:3000 &
kubectl -n diageno port-forward svc/mlflow       5000:5000 &
kubectl -n diageno port-forward svc/minio        9000:9000 9001:9001 &
kubectl -n diageno port-forward svc/n8n          5678:5678 &

# Wait for all background processes
wait
