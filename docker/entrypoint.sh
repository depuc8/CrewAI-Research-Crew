#!/bin/sh
set -eu

case "${SERVICE:-api}" in
  api)
    exec uvicorn research_crew.api:create_app --factory --host 0.0.0.0 --port "${PORT:-8080}"
    ;;
  frontend)
    exec streamlit run frontend/app.py \
      --server.port "${PORT:-8080}" --server.address 0.0.0.0 \
      --server.headless true --browser.gatherUsageStats false
    ;;
  *)
    echo "unknown SERVICE '${SERVICE}' (expected api or frontend)" >&2
    exit 64
    ;;
esac
