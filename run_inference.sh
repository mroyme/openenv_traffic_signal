#!/usr/bin/env bash
set -euo pipefail

# Start server in background
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT

# Wait for server to be ready
echo "Waiting for server..." >&2
for i in $(seq 1 20); do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 1
done

# Run inference for each task
for task in corridor_coordination grid_coordination emergency_response; do
  TRAFFIC_TASK=$task uv run python inference.py --write
  echo "---"
done
