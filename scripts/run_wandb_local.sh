#!/usr/bin/env bash
set -euo pipefail

PORT="${WANDB_LOCAL_PORT:-8080}"
DATA_DIR="${WANDB_LOCAL_DATA_DIR:-/fsx/wandb_local}"
CONTAINER_NAME="${WANDB_LOCAL_CONTAINER:-wandb-local}"

mkdir -p "$DATA_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker and try again."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container ${CONTAINER_NAME} already exists. Starting..."
  docker start "${CONTAINER_NAME}" >/dev/null
else
  echo "Starting W&B Local on port ${PORT} with data dir ${DATA_DIR}"
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:8080" \
    -v "${DATA_DIR}:/vol" \
    wandb/local:latest >/dev/null
fi

echo "W&B Local running at http://localhost:${PORT}"
