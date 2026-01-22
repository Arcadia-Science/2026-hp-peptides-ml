#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scripts/ec2_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

REPO_DIR="${REPO_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/fsx/processed_all}"
GLOBAL_INDEX_DB="${GLOBAL_INDEX_DB:-$OUTPUT_ROOT/global_index.sqlite}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-}"

S3_URI=""
if [[ -n "$S3_BUCKET" && -n "$S3_PREFIX" ]]; then
  S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}"
fi

cmd=(
  "$PYTHON_BIN" "$REPO_DIR/data-gen-pipeline/index_shards_multi.py"
  --shards-root "$OUTPUT_ROOT"
  --output-db "$GLOBAL_INDEX_DB"
  --dataset-version "v1"
  --dataset-source "local"
)

if [[ -n "$S3_URI" ]]; then
  cmd+=(--s3-prefix "$S3_URI")
fi

printf '  %q' "${cmd[@]}"
echo
env PYTHONUNBUFFERED=1 "${cmd[@]}"
echo "Global index: $GLOBAL_INDEX_DB"
