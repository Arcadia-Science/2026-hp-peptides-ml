#!/usr/bin/env bash
set -euo pipefail

# Local loop:
# - reads remote run-id manifest written by queue_spectra_after_dipole_remote.sh
# - rsyncs each run dir from /fsx/model_registry
# - rsyncs live /tmp/<run_id>.log when present

KEY="${KEY:?set KEY to your ssh key path}"
HOST="${HOST:?set HOST to user@host}"
QUEUE_ID="${QUEUE_ID:?set QUEUE_ID}"

REMOTE_REGISTRY_DIR="${REMOTE_REGISTRY_DIR:-/fsx/model_registry}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-artifacts/spectra_queue}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
RUNIDS_FILE_REMOTE="${RUNIDS_FILE_REMOTE:-${REMOTE_REGISTRY_DIR}/${QUEUE_ID}.runids.txt}"

mkdir -p "${LOCAL_ARTIFACT_ROOT}"

while true; do
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] poll runids from ${RUNIDS_FILE_REMOTE}"
  run_ids_raw="$(ssh -i "${KEY}" "${HOST}" "cat '${RUNIDS_FILE_REMOTE}' 2>/dev/null || true" \
    | sed '/^[[:space:]]*$/d' \
    | awk '!seen[$0]++')"

  while IFS= read -r run_id; do
    [[ -z "${run_id}" ]] && continue
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] sync ${run_id}"
    local_dir="${LOCAL_ARTIFACT_ROOT}/${run_id}"
    mkdir -p "${local_dir}"
    rsync -az --delete -e "ssh -i ${KEY}" \
      "${HOST}:${REMOTE_REGISTRY_DIR}/${run_id}/" \
      "${local_dir}/" || true
    rsync -az -e "ssh -i ${KEY}" \
      "${HOST}:/tmp/${run_id}.log" \
      "${local_dir}/" || true
    rsync -az -e "ssh -i ${KEY}" \
      "${HOST}:/tmp/${run_id}.driver.log" \
      "${local_dir}/" || true
  done <<< "${run_ids_raw}"

  sleep "${INTERVAL_SECONDS}"
done
