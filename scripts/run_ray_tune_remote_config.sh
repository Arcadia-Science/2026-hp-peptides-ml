#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT/OVERRIDE THESE ======
KEY="${KEY:-<YOUR-KEY-HERE>}"
HOST="${HOST:-ec2-user@<YOUR-HOST-HERE>}"
LOCAL_REPO="${LOCAL_REPO:-<YOUR-REPO>}"
REMOTE_REPO="${REMOTE_REPO:-/fsx/<YOUR-FS>}"
PY="${PY:-/home/ec2-user/miniforge3/envs/<YOUR-ENV>}"
# ================================

echo ">> Using HOST=$HOST"

echo ">> Sync train scripts"
rsync -avz -e "ssh -i ${KEY}" \
  "${LOCAL_REPO}/train/train_tune.py" \
  "${LOCAL_REPO}/train/train_detanet.py" \
  "${HOST}:${REMOTE_REPO}/train/"

echo ">> Run remote Ray Tune"
ssh -i "${KEY}" "${HOST}" \
  "PY='$PY' REMOTE_REPO='$REMOTE_REPO' bash -s" <<'REMOTE'
set -euo pipefail

PY="$PY"
REPO="$REMOTE_REPO"
RAY_CLI="$(dirname "$PY")/ray"

echo ">> Python path: $PY"
ls -l "$PY" || true

if [ ! -x "$PY" ]; then
  echo "ERROR: Python env not found at $PY"
  exit 1
fi

"$PY" - <<'PY'
import ray
print("Ray version:", ray.__version__)
PY

if [ -x "$RAY_CLI" ]; then
  "$RAY_CLI" stop || true
  "$RAY_CLI" start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats
else
  echo ">> Ray CLI not found; relying on Ray auto-init (no dashboard)."
fi

# Build shard list
find /fsx/processed_all -name 'shard_*.pt' > /tmp/all_shards.txt

# Base args (JSON list)
cat > /tmp/base_args.json <<'JSON'
[
  "--task","Hij",
  "--shard-list","/tmp/all_shards.txt",
  "--checkpoint","/fsx/repos/hp-proteins-ml/capsule-3259363/code/trained_param/qm9spectra/Hij.pth",
  "--no-checkpoint-strict",
  "--checkpoint-relax-embeddings",
  "--checkpoint-relax-mismatch",
  "--split","train",
  "--split-key","number",
  "--split-train","0.8","--split-val","0.1",
  "--epochs","5",
  "--eval-every","1",
  "--amp",
  "--use-elora",
  "--adapter-freeze-base",
  "--no-use-impute-mask",
  "--normalize","none",
  "--exclude-keys","mol_key,subset,source,smile,field_source,field_generated,field_imputed,field_confidence,conformer_id",
  "--tensorboard"
]
JSON

# Param space (pt_shampoo only)
cat > /tmp/param_space.json <<'JSON'
{
  "lr": {"type":"loguniform","min":1e-5,"max":5e-4},
  "batch_size": {"type":"choice","values":[8,12,16]},
  "optimizer": {"type":"choice","values":["pt_shampoo"]},
  "adalora_r": {"type":"choice","values":[8,16]},
  "adalora_alpha": {"type":"choice","values":[16,32]}
}
JSON

WANDB_MODE=offline WANDB_SILENT=true \
"$PY" "$REPO/train/train_tune.py" \
  --registry-dir /fsx/model_registry \
  --run-prefix tune-hij \
  --param-space-file /tmp/param_space.json \
  --base-args "$(tr -d '\n' </tmp/base_args.json)" \
  --num-samples 8 \
  --max-concurrent 4 \
  --cpus-per-trial 8 \
  --gpus-per-trial 2 \
  --scheduler asha \
  --max-t 5 \
  --report-interval 60 \
  --best-copy \
  --best-dir best
REMOTE
