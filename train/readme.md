```python

RUN_ROOT=/fsx/processed_all/run-ip-<INSERT-HERE>.us-west-1.compute.internal-<HERE>/qm7x/1000
  find "$RUN_ROOT" -name 'shard_*.pt' > /tmp/shards.txt
  python /fsx/repos/hp-proteins-ml/train/train_detanet.py \
    --task Hij \
    --shard-list /tmp/shards.txt \
    --checkpoint /fsx/repos/hp-proteins-ml/capsule-3259363/code/trained_param/qm9spectra/Hij.pth \
    --no-checkpoint-strict \
    --checkpoint-relax-embeddings \
    --checkpoint-relax-mismatch \
    --split train \
    --split-key mol_key \
    --split-train 0.8 --split-val 0.1 \
    --optimizer pt_shampoo \
    --lr 3e-4 \
    --batch-size 8 \
    --amp \
    --use-adalora \
    --adalora-r 16 \
    --adalora-alpha 32 \
    --adalora-dropout 0.05 \
    --adapter-freeze-base \
    --no-use-impute-mask \
    --registry-dir /fsx/model_registry \
    --run-id detanet-Hij-$(date +%Y%m%d-%H%M%S) \
    --wandb


```