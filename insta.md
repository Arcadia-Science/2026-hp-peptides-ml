# Some quickstart

```bash
sudo dnf install -y gcc make kernel-devel-$(uname -r)
curl -O https://us.download.nvidia.com/tesla/550.144.03/NVIDIA-Linux-x86_64-550.144.03.run
sudo bash NVIDIA-Linux-x86_64-550.144.03.run --silent
sudo reboot
```

```bash
sudo dnf install -y lustre-client
sudo mkdir -p /fsx
sudo mount -t lustre -o relatime,flock <details-here>
```

```bash
tar -I 'zstd -T0' -xf /fsx/conda/miniforge3.tar.zst -C /home/ec2-user
source /home/ec2-user/miniforge3/etc/profile.d/conda.sh
conda activate hp
```

```bash
cd /fsx/repos/hp-proteins-ml
```

```bash
find /fsx/processed_all -name 'shard_*.pt' > /tmp/all_shards.txt
wc -l /tmp/all_shards.txt
```


```bash
ray stop || true
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

```bash
BASE_ARGS='[
"--task","Hij",
"--shard-list","/tmp/all_shards.txt",
"--checkpoint","/fsx/repos/hp-proteins-ml/capsule-3259363/code/trained_param/qm9spectra/Hij.pth",
"--no-checkpoint-strict",
"--checkpoint-relax-embeddings",
"--checkpoint-relax-mismatch",
"--split-key","number",
"--split-train","0.8","--split-val","0.1",
"--amp",
"--use-elora",
"--adapter-freeze-base",
"--no-use-impute-mask",
"--normalize","none",
"--exclude-keys","mol_key,subset,source,smile,field_source,field_generated,field_imputed,field_confidence,conformer_id",
"--tensorboard"
]'

PARAM_SPACE='{
"lr": {"type":"loguniform","min":1e-5,"max":5e-4},
"batch_size": {"type":"choice","values":[8,12,16]},
"optimizer": {"type":"choice","values":["pt_shampoo","adamw","distributed_shampoo"]},
"adalora_r": {"type":"choice","values":[8,16]},
"adalora_alpha": {"type":"choice","values":[16,32]}
}'

WANDB_MODE=offline WANDB_SILENT=true \
/home/ec2-user/miniforge3/envs/hp/bin/python /fsx/repos/hp-proteins-ml/train/train_tune.py \
--registry-dir /fsx/model_registry \
--run-prefix tune-hij \
--param-space "$PARAM_SPACE" \
--base-args "$BASE_ARGS" \
--num-samples 12 \
--scheduler asha \
--max-t 20 \
--grace-period 1 \
--reduction-factor 2 \
--report-interval 60 \
--cpus-per-trial 6 \
--gpus-per-trial 2 \
--max-concurrent 4
```

```bash
http://<public-ip>:8265
ls /fsx/model_registry/tune-hij-*/train.log
tail -f /fsx/model_registry/tune-hij-*/train.log
ls /fsx/model_registry/best/latest
```