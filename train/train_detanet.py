#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

import json

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "capsule-3259363" / "code"
if str(MODEL_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_ROOT))

from detanet_model.detanet import DetaNet
from detanet_model.model_loader import BASE_MODEL_CONFIG, TASK_CONFIGS


def _list_shards(shard_dir: Optional[str], list_file: Optional[str]) -> List[str]:
    if list_file:
        paths = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
        return paths
    if not shard_dir:
        raise ValueError("Provide --shard-dir or --shard-list.")
    return sorted(str(p) for p in Path(shard_dir).glob("shard_*.pt"))


class ShardIterable(IterableDataset):
    def __init__(
        self,
        shard_paths: List[str],
        task: str,
        mask_mode: str,
        mask_key: str,
        confidence_key: str,
        seed: int = 123,
        shuffle_shards: bool = True,
        shuffle_samples: bool = False,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.task = task
        self.mask_mode = mask_mode
        self.mask_key = mask_key
        self.confidence_key = confidence_key
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _partition(self, paths: List[str]) -> Iterable[str]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        global_workers = world_size * num_workers
        worker_index = rank * num_workers + worker_id
        for idx, path in enumerate(paths):
            if idx % global_workers == worker_index:
                yield path

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.shard_paths)
        if self.shuffle_shards:
            rng.shuffle(paths)

        for shard_path in self._partition(paths):
            data_list = torch.load(shard_path, map_location="cpu", weights_only=False)
            if self.shuffle_samples:
                rng.shuffle(data_list)
            for item in data_list:
                _attach_mask(
                    item,
                    task=self.task,
                    mask_mode=self.mask_mode,
                    mask_key=self.mask_key,
                    confidence_key=self.confidence_key,
                )
                yield item


def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _attach_mask(item, task: str, mask_mode: str, mask_key: str, confidence_key: str) -> None:
    target = getattr(item, task, None)
    if target is None:
        return

    mask_value = 1.0
    imputed = getattr(item, mask_key, None)
    if isinstance(imputed, dict):
        if imputed.get(task, False):
            mask_value = 0.0

    if mask_mode == "confidence":
        conf = getattr(item, confidence_key, None)
        if isinstance(conf, dict):
            conf_val = conf.get(task, None)
            if conf_val is not None:
                try:
                    mask_value = float(conf_val)
                except Exception:
                    pass

    if torch.is_tensor(target):
        mask_tensor = torch.full_like(target, float(mask_value))
    else:
        mask_tensor = torch.tensor(float(mask_value), dtype=torch.float32)
    setattr(item, f"mask_{task}", mask_tensor)


def _compute_stats(
    loader: DataLoader,
    task: str,
    mask_name: str,
    per_atom: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = 0.0
    total_sq = 0.0
    count = 0.0
    for batch in loader:
        batch = batch.to(device)
        target = getattr(batch, task).float()
        mask = getattr(batch, mask_name, None)
        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = mask.float()

        if per_atom:
            counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(device)
            while counts.dim() < target.dim():
                counts = counts.unsqueeze(-1)
            target = target / counts.clamp(min=1.0)

        total += (target * mask).sum().item()
        total_sq += (target * target * mask).sum().item()
        count += mask.sum().item()

    if count == 0:
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
    else:
        mean = torch.tensor(total / count)
        var = max(total_sq / count - mean.item() ** 2, 0.0)
        std = torch.tensor(var ** 0.5 if var > 0 else 1.0)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([mean.item(), std.item(), count], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[2].item() > 0:
            mean = stats[0] / stats[2]
            std = stats[1] / stats[2]
        else:
            mean = torch.tensor(0.0, device=device)
            std = torch.tensor(1.0, device=device)

    return mean.to(device), std.to(device)


def build_model(args) -> nn.Module:
    if args.task not in TASK_CONFIGS:
        raise KeyError(f"Unknown task {args.task}. Available: {sorted(TASK_CONFIGS)}")

    config = dict(BASE_MODEL_CONFIG)
    config.update(TASK_CONFIGS[args.task])
    config.update(
        dict(
            num_features=args.num_features,
            num_block=args.num_block,
            num_radial=args.num_radial,
            attention_head=args.attention_head,
            rc=args.rc,
            dropout=args.dropout,
            elora_path=args.elora_path,
            device=args.device,
        )
    )

    adalora_config = None
    if args.use_adalora:
        try:
            from peft import AdaLoraConfig, TaskType
        except Exception as exc:
            raise RuntimeError("peft is required for AdaLoRA.") from exc
        adalora_config = AdaLoraConfig(
            r=args.adalora_r,
            lora_alpha=args.adalora_alpha,
            lora_dropout=args.adalora_dropout,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            total_step=args.adalora_total_step,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        config["adalora_config"] = adalora_config
        if args.adalora_targets:
            config["adalora_targets"] = [t.strip() for t in args.adalora_targets.split(",") if t.strip()]
        config["adalora_scalar_heads"] = args.adalora_scalar_heads
        config["adalora_attention"] = args.adalora_attention
        config["adapter_freeze_base"] = args.adapter_freeze_base

    model = DetaNet(**config)
    return model


def save_checkpoint(model: nn.Module, save_path: Path, use_fsdp: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()
        torch.save(state, save_path)
        return

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DetaNet with optional AdaLoRA + ELoRA.")
    parser.add_argument("--task", required=True, help="Task name (e.g. energy, polar, Hij).")
    parser.add_argument("--shard-dir", default=None, help="Directory containing shard_*.pt files.")
    parser.add_argument("--shard-list", default=None, help="Text file with shard paths.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-dir", default="trained_param/latest")
    parser.add_argument("--checkpoint", default=None, help="Path to a pretrained checkpoint to load.")
    parser.add_argument(
        "--checkpoint-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strictly enforce that the checkpoint keys match the model.",
    )
    parser.add_argument(
        "--exclude-keys",
        default="field_imputed,field_generated,field_confidence,field_source,smile,source",
        help="Comma-separated list of Data keys to exclude from PyG collation.",
    )

    parser.add_argument("--num-features", type=int, default=128)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--num-radial", type=int, default=32)
    parser.add_argument("--attention-head", type=int, default=8)
    parser.add_argument("--rc", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--use-elora", action="store_true")
    parser.add_argument("--elora-path", default=None, help="Path to ELoRA repo or 'vendored'.")

    parser.add_argument("--use-adalora", action="store_true")
    parser.add_argument("--adalora-r", type=int, default=8)
    parser.add_argument("--adalora-alpha", type=int, default=32)
    parser.add_argument("--adalora-dropout", type=float, default=0.05)
    parser.add_argument("--adalora-tinit", type=int, default=10)
    parser.add_argument("--adalora-tfinal", type=int, default=20)
    parser.add_argument("--adalora-total-step", type=int, default=1000)
    parser.add_argument("--adalora-targets", default=None)
    parser.add_argument("--adalora-scalar-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adalora-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-freeze-base", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--normalize",
        default="none",
        choices=["none", "batch", "dataset", "per_atom", "batch_per_atom", "dataset_per_atom"],
        help="Target normalization strategy.",
    )
    parser.add_argument("--norm-cache", default=None, help="Optional JSON file to cache dataset mean/std.")
    parser.add_argument("--use-impute-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-mode", default="binary", choices=["binary", "confidence"])
    parser.add_argument("--mask-key", default="field_imputed")
    parser.add_argument("--confidence-key", default="field_confidence")

    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    torch.manual_seed(args.seed + rank)

    args.device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if not args.use_elora:
        args.elora_path = None
    elif args.elora_path is None:
        args.elora_path = "vendored"

    shard_paths = _list_shards(args.shard_dir, args.shard_list)
    dataset = ShardIterable(
        shard_paths,
        task=args.task,
        mask_mode=args.mask_mode,
        mask_key=args.mask_key,
        confidence_key=args.confidence_key,
        seed=args.seed,
        shuffle_shards=True,
        shuffle_samples=False,
    )
    exclude_keys = [k.strip() for k in args.exclude_keys.split(",") if k.strip()]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        exclude_keys=exclude_keys,
    )

    # Normalization stats
    norm_mode = args.normalize
    per_atom = norm_mode.endswith("per_atom") or norm_mode == "per_atom"
    base_norm = norm_mode.replace("_per_atom", "")
    if base_norm == "per_atom":
        base_norm = "none"
    mask_name = f"mask_{args.task}"
    norm_mean = torch.tensor(0.0, device=args.device)
    norm_std = torch.tensor(1.0, device=args.device)

    if base_norm == "dataset":
        cache_path = Path(args.norm_cache) if args.norm_cache else None
        if cache_path and cache_path.exists():
            stats = json.loads(cache_path.read_text())
            norm_mean = torch.tensor(stats["mean"], device=args.device)
            norm_std = torch.tensor(stats["std"], device=args.device)
        else:
            stats_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                exclude_keys=exclude_keys,
            )
            norm_mean, norm_std = _compute_stats(
                stats_loader, args.task, mask_name, per_atom=per_atom, device=args.device
            )
            if rank == 0 and cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"mean": norm_mean.item(), "std": norm_std.item()}))

    model = build_model(args).to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            elif "module" in ckpt and isinstance(ckpt["module"], dict):
                ckpt = ckpt["module"]
        if isinstance(ckpt, dict) and all(k.startswith("module.") for k in ckpt):
            ckpt = {k[len("module.") :]: v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(ckpt, strict=args.checkpoint_strict)
        if rank == 0:
            print(f"loaded checkpoint: {args.checkpoint}")
            if missing:
                print(f"missing keys: {len(missing)}")
            if unexpected:
                print(f"unexpected keys: {len(unexpected)}")

    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(model, device_id=args.device if args.device.type == "cuda" else None)
    elif world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if args.device.type == "cuda" else None,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device.type == "cuda")

    global_step = 0
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        step = -1
        for step, batch in enumerate(loader):
            batch = batch.to(args.device)
            target = getattr(batch, args.task).float()
            mask = getattr(batch, mask_name, None)
            if mask is None or not args.use_impute_mask:
                mask = torch.ones_like(target)
            else:
                mask = mask.float()
            if step % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and args.device.type == "cuda"):
                pred = model(z=batch.z, pos=batch.pos, edge_index=batch.edge_index, batch=batch.batch).float()

                if per_atom:
                    counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(args.device)
                    while counts.dim() < target.dim():
                        counts = counts.unsqueeze(-1)
                    pred = pred / counts.clamp(min=1.0)
                    target = target / counts.clamp(min=1.0)

                if base_norm == "batch":
                    denom = mask.sum().clamp(min=1.0)
                    mean = (target * mask).sum() / denom
                    var = ((target - mean) ** 2 * mask).sum() / denom
                    std = torch.sqrt(var + 1e-12)
                elif base_norm == "dataset":
                    mean = norm_mean
                    std = norm_std
                else:
                    mean = 0.0
                    std = 1.0

                pred = (pred - mean) / std
                target = (target - mean) / std

                loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()

            running += loss.item()
            global_step += 1
            if rank == 0 and (global_step % args.log_every == 0):
                avg_loss = running / max(1, args.log_every)
                print(f"epoch={epoch} step={global_step} loss={avg_loss:.6f}")
                running = 0.0

        if step >= 0 and (step + 1) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()

        if rank == 0:
            save_path = Path(args.save_dir) / f"latest_{args.task}.pth"
            save_checkpoint(model, save_path, args.fsdp)
            print(f"saved {save_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
