from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import ray
from ray import tune
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler


def _ensure_ray_runtime_env(repo_root: str) -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "auto"),
        runtime_env={"env_vars": {"PYTHONPATH": repo_root}},
    )


def _load_param_space(args: argparse.Namespace) -> Dict[str, Any]:
    if args.param_space_file:
        return json.loads(Path(args.param_space_file).read_text())
    if args.param_space:
        return json.loads(args.param_space)
    return {}


def _to_tune_space(tune_module, value: Any) -> Any:
    if isinstance(value, list):
        return tune_module.choice(value)
    if isinstance(value, dict):
        kind = value.get("type")
        vals = value.get("values")
        if kind == "choice":
            return tune_module.choice(vals)
        if kind == "grid":
            return tune_module.grid_search(vals)
        if kind == "uniform":
            return tune_module.uniform(value["min"], value["max"])
        if kind == "loguniform":
            return tune_module.loguniform(value["min"], value["max"])
        if kind == "randint":
            return tune_module.randint(value["min"], value["max"])
        if kind == "qrandint":
            return tune_module.qrandint(value["min"], value["max"], value.get("q", 1))
    return value


def _build_tune_space(tune_module, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_tune_space(tune_module, v) for k, v in raw.items()}


def _iter_new_lines(path: Path, start_pos: int) -> Iterable[tuple[int, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        f.seek(start_pos)
        data = f.read()
        new_pos = f.tell()
    if not data:
        return []
    return [(new_pos, line) for line in data.splitlines()]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    _ensure_ray_runtime_env(str(repo_root))

    parser = argparse.ArgumentParser(description="Ray Tune trainable-based HPO (DDP via torchrun).")
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--run-prefix", default="tune")
    parser.add_argument("--param-space", default=None)
    parser.add_argument("--param-space-file", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--metric", default="val_mse")
    parser.add_argument("--mode", default="min", choices=["min", "max"])
    parser.add_argument("--local-dir", default="ray_results")
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--cpus-per-trial", type=int, default=4)
    parser.add_argument("--gpus-per-trial", type=int, default=1)
    parser.add_argument(
        "--scheduler",
        default="asha",
        choices=["none", "asha", "hyperband"],
        help="Scheduler for early stopping/pruning.",
    )
    parser.add_argument("--max-t", type=int, default=None)
    parser.add_argument("--grace-period", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=2)
    # Kept for compatibility with existing launcher flags (unused in trainable-based flow).
    parser.add_argument("--best-copy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--best-dir", default="best")
    parser.add_argument("--base-args", default=None, help="JSON list of args to pass to train script.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base_args: list[str] = []
    if args.base_args:
        base_args = json.loads(args.base_args)
    if args.extra_args:
        base_args += args.extra_args

    raw_space = _load_param_space(args)
    tune_space = _build_tune_space(tune, raw_space)

    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            max_t=args.max_t or args.num_samples,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
    elif args.scheduler == "hyperband":
        scheduler = HyperBandScheduler(
            max_t=args.max_t or args.num_samples,
            reduction_factor=args.reduction_factor,
        )

    def _trainable(config: Dict[str, Any]) -> None:
        ctx = tune.get_context()
        trial_id = None
        trial_name = None
        if ctx is not None:
            try:
                trial_id = ctx.get_trial_id()
            except Exception:
                trial_id = None
            try:
                trial_name = ctx.get_trial_name()
            except Exception:
                trial_name = None
        if not trial_id:
            trial_id = os.environ.get("RAY_TRIAL_ID") or os.environ.get("TUNE_TRIAL_ID") or "trial"
        run_id = f"{args.run_prefix}-{trial_name or trial_id}"
        registry_dir = args.registry_dir or args.local_dir
        run_dir = Path(registry_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        override_args: list[str] = []
        for key, value in config.items():
            flag = "--" + key.replace("_", "-")
            override_args += [flag, str(value)]

        train_script = repo_root / "train" / "train_detanet.py"
        env = os.environ.copy()
        debug_val = env.get("TORCH_DISTRIBUTED_DEBUG")
        if debug_val not in (None, "OFF", "INFO", "DETAIL"):
            env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
        else:
            env.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
        for key in (
            "TORCH_NCCL_BLOCKING_WAIT",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_BLOCKING_WAIT",
            "NCCL_ASYNC_ERROR_HANDLING",
        ):
            val = env.get(key)
            if val is None:
                continue
            if str(val) not in ("0", "1"):
                env[key] = "0"
        trace_key = "TORCH_NCCL_TRACE_BUFFER_SIZE"
        trace_val = env.get(trace_key)
        if trace_val is not None:
            if not str(trace_val).isdigit():
                env[trace_key] = "0"
        else:
            env[trace_key] = "0"
        visible = env.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            nproc = len([v for v in visible.split(",") if v.strip()])
        else:
            nproc = max(1, int(args.gpus_per_trial))

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            str(train_script),
            "--registry-dir",
            registry_dir,
            "--run-id",
            run_id,
        ]
        cmd += base_args
        cmd += override_args

        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(f"TORCH_NCCL_BLOCKING_WAIT={env.get('TORCH_NCCL_BLOCKING_WAIT')}\n")
            log_f.write(f"TORCH_NCCL_ASYNC_ERROR_HANDLING={env.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')}\n")
            log_f.write(f"NCCL_BLOCKING_WAIT={env.get('NCCL_BLOCKING_WAIT')}\n")
            log_f.write(f"NCCL_ASYNC_ERROR_HANDLING={env.get('NCCL_ASYNC_ERROR_HANDLING')}\n")
            log_f.write(f"TORCH_NCCL_TRACE_BUFFER_SIZE={env.get('TORCH_NCCL_TRACE_BUFFER_SIZE')}\n")
            log_f.flush()
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)

            metrics_path = run_dir / "metrics.jsonl"
            last_pos = 0
            while proc.poll() is None:
                for pos, line in _iter_new_lines(metrics_path, last_pos):
                    last_pos = pos
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    metrics = {
                        k: v
                        for k, v in row.items()
                        if isinstance(v, (int, float))
                    }
                    if metrics:
                        if args.metric and args.metric not in metrics:
                            metrics[args.metric] = float("nan")
                        tune.report(metrics)
                time.sleep(2)

            if proc.returncode != 0:
                raise RuntimeError(f"Training failed with exit code {proc.returncode}. See {log_path}")

    trainable = tune.with_resources(
        _trainable,
        {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable,
        param_space=tune_space,
        tune_config=TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
            metric=args.metric,
            mode=args.mode,
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(
            name=args.run_prefix,
            storage_path=args.local_dir,
        ),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
