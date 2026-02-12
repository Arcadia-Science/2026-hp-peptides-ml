from __future__ import annotations

import argparse
import json
import math
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


def _terminate_process(proc: subprocess.Popen, *, kill_after: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=kill_after)
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, bool):
        return False
    return math.isfinite(float(value))


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
    parser.add_argument(
        "--trial-timeout-seconds",
        type=int,
        default=0,
        help="Per-trial wall-clock timeout in seconds (0 disables).",
    )
    parser.add_argument(
        "--flat-loss-threshold",
        type=float,
        default=1e-12,
        help="Treat train_loss_epoch <= threshold as invalid/flat loss.",
    )
    parser.add_argument(
        "--flat-loss-patience",
        type=int,
        default=2,
        help="Abort trial after this many consecutive flat-loss eval reports (0 disables).",
    )
    parser.add_argument(
        "--max-no-metric-seconds",
        type=int,
        default=900,
        help="Abort trial if no metric rows are reported for this long (0 disables).",
    )
    parser.add_argument(
        "--overfit-ratio",
        type=float,
        default=1.2,
        help="Overfitting trigger: val metric exceeds best_val * overfit_ratio.",
    )
    parser.add_argument(
        "--overfit-train-ratio",
        type=float,
        default=0.9,
        help="Overfitting trigger: train loss drops below best_train * overfit_train_ratio.",
    )
    parser.add_argument(
        "--overfit-patience",
        type=int,
        default=0,
        help="Abort trial after this many consecutive overfitting triggers (0 disables).",
    )
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
            time_attr="epoch",
            metric=args.metric,
            mode=args.mode,
            max_t=args.max_t or args.num_samples,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
    elif args.scheduler == "hyperband":
        scheduler = HyperBandScheduler(
            time_attr="epoch",
            metric=args.metric,
            mode=args.mode,
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
            trial_start = time.monotonic()
            last_metric_time = trial_start
            saw_metric = False
            flat_loss_streak = 0
            best_metric = float("inf")
            best_train_loss = float("inf")
            overfit_streak = 0
            while proc.poll() is None:
                now = time.monotonic()
                if args.trial_timeout_seconds > 0 and (now - trial_start) > args.trial_timeout_seconds:
                    _terminate_process(proc)
                    raise RuntimeError(
                        f"Trial timeout after {args.trial_timeout_seconds}s. "
                        f"See {log_path} and {metrics_path}"
                    )
                for pos, line in _iter_new_lines(metrics_path, last_pos):
                    last_pos = pos
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    metric_value = row.get(args.metric)
                    if args.metric in row and not _is_finite_number(metric_value):
                        _terminate_process(proc)
                        raise RuntimeError(
                            f"Non-finite {args.metric} reported by trial. "
                            f"See {log_path} and {metrics_path}"
                        )
                    # Only report rows that actually contain the tuned metric.
                    if not _is_finite_number(metric_value):
                        continue

                    metrics = {k: v for k, v in row.items() if _is_finite_number(v)}
                    if not metrics:
                        continue

                    saw_metric = True
                    last_metric_time = now

                    train_loss_epoch = row.get("train_loss_epoch")
                    train_loss_value = None
                    if _is_finite_number(train_loss_epoch):
                        train_loss_value = float(train_loss_epoch)
                        if train_loss_value <= args.flat_loss_threshold:
                            flat_loss_streak += 1
                        else:
                            flat_loss_streak = 0
                    else:
                        flat_loss_streak = 0

                    if args.flat_loss_patience > 0 and flat_loss_streak >= args.flat_loss_patience:
                        _terminate_process(proc)
                        raise RuntimeError(
                            "Detected flat/near-zero training loss for consecutive eval epochs; "
                            f"threshold={args.flat_loss_threshold}, streak={flat_loss_streak}. "
                            f"See {log_path} and {metrics_path}"
                        )

                    metric_float = float(metric_value)
                    if metric_float < best_metric:
                        best_metric = metric_float
                        if train_loss_value is not None:
                            best_train_loss = min(best_train_loss, train_loss_value)
                        overfit_streak = 0
                    elif (
                        args.overfit_patience > 0
                        and train_loss_value is not None
                        and math.isfinite(best_train_loss)
                        and best_train_loss < float("inf")
                        and metric_float > best_metric * args.overfit_ratio
                        and train_loss_value < best_train_loss * args.overfit_train_ratio
                    ):
                        overfit_streak += 1
                        if overfit_streak >= args.overfit_patience:
                            _terminate_process(proc)
                            raise RuntimeError(
                                "Detected sustained overfitting pattern; "
                                f"val={metric_float:.6g}, best_val={best_metric:.6g}, "
                                f"train={train_loss_value:.6g}, best_train={best_train_loss:.6g}. "
                                f"See {log_path} and {metrics_path}"
                            )
                    else:
                        overfit_streak = 0

                    tune.report(metrics)

                if (
                    args.max_no_metric_seconds > 0
                    and not saw_metric
                    and (now - trial_start) > args.max_no_metric_seconds
                ):
                    _terminate_process(proc)
                    raise RuntimeError(
                        "No valid metric rows were reported in time. "
                        f"threshold={args.max_no_metric_seconds}s. "
                        f"See {log_path} and {metrics_path}"
                    )
                if (
                    args.max_no_metric_seconds > 0
                    and saw_metric
                    and (now - last_metric_time) > args.max_no_metric_seconds
                ):
                    _terminate_process(proc)
                    raise RuntimeError(
                        "Metric reporting stalled. "
                        f"threshold={args.max_no_metric_seconds}s. "
                        f"See {log_path} and {metrics_path}"
                    )
                time.sleep(2)

            # Drain any buffered metric lines after process exits.
            for pos, line in _iter_new_lines(metrics_path, last_pos):
                last_pos = pos
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                metric_value = row.get(args.metric)
                if args.metric in row and not _is_finite_number(metric_value):
                    continue
                if not _is_finite_number(metric_value):
                    continue
                metrics = {k: v for k, v in row.items() if _is_finite_number(v)}
                if metrics:
                    tune.report(metrics)

            if proc.returncode != 0:
                raise RuntimeError(f"Training failed with exit code {proc.returncode}. See {log_path}")

    trainable = tune.with_resources(
        _trainable,
        {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )

    tune_config_kwargs: Dict[str, Any] = {
        "num_samples": args.num_samples,
        "max_concurrent_trials": args.max_concurrent,
        "scheduler": scheduler,
    }
    if scheduler is None:
        tune_config_kwargs["metric"] = args.metric
        tune_config_kwargs["mode"] = args.mode

    tuner = tune.Tuner(
        trainable,
        param_space=tune_space,
        tune_config=TuneConfig(**tune_config_kwargs),
        run_config=tune.RunConfig(
            name=args.run_prefix,
            storage_path=args.local_dir,
        ),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
