#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _is_finite_value(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _iter_items(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    # Most shard files are list[Data]. Keep this fallback defensive.
    try:
        return list(obj)
    except Exception:
        return []


def _field_imputed(item: Any, task: str) -> bool:
    imputed = getattr(item, "field_imputed", None)
    if isinstance(imputed, dict):
        return bool(imputed.get(task, False))
    return False


def _check_item(
    item: Any,
    task: str,
    require_pos_finite: bool,
    require_z_finite: bool,
    exclude_imputed: bool,
) -> Tuple[bool, str]:
    target = getattr(item, task, None)
    if target is None:
        return False, "missing_target"
    if not _is_finite_value(target):
        return False, "nonfinite_target"

    if exclude_imputed and _field_imputed(item, task):
        return False, "imputed_target"

    if require_pos_finite:
        pos = getattr(item, "pos", None)
        if pos is None:
            return False, "missing_pos"
        if not _is_finite_value(pos):
            return False, "nonfinite_pos"

    if require_z_finite:
        z = getattr(item, "z", None)
        if z is None:
            return False, "missing_z"
        if not _is_finite_value(z):
            return False, "nonfinite_z"

    return True, "kept"


def _process_one(job: Tuple[Any, ...]) -> Dict[str, Any]:
    (
        shard_idx,
        src_path_str,
        dst_path_str,
        task,
        require_pos_finite,
        require_z_finite,
        exclude_imputed,
        min_items_per_shard,
    ) = job

    src_path = Path(src_path_str)
    dst_path = Path(dst_path_str)
    counters: Dict[str, int] = {
        "items_total": 0,
        "items_kept": 0,
        "missing_target": 0,
        "nonfinite_target": 0,
        "imputed_target": 0,
        "missing_pos": 0,
        "nonfinite_pos": 0,
        "missing_z": 0,
        "nonfinite_z": 0,
        "load_error": 0,
    }

    try:
        raw_items = torch.load(src_path_str, map_location="cpu", weights_only=False)
        items = _iter_items(raw_items)
    except Exception:
        counters["load_error"] = 1
        return {
            "index": shard_idx,
            "source": src_path_str,
            "output": None,
            "written": False,
            "counters": counters,
            "error": "load_error",
        }

    kept: List[Any] = []
    for item in items:
        counters["items_total"] += 1
        ok, reason = _check_item(
            item,
            task=task,
            require_pos_finite=require_pos_finite,
            require_z_finite=require_z_finite,
            exclude_imputed=exclude_imputed,
        )
        if ok:
            kept.append(item)
            counters["items_kept"] += 1
        else:
            counters[reason] = counters.get(reason, 0) + 1

    if len(kept) < min_items_per_shard:
        return {
            "index": shard_idx,
            "source": src_path_str,
            "output": None,
            "written": False,
            "counters": counters,
            "error": None,
        }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.pt")
    torch.save(kept, tmp_path)
    os.replace(tmp_path, dst_path)

    return {
        "index": shard_idx,
        "source": src_path_str,
        "output": str(dst_path),
        "written": True,
        "counters": counters,
        "error": None,
    }


def _load_source_shards(input_list: Optional[Path], input_roots: List[Path], glob_pattern: str) -> List[str]:
    shards: List[str] = []
    if input_list is not None:
        shards.extend(line.strip() for line in input_list.read_text().splitlines() if line.strip())
    for root in input_roots:
        shards.extend(str(p) for p in sorted(root.rglob(glob_pattern)))
    # Stable de-dup while preserving order.
    seen = set()
    ordered = []
    for s in shards:
        if s in seen:
            continue
        seen.add(s)
        ordered.append(s)
    return ordered


def _sum_counters(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for result in results:
        counters = result.get("counters", {})
        for key, value in counters.items():
            total[key] = total.get(key, 0) + int(value)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel preprocessing for DetaNet tasks: filter out invalid targets and materialize "
            "a reduced shard set for downstream scaffold split/training."
        )
    )
    parser.add_argument("--task", required=True, help="Task field to validate (e.g., Hi, Hij, depolar).")
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help="Optional file containing source shard paths (one per line).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=[],
        help="Optional source root to recursively scan for shard files. Can be provided multiple times.",
    )
    parser.add_argument(
        "--glob",
        default="shard_*.pt",
        help="Glob used with --input-root (default: shard_*.pt).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for filtered shard outputs.")
    parser.add_argument(
        "--output-list",
        type=Path,
        default=None,
        help="Output shard list file (default: <output-dir>/shards.txt).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON path (default: <output-dir>/summary.json).",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument(
        "--min-items-per-shard",
        type=int,
        default=1,
        help="Only write shards with at least this many kept items.",
    )
    parser.add_argument(
        "--require-pos-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require finite `pos` for each kept sample.",
    )
    parser.add_argument(
        "--require-z-finite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require finite `z` for each kept sample.",
    )
    parser.add_argument(
        "--exclude-imputed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop samples where field_imputed[task] is true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output-dir (existing shard_*.pt may be overwritten).",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N completed shards.")
    args = parser.parse_args()

    if args.input_list is None and not args.input_root:
        raise ValueError("Provide at least one of --input-list or --input-root.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.min_items_per_shard < 1:
        raise ValueError("--min-items-per-shard must be >= 1.")
    return args


def main() -> None:
    args = parse_args()

    src_shards = _load_source_shards(args.input_list, args.input_root, args.glob)
    if not src_shards:
        raise RuntimeError("No source shards found.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("shard_*.pt"))
    if existing and not args.overwrite:
        raise RuntimeError(
            f"{output_dir} already contains {len(existing)} shard_*.pt files. "
            "Use --overwrite to allow replacing them."
        )

    output_list = args.output_list or (output_dir / "shards.txt")
    summary_json = args.summary_json or (output_dir / "summary.json")

    jobs = []
    for idx, src in enumerate(src_shards):
        dst = output_dir / f"shard_{idx:06d}.pt"
        jobs.append(
            (
                idx,
                src,
                str(dst),
                args.task,
                args.require_pos_finite,
                args.require_z_finite,
                args.exclude_imputed,
                args.min_items_per_shard,
            )
        )

    results: List[Dict[str, Any]] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, job) for job in jobs]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except Exception:
                failures += 1
                continue
            results.append(res)
            if args.progress_every and (done % args.progress_every == 0 or done == total):
                print(f"progress: {done}/{total} shards")

    ordered = sorted(results, key=lambda x: int(x["index"]))
    written_paths = [r["output"] for r in ordered if r.get("written") and r.get("output")]
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(written_paths) + ("\n" if written_paths else ""))

    counters = _sum_counters(ordered)
    summary = {
        "task": args.task,
        "source_shards": len(src_shards),
        "written_shards": len(written_paths),
        "empty_or_dropped_shards": len(src_shards) - len(written_paths),
        "worker_failures": failures,
        "counters": counters,
        "config": {
            "require_pos_finite": args.require_pos_finite,
            "require_z_finite": args.require_z_finite,
            "exclude_imputed": args.exclude_imputed,
            "min_items_per_shard": args.min_items_per_shard,
            "workers": args.workers,
        },
        "output_list": str(output_list),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    kept = counters.get("items_kept", 0)
    total_items = counters.get("items_total", 0)
    ratio = (kept / total_items) if total_items else 0.0
    print(
        f"done: task={args.task} source_shards={len(src_shards)} "
        f"written_shards={len(written_paths)} kept_items={kept}/{total_items} ({ratio:.2%})"
    )
    print(f"shard_list: {output_list}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()

