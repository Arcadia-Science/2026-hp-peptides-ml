#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _load_rows(metrics_path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_rows = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad_rows += 1
                continue
            rows.append(row)
    return rows, bad_rows


def _status(
    eval_count: int,
    first_val: float,
    last_val: float,
    best_val: float,
    first_train: float | None,
    last_train: float | None,
    nan_rows: int,
) -> str:
    if nan_rows > 0:
        return "bad_nan"
    if eval_count < 2:
        return "warmup"
    if not math.isfinite(last_val) or not math.isfinite(best_val):
        return "bad_nan"

    improving = last_val <= first_val * 0.95
    overfit = False
    if first_train is not None and last_train is not None and math.isfinite(first_train) and math.isfinite(last_train):
        overfit = last_val > best_val * 1.15 and last_train < first_train * 0.9

    if overfit:
        return "overfit_risk"
    if improving:
        return "improving"
    if last_val <= best_val * 1.05:
        return "near_best"
    return "stalled"


def summarize_trial(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    rows, bad_rows = _load_rows(metrics_path)
    val_rows = [r for r in rows if _is_num(r.get("val_mse"))]
    if not val_rows:
        return {
            "run": run_dir.name,
            "evals": 0,
            "status": "no_eval",
            "best_val": math.inf,
            "last_val": math.inf,
            "improvement_pct": 0.0,
            "nan_rows": bad_rows,
        }

    val_seq = [float(r["val_mse"]) for r in val_rows]
    train_seq = [float(r["train_loss_epoch"]) for r in val_rows if _is_num(r.get("train_loss_epoch"))]
    best_val = min(val_seq)
    first_val = val_seq[0]
    last_val = val_seq[-1]
    improvement_pct = 0.0
    if abs(first_val) > 1e-12:
        improvement_pct = (first_val - last_val) / abs(first_val) * 100.0
    first_train = train_seq[0] if train_seq else None
    last_train = train_seq[-1] if train_seq else None
    nan_rows = bad_rows + sum(
        1
        for r in rows
        if ("val_mse" in r and not _is_num(r.get("val_mse")))
        or ("train_loss_epoch" in r and not _is_num(r.get("train_loss_epoch")))
    )
    return {
        "run": run_dir.name,
        "evals": len(val_rows),
        "status": _status(len(val_rows), first_val, last_val, best_val, first_train, last_train, nan_rows),
        "best_val": best_val,
        "last_val": last_val,
        "improvement_pct": improvement_pct,
        "last_train": last_train,
        "nan_rows": nan_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ray Tune trial metrics.")
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.registry_dir)
    trial_dirs = sorted(p for p in root.glob(f"{args.run_prefix}-*") if p.is_dir())
    summaries = [s for p in trial_dirs if (s := summarize_trial(p)) is not None]
    summaries.sort(key=lambda s: s["best_val"])

    print(f"trials={len(summaries)} registry={root} prefix={args.run_prefix}")
    print("status run evals best_val last_val improve% last_train nan_rows")
    for s in summaries[: max(1, args.top_k)]:
        best = s["best_val"]
        last = s["last_val"]
        trn = s.get("last_train")
        best_txt = "inf" if not math.isfinite(best) else f"{best:.6g}"
        last_txt = "inf" if not math.isfinite(last) else f"{last:.6g}"
        trn_txt = "-" if trn is None else f"{trn:.6g}"
        print(
            f"{s['status']:>10} {s['run']} {s['evals']:>4} {best_txt:>10} {last_txt:>10} "
            f"{s['improvement_pct']:>8.2f} {trn_txt:>10} {s['nan_rows']:>8}"
        )


if __name__ == "__main__":
    main()
