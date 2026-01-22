from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch

import pipeline
from deepmd_backend import DeepMDDipoleBackend, DeepMDPolarBackend, DeepMDPotBackend


def sample_items(hdf5_paths: list[Path], limit: int) -> list[pipeline.SmilesItem]:
    cfg = pipeline.PipelineConfig(
        output_dir=Path("."),
        device=torch.device("cpu"),
        hdf5_paths=hdf5_paths,
    )
    items = []
    for item in pipeline.iter_hdf5(cfg):
        items.append(item)
        if limit and len(items) >= limit:
            break
    return items


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def eval_energy(model: DeepMDPotBackend, items: Iterable[pipeline.SmilesItem]) -> tuple[float, int]:
    errors = []
    for item in items:
        if item.energy is None:
            continue
        if not model.supports(item.z):
            continue
        pred = model.energy(item.pos, item.z)
        target = item.energy
        if pred.ndim == 1 and pred.numel() == 1:
            pred = pred.view(1, 1)
        if target.ndim == 0:
            target = target.view(1, 1)
        errors.append(mae(pred, target))
    return (sum(errors) / len(errors), len(errors)) if errors else (float("nan"), 0)


def eval_dipole(model: DeepMDDipoleBackend, items: Iterable[pipeline.SmilesItem]) -> tuple[float, int]:
    errors = []
    for item in items:
        if item.dipole is None:
            continue
        if not model.supports(item.z):
            continue
        pred = model.dipole(item.pos, item.z).view(1, 3)
        target = item.dipole
        if target.ndim == 1:
            target = target.view(1, 3)
        errors.append(mae(pred, target))
    return (sum(errors) / len(errors), len(errors)) if errors else (float("nan"), 0)


def eval_polar(model: DeepMDPolarBackend, items: Iterable[pipeline.SmilesItem]) -> tuple[float, int]:
    errors = []
    for item in items:
        if item.polar is None:
            continue
        if not model.supports(item.z):
            continue
        pred = model.polar(item.pos, item.z).view(1, 3, 3)
        target = item.polar
        if target.ndim == 2:
            target = target.unsqueeze(0)
        errors.append(mae(pred, target))
    return (sum(errors) / len(errors), len(errors)) if errors else (float("nan"), 0)


def try_load_pot(path: Path, type_map: Optional[str]) -> Optional[DeepMDPotBackend]:
    try:
        return DeepMDPotBackend(device=torch.device("cpu"), model_path=str(path), type_map=type_map)
    except Exception:
        return None


def try_load_dipole(path: Path, type_map: Optional[str], dipole_unit: str) -> Optional[DeepMDDipoleBackend]:
    try:
        return DeepMDDipoleBackend(
            device=torch.device("cpu"),
            model_path=str(path),
            type_map=type_map,
            dipole_unit=dipole_unit,
        )
    except Exception:
        return None


def try_load_polar(path: Path, type_map: Optional[str]) -> Optional[DeepMDPolarBackend]:
    try:
        return DeepMDPolarBackend(device=torch.device("cpu"), model_path=str(path), type_map=type_map)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepMD checkpoints on HDF5 datasets.")
    parser.add_argument(
        "--hdf5-path",
        action="append",
        default=[],
        help="HDF5 dataset path (repeatable).",
    )
    parser.add_argument("--checkpoints-dir", default="data-gen-pipeline/checkpoints")
    parser.add_argument("--type-map", default=None, help="Comma-separated element symbols for DeepMD type map.")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to evaluate.")
    parser.add_argument("--dipole-unit", default="au", choices=("au", "debye"))
    args = parser.parse_args()

    hdf5_paths = [Path(p) for p in args.hdf5_path] if args.hdf5_path else []
    if not hdf5_paths:
        default = Path("Datasets/SPICE-2.0.1.hdf5")
        if default.exists():
            hdf5_paths = [default]
        else:
            raise RuntimeError("Provide at least one --hdf5-path.")

    items = sample_items(hdf5_paths, args.limit)
    if not items:
        raise RuntimeError("No samples found in HDF5 datasets.")

    ckpt_dir = Path(args.checkpoints_dir)
    ckpts = sorted([p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in (".pt", ".pth", ".pb", ".model")])
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {ckpt_dir}")

    print(f"Evaluating {len(ckpts)} checkpoints on {len(items)} samples")

    for path in ckpts:
        pot = try_load_pot(path, args.type_map)
        dip = try_load_dipole(path, args.type_map, args.dipole_unit)
        polar = try_load_polar(path, args.type_map)

        if pot is None and dip is None and polar is None:
            print(f"- {path.name}: unable to load as pot/dipole/polar")
            continue

        print(f"- {path.name}")
        if pot is not None:
            mae_e, n_e = eval_energy(pot, items)
            print(f"  energy:  MAE={mae_e:.6g} (n={n_e})")
        if dip is not None:
            mae_d, n_d = eval_dipole(dip, items)
            print(f"  dipole:  MAE={mae_d:.6g} (n={n_d})")
        if polar is not None:
            mae_p, n_p = eval_polar(polar, items)
            print(f"  polar:   MAE={mae_p:.6g} (n={n_p})")


if __name__ == "__main__":
    main()
