"""
Standalone script to build the DFT alignment dataset cache.

Usage:
    python ramanchembl_pipeline/build_dataset_cache.py --max-cases 10000

This extracts the model-loading and inference code from
raman_alignment_pipeline.ipynb into a runnable script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CAPSULE_CODE = REPO_ROOT / "capsule-3259363" / "code"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CAPSULE_CODE))

from train.train_detanet import build_model
from detanet_model.constant import atom_masses
from detanet_model.spectra_simulator import (
    Lorenz_broadening,
    chain_rule_raman,
    get_raman_act,
    get_raman_intensity,
    hessfreq,
)
from torch_geometric.nn import radius_graph

# ---------------------------------------------------------------------------
# Constants (must match raman_alignment_pipeline.ipynb)
# ---------------------------------------------------------------------------
SIGMA = 12.0
TEMP = 298.0
INIT_WL = 532.0
FREQ_SCALE_FACTOR = 0.967
X_MIN, X_MAX, N_POINTS = 500.0, 4000.0, 3501
X_GRID = np.linspace(float(X_MIN), float(X_MAX), int(N_POINTS), dtype=np.float64)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "spectra_queue" / "prodq-depolar-a100x8-20260219-044935"
WEIGHT_PATHS = {
    "Hi": [
        REPO_ROOT / "artifacts" / "hi" / "prod-hi-a10080x8-clean-20260224-182057" / "latest_Hi.pth",
    ],
    "Hij": [
        REPO_ROOT / "artifacts" / "hij" / "prod-hij-a10080x8-2ep-20260224-232300" / "latest_Hij.pth",
    ],
}

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Model loading (extracted from notebook)
# ---------------------------------------------------------------------------
def _build_args_from_config(cfg: dict, device: str, task_override: str | None = None):
    import argparse as _ap
    task_name = task_override or cfg.get("task", "depolar")
    return _ap.Namespace(
        task=task_name,
        num_features=cfg.get("num_features", 160),
        num_block=cfg.get("num_block", 4),
        num_radial=cfg.get("num_radial", 32),
        attention_head=cfg.get("attention_head", 8),
        rc=cfg.get("rc", 5.0),
        dropout=cfg.get("dropout", 0.1),
        pre_layernorm=cfg.get("pre_layernorm", True),
        pre_layernorm_eps=cfg.get("pre_layernorm_eps", 1e-5),
        elora_path=cfg.get("elora_path", "vendored"),
        device=device,
        use_adalora=cfg.get("use_adalora", True),
        adalora_r=cfg.get("adalora_r", 256),
        adalora_alpha=cfg.get("adalora_alpha", 512),
        adalora_dropout=cfg.get("adalora_dropout", 0.1),
        adalora_tinit=cfg.get("adalora_tinit", 10),
        adalora_tfinal=cfg.get("adalora_tfinal", 20),
        adalora_total_step=cfg.get("adalora_total_step", 1000),
        adalora_target_r=cfg.get("adalora_target_r", 128),
        adalora_rslora=cfg.get("adalora_rslora", True),
        adalora_targets=cfg.get("adalora_targets", None),
        adalora_scalar_heads=cfg.get("adalora_scalar_heads", True),
        adalora_attention=cfg.get("adalora_attention", True),
        adalora_all_linears=cfg.get("adalora_all_linears", True),
        adapter_unfreeze_initial=cfg.get("adapter_unfreeze_initial", True),
        adapter_unfreeze_prefixes=cfg.get("adapter_unfreeze_prefixes", None),
        adapter_freeze_base=cfg.get("adapter_freeze_base", True),
    )


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint payload is not dict-like: {type(obj)}")
    if any(k.startswith("module.") for k in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def load_model(task: str, ckpt_paths: list[Path], base_cfg: dict, device: str):
    for ckpt in ckpt_paths:
        if not ckpt.exists():
            continue
        args = _build_args_from_config(base_cfg, device=device, task_override=task)
        model = build_model(args)
        loaded = torch.load(ckpt, map_location=device, weights_only=False)
        state = _extract_state_dict(loaded)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[{task}] loaded {ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
        return model
    raise RuntimeError(f"No checkpoint found for {task}")


def load_depolar(artifact_dir: Path, device: str):
    config_path = artifact_dir / "config.json"
    weights_path = artifact_dir / "latest_depolar.pth"
    cfg = json.loads(config_path.read_text())
    args = _build_args_from_config(cfg, device=device, task_override="depolar")
    model = build_model(args)
    loaded = torch.load(weights_path, map_location=device, weights_only=False)
    state = _extract_state_dict(loaded)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[depolar] loaded {weights_path}")
    return model, cfg


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------
def normalize_signal(y):
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, None)
    m = float(np.max(y)) if y.size else 0.0
    if m <= 0:
        return np.zeros_like(y)
    return y / m


def lines_to_norm_spectrum(freq, activity, x_grid, sigma=SIGMA, temp=TEMP, init_wl=INIT_WL):
    freq = np.asarray(freq, dtype=np.float64)
    activity = np.asarray(activity, dtype=np.float64)
    valid = np.isfinite(freq) & np.isfinite(activity) & (freq > 1e-8)
    freq, activity = freq[valid], activity[valid]
    if freq.size == 0:
        return np.zeros_like(x_grid, dtype=np.float64)
    x_t = torch.as_tensor(x_grid, dtype=torch.float64)
    f_t = torch.as_tensor(freq, dtype=torch.float64)
    a_t = torch.as_tensor(activity, dtype=torch.float64)
    broadened = Lorenz_broadening(f_t, a_t, c=x_t, sigma=float(sigma))
    spec = get_raman_intensity(x_t, broadened, temp=float(temp), init_wl=float(init_wl)).detach().cpu().numpy()
    return normalize_signal(spec)


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def make_predict_fn(depolar_model, hi_model, hij_model, device_t):
    def predict_raman_from_geometry(pos, z, x_grid):
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device_t, requires_grad=True)
        z_t = torch.tensor(z, dtype=torch.long, device=device_t)
        edge_index = radius_graph(x=pos_t, r=5.0)

        with torch.enable_grad():
            hi = hi_model(pos=pos_t, z=z_t)
            hij = hij_model(pos=pos_t, z=z_t, edge_index=edge_index)
            dp = depolar_model(z=z_t, pos=pos_t)
            freq, modes = hessfreq(
                Hi=hi, Hij=hij, edge_index=edge_index,
                masses=atom_masses[z_t], normal=False, linear=False, scale=1.0,
            )
            raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

        freq = torch.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
        raman_act = torch.nan_to_num(raman_act, nan=0.0, posinf=0.0, neginf=0.0)
        freq_np = freq.detach().cpu().numpy().astype(np.float64)
        act_np = raman_act.detach().cpu().numpy().astype(np.float64)
        valid = np.isfinite(freq_np) & np.isfinite(act_np) & (freq_np > 1e-8)
        freq_np = freq_np[valid] * float(FREQ_SCALE_FACTOR)
        act_np = act_np[valid]
        y_pred = lines_to_norm_spectrum(freq_np, act_np, x_grid)
        return y_pred, freq_np, act_np

    return predict_raman_from_geometry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build DFT alignment dataset cache")
    parser.add_argument("--max-cases", type=int, default=10000)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print(f"Loading models on {args.device} ...")
    device_t = torch.device(args.device)
    depolar_model, depolar_cfg = load_depolar(ARTIFACT_DIR, args.device)
    hi_model = load_model("Hi", WEIGHT_PATHS["Hi"], depolar_cfg, args.device)
    hij_model = load_model("Hij", WEIGHT_PATHS["Hij"], depolar_cfg, args.device)

    predict_fn = make_predict_fn(depolar_model, hi_model, hij_model, device_t)

    import ramanchembl_pipeline.alignment_notebook_lib as lib
    import ramanchembl_pipeline.stats_notebook_lib as stats_lib
    lib.stats_lib = stats_lib

    cache_dir = REPO_ROOT / "ramanchembl_pipeline" / "artifacts" / "alignment" / "cache"
    db_path = REPO_ROOT / "ramanchembl_pipeline" / "dataset" / "molecule.db"

    print(f"Building dataset cache: max_cases={args.max_cases}, db={db_path}")
    t0 = time.time()

    ds = lib.build_dft_mode_alignment_dataset(
        db_path=db_path,
        predict_fn=predict_fn,
        x_grid=X_GRID,
        lines_to_spectrum_fn=lambda freq, inten, xg: lines_to_norm_spectrum(freq, inten, xg),
        cache_dir=cache_dir,
        max_cases=args.max_cases,
        sample_seed=20260309,  # match SpectralAlignmentTrainConfig.seed
        pred_freq_scale_factor=1.0,
        refresh=True,
    )

    elapsed = time.time() - t0
    print(f"Built {len(ds)} cases in {elapsed/60:.1f} minutes")
    print(f"Cache: {cache_dir}/dft_point_v1_{args.max_cases}.npz")


if __name__ == "__main__":
    main()
