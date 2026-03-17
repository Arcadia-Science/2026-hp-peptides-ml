"""
Parallel DFT alignment dataset builder.

Architecture:
  1. Pre-decode all DFT blobs (fast, single pass)
  2. Batched GNN inference (8 molecules per forward pass — ~8x faster)
  3. 12 worker processes for hessfreq + broadening + mode features
  4. Bounded in-flight queue to control memory

Usage:
    python -u ramanchembl_pipeline/build_dataset_cache.py --max-cases 10000 --workers 12 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Element type mapping for mode features (10 types)
ELEMENT_BINS = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 9: 5, 17: 6, 35: 7, 15: 8}  # H,C,N,O,S,F,Cl,Br,P
N_ELEMENT_TYPES = 10  # 9 named + 1 "other"
N_MODE_FEATURES = N_ELEMENT_TYPES + 2  # element_rms(10) + participation_ratio(1) + max_disp(1)


# ---------------------------------------------------------------------------
# Model loading
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
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[{task}] loaded {ckpt}", flush=True)
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
    print(f"[depolar] loaded {weights_path}", flush=True)
    return model, cfg


# ---------------------------------------------------------------------------
# Post-GNN processing (runs in worker processes)
# ---------------------------------------------------------------------------
def _compute_mode_features(modes_np, z_np):
    """
    Extract per-mode character features from eigenvectors.

    modes_np: (num_modes, num_atoms, 3) — displacement vectors
    z_np: (num_atoms,) — atomic numbers

    Returns: (num_modes, N_MODE_FEATURES) float32 array
        [0:10]  per-element-type RMS displacement (H, C, N, O, S, F, Cl, Br, P, other)
        [10]    participation ratio
        [11]    max atom displacement magnitude
    """
    n_modes, n_atoms, _ = modes_np.shape
    feats = np.zeros((n_modes, N_MODE_FEATURES), dtype=np.float32)

    disp_mag = np.linalg.norm(modes_np, axis=2)
    atom_bins = np.array([ELEMENT_BINS.get(int(z), N_ELEMENT_TYPES - 1) for z in z_np])

    for k in range(N_ELEMENT_TYPES):
        mask = atom_bins == k
        if mask.any():
            feats[:, k] = np.sqrt((disp_mag[:, mask] ** 2).mean(axis=1))

    d2 = disp_mag ** 2
    sum_d2 = d2.sum(axis=1)
    sum_d4 = (d2 ** 2).sum(axis=1)
    feats[:, N_ELEMENT_TYPES] = np.where(
        sum_d4 > 1e-30, sum_d2 ** 2 / (n_atoms * sum_d4), 0.0
    )
    feats[:, N_ELEMENT_TYPES + 1] = disp_mag.max(axis=1)

    return feats


def process_post_gnn(args):
    """Worker function: hessfreq + chain_rule_raman + broadening + mode features."""
    os.environ['OMP_NUM_THREADS'] = '1'

    (mol_id, smiles, hi_np, hij_np, dp_np, edge_index_np,
     masses_np, pos_np, z_np, x_grid, freq_scale) = args

    hi = torch.as_tensor(hi_np)
    hij = torch.as_tensor(hij_np)
    dp = torch.as_tensor(dp_np)
    edge_index = torch.as_tensor(edge_index_np)
    masses = torch.as_tensor(masses_np)

    freq, modes = hessfreq(
        Hi=hi, Hij=hij, edge_index=edge_index,
        masses=masses, normal=False, linear=False, scale=1.0,
    )
    raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

    freq = torch.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
    raman_act = torch.nan_to_num(raman_act, nan=0.0, posinf=0.0, neginf=0.0)

    freq_np = freq.detach().numpy().astype(np.float64)
    act_np = raman_act.detach().numpy().astype(np.float64)
    modes_ret = modes.detach().numpy().astype(np.float32)

    valid = np.isfinite(freq_np) & np.isfinite(act_np) & (freq_np > 1e-8)
    freq_np = freq_np[valid] * freq_scale
    act_np = act_np[valid]
    modes_ret = modes_ret[valid]

    if modes_ret.shape[0] > 0:
        mode_feats = _compute_mode_features(modes_ret, z_np)
    else:
        mode_feats = np.zeros((0, N_MODE_FEATURES), dtype=np.float32)

    x_t = torch.as_tensor(x_grid, dtype=torch.float64)
    if freq_np.size > 0:
        f_t = torch.as_tensor(freq_np, dtype=torch.float64)
        a_t = torch.as_tensor(act_np, dtype=torch.float64)
        broadened = Lorenz_broadening(f_t, a_t, c=x_t, sigma=SIGMA)
        spec = get_raman_intensity(x_t, broadened, temp=TEMP, init_wl=INIT_WL).numpy()
        y_pred = spec / (spec.max() + 1e-30) if spec.max() > 0 else np.zeros_like(spec)
    else:
        y_pred = np.zeros(len(x_grid), dtype=np.float64)

    return {
        'mol_id': mol_id,
        'smiles': smiles,
        'pred_freq': freq_np.astype(np.float32),
        'pred_intensity': act_np.astype(np.float32),
        'mode_features': mode_feats,
        'y_pred_spec': y_pred.astype(np.float32),
    }


def normalize_signal(y):
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, None)
    m = float(np.max(y)) if y.size else 0.0
    return y / m if m > 0 else np.zeros_like(y)


# ---------------------------------------------------------------------------
# Batched GNN forward — B molecules in one forward pass
# Follows spectra_simulator.py lines 140-175 for correct batching
# ---------------------------------------------------------------------------
def batched_gnn_forward(mol_batch, hi_model, hij_model, depolar_model, device_t):
    """
    Run GNN on a batch of molecules simultaneously.

    mol_batch: list of (mid, smiles, pos_np, z_np) tuples
    Returns: list of per-molecule dicts with GNN outputs as numpy arrays
    """
    all_pos, all_z, batch_idx = [], [], []
    offsets = [0]
    for i, (mid, smiles, pos, z) in enumerate(mol_batch):
        n = len(z)
        all_pos.append(torch.tensor(pos, dtype=torch.float32))
        all_z.append(torch.tensor(z, dtype=torch.long))
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        offsets.append(offsets[-1] + n)

    pos_t = torch.cat(all_pos).to(device_t).requires_grad_(True)
    z_t = torch.cat(all_z).to(device_t)
    batch_t = torch.cat(batch_idx).to(device_t)

    # radius_graph with batch → no cross-molecule edges
    edge_index = radius_graph(x=pos_t, r=5.0, batch=batch_t)

    # Batched GNN forward (matches spectra_simulator.py lines 144-147)
    with torch.enable_grad():
        Hi = hi_model(pos=pos_t, z=z_t, batch=batch_t)
        Hij = hij_model(pos=pos_t, z=z_t, edge_index=edge_index, batch=batch_t)
        dp = depolar_model(z=z_t, pos=pos_t, batch=batch_t)

    # Split per molecule (matches spectra_simulator.py lines 158-167)
    out = []
    for i, (mid, smiles, pos, z) in enumerate(mol_batch):
        atom_mask = batch_t == i

        hi_i = Hi[atom_mask].detach().cpu().numpy()
        dp_i = dp[atom_mask].detach().cpu().numpy()

        # Edges for molecule i
        edge_mask = atom_mask[edge_index[0]] & atom_mask[edge_index[1]]
        hij_i = Hij[edge_mask].detach().cpu().numpy()
        edge_index_i = (edge_index[:, edge_mask] - offsets[i]).cpu().numpy()

        out.append({
            'mol_id': mid,
            'smiles': smiles,
            'hi': hi_i,
            'hij': hij_i,
            'dp': dp_i,
            'edge_index': edge_index_i,
            'masses': atom_masses[z_t[atom_mask]].cpu().numpy(),
            'pos': np.array(pos, dtype=np.float32),
            'z': np.array(z, dtype=np.int64),
        })

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build DFT alignment dataset cache (parallel)")
    parser.add_argument("--max-cases", type=int, default=10000)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="GNN batch size (molecules per forward pass)")
    parser.add_argument("--seed", type=int, default=20260309)
    args = parser.parse_args()

    print(f"Config: workers={args.workers}, gnn_batch={args.batch_size}, "
          f"device={args.device}, max_cases={args.max_cases}", flush=True)

    print(f"Loading models on {args.device} ...", flush=True)
    device_t = torch.device(args.device)
    depolar_model, depolar_cfg = load_depolar(ARTIFACT_DIR, args.device)
    hi_model = load_model("Hi", WEIGHT_PATHS["Hi"], depolar_cfg, args.device)
    hij_model = load_model("Hij", WEIGHT_PATHS["Hij"], depolar_cfg, args.device)

    import ramanchembl_pipeline.stats_notebook_lib as stats_lib

    db_path = REPO_ROOT / "ramanchembl_pipeline" / "dataset" / "molecule.db"
    con = sqlite3.connect(str(db_path))
    all_ids = [r[0] for r in con.execute("SELECT id FROM molecule").fetchall()]
    rng = np.random.default_rng(args.seed)
    sel_ids = rng.choice(all_ids, size=min(args.max_cases, len(all_ids)), replace=False).tolist()
    placeholders = ",".join("?" * len(sel_ids))
    rows = con.execute(
        f"SELECT id, SMILES, blob_data FROM molecule WHERE id IN ({placeholders})", sel_ids
    ).fetchall()
    con.close()
    print(f"Loaded {len(rows)} molecules from DB", flush=True)

    # Step 1: Pre-decode all blobs
    from tqdm import tqdm
    t0 = time.time()

    decoded = []  # (mid, smiles, pos, z)
    blob_lookup = {}  # mid → blob (for DFT targets later)
    for mid, smiles, blob in tqdm(rows, desc="Decoding blobs", unit="mol"):
        try:
            payload = stats_lib._decode_dft_blob(blob)
            decoded.append((mid, smiles, payload["coord"], payload["atoms"]))
            blob_lookup[mid] = blob
        except Exception as e:
            print(f"Decode error mol {mid}: {e}", flush=True)

    print(f"Decoded {len(decoded)} in {time.time()-t0:.0f}s", flush=True)

    # Step 2: Batched GNN → parallel hessfreq
    cache_dir = REPO_ROOT / "ramanchembl_pipeline" / "artifacts" / "alignment" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    errors = 0
    max_inflight = args.workers * 3

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        batch_buf = []
        pbar = tqdm(total=len(decoded), desc="GNN+hessfreq", unit="mol")

        def drain_completed():
            nonlocal errors
            done = [f for f in futures if f.done()]
            for f in done:
                mid = futures.pop(f)
                try:
                    results[mid] = f.result()
                except Exception as e:
                    print(f"Worker error mol {mid}: {e}", flush=True)
                    errors += 1
                pbar.update(1)

        def flush_batch():
            nonlocal errors
            if not batch_buf:
                return
            try:
                gnn_outputs = batched_gnn_forward(
                    batch_buf, hi_model, hij_model, depolar_model, device_t
                )
                for gout in gnn_outputs:
                    worker_args = (
                        gout['mol_id'], gout['smiles'],
                        gout['hi'], gout['hij'], gout['dp'],
                        gout['edge_index'], gout['masses'],
                        gout['pos'], gout['z'],
                        X_GRID, FREQ_SCALE_FACTOR,
                    )
                    future = pool.submit(process_post_gnn, worker_args)
                    futures[future] = gout['mol_id']
            except Exception as e:
                for mid, smiles, pos, z in batch_buf:
                    print(f"GNN batch error (mol {mid}): {e}", flush=True)
                    errors += 1
                    pbar.update(1)

        for item in decoded:
            batch_buf.append(item)

            if len(batch_buf) >= args.batch_size:
                flush_batch()
                batch_buf = []

                # Bound in-flight futures
                while len(futures) >= max_inflight:
                    drain_completed()
                    if len(futures) >= max_inflight:
                        next(as_completed(futures))
                        drain_completed()

        # Flush last partial batch
        flush_batch()
        batch_buf = []

        # Drain all remaining
        for f in as_completed(list(futures.keys())):
            mid = futures.pop(f)
            try:
                results[mid] = f.result()
            except Exception as e:
                print(f"Worker error mol {mid}: {e}", flush=True)
                errors += 1
            pbar.update(1)

        pbar.close()

    print(f"Done: {len(results)} OK, {errors} errors, "
          f"{time.time()-t0:.0f}s elapsed", flush=True)

    # Step 3: Assemble arrays + save
    print("Building arrays...", flush=True)

    pf_l, pi_l, tf_l, ti_l, mf_l, yps_l, yts_l, mode_feats_l, meta = (
        [], [], [], [], [], [], [], [], []
    )

    for mid, smiles, pos, z in tqdm(decoded, desc="Assembling", unit="mol"):
        if mid not in results:
            continue
        r = results[mid]
        payload = stats_lib._decode_dft_blob(blob_lookup[mid])
        tf, ti = payload["freq"], payload["Raman Activ"]

        pos_arr = np.array(pos, dtype=np.float32)
        z_arr = np.array(z, dtype=np.int64)
        centroid = pos_arr.mean(axis=0)
        dists = np.linalg.norm(pos_arr - centroid, axis=1)
        n_atoms = len(z_arr)
        geom_feats = np.array([
            n_atoms,
            dists.mean(), dists.std(), dists.max(),
            (z_arr == 1).sum(), (z_arr == 6).sum(), (z_arr == 7).sum(),
            (z_arr == 8).sum(), (z_arr == 16).sum(), (z_arr == 9).sum(),
            (z_arr == 17).sum(), (z_arr == 35).sum(), (z_arr == 15).sum(),
            pos_arr.std(axis=0).mean(),
            np.linalg.norm(pos_arr.max(axis=0) - pos_arr.min(axis=0)),
            float(n_atoms > 50),
        ], dtype=np.float32)

        pf_l.append(r['pred_freq'])
        pi_l.append(r['pred_intensity'])
        tf_l.append(np.asarray(tf, dtype=np.float32))
        ti_l.append(np.asarray(ti, dtype=np.float32))
        mf_l.append(geom_feats)
        mode_feats_l.append(r['mode_features'])
        yps_l.append(r['y_pred_spec'])

        y_target = normalize_signal(
            Lorenz_broadening(
                torch.as_tensor(np.asarray(tf, dtype=np.float64)),
                torch.as_tensor(np.asarray(ti, dtype=np.float64)),
                c=torch.as_tensor(X_GRID, dtype=torch.float64),
                sigma=SIGMA,
            ).numpy()
        )
        yts_l.append(y_target.astype(np.float32))
        meta.append({"molecule_id": mid, "smiles": smiles})

    if not meta:
        print("ERROR: No molecules processed!", flush=True)
        sys.exit(1)

    def pad(lst, val=0):
        ml = max(len(x) for x in lst) if lst else 0
        res = np.full((len(lst), ml), val, dtype=np.float32)
        mask = np.zeros((len(lst), ml), dtype=np.float32)
        for i, x in enumerate(lst):
            n = min(len(x), ml)
            res[i, :n] = x[:n]
            mask[i, :n] = 1.0
        return res, mask

    pf, pm = pad(pf_l)
    pi, _ = pad(pi_l)
    tf, tm = pad(tf_l)
    ti, _ = pad(ti_l)

    max_modes = pf.shape[1]
    mode_features_arr = np.zeros((len(mode_feats_l), max_modes, N_MODE_FEATURES), dtype=np.float32)
    for i, mf in enumerate(mode_feats_l):
        n = min(len(mf), max_modes)
        if n > 0:
            mode_features_arr[i, :n] = mf[:n]

    import pandas as pd
    from scipy.optimize import linear_sum_assignment as _lsa

    MODE_TRAIN_MATCH_CUTOFF_CM = 60.0
    N = pf.shape[0]
    max_p = pf.shape[1]
    mi_arr = np.full((N, max_p), -1, dtype=np.int32)
    mm_arr = np.zeros((N, max_p), dtype=np.float32)
    for i in range(N):
        p_valid = np.where(pm[i] > 0.5)[0]
        t_valid = np.where(tm[i] > 0.5)[0]
        if len(p_valid) == 0 or len(t_valid) == 0:
            continue
        cost = np.abs(pf[i][p_valid, None] - tf[i][None, t_valid])
        p_idx, t_idx = _lsa(cost)
        for pi_idx, ti_idx in zip(p_idx, t_idx):
            if cost[pi_idx, ti_idx] <= MODE_TRAIN_MATCH_CUTOFF_CM:
                mi_arr[i, p_valid[pi_idx]] = int(t_valid[ti_idx])
                mm_arr[i, p_valid[pi_idx]] = 1.0

    npz_p = cache_dir / f"dft_point_v1_{args.max_cases}.npz"
    csv_p = cache_dir / f"dft_point_v1_{args.max_cases}.csv"

    np.savez_compressed(
        npz_p,
        x_grid=X_GRID,
        mol_features=np.stack(mf_l),
        pred_freq=pf, pred_intensity=pi, pred_mask=pm,
        target_freq=tf, target_intensity=ti, target_mask=tm,
        match_target_idx=mi_arr, match_mask=mm_arr,
        y_pred_spec=np.stack(yps_l), y_target_spec=np.stack(yts_l),
        mode_features=mode_features_arr,
    )
    pd.DataFrame(meta).to_csv(csv_p, index=False)

    elapsed = time.time() - t0
    print(f"\nBuilt {len(meta)} cases in {elapsed/60:.1f} minutes", flush=True)
    print(f"Cache: {npz_p} ({npz_p.stat().st_size / 1e6:.0f} MB)", flush=True)
    print(f"Mode features shape: {mode_features_arr.shape}", flush=True)


if __name__ == "__main__":
    main()
