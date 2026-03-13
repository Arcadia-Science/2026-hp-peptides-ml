from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from scipy.optimize import linear_sum_assignment as _lsa

if os.environ.get("MPLCONFIGDIR") is None:
    _mpl_dir = Path.cwd() / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from ramanchembl_pipeline import stats_notebook_lib as stats_lib
except ImportError:
    import stats_notebook_lib as stats_lib  # standalone / Modal container

EPS = 1e-12
ALIGNMENT_TOLS = (5.0, 10.0, 15.0, 20.0)
MODE_TRAIN_MATCH_CUTOFF_CM = 60.0

@dataclass
class AlignmentDatasetBundle:
    domain: str
    x_grid: np.ndarray
    y_pred: np.ndarray
    y_target: np.ndarray
    mask: np.ndarray
    mol_features: np.ndarray
    metadata: pd.DataFrame
    cache_npz: Path
    cache_csv: Path

    def __len__(self) -> int:
        return int(self.y_pred.shape[0])

@dataclass
class DFTModeAlignmentDatasetBundle:
    domain: str
    x_grid: np.ndarray
    mol_features: np.ndarray
    pred_freq: np.ndarray
    pred_intensity: np.ndarray
    pred_mask: np.ndarray
    target_freq: np.ndarray
    target_intensity: np.ndarray
    target_mask: np.ndarray
    match_target_idx: np.ndarray
    match_mask: np.ndarray
    y_pred_spec: np.ndarray
    y_target_spec: np.ndarray
    metadata: pd.DataFrame
    cache_npz: Path
    cache_csv: Path

    def __len__(self) -> int:
        return int(self.pred_freq.shape[0])

@dataclass
class AlignmentTrainConfig:
    seed: int = 20260309
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-3
    # Architecture — scale with dataset size:
    #   <500 mols:  latent=64,  heads=4, layers=2
    #   500-2000:   latent=128, heads=8, layers=4  (defaults below)
    #   >2000:      latent=256, heads=8, layers=6
    latent_dim: int = 128
    mol_latent_dim: int = 64
    transformer_heads: int = 8
    transformer_layers: int = 4
    string_feature_dim: int = 128  # Morgan fingerprint dim
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_freq_delta: float = 150.0  # Max cm^-1 shift the model can output
    coverage_loss_weight: float = 2.0
    coverage_target_cm: float = 10.0  # cm^-1 where Huber transitions quadratic→linear
    confidence_loss_weight: float = 1.0
    confidence_threshold: float = 0.5  # eval-time cutoff
    repulsion_loss_weight: float = 0.5
    repulsion_radius_cm: float = 5.0
    match_cutoff: float = 15.0  # cm^-1 cutoff for ground-truth matching labels
    freq_loss_weight: float = 1.0  # weight for frequency correction loss

class ModeArrayDataset(Dataset):
    def __init__(self, mol_features, pf, pi, pm, tf, ti, tm, mi, mm):
        self.mol_features = torch.as_tensor(mol_features, dtype=torch.float32)
        self.pf = torch.as_tensor(pf, dtype=torch.float32)
        self.pi = torch.as_tensor(pi, dtype=torch.float32)
        self.pm = torch.as_tensor(pm, dtype=torch.float32)
        self.tf = torch.as_tensor(tf, dtype=torch.float32)
        self.ti = torch.as_tensor(ti, dtype=torch.float32)
        self.tm = torch.as_tensor(tm, dtype=torch.float32)
        self.mi = torch.as_tensor(mi, dtype=torch.long)
        self.mm = torch.as_tensor(mm, dtype=torch.float32)

    def __len__(self): return len(self.pf)

    def __getitem__(self, idx):
        return (self.mol_features[idx], self.pf[idx], self.pi[idx], self.pm[idx], 
                self.tf[idx], self.ti[idx], self.tm[idx], self.mi[idx], self.mm[idx])

class PeakCoordinateTransformer(nn.Module):
    """
    Function f(SMILES hash, predicted peaks) -> Corrected peaks.
    Uses point registration logic to align (x, y) coordinates exactly.
    """
    def __init__(self, mol_dim: int, cfg: AlignmentTrainConfig, x_grid: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("x_grid", torch.as_tensor(x_grid, dtype=torch.float32))
        
        # Molecule identity encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, cfg.mol_latent_dim),
            nn.GELU(),
            nn.Linear(cfg.mol_latent_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim)
        )
        
        # Peak property encoder (frequency, intensity, rank, local gaps)
        self.peak_embed = nn.Linear(8, cfg.latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.latent_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        
        # Correction heads
        self.freq_shift_head = nn.Linear(cfg.latent_dim, 1)
        self.intensity_head = nn.Linear(cfg.latent_dim, 1)
        # v2: confidence head — "is this predicted mode real or noise?"
        self.confidence_head = nn.Linear(cfg.latent_dim, 1)

    def forward(self, mol_features, pred_freq, pred_intensity, pred_mask):
        # 1. Molecule conditioning
        mol_token = self.mol_encoder(mol_features).unsqueeze(1)

        # 2. Extract peak-level features (local structure)
        peak_feats = _build_mode_features(pred_freq, pred_intensity, pred_mask, self.x_grid)
        peak_tokens = self.peak_embed(peak_feats)

        # 3. Transformer processing (Attention over all peaks + molecule context)
        combined = torch.cat([mol_token, peak_tokens], dim=1)
        # Pad mask: True means "ignore"
        padding_mask = torch.cat([
            torch.zeros((mol_features.shape[0], 1), device=mol_features.device, dtype=torch.bool),
            pred_mask < 0.5
        ], dim=1)

        encoded = self.transformer(combined, src_key_padding_mask=padding_mask)
        peak_out = encoded[:, 1:, :] # Extract only peak tokens

        # 4. Predict Coordinate Adjustments
        # Clamp instead of tanh — gradient is 1.0 everywhere inside the window,
        # so the model can learn large corrections without fighting saturation
        delta_f = torch.clamp(self.freq_shift_head(peak_out).squeeze(-1),
                              -self.cfg.max_freq_delta, self.cfg.max_freq_delta)
        corrected_f = (pred_freq + delta_f) * pred_mask

        # Intensity must be [0, 1]
        corrected_i = torch.sigmoid(self.intensity_head(peak_out).squeeze(-1)) * pred_mask

        # v2: confidence — probability this mode is real (has a DFT match)
        confidence = torch.sigmoid(self.confidence_head(peak_out).squeeze(-1)) * pred_mask

        return {"corrected_freq": corrected_f, "corrected_intensity": corrected_i, "confidence": confidence}

def _supervised_alignment_loss(out, tf, ti, pm, tm, mi, mm, cfg):
    """
    v4 loss: direct supervised regression with per-component logging.

    Key change from v3: frequency loss in raw cm⁻¹ space (no /10 normalization)
    with delta=10.0 Huber, giving 10x stronger gradients for frequency correction.
    Confidence and repulsion can be dialed via config weights (set to 0 to disable).
    """
    pf, pi, conf = out["corrected_freq"], out["corrected_intensity"], out["confidence"]
    batch_size = pf.shape[0]
    total_loss = 0.0
    acc_freq = acc_int = acc_conf = 0.0
    count = 0

    for b in range(batch_size):
        idx_p = pm[b] > 0.5
        if not idx_p.any():
            continue

        p_f = pf[b][idx_p]
        p_i = pi[b][idx_p]
        p_conf = conf[b][idx_p]

        m_idx = mi[b][idx_p]
        m_mask = mm[b][idx_p]
        matched = m_mask > 0.5
        n_matched = matched.sum().item()

        # 1. Frequency + intensity on matched modes (raw cm⁻¹, delta=10)
        if n_matched > 0:
            target_idx = m_idx[matched]
            t_f_matched = tf[b][target_idx]
            t_i_matched_raw = ti[b][target_idx]

            idx_t = tm[b] > 0.5
            t_i_max = ti[b][idx_t].max() + EPS if idx_t.any() else torch.tensor(EPS)
            t_i_matched = t_i_matched_raw / t_i_max

            loss_freq = F.huber_loss(p_f[matched], t_f_matched, delta=10.0)
            loss_int = F.l1_loss(p_i[matched], t_i_matched)
        else:
            loss_freq = pf[b].sum() * 0.0
            loss_int = pf[b].sum() * 0.0

        # 2. Confidence BCE (can be disabled via weight=0)
        if cfg.confidence_loss_weight > 0:
            loss_conf = F.binary_cross_entropy(
                p_conf.clamp(1e-6, 1 - 1e-6), m_mask.float(), reduction="mean"
            )
        else:
            loss_conf = pf[b].sum() * 0.0

        # 3. Repulsion (can be disabled via weight=0)
        loss_repulsion = pf[b].sum() * 0.0
        if cfg.repulsion_loss_weight > 0:
            confident = p_conf > cfg.confidence_threshold
            if confident.sum() > 1:
                cp = p_f[confident]
                pair_dist = torch.abs(cp.unsqueeze(0) - cp.unsqueeze(1))
                repel = torch.clamp(cfg.repulsion_radius_cm - pair_dist, min=0.0)
                repel = repel - torch.diag(repel.diag())
                loss_repulsion = repel.sum() / (len(cp) * cfg.repulsion_radius_cm + EPS)

        total_loss = (total_loss
                      + cfg.freq_loss_weight * (loss_freq + loss_int)
                      + cfg.confidence_loss_weight * loss_conf
                      + cfg.repulsion_loss_weight * loss_repulsion)
        acc_freq += loss_freq.item()
        acc_int += loss_int.item()
        acc_conf += loss_conf.item() if isinstance(loss_conf, torch.Tensor) else 0.0
        count += 1

    if count == 0:
        return pf.sum() * 0.0, {}
    components = {"freq": acc_freq / count, "int": acc_int / count, "conf": acc_conf / count}
    return total_loss / count, components

def _build_mode_features(pred_freq, pred_intensity, pred_mask, x_grid):
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    x_scale = max(x_max - x_min, 1.0)
    
    freq_norm = ((pred_freq - x_min) / x_scale) * pred_mask
    pred_log = torch.log(torch.clamp(pred_intensity, min=EPS)) * pred_mask
    
    # Context: distances to neighbors
    prev_f = torch.cat([pred_freq[:, :1], pred_freq[:, :-1]], dim=1)
    next_f = torch.cat([pred_freq[:, 1:], pred_freq[:, -1:]], dim=1)
    gap_p = torch.clamp((pred_freq - prev_f) / 200.0, 0, 5) * pred_mask
    gap_n = torch.clamp((next_f - pred_freq) / 200.0, 0, 5) * pred_mask
    
    # Rank in the spectrum
    rank = (torch.cumsum(pred_mask, dim=1) - 1.0) * pred_mask
    rank /= torch.clamp(pred_mask.sum(dim=1, keepdim=True), min=1.0)
    
    # Global density features
    count = (pred_mask.sum(dim=1, keepdim=True) / 100.0).expand_as(pred_freq)
    
    # Intensity rank (by descending intensity, normalized) — which peaks are strongest
    int_rank = torch.zeros_like(pred_intensity)
    for b in range(pred_intensity.shape[0]):
        valid_idx = (pred_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        if len(valid_idx) > 1:
            order = torch.argsort(pred_intensity[b][valid_idx], descending=True)
            int_rank[b][valid_idx[order]] = torch.arange(len(valid_idx), dtype=torch.float32, device=pred_intensity.device) / (len(valid_idx) - 1)
    int_rank = int_rank * pred_mask

    # Local gap asymmetry: how asymmetrically placed this peak is between neighbors
    gap_total = gap_p + gap_n + EPS
    gap_asym = (gap_p - gap_n) / gap_total * pred_mask

    return torch.stack([freq_norm, pred_log, gap_p, gap_n, rank, count, int_rank, gap_asym], dim=-1)

def _morgan_features(smiles_list, n_bits=128, radius=2):
    """ECFP4 Morgan fingerprints via RDKit. Falls back to SMILES hash if RDKit unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        feats = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smi), sanitize=False)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    feats.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
                else:
                    feats.append(_string_hash_features([smi], dim=n_bits)[0])
            except Exception:
                feats.append(_string_hash_features([smi], dim=n_bits)[0])
        return np.stack(feats)
    except ImportError:
        return _string_hash_features(smiles_list, dim=n_bits)

def _string_hash_features(strings, dim=64):
    feats = np.zeros((len(strings), dim), dtype=np.float32)
    for i, s in enumerate(strings):
        text = f"<{s}>"
        for n in (1, 2, 3):
            for j in range(max(len(text) - n + 1, 0)):
                bucket = hash((n, text[j:j+n])) % dim
                feats[i, bucket] += 1.0
        norm = np.linalg.norm(feats[i])
        if norm > EPS: feats[i] /= norm
    return feats

def _augment_mol_features(mol_features, metadata, dim=64):
    key_col = next((c for c in ("smiles", "component", "cid") if c in metadata.columns), None)
    if key_col:
        strings = metadata[key_col].astype(str).tolist()
        # Use Morgan fingerprints for SMILES (chemistry-aware > hash)
        str_feats = _morgan_features(strings, n_bits=dim) if key_col == "smiles" \
                    else _string_hash_features(strings, dim=dim)
        return np.concatenate([mol_features, str_feats], axis=1)
    return mol_features

def _compute_match_indices(pred_freq, pred_mask, target_freq, target_mask, cutoff=60.0):
    """
    For each molecule, Hungarian-match predicted modes to target modes within `cutoff` cm^-1.
    Returns:
        match_target_idx: (N, max_pred) int array; -1 for unmatched pred modes
        match_mask:       (N, max_pred) float array; 1.0 if matched, else 0.0
    """
    N, max_p = pred_freq.shape
    mi = np.full((N, max_p), -1, dtype=np.int32)
    mm = np.zeros((N, max_p), dtype=np.float32)
    for i in range(N):
        p_valid = np.where(pred_mask[i] > 0.5)[0]
        t_valid = np.where(target_mask[i] > 0.5)[0]
        if len(p_valid) == 0 or len(t_valid) == 0:
            continue
        cost = np.abs(pred_freq[i][p_valid, None] - target_freq[i][None, t_valid])
        p_idx, t_idx = _lsa(cost)
        for pi, ti in zip(p_idx, t_idx):
            if cost[pi, ti] <= cutoff:
                mi[i, p_valid[pi]] = int(t_valid[ti])
                mm[i, p_valid[pi]] = 1.0
    return mi, mm


# ---------------------------------------------------------------------------
# Frequency-dependent calibration (non-parametric)
# ---------------------------------------------------------------------------

def fit_frequency_calibration(pred_freq, target_freq, pred_mask, target_mask,
                              match_target_idx, match_mask, n_bins=80):
    """
    Fit a smooth correction curve: corrected = pred_freq + correction(pred_freq).

    Collects all matched (pred_freq, target_freq) pairs from training data,
    bins by pred_freq, computes median correction per bin, and fits a smooth
    PCHIP interpolant. This captures DeTaNet's systematic frequency-dependent
    error — it generalizes perfectly since it's a function of frequency alone.

    Returns a callable: correction_fn(freq_array) -> correction_array (cm⁻¹).
    """
    # Collect all matched pairs
    all_pf, all_tf = [], []
    N = pred_freq.shape[0]
    for i in range(N):
        p_valid = pred_mask[i] > 0.5
        m_valid = match_mask[i] > 0.5
        both = p_valid & m_valid
        if not both.any():
            continue
        pf_i = pred_freq[i][both]
        tidx = match_target_idx[i][both]
        tf_i = target_freq[i][tidx]
        all_pf.append(pf_i)
        all_tf.append(tf_i)

    all_pf = np.concatenate(all_pf)
    all_tf = np.concatenate(all_tf)
    corrections = all_tf - all_pf  # positive = need to shift pred UP

    # Bin by predicted frequency and compute robust statistics
    pf_min, pf_max = all_pf.min(), all_pf.max()
    bin_edges = np.linspace(pf_min, pf_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_medians = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for j in range(n_bins):
        mask = (all_pf >= bin_edges[j]) & (all_pf < bin_edges[j + 1])
        if j == n_bins - 1:  # include right edge
            mask |= (all_pf == bin_edges[j + 1])
        bin_counts[j] = mask.sum()
        if bin_counts[j] >= 5:
            bin_medians[j] = np.median(corrections[mask])

    # Fill empty bins by interpolation from neighbors
    valid = bin_counts >= 5
    if valid.sum() < 3:
        # Not enough data — return identity (no correction)
        return lambda freq: np.zeros_like(freq)

    # Smooth the medians slightly to avoid overfitting to bin noise
    smoothed = gaussian_filter1d(bin_medians[valid], sigma=1.5)
    interp = PchipInterpolator(bin_centers[valid], smoothed, extrapolate=True)

    stats = {
        "n_pairs": len(all_pf),
        "median_correction": float(np.median(corrections)),
        "mean_correction": float(np.mean(corrections)),
        "std_correction": float(np.std(corrections)),
        "freq_range": (float(pf_min), float(pf_max)),
    }
    print(f"Calibration: {stats['n_pairs']} matched pairs, "
          f"median correction = {stats['median_correction']:.2f} cm⁻¹, "
          f"std = {stats['std_correction']:.2f} cm⁻¹")

    return interp


def apply_frequency_calibration(pred_freq, pred_mask, calibration_fn):
    """Apply the calibration correction to predicted frequencies."""
    corrected = pred_freq.copy()
    for i in range(pred_freq.shape[0]):
        valid = pred_mask[i] > 0.5
        if valid.any():
            corrected[i][valid] += calibration_fn(pred_freq[i][valid])
    return corrected


# Core Dataset Construction Logic

def _geometry_mol_features(pos, z):
    pos = np.asarray(pos); z = np.asarray(z)
    if pos.size == 0: return np.zeros(16, dtype=np.float32)
    center = pos.mean(axis=0)
    rad = np.linalg.norm(pos - center, axis=1)
    return np.asarray([
        len(z)/100, np.mean(z)/20, np.std(z)/10, 
        np.mean(rad)/10, np.std(rad)/5, np.max(rad)/20,
        float(np.sum(z==6)/len(z)), float(np.sum(z==7)/len(z)), 
        float(np.sum(z==8)/len(z)), float(np.sum(z==16)/len(z)),
        0, 0, 0, 0, 0, 0 # Padding
    ], dtype=np.float32)

def build_experimental_alignment_dataset(
    *, exp_df, resolver_cache, predict_fn, x_grid, cache_dir, max_rows=None, max_atoms=120, refresh=False
):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    rows_tag = "all" if max_rows is None else str(int(max_rows))
    npz_p = cache_dir / f"exp_point_v1_{rows_tag}.npz"
    csv_p = cache_dir / f"exp_point_v1_{rows_tag}.csv"
    
    if npz_p.exists() and csv_p.exists() and not refresh:
        return _load_dataset_bundle(npz_p, csv_p, "experimental")

    df = exp_df.iloc[:max_rows] if max_rows else exp_df
    y_pred_all, y_target_all, mask_all, mol_feats, meta_rows = [], [], [], [], []
    
    for idx, row in df.iterrows():
        comp = str(row["component"])
        res = resolver_cache.get(comp, {})
        pos, z = res.get("pos"), res.get("z")
        if res.get("status") != "resolved" or pos is None or len(z) > max_atoms: continue
        
        try:
            yp, _, _ = predict_fn(pos, z, x_grid)
            interpolator = PchipInterpolator(row["wavenumbers_arr"], row["intensity_arr"], extrapolate=False)
            yt = np.nan_to_num(interpolator(x_grid), nan=0.0)
            yt = _normalize_signal(gaussian_filter1d(yt, 1.25))
            mask = ((x_grid >= row["wavenumbers_arr"][0]) & (x_grid <= row["wavenumbers_arr"][-1])).astype(np.float32)
            
            y_pred_all.append(_normalize_signal(yp))
            y_target_all.append(yt)
            mask_all.append(mask)
            mol_feats.append(_geometry_mol_features(pos, z))
            meta_rows.append({"component": comp, "cid": res.get("cid"), "n_atoms": len(z)})
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    np.savez_compressed(npz_p, x_grid=x_grid, y_pred=np.stack(y_pred_all), y_target=np.stack(y_target_all), 
                        mask=np.stack(mask_all), mol_features=np.stack(mol_feats))
    pd.DataFrame(meta_rows).to_csv(csv_p, index=False)
    return _load_dataset_bundle(npz_p, csv_p, "experimental")

def build_dft_mode_alignment_dataset(
    *, db_path, predict_fn, x_grid, lines_to_spectrum_fn, cache_dir, max_cases=1000, sample_seed=2026, 
    pred_freq_scale_factor=1.0, refresh=False
):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    npz_p = cache_dir / f"dft_point_v1_{max_cases}.npz"
    csv_p = cache_dir / f"dft_point_v1_{max_cases}.csv"
    
    if npz_p.exists() and csv_p.exists() and not refresh:
        return _load_dft_mode_dataset_bundle(npz_p, csv_p)

    con = sqlite3.connect(str(db_path))
    all_ids = [r[0] for r in con.execute("SELECT id FROM molecule").fetchall()]
    rng_s = np.random.default_rng(sample_seed)
    sel_ids = rng_s.choice(all_ids, size=min(max_cases, len(all_ids)), replace=False).tolist()
    placeholders = ",".join("?" * len(sel_ids))
    rows = con.execute(
        f"SELECT id, SMILES, blob_data FROM molecule WHERE id IN ({placeholders})", sel_ids
    ).fetchall()
    con.close()

    mol_feats, pf_l, pi_l, tf_l, ti_l, yps_l, yts_l, meta = [], [], [], [], [], [], [], []
    
    for mid, smiles, blob in rows:
        try:
            payload = stats_lib._decode_dft_blob(blob)
            pos, z = payload["coord"], payload["atoms"]
            _, prf, pra = predict_fn(pos, z, x_grid)
            prf = prf * pred_freq_scale_factor

            tf, ti = payload["freq"], payload["Raman Activ"]
            
            mol_feats.append(_geometry_mol_features(pos, z))
            pf_l.append(np.asarray(prf, dtype=np.float32)); pi_l.append(np.asarray(pra, dtype=np.float32))
            tf_l.append(np.asarray(tf, dtype=np.float32)); ti_l.append(np.asarray(ti, dtype=np.float32))
            yps_l.append(_normalize_signal(lines_to_spectrum_fn(prf, pra, x_grid)))
            yts_l.append(_normalize_signal(lines_to_spectrum_fn(tf, ti, x_grid)))
            meta.append({"molecule_id": mid, "smiles": smiles})
        except Exception as e:
            print(f"Error processing row {mid}: {e}")
            continue

    def pad(l, val=0):
        ml = max(len(x) for x in l)
        res = np.full((len(l), ml), val, dtype=np.float32)
        mask = np.zeros((len(l), ml), dtype=np.float32)
        for i, x in enumerate(l): 
            res[i, :len(x)] = x[:ml]
            mask[i, :len(x)] = 1.0
        return res, mask

    pf, pm = pad(pf_l); pi, _ = pad(pi_l)
    tf, tm = pad(tf_l); ti, _ = pad(ti_l)

    mi, mm = _compute_match_indices(pf, pm, tf, tm, cutoff=MODE_TRAIN_MATCH_CUTOFF_CM)
    np.savez_compressed(npz_p, x_grid=x_grid, mol_features=np.stack(mol_feats), pred_freq=pf, pred_intensity=pi,
                        pred_mask=pm, target_freq=tf, target_intensity=ti, target_mask=tm,
                        match_target_idx=mi, match_mask=mm, y_pred_spec=np.stack(yps_l), y_target_spec=np.stack(yts_l))
    pd.DataFrame(meta).to_csv(csv_p, index=False)
    return _load_dft_mode_dataset_bundle(npz_p, csv_p)

# Utility Functions

def _safe_array(v, dtype=np.float64):
    return np.nan_to_num(np.asarray(v, dtype=dtype), nan=0, posinf=0, neginf=0)

def _normalize_signal(y):
    y = _safe_array(y)
    m = np.max(y) if y.size else 0
    return y / m if m > EPS else np.zeros_like(y)

def _lorentz_lines_to_spectrum(freq, intensity, x_grid, sigma=12.0):
    """Lorentz broadening — matches spectra_simulator.Lorenz_broadening."""
    freq = np.asarray(freq, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if freq.size == 0:
        return np.zeros_like(x_grid, dtype=np.float32)
    lx = freq[:, None] - x_grid[None, :]  # (n_modes, n_grid)
    ly = (sigma / (2 * np.pi)) / (lx ** 2 + 0.25 * sigma ** 2)
    y = (intensity[:, None] * ly).sum(axis=0)
    return _normalize_signal(y).astype(np.float32)

def _split_indices(n, seed, val_f, test_f):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_t, n_v = int(n * test_f), int(n * val_f)
    return {"train": idx[n_t+n_v:], "val": idx[n_t:n_t+n_v], "test": idx[:n_t]}

def _load_dataset_bundle(npz_p, csv_p, domain):
    data = np.load(npz_p); meta = pd.read_csv(csv_p)
    return AlignmentDatasetBundle(domain, data["x_grid"], data["y_pred"], data["y_target"], 
                                  data["mask"], data["mol_features"], meta, npz_p, csv_p)

def _load_dft_mode_dataset_bundle(npz_path: Path, csv_path: Path) -> DFTModeAlignmentDatasetBundle:
    data = np.load(npz_path)
    meta = pd.read_csv(csv_path)
    return DFTModeAlignmentDatasetBundle(
        domain="dft", x_grid=data["x_grid"], mol_features=data["mol_features"],
        pred_freq=data["pred_freq"], pred_intensity=data["pred_intensity"], pred_mask=data["pred_mask"],
        target_freq=data["target_freq"], target_intensity=data["target_intensity"], target_mask=data["target_mask"],
        match_target_idx=data["match_target_idx"], match_mask=data["match_mask"],
        y_pred_spec=data["y_pred_spec"], y_target_spec=data["y_target_spec"],
        metadata=meta, cache_npz=npz_path, cache_csv=csv_path
    )

# Evaluation Logic

_EVAL_TOLS = (5.0, 10.0, 15.0, 20.0)

def _evaluate_coordinate_alignment(pf, pi, pm, tf, ti, tm, confidence=None, conf_threshold=0.5):
    """
    Coverage-honest evaluation metrics.
    If confidence is provided, only predicted modes with confidence > conf_threshold
    are kept — this is how the v2 confidence head improves precision.
    """
    base = {f"coverage@{int(t)}": 0.0 for t in _EVAL_TOLS}
    base.update({f"cwmae@{int(t)}": float(t) for t in _EVAL_TOLS})
    base.update({f"f1@{int(t)}": 0.0 for t in _EVAL_TOLS})
    base["point_rmse"] = 0.0
    base["intensity_mae"] = 0.0
    base["n_pred_kept"] = 0
    base["n_pred_total"] = 0
    base["n_target"] = 0

    idx_t = tm > 0.5
    if not idx_t.any():
        return base

    t_f, t_i = tf[idx_t], ti[idx_t]
    n_target = len(t_f)
    base["n_target"] = n_target

    # Apply confidence filter if provided
    idx_p = pm > 0.5
    base["n_pred_total"] = int(idx_p.sum())
    if confidence is not None:
        idx_p = idx_p & (confidence > conf_threshold)
    base["n_pred_kept"] = int(idx_p.sum())

    if not idx_p.any():
        return base

    p_f, p_i = pf[idx_p], pi[idx_p]
    n_pred = len(p_f)

    dist = np.abs(p_f[:, None] - t_f[None, :])  # (n_pred, n_target)
    nearest_to_target = dist.min(axis=0)         # for each target, closest pred (cm^-1)

    # Compute Hungarian assignment once — same dist matrix for all tolerances
    p_idx_h, t_idx_h = stats_lib.linear_sum_assignment(dist)

    for t in _EVAL_TOLS:
        ti_int = int(t)
        # coverage@T: fraction of targets with a predicted mode within T
        base[f"coverage@{ti_int}"] = float(np.mean(nearest_to_target <= t))
        # cwmae@T: mean(min(nearest, T)) — honest because unmatched penalised at T
        base[f"cwmae@{ti_int}"] = float(np.minimum(nearest_to_target, t).mean())

        # F1@T via Hungarian matching
        keep = dist[p_idx_h, t_idx_h] <= t
        tp = float(keep.sum())
        fp = float(n_pred - tp)
        fn = float(n_target - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        base[f"f1@{ti_int}"] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Conditional accuracy on @10 matched pairs (reuse same assignment)
    keep10 = dist[p_idx_h, t_idx_h] <= 10.0
    if keep10.any():
        base["point_rmse"] = float(np.sqrt(np.mean(
            (p_f[p_idx_h[keep10]] - t_f[t_idx_h[keep10]]) ** 2)))
        base["intensity_mae"] = float(np.mean(
            np.abs(p_i[p_idx_h[keep10]] - t_i[t_idx_h[keep10]])))

    return base

def run_alignment_study(*, experimental_dataset, dft_dataset, out_dir, device="cpu", train_config=None, **kwargs):
    cfg = train_config or AlignmentTrainConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dft_mol = _augment_mol_features(dft_dataset.mol_features, dft_dataset.metadata, cfg.string_feature_dim)
    splits = _split_indices(len(dft_dataset), cfg.seed, cfg.val_fraction, cfg.test_fraction)

    # -----------------------------------------------------------------------
    # Recompute match indices with training-time cutoff
    # (tighter cutoff = harder classification task = better confidence filtering)
    # -----------------------------------------------------------------------
    tight_mi, tight_mm = _compute_match_indices(
        dft_dataset.pred_freq, dft_dataset.pred_mask,
        dft_dataset.target_freq, dft_dataset.target_mask,
        cutoff=cfg.match_cutoff,
    )
    orig_matched = dft_dataset.match_mask.sum()
    tight_matched = tight_mm.sum()
    match_rate = tight_matched / max(dft_dataset.pred_mask.sum(), 1) * 100
    print(f"Match cutoff={cfg.match_cutoff} cm⁻¹: {tight_matched:.0f} matched modes "
          f"({match_rate:.1f}% of predictions, was {orig_matched:.0f} at 60 cm⁻¹)")

    # -----------------------------------------------------------------------
    # Neural model
    # -----------------------------------------------------------------------
    model = PeakCoordinateTransformer(dft_mol.shape[1], cfg, dft_dataset.x_grid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = ModeArrayDataset(dft_mol[splits["train"]], dft_dataset.pred_freq[splits["train"]],
                                dft_dataset.pred_intensity[splits["train"]], dft_dataset.pred_mask[splits["train"]],
                                dft_dataset.target_freq[splits["train"]], dft_dataset.target_intensity[splits["train"]],
                                dft_dataset.target_mask[splits["train"]], tight_mi[splits["train"]],
                                tight_mm[splits["train"]])
    
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=cfg.lr / 20)
    best_val, patience_cnt = float("inf"), 0
    best_state = None

    # Build tiny val loader for early stopping
    val_ds = ModeArrayDataset(dft_mol[splits["val"]], dft_dataset.pred_freq[splits["val"]],
                              dft_dataset.pred_intensity[splits["val"]], dft_dataset.pred_mask[splits["val"]],
                              dft_dataset.target_freq[splits["val"]], dft_dataset.target_intensity[splits["val"]],
                              dft_dataset.target_mask[splits["val"]], tight_mi[splits["val"]],
                              tight_mm[splits["val"]])
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    print(f"Training on {device} | train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    for epoch in range(cfg.max_epochs):
        model.train()
        l_acc = 0.0
        for b in loader:
            mol, pf, pi, pm, tf, ti, tm, mi, mm = [x.to(device) for x in b]
            opt.zero_grad()
            out = model(mol, pf, pi, pm)
            loss, comp = _supervised_alignment_loss(out, tf, ti, pm, tm, mi, mm, cfg)
            loss.backward()
            opt.step()
            l_acc += loss.item()
        scheduler.step()

        # Validation for early stopping
        model.eval(); v_acc = 0.0; v_comp_acc = {"freq": 0.0, "int": 0.0, "conf": 0.0}
        with torch.no_grad():
            for b in val_loader:
                mol, pf, pi, pm, tf, ti, tm, mi, mm = [x.to(device) for x in b]
                out = model(mol, pf, pi, pm)
                v_loss_b, v_comp = _supervised_alignment_loss(out, tf, ti, pm, tm, mi, mm, cfg)
                v_acc += v_loss_b.item()
                for k in v_comp_acc:
                    v_comp_acc[k] += v_comp.get(k, 0.0)
        n_vb = max(len(val_loader), 1)
        v_loss = v_acc / n_vb
        if v_loss < best_val:
            best_val = v_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if epoch % 10 == 0:
            vc = {k: v / n_vb for k, v in v_comp_acc.items()}
            print(f"Epoch {epoch:4d} | train={l_acc/len(loader):.4f} val={v_loss:.4f} "
                  f"[freq={vc['freq']:.4f} int={vc['int']:.4f} conf={vc['conf']:.4f}] patience={patience_cnt}")
        if patience_cnt >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best checkpoint before eval
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt_path = out_dir / "alignment_model.pth"
    torch.save({"model_state": best_state or model.state_dict(),
                "cfg": cfg, "mol_dim": dft_mol.shape[1],
                "x_grid": dft_dataset.x_grid}, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(dft_mol, device=device),
                     torch.as_tensor(dft_dataset.pred_freq, device=device, dtype=torch.float32),
                     torch.as_tensor(dft_dataset.pred_intensity, device=device, dtype=torch.float32),
                     torch.as_tensor(dft_dataset.pred_mask, device=device, dtype=torch.float32))
        pf_corr = pred["corrected_freq"].cpu().numpy()
        pi_corr = pred["corrected_intensity"].cpu().numpy()
        conf_arr = pred["confidence"].cpu().numpy()

    case_rows = []
    for i in range(len(dft_dataset)):
        # Eval WITHOUT confidence filter — report both filtered and unfiltered
        metrics_all = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
        )
        metrics_conf = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
            confidence=conf_arr[i], conf_threshold=cfg.confidence_threshold,
        )
        row = {"case_index": i, "model": "point_transformer_v2"}
        row.update(metrics_all)
        row.update({f"conf_{k}": v for k, v in metrics_conf.items()})
        case_rows.append(row)
    
    case_df = pd.DataFrame(case_rows)
    case_csv = out_dir / "dft_alignment_cases.csv"; case_df.to_csv(case_csv, index=False)

    # -----------------------------------------------------------------------
    # Intensity-based filtering sweep (no neural model — simple baseline)
    # Find the best top-K or intensity threshold on val, report on test
    # -----------------------------------------------------------------------
    print("\n=== Intensity threshold sweep (val set) ===")
    best_k, best_f1_val = 0, 0.0
    for top_k in [40, 50, 60, 70, 80, 90, 100, 110, 120, 136]:
        f1s = []
        for i in splits["val"]:
            pm_i = dft_dataset.pred_mask[i]
            valid = pm_i > 0.5
            n_valid = valid.sum()
            if n_valid <= top_k:
                # Keep all
                f1s.append(_evaluate_coordinate_alignment(
                    pf_corr[i], pi_corr[i], pm_i,
                    dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                    dft_dataset.target_mask[i])["f1@10"])
                continue
            # Keep top_k by predicted intensity (raw pred, not model output)
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-top_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
            f1s.append(_evaluate_coordinate_alignment(
                pf_corr[i], pi_corr[i], topk_mask,
                dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                dft_dataset.target_mask[i])["f1@10"])
        mean_f1 = np.mean(f1s)
        print(f"  top_k={top_k:3d} | val F1@10={mean_f1:.3f}")
        if mean_f1 > best_f1_val:
            best_f1_val = mean_f1
            best_k = top_k

    # Evaluate best top_k on test set
    test_f1s, test_cov = [], []
    for i in splits["test"]:
        pm_i = dft_dataset.pred_mask[i]
        valid = pm_i > 0.5
        n_valid = valid.sum()
        if n_valid <= best_k:
            topk_mask = pm_i.copy()
        else:
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-best_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
        m = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], topk_mask,
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i])
        test_f1s.append(m["f1@10"])
        test_cov.append(m["coverage@10"])
    print(f"  BEST: top_k={best_k} | test F1@10={np.mean(test_f1s):.3f}  "
          f"Coverage@10={np.mean(test_cov):.3f}")

    summary_rows = []
    for s_name, s_idx in splits.items():
        sub = case_df.iloc[s_idx]
        row = {
            "model": "point_transformer_v2", "split": s_name, "n_cases": len(sub),
            "f1@5": sub["f1@5"].mean(), "f1@10": sub["f1@10"].mean(),
            "f1@15": sub["f1@15"].mean(), "f1@20": sub["f1@20"].mean(),
            "cwmae@10": sub["cwmae@10"].mean(), "cwmae@5": sub["cwmae@5"].mean(),
            "coverage@10": sub["coverage@10"].mean(), "coverage@5": sub["coverage@5"].mean(),
            "point_rmse": sub["point_rmse"].mean(), "intensity_mae": sub["intensity_mae"].mean(),
            "avg_pred_kept": sub["n_pred_kept"].mean(), "avg_pred_total": sub["n_pred_total"].mean(),
            "avg_target": sub["n_target"].mean(),
        }
        # Add confidence-filtered metrics to summary
        for col in ["conf_f1@10", "conf_coverage@10", "conf_cwmae@10", "conf_n_pred_kept"]:
            if col in sub.columns:
                row[col] = sub[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "dft_alignment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    test_row = summary_df[summary_df["split"] == "test"].iloc[0] if "test" in summary_df["split"].values else summary_df.iloc[0]
    kept_pct = test_row['avg_pred_kept'] / (test_row['avg_pred_total'] + 1e-9) * 100
    conf_f1 = test_row.get('conf_f1@10', float('nan'))
    conf_cov = test_row.get('conf_coverage@10', float('nan'))
    conf_kept = test_row.get('conf_n_pred_kept', float('nan'))
    report = (
        f"### DFT Alignment Results (test set)\n"
        f"- Unfiltered: F1@10={test_row['f1@10']:.3f}  Coverage@10={test_row['coverage@10']:.3f}  "
        f"CWMAE@10={test_row['cwmae@10']:.2f} cm⁻¹  ({test_row['avg_pred_kept']:.0f}/{test_row['avg_pred_total']:.0f} modes)\n"
        f"- Filtered:   F1@10={conf_f1:.3f}  Coverage@10={conf_cov:.3f}  "
        f"({conf_kept:.0f} modes kept, threshold={cfg.confidence_threshold})\n"
        f"- Point RMSE (matched@10): {test_row['point_rmse']:.2f} cm⁻¹\n"
        f"- Match cutoff: {cfg.match_cutoff} cm⁻¹  conf_weight: {cfg.confidence_loss_weight}"
    )
    return {
        "domains": {
            "dft": {"best_model": "point_transformer", "summary_csv": str(summary_csv),
                    "case_csv": str(case_csv), "report_markdown": report},
            "experimental": {"best_model": "uncorrected", "summary_csv": str(summary_csv),
                             "report_markdown": "Experimental study results pending."}
        },
        "summary_json": str(out_dir / "summary.json"),
        "checkpoint": str(ckpt_path),
    }

def modal_notebook_guidance(v="/mnt/raman"):
    return f"Projected runtime high. Use `ALIGNMENT_USE_MODAL_VOLUME=1` at {v}"

def _runtime_estimate_minutes(ds, cfg, dev):
    return len(ds) * cfg.max_epochs * 0.0005